import os
import numpy as np

import matplotlib.pyplot as plt

from devito.mpi.distributed import MPI
from devito.symbolics import retrieve_functions, INT
from devito import (configuration, Operator, Eq, Grid, Function,
                    SubDomain, switchconfig, norm)
from devito.petsc import petscsolve
from devito.petsc.initialize import PetscInitialize
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

# Modified Helmholtz equation
# Ref - https://www.firedrakeproject.org/demos/helmholtz.py.html


# RUN WITH CG if you're ensuring the matrix is symmetric
# run with: DEVITO_MPI=1 mpiexec -n 4 python3 2d_modified_helmholtz_mpi.py -ksp_converged_reason -ksp_type cg -ksp_rtol 1e-8

PetscInitialize()


Lx = 1.
Ly = Lx

so = 2


class SubTop(SubDomain):
    name = 'subtop'

    def __init__(self, S_O, grid=None):
        self.S_O = S_O
        super().__init__(grid=grid)

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('right', self.S_O//2)}


class SubBottom(SubDomain):
    name = 'subbottom'

    def __init__(self, S_O, grid=None):
        self.S_O = S_O
        super().__init__(grid=grid)

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('left', self.S_O//2)}


class SubLeft(SubDomain):
    name = 'subleft'

    def __init__(self, S_O, grid=None):
        self.S_O = S_O
        super().__init__(grid=grid)

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', self.S_O//2), y: ('middle', 1, 1)}


class SubRight(SubDomain):
    name = 'subright'

    def __init__(self, S_O, grid=None):
        self.S_O = S_O
        super().__init__(grid=grid)

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', self.S_O//2), y: ('middle', 1, 1)}


class SubPointBottomLeft(SubDomain):
    name = 'subpointbottomleft'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: ('left', 1)}


class SubPointBottomRight(SubDomain):
    name = 'subpointbottomright'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: ('left', 1)}


class SubPointTopLeft(SubDomain):
    name = 'subpointtopleft'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: ('right', 1)}


class SubPointTopRight(SubDomain):
    name = 'subpointtopright'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: ('right', 1)}


def neumann_bottom(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get vertical subdimension and its parent
    yfs = subdomain.dimensions[-1]
    y = yfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    mapper = {}
    for f in funcs:
        # Get the y index
        yind = f.indices[-1]
        if (yind - y).as_coeff_Mul()[0] < 0:
            mapper.update({f: f.subs({yind: INT(abs(yind))})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def neumann_top(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get vertical subdimension and its parent
    yfs = subdomain.dimensions[-1]
    y = yfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    mapper = {}
    for f in funcs:
        # Get the y index
        yind = f.indices[-1]
        if (yind - y).as_coeff_Mul()[0] > 0:
            # Symmetric mirror
            tmp = y - INT(abs(y.symbolic_max - yind))
            mapper.update({f: f.subs({yind: tmp})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def neumann_left(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get horizontal subdimension and its parent
    xfs = subdomain.dimensions[0]
    x = xfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    mapper = {}
    for f in funcs:
        # Get the x index
        xind = f.indices[-2]
        if (xind - x).as_coeff_Mul()[0] < 0:
            # Symmetric mirror
            # Substitute where index is negative for +ve where
            # index is positive
            mapper.update({f: f.subs({xind: INT(abs(xind))})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def neumann_right(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get horizontal subdimension and its parent
    xfs = subdomain.dimensions[0]
    x = xfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    mapper = {}
    for f in funcs:
        # Get the x index
        xind = f.indices[-2]
        if (xind - x).as_coeff_Mul()[0] > 0:
            tmp = x - INT(abs(x.symbolic_max - xind))
            mapper.update({f: f.subs({xind: tmp})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def analytical_solution(x, y):
    return np.cos(2*np.pi*x)*np.cos(2*np.pi*y)

n_values = [9, 17, 33, 65, 129, 257, 513, 1025, 2049, 4097]
h = np.array([Lx/(n-1) for n in n_values])
infinity_norms = []


for n in n_values:
    grid = Grid(
        shape=(n, n), extent=(Lx, Ly), dtype=np.float64
    )
    top = SubTop(so, grid=grid)
    bottom = SubBottom(so, grid=grid)
    left = SubLeft(so, grid=grid)
    right = SubRight(so, grid=grid)
    bottom_left = SubPointBottomLeft(grid=grid)
    bottom_right = SubPointBottomRight(grid=grid)
    top_left = SubPointTopLeft(grid=grid)
    top_right = SubPointTopRight(grid=grid)

    time = grid.time_dim
    t = grid.stepping_dim
    x, y = grid.dimensions

    u = Function(name='u', grid=grid, space_order=so, dtype=np.float64)
    f = Function(name='f', grid=grid, space_order=so, dtype=np.float64)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)
    Y, X = np.meshgrid(tmpx, tmpy)
    f.data[:] = (1.+(8.*(np.pi**2)))*np.cos(2.*np.pi*X)*np.cos(2.*np.pi*Y)

    eqn = Eq(-u.laplace+u, f, subdomain=grid.interior)

    # The reason for the added complexity here is to show how to mantain the symmetry of the 
    # matrix system -> can divide by 0.5 or 0.25 if at a corner
    # check using mat convert if it acc is symmetric
    bc_top = neumann_top(eqn, top)
    bc_top = Eq(0.5*bc_top.lhs, 0.5*bc_top.rhs, subdomain=top)
    bc_bottom = neumann_bottom(eqn, bottom)
    bc_bottom = Eq(0.5*bc_bottom.lhs, 0.5*bc_bottom.rhs, subdomain=bottom)
    bc_left = neumann_left(eqn, left)
    bc_left = Eq(0.5*bc_left.lhs, 0.5*bc_left.rhs, subdomain=left)
    bc_right = neumann_right(eqn, right)
    bc_right = Eq(0.5*bc_right.lhs, 0.5*bc_right.rhs, subdomain=right)
    bc_bottom_left = neumann_left(neumann_bottom(eqn, bottom_left), bottom_left)
    bc_bottom_left = Eq(0.25*bc_bottom_left.lhs, 0.25*bc_bottom_left.rhs, subdomain=bottom_left)
    bc_bottom_right = neumann_right(neumann_bottom(eqn, bottom_right), bottom_right)
    bc_bottom_right = Eq(0.25*bc_bottom_right.lhs, 0.25*bc_bottom_right.rhs, subdomain=bottom_right)
    bc_top_left = neumann_left(neumann_top(eqn, top_left), top_left)
    bc_top_left = Eq(0.25*bc_top_left.lhs, 0.25*bc_top_left.rhs, subdomain=top_left)
    bc_top_right = neumann_right(neumann_top(eqn, top_right), top_right)
    bc_top_right = Eq(0.25*bc_top_right.lhs, 0.25*bc_top_right.rhs, subdomain=top_right)
    bcs = [bc_top, bc_bottom, bc_left, bc_right,
           bc_bottom_left, bc_bottom_right, bc_top_left, bc_top_right]

    solver = petscsolve(
        [eqn]+bcs,
        target=u,
        solver_parameters={'ksp_rtol': 1e-8, 'ksp_type': 'cg', 'pc_type': 'none'}
    )

    op = Operator(solver, language='petsc')
    op.apply()

    analytical = Function(name='analytical', grid=grid, space_order=2)
    analytical.data[:] = analytical_solution(X, Y)

    diff = Function(name='diff', grid=grid, space_order=2)
    diff.data[:] = analytical.data[:] - u.data[:]

    gathered = diff.data._gather()
    comm = grid.comm

    if comm is not None and configuration['mpi']:
        if comm != MPI.COMM_NULL and comm.rank == 0:
            infinity_norm_mpi = np.linalg.norm(np.asarray(gathered).ravel(), ord=np.inf)
        else:
            infinity_norm_mpi = None
    else:
        infinity_norm_mpi = None

    infinity_norms.append(infinity_norm_mpi)


size = comm.size
if comm.rank == 0:
    print(infinity_norms)
    slope, intercept = np.polyfit(np.log(h), np.log(infinity_norms), 1)
    assert slope > 1.9
    assert slope < 2.1


    plt.figure(figsize=(6, 5))
    plt.loglog(h, infinity_norms, 'o-', label=f'Observed rate ≈ {slope:.3f}', color='orange')
    plt.loglog(
        h, np.exp(intercept) * h**2,
        'k--',
        label=r'Reference slope $O(h^2)$'
    )
    plt.xlabel(r'Grid spacing h')
    plt.ylabel(r'$\infty$-norm error')
    plt.title(f'Convergence Plot (MPI processes = {size})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"helmholtz_neumann_mpi_procs{size}.png", dpi=200)
    plt.show()

