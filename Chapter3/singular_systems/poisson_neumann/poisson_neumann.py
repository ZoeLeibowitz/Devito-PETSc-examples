import os
import numpy as np

import matplotlib.pyplot as plt

from devito.symbolics import retrieve_functions, INT
from devito import configuration, Operator, Eq, Grid, Function, SubDomain
from devito.petsc import petscsolve, NullSpace
from devito.petsc.initialize import PetscInitialize
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

# Poisson equation with homogeneous Neumann conditions on the whole boundary,
#   -lap u = f in (0, 1)^2,  grad u . n = 0 on the boundary.
# The operator is singular: its null space, and that of its transpose, are the
# constant vectors. The manufactured solution
#   u = x^2 (1 - x)^2 y^2 (1 - y)^2
# satisfies the boundary conditions exactly and its source integrates to zero,
# but the source weighted by the control volume of each point does not sum to
# zero on the grid, so the right hand side is only compatible up to O(h^2).
# The null space and the null space of the transpose are both passed to
# petscsolve, and the error is measured up to a constant.


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
    return x**2*(1 - x)**2*y**2*(1 - y)**2


def source(x, y):
    # f = -lap u
    return -((12*x**2 - 12*x + 2)*y**2*(1 - y)**2
             + x**2*(1 - x)**2*(12*y**2 - 12*y + 2))


def solve(n, nullspace=None, transpose_nullspace=None, solver_parameters=None):
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

    u = Function(name='u', grid=grid, space_order=so, dtype=np.float64)
    f = Function(name='f', grid=grid, space_order=so, dtype=np.float64)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)
    Y, X = np.meshgrid(tmpx, tmpy)
    f.data[:] = source(X, Y)

    eqn = Eq(-u.laplace, f, subdomain=grid.interior)

    # Boundary rows scaled to keep the Jacobian symmetric, as in the
    # modified Helmholtz problem: one half on the sides, one quarter at
    # the corners
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
        [eqn]+bcs, target=u, nullspace=nullspace,
        transpose_nullspace=transpose_nullspace,
        solver_parameters=solver_parameters,
        options_prefix='poisson_neumann',
        get_info=['kspgetiterationnumber', 'kspgetconvergedreason']
    )

    op = Operator(solver, language='petsc')
    summary = op.apply()
    info = summary.petsc.get_entry('section0', 'poisson_neumann')

    # The solution is only defined up to a constant, so the error is
    # measured up to a constant by removing its mean
    error = u.data[:] - analytical_solution(X, Y)
    infinity_norm = np.abs(error - error.mean()).max()

    return (infinity_norm, info.KSPGetIterationNumber,
            info.KSPGetConvergedReason, np.sum(u.data[:]))


nsp = NullSpace(constant=True)
solver_parameters = {'ksp_rtol': 1e-12, 'ksp_type': 'cg', 'pc_type': 'none'}

n_values = [17, 33, 65, 129, 257, 513]
h = np.array([Lx/(n-1) for n in n_values])


# CG fails without the null spaces, and here also with the null space of the
# operator alone, since the right hand side is not compatible
for name, kwargs in (('no null spaces', {}),
                     ('null space only', {'nullspace': nsp})):
    for n in n_values[:4]:
        _, its, reason, _ = solve(n, solver_parameters=solver_parameters, **kwargs)
        print(f'{name:16s} n = {n:4d}  iterations = {its:5d}  reason = {reason}')
        assert reason < 0

# With both null spaces CG converges, and the solution has zero mean
infinity_norms = []
for n in n_values:
    err, its, reason, total = solve(n, nullspace=nsp, transpose_nullspace=nsp,
                                    solver_parameters=solver_parameters)
    print(f'both null spaces n = {n:4d}  iterations = {its:5d}  reason = {reason}  '
          f'sum(u) = {total:+.1e}  ||e||_inf = {err:.3e}')
    assert reason > 0
    infinity_norms.append(err)

slope, intercept = np.polyfit(np.log(h), np.log(infinity_norms), 1)
print('observed order', slope)
assert slope > 1.9
assert slope < 2.1


plt.figure(figsize=(6, 5))
plt.loglog(h, infinity_norms, 'o-', label=rf'Observed order $\hat{{q}} = {slope:.3f}$', color='orange')
plt.loglog(
    h, np.exp(intercept) * h**2,
    'k--',
    label='Reference line of slope 2'
)
plt.xlabel(r'Grid spacing $h$')
plt.ylabel(r'$\|e\|_\infty$')
plt.legend()
plt.tight_layout()
plt.savefig("poisson_neumann.png", dpi=200)
plt.show()
