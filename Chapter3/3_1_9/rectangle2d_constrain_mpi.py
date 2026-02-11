import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain)
from devito.symbolics import retrieve_functions, INT
from devito.mpi.distributed import MPI

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize


import matplotlib
matplotlib.use("Agg")  # Fully deterministic non-interactive backend
import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# DEVITO_MPI=1 mpiexec -n 4 python3 rectangle2d_constrain_mpi.py -ksp_converged_reason -ksp_type gmres -pc_type none -ksp_max_it 500000 -ksp_rtol 1e-11

# solves laplace equation in 2D with dirichlet and neumann boundary conditions
# u(0,y) = 0
# u(2,y) = cos(pi*y)
# du/dy(x,0) = 0
# du/dy(x,1) = 0

# yields analytical solution:
# u(x,y) = (1/sinh(2*pi)) * cos(pi*y) * sinh(pi*x)


PetscInitialize()

# Subdomains to implement BCs
class SubTop(SubDomain):
    name = 'subtop'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('right', 1)}


class SubBottom(SubDomain):
    name = 'subbottom'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('left', 1)}


class SubLeft(SubDomain):
    name = 'subleft'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: y}


class SubRight(SubDomain):
    name = 'subright'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: y}
    

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


sub1 = SubTop()
sub2 = SubBottom()
sub3 = SubLeft()
sub4 = SubRight()


subdomains = (sub1, sub2, sub3, sub4)

def exact(x, y):
    tmp1 = np.float64(np.sinh(2.0*np.float64(np.pi)))
    tmp2 = np.float64(np.cos(np.float64(np.pi)*y))
    tmp3 = np.float64(np.sinh(np.float64(np.pi)*x))
    tmp4 = np.float64(1.)/tmp1

    return np.float64(tmp4*tmp2*tmp3)

Lx = np.float64(2.)
Ly = np.float64(1.)


k_vals = range(5, 11)
nx_values = [2**k + 1 for k in k_vals]
ny_values = [int(0.5 * (nx - 1) + 1) for nx in nx_values]

print("nx_values =", nx_values)
print("ny_values =", ny_values)

hx = np.array([Lx/(n-1) for n in nx_values])
hy = np.array([Ly/(n-1) for n in ny_values])
print("hx =", hx)
print("hy =", hy)


h = np.array([Lx/(n-1) for n in nx_values])
infinity_norms = []
discrete_l2_norms = []
ksp_iters = []

for nx, ny in zip(nx_values, ny_values):
    grid = Grid(
        shape=(nx, ny), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
    )

    u = Function(name='u', grid=grid, space_order=2)
    f = Function(name='f', grid=grid, space_order=2)
    bc = Function(name='bc', grid=grid, space_order=2)

    eqn = Eq(u.laplace, f, subdomain=grid.interior)

    tmpx = np.linspace(0, Lx, nx).astype(np.float64)
    tmpy = np.linspace(0, Ly, ny).astype(np.float64)

    X, Y = np.meshgrid(tmpx, tmpy, indexing='ij')

    f.data[:] = 0.0

    bc.data[0, :] = np.float64(0.) # u(0,y) = 0
    bc.data[nx-1, :] = np.float64(np.cos(np.float64(np.pi * tmpy))) # u(2,y) = cos(pi*y)

    # Create boundary condition expressions using subdomains
    bcs = [EssentialBC(u, bc, subdomain=sub3)]
    bcs += [EssentialBC(u, bc, subdomain=sub4)]
    # TODO: maintain symmetry by dividing?
    bcs += [neumann_bottom(eqn, sub2)]
    bcs += [neumann_top(eqn, sub1)]

    exprs = [eqn] + bcs
    petsc = petscsolve(
        exprs,
        target=u,
        solver_parameters={'ksp_rtol': 1e-11, 'ksp_type': 'gmres', 'pc_type': 'none', 'ksp_max_it': 500000},
        constrain_bcs=True
    )

    op = Operator(petsc, language='petsc')
    summary = op.apply()

    u_exact = Function(name='u_exact', grid=grid, space_order=2)
    u_exact.data[:] = exact(X, Y)

    diff = Function(name='diff', grid=grid, space_order=2)
    diff.data[:] = u_exact.data[:] - u.data[:]

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
if comm.rank==0:
    print(infinity_norms)
    # print(ksp_iters)
    slope, intercept = np.polyfit(np.log(h), np.log(infinity_norms), 1)
    print(slope)
    assert slope > 1.9
    assert slope < 2.1

    # Plot
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
    plt.savefig(f"3_1_9_constrain_mpi_procs{size}.png", dpi=200, metadata={})
    plt.show()
