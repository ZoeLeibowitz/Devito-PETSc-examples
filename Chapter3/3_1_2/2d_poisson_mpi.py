import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, mmax)

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

from devito.mpi.distributed import MPI

import matplotlib
matplotlib.use("Agg")  # Fully deterministic non-interactive backend
import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# 2D test
# Solving -u_xx - u_yy = f(x,y)
# Dirichlet BCs: u(0,y) = 0, u(1,y)=-e^y, u(x,0) = -x, u(x,1)=-xe
# Manufactured solution: u(x,y) = -xe^(y), with corresponding RHS f(x,y) = xe^(y)
# ref - https://github.com/bueler/p4pdes/blob/master/c/ch6/fish.c

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


sub1 = SubTop()
sub2 = SubBottom()
sub3 = SubLeft()
sub4 = SubRight()

subdomains = (sub1, sub2, sub3, sub4)

def exact(x, y):
    return -x*np.float64(np.exp(y))

Lx = np.float64(1.)
Ly = np.float64(1.)

# n = 9, 17, 33, 65, 129, 257, 513, 1025
n_values = [2**k + 1 for k in range(3, 11)]
h = np.array([Lx/(n-1) for n in n_values])
infinity_norms = []
discrete_l2_norms = []
ksp_iters = []

for n in n_values:
    grid = Grid(
        shape=(n, n), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
    )

    u = Function(name='u', grid=grid, space_order=2)
    f = Function(name='f', grid=grid, space_order=2)
    bc = Function(name='bc', grid=grid, space_order=2)

    eqn = Eq(-u.laplace, f, subdomain=grid.interior)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)

    Y, X = np.meshgrid(tmpx, tmpy)

    f.data[:] = X*np.float64(np.exp(Y))

    bc.data[0, :] = 0.
    bc.data[-1, :] = -np.exp(tmpy)
    bc.data[:, 0] = -tmpx
    bc.data[:, -1] = -tmpx*np.exp(1)

    # # Create boundary condition expressions using subdomains
    bcs = [EssentialBC(u, bc, subdomain=sub1)]
    bcs += [EssentialBC(u, bc, subdomain=sub2)]
    bcs += [EssentialBC(u, bc, subdomain=sub3)]
    bcs += [EssentialBC(u, bc, subdomain=sub4)]

    exprs = [eqn] + bcs

    petsc = petscsolve(
        exprs,
        target=u,
        solver_parameters={'ksp_rtol': 1e-12, 'ksp_type': 'cg', 'pc_type': 'none'},
        options_prefix='poisson_2d'
    )

    # with switchconfig():
    op = Operator(petsc, language='petsc')
    # summary = op.apply()
    op.apply()

    # iters = summary.petsc[('section0', 'poisson_2d')].KSPGetIterationNumber
    # ksp_iters.append(iters)

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
if comm.rank == 0:
    slope, intercept = np.polyfit(np.log(h), np.log(infinity_norms), 1)
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
    plt.savefig(f"3_1_2_mpi_procs{size}.png", dpi=200, metadata={})
    plt.show()


    # TODO: Note, I ran with mpiexec -n 1 to check that the infinity norm computation is the exact same as below (the serial case)
    serial_infinity_norms = [
        np.float64(8.59058043676253e-05),
        np.float64(2.183883155537636e-05),
        np.float64(5.477518565166761e-06),
        np.float64(1.371033316877046e-06),
        np.float64(3.428173114272681e-07),
        np.float64(8.571514120703227e-08),
        np.float64(2.1436735497815107e-08),
        np.float64(5.381981305063732e-09)
    ]

    # taken from output:
    parallel_infinity_norms = [
        np.float64(8.590580436740325e-05),
        np.float64(2.1838831555598404e-05),
        np.float64(5.477518564056538e-06),
        np.float64(1.371033316877046e-06),
        np.float64(3.428173112052235e-07),
        np.float64(8.571514054089846e-08),
        np.float64(2.1436737274171946e-08),
        np.float64(5.3819804168853125e-09)
    ]
