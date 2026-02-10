import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, mmax)

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt
from devito.mpi.distributed import MPI

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# DEVITO_MPI=1 mpiexec -n 4 python3 2d_laplace_sine_constrain_mpi.py -ksp_converged_reason -ksp_type cg -ksp_rtol 1e-12 -pc_type none

# 2D test
# Solving u_xx + u_yy = 0
# Dirichlet BCs: u(0,y) = 0, u(1,y)=0, u(x,0) = sin(pix), u(x,1)=e^(-pi)*sin(pix)
# ref - https://www.scirp.org/journal/paperinformation?paperid=113731#f2
# example 2 -> note they wrote u(x,1) bc wrong, it should be u(x,1) = e^-pi*sin(pix)
# Analytical solution: u(x,y) = e^(-pi*y)*sin(pi*x)

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
    return np.float64(np.exp(-y*np.pi)) * np.float64(np.sin(np.pi*x))

Lx = np.float64(1.)
Ly = np.float64(1.)

# n = 9, 17, 33, 65, 129, 257, 513, 1025, 2049, 4097
n_values = [2**k + 1 for k in range(3, 13)]
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

    eqn = Eq(u.laplace, f, subdomain=grid.interior)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)

    Y, X = np.meshgrid(tmpx, tmpy)

    f.data[:] = 0.0

    bc.data[:, 0] = np.sin(np.pi*tmpx)
    bc.data[:, -1] = np.exp(-np.pi)*np.sin(np.pi*tmpx)
    bc.data[0, :] = 0.
    bc.data[-1, :] = 0.

    # Create boundary condition expressions using subdomains
    bcs = [EssentialBC(u, bc, subdomain=sub1)]
    bcs += [EssentialBC(u, bc, subdomain=sub2)]
    bcs += [EssentialBC(u, bc, subdomain=sub3)]
    bcs += [EssentialBC(u, bc, subdomain=sub4)]

    exprs = [eqn] + bcs
    petsc = petscsolve(
        exprs,
        target=u,
        solver_parameters={'ksp_rtol': 1e-12, 'ksp_type': 'cg'},
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


size = comm.rank
if comm.rank == 0:
    print(infinity_norms)
    print(ksp_iters)
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
    plt.savefig(f"3_1_4_constrain_mpi_procs{size}.png", dpi=200)


    serial_infinity_norms = [
        np.float64(0.004403459551176603),
        np.float64(0.0011343085906865835),
        np.float64(0.00028433967896346335),
        np.float64(7.114298528454466e-05),
        np.float64(1.779375875687883e-05),
        np.float64(4.448628978415137e-06),
        np.float64(1.1121690737248002e-06),
        np.float64(2.7804420504873306e-07),
        np.float64(6.951107367481058e-08),
        np.float64(1.7377628502845965e-08)
    ]

    # taken from output:
    parallel_infinity_norms = [
        np.float64(0.004403459551172995),
        np.float64(0.001134308590686528),
        np.float64(0.000284339678951695),
        np.float64(7.114298528404506e-05),
        np.float64(1.7793758751716293e-05),
        np.float64(4.448628978193092e-06),
        np.float64(1.1121690760562686e-06),
        np.float64(2.780442084349133e-07),
        np.float64(6.95110903836671e-08),
        np.float64(1.7377738803503462e-08)
    ]

    # check iters are exact same 
    serial_kspiters = [
        7,
        19,
        44,
        96,
        191,
        377,
        746,
        1478,
        2936,
        5844
    ]

    # assert ksp_iters == serial_kspiters

