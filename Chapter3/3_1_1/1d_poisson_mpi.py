import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

from devito.mpi.distributed import MPI

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# DEVITO_MPI=1 mpiexec -n 4 python3 1d_poisson_mpi.py -ksp_converged_reason -ksp_type cg -ksp_rtol 1e-12 -pc_type none

# 1D test
# Solving -u.laplace = f(x)
# Dirichlet BCs: u(0) = -1, u(1) = -e
# Manufactured solution: u(x) = -e^(x), with corresponding RHS f(x) = e^(x)
# ref - https://github.com/bueler/p4pdes/blob/master/c/ch6/fish.c

PetscInitialize()

# Subdomains to implement BCs
class SubLeft(SubDomain):
    name = 'subleft'

    def define(self, dimensions):
        x, = dimensions
        return {x: ('left', 1)}


class SubRight(SubDomain):
    name = 'subright'

    def define(self, dimensions):
        x, = dimensions
        return {x: ('right', 1)}


sub1 = SubLeft()
sub2 = SubRight()

subdomains = (sub1, sub2,)


def exact(x):
    return -np.float64(np.exp(x))


Lx = np.float64(1.)

# n = 9, 17, 33, 65, 129, 257, 513, 1025, 2049, 4097, 8193
n_values = [2**k + 1 for k in range(3, 14)]
dx = np.array([Lx/(n-1) for n in n_values])

infinity_norms = []
discrete_l2_norms = []
ksp_iters = []

for n in n_values:
    grid = Grid(
        shape=(n,), extent=(Lx,), subdomains=subdomains, dtype=np.float64
    )

    u = Function(name='u', grid=grid, space_order=2)
    f = Function(name='f', grid=grid, space_order=2)
    bc = Function(name='bc', grid=grid, space_order=2)

    eqn = Eq(-u.laplace, f, subdomain=grid.interior)

    X = np.linspace(0, Lx, n).astype(np.float64)
    f.data[:] = np.float64(np.exp(X))

    bc.data[0] = -np.float64(1.0)  # u(0) = -1
    bc.data[-1] = -np.float64(np.exp(1.0))  # u(1) = -e

    # Create boundary condition expressions using subdomains
    bcs = [EssentialBC(u, bc, subdomain=sub1)]
    bcs += [EssentialBC(u, bc, subdomain=sub2)]

    exprs = [eqn] + bcs
    # TODO: set ksp type to CG
    petsc = PETScSolve(exprs, target=u, solver_parameters={'ksp_rtol': 1e-12})

    with switchconfig(log_level='DEBUG'):
        op = Operator(petsc, language='petsc')
        summary = op.apply()

    iters = summary.petsc[('section0', None)].KSPGetIterationNumber
    ksp_iters.append(iters)

    u_exact = Function(name='u_exact', grid=grid, space_order=2)
    u_exact.data[:] = exact(X)

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

print(infinity_norms)
print(ksp_iters)
slope, intercept = np.polyfit(np.log(dx), np.log(infinity_norms), 1)

assert slope > 1.9
assert slope < 2.1

# Convergence Plot
plt.figure(figsize=(6, 5))
plt.loglog(dx, infinity_norms, 'o-', label=f'Observed rate ≈ {slope:.3f}', color='orange')
plt.loglog(
    dx, np.exp(intercept) * dx**2,
    'k--',
    label=r'Reference slope $O(h^2)$'
)
plt.xlabel(r'Grid spacing h')
plt.ylabel(r'$\infty$-norm error')
plt.title('Convergence Plot')
plt.legend()
plt.tight_layout()
plt.savefig("3_1_1_mpi.png", dpi=200)
plt.show()

#rerun these
serial_infinity_norms = [
    np.float64(0.00027376999356110154),
    np.float64(6.882733511814898e-05),
    np.float64(1.7233852468212518e-05),
    np.float64(4.309849699124513e-06),
    np.float64(1.0775847982813502e-06),
    np.float64(2.693994516356213e-07),
    np.float64(6.735061530704911e-08),
    np.float64(1.6837652383472346e-08),
    np.float64(4.209424364631786e-09),
    np.float64(1.0523915072724321e-09),
    np.float64(2.6304691758127774e-10)
]

# Taken from output:
devito_mpi_norms = [
    np.float64(0.00027376999356154563),
    np.float64(6.882733511570649e-05),
    np.float64(1.723385246643616e-05),
    np.float64(4.309849703121316e-06),
    np.float64(1.0775847980593056e-06),
    np.float64(2.693994520797105e-07),
    np.float64(6.735060997797859e-08),
    np.float64(1.6837660377078123e-08),
    np.float64(4.209419257605873e-09),
    np.float64(1.0523486526636816e-09),
    np.float64(2.6310287282171885e-10)
]

serial_kspiters = [
    7,
    15,
    31,
    63,
    127,
    255,
    511,
    1023,
    2047,
    4095,
    8191,
]

assert ksp_iters == serial_kspiters
