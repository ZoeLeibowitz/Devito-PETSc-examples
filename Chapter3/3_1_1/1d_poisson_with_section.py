import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain)

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

from devito.mpi.distributed import MPI

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# DEVITO_MPI=1 mpiexec -n 4 python3 1d_poisson_mpi.py

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
# dx = np.array([Lx/(n-1) for n in n_values])


n = 17

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
petsc = petscsolve(
    exprs, target=u,
    solver_parameters={'ksp_rtol': 1e-12, 'ksp_type': 'cg'},
    options_prefix='poisson_1d',
    constrain_bcs=True
)

with switchconfig(log_level='DEBUG'):
    op = Operator(petsc, language='petsc')
    # summary = op.apply()
    print(op.ccode)

# iters = summary.petsc[('section0', 'poisson_1d')].KSPGetIterationNumber
# print(iters)

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


print(f"Infinity Norm={infinity_norm_mpi}")

