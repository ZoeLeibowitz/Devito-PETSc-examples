import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, mmax)

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# python3 1d_poisson.py -ksp_converged_reason -ksp_type cg

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
errors = []

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

    op = Operator(petsc, language='petsc')
    op.apply()

    u_exact = Function(name='u_exact', grid=grid, space_order=2)
    u_exact.data[:] = exact(X)

    diff = Function(name='diff', grid=grid, space_order=2)
    diff.data[:] = u_exact.data[:] - u.data[:]

    # Compute infinity norm using numpy
    # TODO: Figure out how to compute the infinity norm using Devito
    diff_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
    u_error = diff_norm / np.linalg.norm(u_exact.data[:].ravel(), ord=np.inf)

    errors.append(u_error)


slope, intercept = np.polyfit(np.log(dx), np.log(errors), 1)

assert slope > 1.9
assert slope < 2.1

# Plot
plt.figure(figsize=(6, 5))
plt.loglog(dx, errors, 'o-', label=f'Observed rate ≈ {slope:.2f}', color='orange')
plt.loglog(
    dx, np.exp(intercept) * dx**2,
    'k--',
    label=r'Reference slope $O(\Delta x^2)$'
)
plt.xlabel(r'Grid spacing $\Delta x$')
plt.ylabel(r'Relative $\infty$-norm error')
plt.title('Convergence Plot')

plt.legend()
plt.tight_layout()

plt.savefig("3_1_1.png", dpi=200)

plt.show()
