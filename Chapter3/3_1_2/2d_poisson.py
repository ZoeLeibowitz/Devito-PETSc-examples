import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, mmax)

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# python3 2d_poisson.py -ksp_converged_reason -ksp_type cg

# 2D test
# Solving -u_xx - u_yy = f(x)
# Dirichlet BCs: u(0,y) = 0, u(1,y)=-e^y, u(x,0) = -x, u(x,1)=-xe
# Manufactured solution: u(x,y) = -xe^(y), with corresponding RHS f(x) = xe^(y)
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

# n = 9, 17, 33, 65, 129, 257, 513, 1025, 2049
n_values = [2**k + 1 for k in range(3, 12)]
h = np.array([Lx/(n-1) for n in n_values])
errors = []

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
    # TODO: set ksp type to CG
    petsc = PETScSolve(exprs, target=u, solver_parameters={'ksp_rtol': 1e-12})

    op = Operator(petsc, language='petsc')
    op.apply()

    u_exact = Function(name='u_exact', grid=grid, space_order=2)
    u_exact.data[:] = exact(X, Y)

    diff = Function(name='diff', grid=grid, space_order=2)
    diff.data[:] = u_exact.data[:] - u.data[:]

    # Compute infinity norm using numpy
    # TODO: Figure out how to compute the infinity norm using Devito
    diff_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
    u_error = diff_norm / np.linalg.norm(u_exact.data[:].ravel(), ord=np.inf)
    errors.append(u_error)

slope, intercept = np.polyfit(np.log(h), np.log(errors), 1)

assert slope > 1.9
assert slope < 2.1

# Plot
plt.figure(figsize=(6, 5))
plt.loglog(h, errors, 'o-', label=f'Observed rate ≈ {slope:.2f}', color='orange')
plt.loglog(
    h, np.exp(intercept) * h**2,
    'k--',
    label=r'Reference slope $O(h^2)$'
)
plt.xlabel(r'Grid spacing h')
plt.ylabel(r'Relative $\infty$-norm error')
plt.title('Convergence Plot')

plt.legend()
plt.tight_layout()

plt.savefig("3_1_2.png", dpi=200)

plt.show()
