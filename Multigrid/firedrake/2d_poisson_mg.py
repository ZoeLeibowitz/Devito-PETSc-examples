import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)

from devito.petsc import petscsolve, EssentialBC, GridHierarchy
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# 2D test
# Solving -u_xx - u_yy = f(x,y)
# copying firedrake test

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
    return np.sin(np.pi*x) * np.tan(np.pi*x*0.25) * np.sin(np.pi*y)


def rhs(x, y):
    return -0.5*np.pi**2*(4*np.cos(np.pi*x) - 5*np.cos(np.pi*x*0.5) + 2)*np.sin(np.pi*y)


Lx = np.float64(1.)
Ly = np.float64(1.)

# n = 9, 17, 33, 65
n_values = [2**k + 1 for k in range(3, 7)]
dx = np.array([Lx/(n-1) for n in n_values])
infinity_norms = []
discrete_l2_norms = []
ksp_iters = []

for n in n_values:
    grid = Grid(
        shape=(n, n), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
    )


    nlevels = int(np.log2(n - 1))
    hierarchy = GridHierarchy(grid, nlevels=nlevels)

    u = Function(name='u', grid=grid, space_order=2)
    f = Function(name='f', grid=grid, space_order=2)
    bc = Function(name='bc', grid=grid, space_order=2)

    eqn = Eq(-u.laplace, f, subdomain=grid.interior)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)

    X, Y = np.meshgrid(tmpx, tmpy, indexing='ij')

    f.data[:] = rhs(X, Y)

    bc.data[0, :] = 0.
    bc.data[-1, :] = 0.
    bc.data[:, 0] = 0.
    bc.data[:, -1] = 0.

    # Create boundary condition expressions using subdomains
    bcs = [EssentialBC(u, bc, subdomain=sub1)]
    bcs += [EssentialBC(u, bc, subdomain=sub2)]
    bcs += [EssentialBC(u, bc, subdomain=sub3)]
    bcs += [EssentialBC(u, bc, subdomain=sub4)]

    exprs = [eqn] + bcs
    petsc = petscsolve(
        exprs, target=u,
        hierarchy=hierarchy,
        solver_parameters={
            'snes_type': 'ksponly',
            'ksp_type': 'preonly',
            'pc_type': 'mg',
            'pc_mg_type': 'full',
            'mg_levels_ksp_type': 'chebyshev',
            'mg_levels_ksp_max_it': 3,
            'mg_levels_pc_type': 'jacobi',

            'mg_coarse_ksp_type': 'gmres',
            'mg_coarse_pc_type': 'none',
        },
        options_prefix='poisson_2d_mg'
    )

    with switchconfig(log_level='DEBUG'):
        op = Operator(petsc, language='petsc')
        summary = op.apply()

    iters = summary.petsc[('section0', 'poisson_2d_mg')].KSPGetIterationNumber
    ksp_iters.append(iters)

    u_exact = Function(name='u_exact', grid=grid, space_order=2)
    u_exact.data[:] = exact(X, Y)

    diff = Function(name='diff', grid=grid, space_order=2)
    diff.data[:] = u_exact.data[:] - u.data[:]

    # Compute infinity norm using numpy
    # TODO: Figure out how to compute the infinity norm using Devito
    infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
    infinity_norms.append(infinity_norm)

    # Compute discrete L2 norm (RMS error)
    n_interior = np.prod([s - 1 for s in grid.shape])
    discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
    discrete_l2_norms.append(discrete_l2_norm)


slope, intercept = np.polyfit(np.log(dx), np.log(infinity_norms), 1)
print("ITERS:", ksp_iters)
print(infinity_norms)
print(f"Observed convergence rate: {slope:.3f}")
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
plt.savefig("convergence_2d_mg.png", dpi=200)
plt.show()

# KSP iterations vs. problem size
plt.figure(figsize=(6, 5))
plt.semilogx(n_values, ksp_iters, 'o-', color='darkgreen', base=2)
plt.xlabel('n (grid points per direction)')
plt.ylabel('KSP Iterations')
plt.title('KSP Iterations vs. Problem Size (FMG, matfree substitute)')
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig("iters_vs_n_2d_mg.png", dpi=200)
plt.show()

