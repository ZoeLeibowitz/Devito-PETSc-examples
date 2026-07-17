import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)

from devito.petsc import petscsolve, EssentialBC, GridHierarchy
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# 1D test
# Solving -phi.laplace + k^2*phi = f(x)
# Dirichlet BCs: phi(0) = 0, phi(1) = 1
# Manufactured solution: phi(x) = 1 -x^2 - cos(l*pi*x), with corresponding RHS f(x) = 2 + k^2(1 - x^2) - (k^2 + l^2*pi^2)*cos(l*pi*x)
# ref - Jack thesis 

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


def exact(x, l=3.):
    return 1 - x**2 - np.cos(l * np.pi * x)


Lx = np.float64(1.)
k = np.float64(1.)
l = np.float64(3.)

n_values = [2**k + 1 for k in range(4, 15)]
dx = np.array([Lx/(n-1) for n in n_values])

infinity_norms = []
discrete_l2_norms = []
ksp_iters = []

for n in n_values:
    grid = Grid(
        shape=(n,), extent=(Lx,), subdomains=subdomains, dtype=np.float64
    )

    nlevels = int(np.log2(n - 1))
    hierarchy = GridHierarchy(grid, nlevels=nlevels)

    phi = Function(name='phi', grid=grid, space_order=2)
    f = Function(name='f', grid=grid, space_order=2)
    bc = Function(name='bc', grid=grid, space_order=2)

    eqn = Eq(-phi.laplace + k**2 * phi, f, subdomain=grid.interior)

    X = np.linspace(0, Lx, n).astype(np.float64)
    f.data[:] = np.float64(
        2.0 + k**2 * (1 - X**2) - (k**2 + (l * np.pi)**2) * np.cos(l * np.pi * X)
    )

    phi.data[:] = X

    bc.data[0] = np.float64(0.0)  # phi(0) = 0
    bc.data[-1] = np.float64(1.0)  # phi(1) = 1

    # Create boundary condition expressions using subdomains
    bcs = [EssentialBC(phi, bc, subdomain=sub1)]
    bcs += [EssentialBC(phi, bc, subdomain=sub2)]

    exprs = [eqn] + bcs
    # petsc = petscsolve(
    #     exprs, target=phi,
    #     hierarchy=hierarchy,
    #     solver_parameters={
    #         'ksp_rtol': 1e-12,
    #         'ksp_type': 'cg',
    #         'pc_type': 'mg',
    #         'snes_type': 'ksponly',
    #         'mg_levels_ksp_type': 'chebyshev',
    #         'mg_levels_pc_type': 'jacobi',
    #         'mg_coarse_ksp_type': 'gmres',
    #         'mg_coarse_pc_type': 'none',
    #         'mg_levels_ksp_max_it': 4,
    #     },
    #     options_prefix='helmholtz_mg_1d',
    #     constrain_bcs=True

    # )

    # 'pc_mg_type': 'full'
    petsc = petscsolve(
        exprs, target=phi,
        hierarchy=hierarchy,
        solver_parameters={
            'ksp_rtol': 1e-12,
            'ksp_type': 'preonly',
            'pc_type': 'mg',
            'snes_type': 'ksponly',
            'mg_levels_ksp_type': 'chebyshev',
            'mg_levels_pc_type': 'jacobi',
            'mg_levels_ksp_max_it': 2,

            'mg_coarse_ksp_type': 'gmres',
            'mg_coarse_ksp_rtol': 1e-12,
            'mg_coarse_pc_type': 'none',
            
            'pc_mg_type': 'full',
        },
        options_prefix='helmholtz_mg_1d',
        constrain_bcs=True

    )

    with switchconfig(log_level='DEBUG'):
        op = Operator(petsc, language='petsc')
        summary = op.apply()

    iters = summary.petsc[('section0', 'helmholtz_mg_1d')].KSPGetIterationNumber
    ksp_iters.append(iters)

    phi_exact = Function(name='phi_exact', grid=grid, space_order=2)
    phi_exact.data[:] = exact(X, l=l)

    diff = Function(name='diff', grid=grid, space_order=2)
    diff.data[:] = phi_exact.data[:] - phi.data[:]

    # Compute infinity norm using numpy
    # TODO: Figure out how to compute the infinity norm using Devito
    infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
    infinity_norms.append(infinity_norm)

    # Compute discrete L2 norm (RMS error)
    n_interior = np.prod([s - 1 for s in grid.shape])
    discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
    discrete_l2_norms.append(discrete_l2_norm)


slope, intercept = np.polyfit(np.log(dx), np.log(infinity_norms), 1)
print(infinity_norms)
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
plt.savefig("convergence_mg.png", dpi=200)
plt.show()


# KSP iterations vs. problem size
plt.figure(figsize=(6, 5))
plt.semilogx(n_values, ksp_iters, 'o-', color='darkgreen', base=2)
plt.xlabel('n (grid points)')
plt.ylabel('KSP Iterations')
plt.title('KSP Iterations vs. Problem Size (with multigrid)')
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig("iters_vs_n_mg.png", dpi=200)
plt.show()
