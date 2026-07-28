import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, mmax)

from devito.petsc import petscsolve, EssentialBC, GridHierarchy
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# 3D test
# Solving -u_xx - u_yy - u_zz = f(x,y,z)
# Dirichlet BCs: u = 0 on the entire boundary (all 6 faces of the unit cube)
# Manufactured solution (3D extension of MMS4): u(x,y,z) = w(x)w(y)w(z),
# w(t) = t^4 - t^2, with corresponding RHS
#   f(x,y,z) = (2 - 12x^2)w(y)w(z) + (2 - 12y^2)w(x)w(z) + (2 - 12z^2)w(x)w(y)
# ref - PETSc SNES ex5.c tutorial, MMSSolution4/MMSForcing4 with -par 0.0,
# extended to 3D as a tensor product (see mms4.py for the 2D version)

PetscInitialize()

# Subdomains to implement BCs

# Subdomain for z = 1 (top)
class SubTop(SubDomain):
    name = 'subtop'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('middle', 1, 1), y: ('middle', 1, 1), z: ('right', 1)}

# Subdomain for z = 0 (bottom)
class SubBottom(SubDomain):
    name = 'subbottom'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('middle', 1, 1), y: ('middle', 1, 1), z: ('left', 1)}

# Subdomain for y = 1 (back)
class SubBack(SubDomain):
    name = 'subback'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('middle', 1, 1), y: ('right', 1), z: z}

# Subdomain for y = 0 (front)
class SubFront(SubDomain):
    name = 'subfront'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('middle', 1, 1), y: ('left', 1), z: z}

# Subdomain for x = 0 (left)
class SubLeft(SubDomain):
    name = 'subleft'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('left', 1), y: y, z: z}

# Subdomain for x = 1 (right)
class SubRight(SubDomain):
    name = 'subright'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('right', 1), y: y, z: z}


sub1 = SubTop()
sub2 = SubBottom()
sub3 = SubLeft()
sub4 = SubRight()
sub5 = SubBack()
sub6 = SubFront()

subdomains = (sub1, sub2, sub3, sub4, sub5, sub6)

def w(t):
    return t**4 - t**2

def exact(x, y, z):
    return w(x)*w(y)*w(z)

Lx = np.float64(1.)
Ly = np.float64(1.)
Lz = np.float64(1.)

# n = 17, 33, 65, 129
n_values = [2**k + 1 for k in range(6, 9)]
h = np.array([Lx/(n-1) for n in n_values])
infinity_norms = []
discrete_l2_norms = []
ksp_iters = []

for n in n_values:
    grid = Grid(
        shape=(n, n, n), extent=(Lx, Ly, Lz), subdomains=subdomains, dtype=np.float64
    )

    nlevels = int(np.log2(n - 1))
    print(f"n = {n}, nlevels = {nlevels}")
    hierarchy = GridHierarchy(grid, nlevels=nlevels)

    u = Function(name='u', grid=grid, space_order=2)
    f = Function(name='f', grid=grid, space_order=2)
    bc = Function(name='bc', grid=grid, space_order=2)

    eqn = Eq(-u.laplace, f, subdomain=grid.interior)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)
    tmpz = np.linspace(0, Lz, n).astype(np.float64)

    X, Y, Z = np.meshgrid(tmpx, tmpy, tmpz, indexing="ij")

    f.data[:] = (2 - 12*X**2)*w(Y)*w(Z) + (2 - 12*Y**2)*w(X)*w(Z) + (2 - 12*Z**2)*w(X)*w(Y)

    # u = 0 on the entire boundary
    bc.data[:] = 0.

    # # Create boundary condition expressions using subdomains
    bcs = []
    bcs += [EssentialBC(u, bc, subdomain=sub1)]  # top
    bcs += [EssentialBC(u, bc, subdomain=sub2)]  # bottom
    bcs += [EssentialBC(u, bc, subdomain=sub3)]  # subleft
    bcs += [EssentialBC(u, bc, subdomain=sub4)]  # subright
    bcs += [EssentialBC(u, bc, subdomain=sub5)]  # subback
    bcs += [EssentialBC(u, bc, subdomain=sub6)]  # subfront

    exprs = [eqn] + bcs
    petsc = petscsolve(
        exprs, target=u,
        solver_parameters={
            # Same single-full-multigrid-cycle setup as mms4.py, extended to 3D:
            # ksponly + loose ksp_atol so the default outer KSP converges in a
            # single application of the FMG-preconditioned solve.
            'snes_type': 'ksponly',
            'ksp_atol': 1e-1,
            'pc_type': 'mg',
            'pc_mg_type': 'full',

            'mg_levels_ksp_type': 'chebyshev',
            'mg_levels_pc_type': 'jacobi',
            'mg_levels_ksp_max_it': 5,

            'mg_coarse_ksp_type': 'gmres',
            'mg_coarse_ksp_rtol': 1e-12,
            'mg_coarse_pc_type': 'none',
        },
        options_prefix='mms4_3d',
        hierarchy=hierarchy,
        # constrain_bcs=True

    )

    with switchconfig(log_level='DEBUG'):
        op = Operator(petsc, language='petsc')
        summary = op.apply()

    iters = summary.petsc[('section0', 'mms4_3d')].KSPGetIterationNumber
    ksp_iters.append(iters)

    u_exact = Function(name='u_exact', grid=grid, space_order=2)
    u_exact.data[:] = exact(X, Y, Z)

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


slope, intercept = np.polyfit(np.log(h), np.log(infinity_norms), 1)

print("infinity_norms:", infinity_norms)
print("ksp_iters:", ksp_iters)
print("slope:", slope)

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
plt.title('Convergence Plot')
plt.legend()
plt.tight_layout()
plt.savefig("mms4_3d.png", dpi=200)
plt.show()
