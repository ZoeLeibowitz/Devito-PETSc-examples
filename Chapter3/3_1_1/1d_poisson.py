import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# python3 1d_poisson.py -ksp_converged_reason -ksp_type cg -ksp_rtol 1e-12 -pc_type none

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

    # Compute infinity norm using numpy
    # TODO: Figure out how to compute the infinity norm using Devito
    infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
    infinity_norms.append(infinity_norm)

    # Compute discrete L2 norm (RMS error)
    n_interior = np.prod([s - 1 for s in grid.shape])
    discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
    discrete_l2_norms.append(discrete_l2_norm)


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
plt.savefig("3_1_1.png", dpi=200)
plt.show()



# Error vs iterations plot
plt.figure(figsize=(6, 5))
plt.semilogy(ksp_iters, infinity_norms, 'o-', color='darkgreen')
plt.xlabel('KSP Iterations')
plt.ylabel(r'Infinity Norm Error')
plt.title('Error vs. KSP Iteration Count')
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig("error_vs_iterations.png", dpi=200)
plt.show()


# Make a table comparing solution against petsc4pdes solution
# The command used for comparison:
# export PETSC_ARCH=v3.22.4
# ./fish -fsh_dim 1 -ksp_type cg -ksp_rtol 1e-12 -pc_type none -ksp_converged_reason -da_refine 2


# I have changed it on the .fish file to print the infinity norm with 'error |u-uexact|_inf = %.22e'
# That way, I can easily adjust the precision shown on my table
petsc4pdes_infinity_norms = [
    2.7376999356110154337784e-04,  # n=9
    6.8827335118148980086517e-05,  # n=17
    1.7233852468212518260771e-05,  # n=33
    4.3098496991245127674119e-06,  # n=65
    1.0775847982813502312638e-06,  # n=129
    2.6939945163562128982448e-07,  # n=257
    6.7350615307049110924709e-08,  # n=513
    1.6837652383472345718474e-08,  # n=1025
    4.2094243646317863749573e-09,  # n=2049
    1.0523915072724321362330e-09,  # n=4097
    2.6304691758127773937304e-10,  # n=8193
]
formatted_petsc4pdes_infinity_norms = [f"{v:.5e}" for v in petsc4pdes_infinity_norms]
formatted_devito_infinity_norms = [f"{v:.5e}" for v in infinity_norms]

petsc4pdes_kspiters = [
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


# I have changed it on the .fish file to print the infinity norm with '|u-uexact|_h = %.22e\n'
# That way, I can easily adjust the precision shown on my table
petsc4pdes_l2_norms = [
    2.0088630181614019619592e-04, # n=9
    5.0257888931275251974493e-05, # n=17
    1.2566420594863539537630e-05, # n=33
    3.1417218831170241652548e-06, # n=65
    7.8543768794078292249446e-07, # n=129
    1.9635987459528171788686e-07, # n=257
    4.9089997585334111635677e-08, # n=513
    1.2272499111161400383430e-08, # n=1025
    3.0681294538312826999851e-09, # n=2049
    7.6706227387254335136462e-10, # n=4097
    1.9172366780671051788854e-10, # n=8193
]
formatted_petsc4pdes_l2_norms = [f"{v:.5e}" for v in petsc4pdes_l2_norms]
formatted_devito_l2_norms = [f"{v:.5e}" for v in discrete_l2_norms]


# print infinity norms to screen with a line break
print("Petsc4pdes Infinity Norms: %s\n" % formatted_petsc4pdes_infinity_norms)
print("Devito Infinity Norms: %s\n" % formatted_devito_infinity_norms)

# print l2 discrete norms to screen with a line break
print("Petsc4pdes L2 Norms: %s\n" % formatted_petsc4pdes_l2_norms)
print("Devito L2 Norms: %s\n" % formatted_devito_l2_norms)


assert formatted_petsc4pdes_infinity_norms == formatted_devito_infinity_norms
assert formatted_petsc4pdes_l2_norms == formatted_devito_l2_norms
assert ksp_iters == petsc4pdes_kspiters