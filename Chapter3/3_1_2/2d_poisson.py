import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, mmax)

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

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
    petsc = PETScSolve(
        exprs, target=u,
        solver_parameters={'ksp_rtol': 1e-12, 'ksp_type': 'cg', 'pc_type': 'none'},
        options_prefix='poisson_2d'
    )

    with switchconfig(log_level='DEBUG'):
        op = Operator(petsc, language='petsc')
        summary = op.apply()

    iters = summary.petsc[('section0', 'poisson_2d')].KSPGetIterationNumber
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
plt.title('Convergence Plot')
plt.legend()
plt.tight_layout()
plt.savefig("3_1_2.png", dpi=200)
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
# ./fish -fsh_dim 2 -ksp_type cg -ksp_rtol 1e-12 -pc_type none -ksp_converged_reason -da_refine 2


# I have changed it on the .fish file to print the infinity norm with 'error |u-uexact|_inf = %.22e'
# That way, I can easily adjust the precision shown on my table
petsc4pdes_infinity_norms = [
    8.5905804367625293593846e-05,  # n=9
    2.1838831554932269796154e-05,  # n=17
    5.4775185651667612773963e-06,  # n=33
    1.3710333173211353141596e-06,  # n=65
    3.4281731053908970352495e-07,  # n=129
    8.5715140540898460130848e-08,  # n=257
    2.1436736385993526710081e-08,  # n=513
    5.3819839695989912797813e-09,  # n=1025
]
formatted_petsc4pdes_infinity_norms = [f"{v:.5e}" for v in petsc4pdes_infinity_norms]
formatted_devito_infinity_norms = [f"{v:.5e}" for v in infinity_norms]

petsc4pdes_kspiters = [
    25,
    60,
    124,
    246,
    486,
    956,
    1886,
    3720,
]

# I have changed it on the .fish file to print the infinity norm with '|u-uexact|_h = %.22e\n'
# That way, I can easily adjust the precision shown on my table
petsc4pdes_l2_norms = [
    4.5863862525532358602968e-05,  # n=9
    1.1595157222403673459642e-05,  # n=17
    2.9066440958240641845528e-06,  # n=33
    7.2714812397916602744799e-07,  # n=65
    1.8181741287290787655741e-07,  # n=129
    4.5456251344289833345864e-08,  # n=257
    1.1364183941767951440089e-08,  # n=513
    2.8410842375103498718711e-09,  # n=1025
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
