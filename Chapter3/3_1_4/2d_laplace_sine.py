import os
import numpy as np
import matplotlib.pyplot as plt

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, mmax)

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# python3 2d_laplace_sine.py

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
        exprs, target=u,
        solver_parameters={'ksp_rtol': 1e-12, 'ksp_type': 'cg', 'pc_type': 'none'},
        options_prefix='laplace_2d'
    )

    with switchconfig(log_level='DEBUG'):
        op = Operator(petsc, language='petsc')
        summary = op.apply()

    iters = summary.petsc[('section0', 'laplace_2d')].KSPGetIterationNumber
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
plt.title('Convergence Plot')
plt.legend()
plt.tight_layout()
plt.savefig("3_1_4.png", dpi=200)
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
