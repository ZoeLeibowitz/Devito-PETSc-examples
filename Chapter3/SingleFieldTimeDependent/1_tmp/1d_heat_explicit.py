import os
import numpy as np

from devito import (Grid, Function, TimeFunction, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)
from devito.symbolics import retrieve_functions, INT

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# python3 1d_heat_explicit.py

# 1D test
# Solving T.dt = alpha * T.laplace + sigma(x, t)
# Dirichlet BCs: T(0, t) = 0, T(L, t) = 0
# Initial condition: T_0(x) = sin(2pix)
# Manufactured solution: T(x,t) = exp(-4*pi^2*alpha*t)*sin(2pix) + (2/(pi^2*alpha))*(1-exp(-pi^2*alpha*t))*sin(pix), with corresponding RHS sigma(x, t) = 2sin(pix)
# ref - https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/04_PartialDifferentialEquations/04_03_Diffusion_Explicit.html

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


def exact(x, t, alpha):
    tmp1 = np.exp(-4.0 * np.pi**2 * alpha * t) * np.sin(2.0 * np.pi * x)
    tmp2 = (2.0 / (np.pi**2 * alpha)) * (1.0 - np.exp(-np.pi**2 * alpha * t)) * np.sin(np.pi * x)
    return tmp1 + tmp2

Lx = np.float64(1.)

# n = 9, 17, 33, 65, 129, 257, 513, 1025, 2049, 4097, 8193
n_values = [2**k + 1 for k in range(3, 14)]
n_values = [21]

dx = np.array([Lx/(n-1) for n in n_values])

alpha = 0.1
ti = 0.
tf = 5.0
# For this explicit scheme, fourier=0.53 becomes unstable
fourier = 0.49
dt = fourier * dx[0]**2 / alpha
nt = int((tf - ti) / dt)

infinity_norms = []
discrete_l2_norms = []
ksp_iters = []

for n in n_values:
    grid = Grid(
        shape=(n,), extent=(Lx,), subdomains=subdomains, dtype=np.float64
    )

    T = TimeFunction(name='T', grid=grid, space_order=2, save=nt+1)
    bc = Function(name='bc', grid=grid, space_order=2)
    sigma = Function(name='sigma', grid=grid, space_order=2)

    X = np.linspace(0, Lx, n).astype(np.float64)

    T.data[0] = np.sin(2.0 * np.pi * X)  # Initial condition

    eqn = Eq(T.dt, alpha*T.laplace + sigma, subdomain=grid.interior)

    bc.data[:] = np.float64(0.0)

    sigma.data[:] = 2.0 * np.sin(np.pi * X)

    # Create boundary condition expressions using subdomains
    bcs = [EssentialBC(T.forward, bc, subdomain=sub1)]
    bcs += [EssentialBC(T.forward, bc, subdomain=sub2)]

    exprs = [eqn] + bcs
    petsc = petscsolve(
        exprs,
        target=T.forward,
        solver_parameters={'ksp_rtol': 1e-10, 'ksp_type': 'gmres', 'pc_type': 'none'},
        options_prefix='heat_explicit'
    )

    with switchconfig(log_level='DEBUG'):
        op = Operator(petsc, language='petsc')
        summary = op.apply(dt=dt)
        print(op.ccode)

    # u_exact = Function(name='u_exact', grid=grid, space_order=2)
    # u_exact.data[:] = exact(X)

    # diff = Function(name='diff', grid=grid, space_order=2)
    # diff.data[:] = u_exact.data[:] - u.data[:]

#     # Compute infinity norm using numpy
#     # TODO: Figure out how to compute the infinity norm using Devito
#     infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
#     infinity_norms.append(infinity_norm)

#     # Compute discrete L2 norm (RMS error)
#     n_interior = np.prod([s - 1 for s in grid.shape])
#     discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
#     discrete_l2_norms.append(discrete_l2_norm)


u_exact = Function(name='u_exact', grid=grid, space_order=2)
u_exact.data[:] = exact(X, 5.0, alpha)

from matplotlib import pyplot

# Set the font family and size to use for Matplotlib figures.
pyplot.rcParams['font.family'] = 'serif'
pyplot.rcParams['font.size'] = 16


# Plot the temperature along the rod.
pyplot.figure(figsize=(10.0, 5.0))
pyplot.xlabel('Distance [m]')
pyplot.ylabel('Temperature [C]')
# add title
pyplot.title('Heat Transport with Forward Euler Scheme - forward finite differences', fontsize=13)
pyplot.grid(False)
pyplot.plot(X, T.data[0], color='C2', linewidth=2, label='Initial condition')
pyplot.plot(X, T.data[-1], color='brown',linewidth=2, label=f'$t={tf}$')
pyplot.plot(X, u_exact.data[:], color='C1', linestyle='dotted', linewidth=2, label='Exact solution at $t=5$')
pyplot.xlim(0.0, 1.)
pyplot.ylim(-1.2, 2.3)
pyplot.legend(fontsize=10)

# Save fig
fig_path = '1d_heat_explicit.png'
pyplot.savefig(fig_path, bbox_inches='tight', dpi=300)