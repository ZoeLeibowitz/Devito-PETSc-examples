import os
import numpy as np

from devito import (Grid, Function, TimeFunction, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'



# 1D test
# ref - https://webspace.science.uu.nl/~zegel101/MOLMODWISK/FDheat2.pdf
# finite difference approximations to the heat equation - Gerald W Recktenwald
# comparison between FTCS (forward in time, centered in space), BTCS (backward time, centered in space) and Crank-Nicolson schemes for 1D heat equation
# with some convergence plots

# stable solution


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


def exact(x, t, alpha, L=1.0):
    tmp1 = np.exp((-alpha * (np.pi**2) * t) / L**2) * np.sin(np.pi * x / L)
    return tmp1

Lx = np.float64(1.)

n = 20
dx = Lx/(n-1)

alpha = 0.1
ti = 0.
tf = 1.0

# this ensures r<1/2
dt = 0.01

r = alpha * dt / dx**2
print(f"r = {r}")
assert r < 0.5, "r must be < 0.5 for stability"

# nt will be 100
nt = int((tf - ti) / dt)
print(f"nt = {nt}")

infinity_norms = []
discrete_l2_norms = []
ksp_iters = []

grid = Grid(
    shape=(n,), extent=(Lx,), subdomains=subdomains, dtype=np.float64
)

phi = TimeFunction(name='phi', grid=grid, space_order=2, save=nt+1)
bc = Function(name='bc', grid=grid, space_order=2)

X = np.linspace(0, Lx, n).astype(np.float64)

phi.data[0] = np.sin(np.pi * X / Lx)  # Initial condition

# FTCS scheme
eqn = Eq(phi.dt, alpha * phi.laplace, subdomain=grid.interior)

bc.data[:] = np.float64(0.0)

# Create boundary condition expressions using subdomains
bcs = [EssentialBC(phi.forward, bc, subdomain=sub1)]
bcs += [EssentialBC(phi.forward, bc, subdomain=sub2)]

exprs = [eqn] + bcs
petsc = petscsolve(
    exprs,
    target=phi.forward,
    solver_parameters={'ksp_rtol': 1e-7, 'ksp_type': 'gmres', 'pc_type': 'none'},
    options_prefix='heat_explicit'
)

with switchconfig(log_level='DEBUG'):
    op = Operator(petsc, language='petsc')
    summary = op.apply(dt=dt)


phi_exact = Function(name='phi_exact', grid=grid, space_order=2)
phi_exact.data[:] = exact(X, tf, alpha)

diff = Function(name='diff', grid=grid, space_order=2)
diff.data[:] = phi_exact.data[:] - phi.data[-1]

# Compute norm
n_interior = np.prod([s - 1 for s in grid.shape])
# pretty sure they don't scale the norm in the paper so not doing that here for comparison
# discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
discrete_l2_norm = norm(diff)
discrete_l2_norms.append(discrete_l2_norm)

print(f"Discrete L2 norm: {discrete_l2_norm}")

from matplotlib import pyplot

# Set the font family and size to use for Matplotlib figures.
pyplot.rcParams['font.family'] = 'serif'
pyplot.rcParams['font.size'] = 16

pyplot.figure()
pyplot.xlabel('$x$')
pyplot.ylabel(r'$\phi$')
# add title
# pyplot.title('', fontsize=13)
pyplot.grid(False)
pyplot.plot(X, phi.data[-1], color='brown',linestyle='dotted', linewidth=2, label=f'FTCS at $t={tf}$')
pyplot.plot(X, phi_exact.data[:], color='C1',linewidth=2, label=f'Exact at $t={tf}$')
pyplot.xlim(0.0, 1.)
pyplot.ylim(0.0, 0.4)
pyplot.legend(fontsize=10)


# Save fig
fig_path = '3_2_1_ftcs_stable.png'
pyplot.savefig(fig_path, bbox_inches='tight', dpi=300)