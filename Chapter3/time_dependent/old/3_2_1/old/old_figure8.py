import os
import numpy as np

from devito import (Grid, Function, TimeFunction, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

# NOTE: Since reproducing figure 8 from the reference, I don't think it is a great plot and therefore
# will not use it - using file spatial_convergence.py instead to reproduce plot from different reference.

# 1D test
# ref - https://webspace.science.uu.nl/~zegel101/MOLMODWISK/FDheat2.pdf
# finite difference approximations to the heat equation - Gerald W Recktenwald
# comparison between FTCS (forward in time, centered in space), BTCS (backward time, centered in space) and Crank-Nicolson schemes for 1D heat equation
# with some convergence plots

# This file reproduces figure 8 from the reference paper


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



n_values = [4, 7, 17, 33, 65, 129, 335]
dx = np.array([Lx/(n-1) for n in n_values])

alpha = 0.1
ti = 0.
tf = 0.5
# Doing a spatial convergence test, so keep dt constant and small so that
# errors from time discretization don't domainate
dt = 5.e-6

ftcs_l2_norms = []
btcs_l2_norms = []
cn_l2_norms = []

nt = int((tf + dt) / dt)
print(f"nt: {nt}")


for n in n_values:

    grid = Grid(
        shape=(n,), extent=(Lx,), subdomains=subdomains, dtype=np.float64
    )

    u0 = TimeFunction(name='u0', grid=grid, space_order=2, save=nt)
    u1 = TimeFunction(name='u1', grid=grid, space_order=2, save=nt)
    u2 = TimeFunction(name='u2', grid=grid, space_order=2, save=nt)

    bc = Function(name='bc', grid=grid, space_order=2)

    X = np.linspace(0, Lx, n).astype(np.float64)

    u0.data[0] = np.sin(np.pi * X / Lx)  # Initial condition
    u1.data[0] = np.sin(np.pi * X / Lx)  # Initial condition
    u2.data[0] = np.sin(np.pi * X / Lx)  # Initial condition

    # FTCS scheme
    eqn0 = Eq(u0.dt, alpha * u0.laplace, subdomain=grid.interior)

    # BTCS scheme
    eqn1 = Eq(u1.dt, alpha * u1.forward.laplace, subdomain=grid.interior)

    # CN scheme
    eqn2 = Eq(u2.dt, (alpha/2.)*(u2.laplace + u2.forward.laplace), subdomain=grid.interior)

    bc.data[:] = np.float64(0.0)

    # Create boundary condition expressions using subdomains
    bcs0 = [EssentialBC(u0.forward, bc, subdomain=sub1)]
    bcs0 += [EssentialBC(u0.forward, bc, subdomain=sub2)]

    bcs1 = [EssentialBC(u1.forward, bc, subdomain=sub1)]
    bcs1 += [EssentialBC(u1.forward, bc, subdomain=sub2)]

    bcs2 = [EssentialBC(u2.forward, bc, subdomain=sub1)]
    bcs2 += [EssentialBC(u2.forward, bc, subdomain=sub2)]

    ftcs_exprs = [eqn0] + bcs0
    ftcs_solver = petscsolve(
        ftcs_exprs,
        target=u0.forward,
        solver_parameters={'ksp_rtol': 1e-7, 'pc_type': 'none'},
        options_prefix='ftcs'
    )

    btcs_exprs = [eqn1] + bcs1
    btcs_solver = petscsolve(
        btcs_exprs,
        target=u1.forward,
        solver_parameters={'ksp_rtol': 1e-7, 'pc_type': 'none'},
        options_prefix='btcs'
    )

    cn_exprs = [eqn2] + bcs2
    cn_solver = petscsolve(
        cn_exprs,
        target=u2.forward,
        solver_parameters={'ksp_rtol': 1e-7, 'pc_type': 'none'},
        options_prefix='cn'
    )

    with switchconfig(log_level='DEBUG'):
        op = Operator([ftcs_solver, btcs_solver, cn_solver], language='petsc')
        summary = op.apply(dt=dt)

    u_exact = Function(name='u_exact', grid=grid, space_order=2)
    u_exact.data[:] = exact(X, tf, alpha)

    diff0 = Function(name='diff0', grid=grid, space_order=2)
    diff0.data[:] = u_exact.data[:] - u0.data[-1]

    diff1 = Function(name='diff1', grid=grid, space_order=2)
    diff1.data[:] = u_exact.data[:] - u1.data[-1]

    diff2 = Function(name='diff2', grid=grid, space_order=2)
    diff2.data[:] = u_exact.data[:] - u2.data[-1]

    # Compute norm
    n_interior = np.prod([s - 1 for s in grid.shape])

    # ftcs_l2_norm = norm(diff0) / np.sqrt(n_interior)
    ftcs_l2_norm = norm(diff0)
    ftcs_l2_norms.append(ftcs_l2_norm)

    # btcs_l2_norm = norm(diff1) / np.sqrt(n_interior)
    btcs_l2_norm = norm(diff1)
    btcs_l2_norms.append(btcs_l2_norm)

    # cn_l2_norm = norm(diff2) / np.sqrt(n_interior)
    cn_l2_norm = norm(diff2)
    cn_l2_norms.append(cn_l2_norm)

    print(f"FTCS discrete L2 norm: {ftcs_l2_norm}")
    print(f"BTCS discrete L2 norm: {btcs_l2_norm}")
    print(f"CN discrete L2 norm: {cn_l2_norm}")

from matplotlib import pyplot

# Set the font family and size to use for Matplotlib figures.
pyplot.rcParams['font.family'] = 'serif'
pyplot.rcParams['font.size'] = 16


# Make a convergence plot

print(ftcs_l2_norms)
ftcs_slope, ftcs_intercept = np.polyfit(np.log(dx), np.log(ftcs_l2_norms), 1)
btcs_slope, btcs_intercept = np.polyfit(np.log(dx), np.log(btcs_l2_norms), 1)
cn_slope, cn_intercept = np.polyfit(np.log(dx), np.log(cn_l2_norms), 1)


# Convergence Plot
plt.figure(figsize=(6, 5))
plt.loglog(dx, ftcs_l2_norms, 'o', markersize=4, label=f'FTCS', color='orange')
plt.loglog(dx, btcs_l2_norms, 'o', markersize=4, label=f'BTCS', color='blue')
plt.loglog(dx, cn_l2_norms, 'o', markersize=4, label=f'CN', color='green')

x0 = dx[2]
y0 = cn_l2_norms[2]


ideal_slope = y0 * (dx / x0)**2

plt.loglog(
    dx, ideal_slope,
    'k--',
    label=r'ideal'
)
plt.xlabel(r'$\Delta x$')
plt.ylabel(r'L2 error norm')
plt.title('Convergence Plot')
plt.xlim(10e-4, 1.)
plt.ylim(10e-6, 10e-2)
# pyplot.legend(fontsize=8)
# make the y axis go up in 0.1
# plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(fontsize=8)
plt.tight_layout()

# Save fig
fig_path = 'comparison_figure8.png'
pyplot.savefig(fig_path, bbox_inches='tight', dpi=300)



