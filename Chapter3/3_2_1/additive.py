import os
import numpy as np

from devito import (Grid, Function, TimeFunction, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt
from matplotlib import pyplot

# Set the font family and size to use for Matplotlib figures.
pyplot.rcParams['font.family'] = 'serif'
pyplot.rcParams['font.size'] = 16

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# 1D test

# https://web.cecs.pdx.edu/~gerry/class/ME448/notes/1Dmodels/pdf/CN_slides.pdf


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
    return np.exp((-alpha * (np.pi**2) * t) / L**2) * np.sin(np.pi * x / L)

Lx = np.float64(1.)

nts = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6096, 8096, 10096, 12096])
nxs = [400, 800, 1600, 3200, 6400, 12800]


alpha = 0.1
ti = 0.
tf = 2.0


dt_values = tf / (nts - 1)

plt.figure(figsize=(6, 5))


colors = ['red', 'blue', 'black', 'magenta', 'red', 'blue']
markerfacecolors = ['none', 'blue', 'none', 'none', 'none', 'none']
markers = ['o', '*', 'v', 'x', 'd', 'p']

for i, n in enumerate(nxs):

    print(f"nx is: {n}")

    cn_l2_norms = []

    for dt in dt_values:

        nt = int((tf + dt) / dt) - 1
        print(f"nt: {nt}")


        grid = Grid(
            shape=(n,), extent=(Lx,), subdomains=subdomains, dtype=np.float64
        )

        phi = TimeFunction(name='phi', grid=grid, space_order=2, save=nt+1)

        bc = Function(name='bc', grid=grid, space_order=2)

        X = np.linspace(0, Lx, n).astype(np.float64)

        phi.data[0] = np.sin(np.pi * X / Lx)  # Initial condition

        # CN scheme
        eqn = Eq(phi.dt, (alpha/2.)*(phi.laplace + phi.forward.laplace), subdomain=grid.interior)

        bc.data[:] = np.float64(0.0)

        # Create boundary condition expressions using subdomains
        bcs = [EssentialBC(phi.forward, bc, subdomain=sub1)]
        bcs += [EssentialBC(phi.forward, bc, subdomain=sub2)]

        cn_exprs = [eqn] + bcs
        cn_solver = petscsolve(
            cn_exprs,
            target=phi.forward,
            solver_parameters={'ksp_rtol': 1e-11, 'pc_type': 'none', 'ksp_type': 'cg'},
            options_prefix='cn'
        )

        with switchconfig(log_level='DEBUG'):
            op = Operator([cn_solver], language='petsc')
            summary = op.apply(dt=dt)

        phi_exact = Function(name='phi_exact', grid=grid, space_order=2)
        phi_exact.data[:] = exact(X, tf, alpha)

        diff = Function(name='diff', grid=grid, space_order=2)
        diff.data[:] = phi_exact.data[:] - phi.data[-1]

        # Compute norm
        n_interior = np.prod([s - 1 for s in grid.shape])
        cn_l2_norm = norm(diff) / np.sqrt(n_interior)
        cn_l2_norms.append(cn_l2_norm)

        print(f"CN discrete L2 norm: {cn_l2_norm}")

    plt.loglog(dt_values, cn_l2_norms, marker=markers[i], markerfacecolor=markerfacecolors[i], markersize=4, linestyle=':', label=f'nx = {n}', color=colors[i])



cn_slope, cn_intercept = np.polyfit(np.log(dt_values), np.log(cn_l2_norms), 1)


dt0 = dt_values[0]
y0 = cn_l2_norms[0]
quad_ideal = y0 * (dt_values / dt0)**2


plt.loglog(
    dt_values, quad_ideal,
    color='red',
    linestyle='--',
    label=r'$E \propto \Delta t^2$'
)


plt.xlabel(r'$\Delta t$', fontsize=12)
plt.ylabel(r'Error: $\|\phi - \phi_e\|_2$', fontsize=12)

# place legend in lower right corner
plt.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.tick_params(axis='both', which='major', labelsize=8)

# Save fig
fig_path = '3_2_1_additive.png'
pyplot.savefig(fig_path, bbox_inches='tight', dpi=300)
