import os
import numpy as np

from devito import (Grid, Function, TimeFunction, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)
from devito.symbolics import retrieve_functions, INT

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'



# 2D test
# Solving utt = c^2 * uxx + f(x,t)
# ref - https://hplgit.github.io/num-methods-for-PDEs/doc/pub/wave/pdf/wave-4print-A4-2up.pdf?
# ref - file:///Users/zoeleibowitz/Downloads/wave-4print-A4-2up.pdf


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


def exact(x, y, t, Lx=2.5, Ly=2.5):
    return x*(Lx - x)*y*(Ly - y)*(1 + 0.5*t)

def I(x, y):
    return exact(x, y, 0)

def V(x, y):
    return 0.5*exact(x, y, 0)

def f(x, y, t, c=1.5, Lx=2.5, Ly=2.5):
    return 2.*(1. + 0.5*t)*(y*(Ly-y)+ x*(Lx-x))*(c**2)


Lx = np.float64(2.5)
Ly = np.float64(2.5)

n = 7  # Very coarse mesh for this exact test

dx = Lx/(n-1)
dy = Ly/(n-1)


ti = 0.
tf = 18.0
c = 2.0

# For implicit, can use much larger dt
dt = 5.0
print(f"dt: {dt}")

C = c*dt*(1/dx)  # Courant number
nt = int((tf - ti) / dt)

infinity_norms = []
discrete_l2_norms = []
ksp_iters = []


grid = Grid(
    shape=(n, n), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
)

u = TimeFunction(name='u', grid=grid, space_order=2, time_order=2, save=nt+1)
bc = Function(name='bc', grid=grid, space_order=2)

tmpx = np.linspace(0, Lx, n).astype(np.float64)
tmpy = np.linspace(0, Ly, n).astype(np.float64)

Y, X = np.meshgrid(tmpx, tmpy)

u.data[0] = I(X, Y)

lap = (
    u.data[0][1:-1, :-2] +  # left
    u.data[0][1:-1, 2:]  +  # right
    u.data[0][:-2, 1:-1] +  # down
    u.data[0][2:, 1:-1]  -  # up
    4.0 * u.data[0][1:-1, 1:-1]  # center
)

u.data[1][1:-1, 1:-1] = (
    u.data[0][1:-1, 1:-1]
    + dt * V(X[1:-1, 1:-1], Y[1:-1, 1:-1])
    + 0.5 * (C**2) * lap
    + 0.5 * dt**2 * f(X[1:-1, 1:-1], Y[1:-1, 1:-1], 0, c)
)

# u.data[1][0] = 0.
# u.data[1][-1] = 0.

t = grid.time_dim
x,y = grid.dimensions

h_x, h_y = grid.spacing

# Should it be t or t+1? - i think t for explicit, t+1 for implicit
eqn = Eq(u.dt2, (c**2)*u.forward.laplace + 2.*(1. + 0.5*((t+1)*dt))*((y*h_y)*(Ly-(y*h_y))+(x*h_x)*(Lx-(x*h_x)))*(c**2), subdomain=grid.interior)

bc.data[:] = np.float64(0.0)

# Create boundary condition expressions using subdomains
bcs = [EssentialBC(u.forward, bc, subdomain=sub1)]
bcs += [EssentialBC(u.forward, bc, subdomain=sub2)]
bcs += [EssentialBC(u.forward, bc, subdomain=sub3)]
bcs += [EssentialBC(u.forward, bc, subdomain=sub4)]

exprs = [eqn] + bcs
petsc = PETScSolve(
    exprs,
    target=u.forward,
    solver_parameters={'ksp_rtol': 1e-10, 'ksp_type': 'gmres', 'pc_type': 'none'},
    options_prefix='wave_2d_btcs'
)

with switchconfig(log_level='DEBUG'):
    op = Operator(petsc, language='petsc')
    summary = op.apply(dt=dt)
    print(op.arguments(dt=dt))

idx = 3

t_to_compare = idx*dt

u_exact = Function(name='u_exact', grid=grid, space_order=2)
u_exact.data[:] = exact(X, Y, t_to_compare, Lx, Ly)

diff = Function(name='diff', grid=grid, space_order=2)
diff.data[:] = u.data[idx][:] - u_exact.data[:]

# Compute infinity norm using numpy
infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
infinity_norms.append(infinity_norm)

# Compute discrete L2 norm (RMS error)
n_interior = np.prod([s - 1 for s in grid.shape])
discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
discrete_l2_norms.append(discrete_l2_norm)

print(f"Infinity norm: {infinity_norm}")
print(f"Discrete L2 norm: {discrete_l2_norm}")

# save exact solution as plt.imshow plot to file
plt.imshow(diff.data[:].T, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
plt.colorbar(label='u_exact')
plt.title(f'Error at t={t_to_compare:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('wave_2d_btcs_error.png', dpi=300)


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


plt.rcParams.update({
    'font.size': 75, 
    'axes.titlesize': 90,
    'axes.labelsize': 85,
    'xtick.labelsize': 60,
    'ytick.labelsize': 60,   
    'legend.fontsize': 70  
})


# Create large figure
fig = plt.figure(figsize=(75, 35))  # Massive size

# Use GridSpec for layout
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.25)

# Subplots
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
cax = plt.subplot(gs[2])  # For colorbar

# Devito solution
c1 = ax0.contourf(X, Y, u.data[idx][:], levels=100, cmap='viridis')
ax0.set_title(f'Devito Solution at t={t_to_compare:.2f}')
ax0.set_xlabel('$x$')
ax0.set_ylabel('$y$')

# Analytical solution
c2 = ax1.contourf(X, Y, u_exact.data[:], levels=100, cmap='viridis')
ax1.set_title(f'Analytical Solution at t={t_to_compare:.2f}')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')

# Sync color scales
vmin = min(u.data[idx][:].min(), u_exact.data[:].min())
vmax = max(u.data[idx][:].max(), u_exact.data[:].max())
c1.set_clim(vmin, vmax)
c2.set_clim(vmin, vmax)

# Colorbar
cb = fig.colorbar(c2, cax=cax)
cb.set_label('$u$')


for ax in [ax0, ax1]:
    ax.tick_params(axis='x', pad=20)
    ax.tick_params(axis='y', pad=20)

# Layout adjustment
plt.subplots_adjust(left=0.02, right=0.95, top=0.92, bottom=0.12, wspace=0.25)

# Save output
plt.savefig("wave_2d_btcs.png", dpi=200, bbox_inches='tight', pad_inches=0.2)
plt.show()
