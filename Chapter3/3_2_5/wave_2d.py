import os
import numpy as np

from devito import (Grid, Function, TimeFunction, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)
from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# 2D test
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

c = 1.5
stability_limit = (1/np.float64(c))*(1/np.sqrt(1/dx**2 + 1/dy**2))

# Using a dt that satisfies the stability limit for FTCS scheme. Even though could use bigger time
# step for BTCS scheme, we will keep it the same for simplicity.
dt = stability_limit

C = c*dt*(1/dx)  # Courant number
nt = int((tf - ti) / dt)


grid = Grid(
    shape=(n, n), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
)

# One function for explicit FTCS scheme, one function for implicit BTCS scheme
u0 = TimeFunction(name='u0', grid=grid, space_order=2, time_order=2, save=nt+1)
u1 = TimeFunction(name='u1', grid=grid, space_order=2, time_order=2, save=nt+1)

bc = Function(name='bc', grid=grid, space_order=2)

tmpx = np.linspace(0, Lx, n).astype(np.float64)
tmpy = np.linspace(0, Ly, n).astype(np.float64)

Y, X = np.meshgrid(tmpx, tmpy)

u0.data[0] = I(X, Y)
u1.data[0] = I(X, Y)

lap0 = (
    u0.data[0][1:-1, :-2] +  # left
    u0.data[0][1:-1, 2:]  +  # right
    u0.data[0][:-2, 1:-1] +  # down
    u0.data[0][2:, 1:-1]  -  # up
    4.0 * u0.data[0][1:-1, 1:-1]  # center
)

lap1 = (
    u1.data[0][1:-1, :-2] +  # left
    u1.data[0][1:-1, 2:]  +  # right
    u1.data[0][:-2, 1:-1] +  # down
    u1.data[0][2:, 1:-1]  -  # up
    4.0 * u1.data[0][1:-1, 1:-1]  # center
)

u0.data[1][1:-1, 1:-1] = (
    u0.data[0][1:-1, 1:-1]
    + dt * V(X[1:-1, 1:-1], Y[1:-1, 1:-1])
    + 0.5 * (C**2) * lap0
    + 0.5 * dt**2 * f(X[1:-1, 1:-1], Y[1:-1, 1:-1], 0, c)
)

u1.data[1][1:-1, 1:-1] = (
    u1.data[0][1:-1, 1:-1]
    + dt * V(X[1:-1, 1:-1], Y[1:-1, 1:-1])
    + 0.5 * (C**2) * lap1
    + 0.5 * dt**2 * f(X[1:-1, 1:-1], Y[1:-1, 1:-1], 0, c)
)


t = grid.time_dim
x,y = grid.dimensions

h_x, h_y = grid.spacing

# t for explicit
eqn_ftcs = Eq(u0.dt2, (c**2)*u0.laplace + 2.*(1. + 0.5*(t*dt))*((y*h_y)*(Ly-(y*h_y))+(x*h_x)*(Lx-(x*h_x)))*(c**2), subdomain=grid.interior)
eqn_btcs = Eq(u1.dt2, (c**2)*u1.forward.laplace + 2.*(1. + 0.5*((t+1)*dt))*((y*h_y)*(Ly-(y*h_y))+(x*h_x)*(Lx-(x*h_x)))*(c**2), subdomain=grid.interior)


bc.data[:] = np.float64(0.0)

# Create boundary condition expressions using subdomains
bcs0 = [EssentialBC(u0.forward, bc, subdomain=sub1)]
bcs0 += [EssentialBC(u0.forward, bc, subdomain=sub2)]
bcs0 += [EssentialBC(u0.forward, bc, subdomain=sub3)]
bcs0 += [EssentialBC(u0.forward, bc, subdomain=sub4)]

bcs1 = [EssentialBC(u1.forward, bc, subdomain=sub1)]
bcs1 += [EssentialBC(u1.forward, bc, subdomain=sub2)]
bcs1 += [EssentialBC(u1.forward, bc, subdomain=sub3)]
bcs1 += [EssentialBC(u1.forward, bc, subdomain=sub4)]

exprs_ftcs = [eqn_ftcs] + bcs0
ftcs_solver = petscsolve(
    exprs_ftcs,
    target=u0.forward,
    solver_parameters={'ksp_rtol': 1e-12, 'ksp_type': 'gmres', 'pc_type': 'none'},
    options_prefix='wave_explicit'
)

exprs_btcs = [eqn_btcs] + bcs1
btcs_solver = petscsolve(
    exprs_btcs,
    target=u1.forward,
    solver_parameters={'ksp_rtol': 1e-12, 'ksp_type': 'gmres', 'pc_type': 'none'},
    options_prefix='wave_implicit'
)


with switchconfig(log_level='DEBUG'):
    op = Operator([ftcs_solver, btcs_solver], language='petsc')
    summary = op.apply(dt=dt)
    print(op.arguments(dt=dt))

idx = 91
t_to_compare = idx*dt

u_exact = Function(name='u_exact', grid=grid, space_order=2)
u_exact.data[:] = exact(X, Y, t_to_compare, Lx, Ly)

diff_ftcs = Function(name='diff_ftcs', grid=grid, space_order=2)
diff_ftcs.data[:] = u0.data[idx][:] - u_exact.data[:]

diff_btcs = Function(name='diff_btcs', grid=grid, space_order=2)
diff_btcs.data[:] = u1.data[idx][:] - u_exact.data[:]

# Compute infinity norm using numpy
infinity_norm_ftcs = np.linalg.norm(diff_ftcs.data[:].ravel(), ord=np.inf)
infinity_norm_btcs = np.linalg.norm(diff_btcs.data[:].ravel(), ord=np.inf)

# Compute discrete L2 norm (RMS error)
n_interior = np.prod([s - 1 for s in grid.shape])
discrete_l2_norm_ftcs = norm(diff_ftcs) / np.sqrt(n_interior)
discrete_l2_norm_btcs = norm(diff_btcs) / np.sqrt(n_interior)

print(f"FTCS infinity norm: {infinity_norm_ftcs}")
print(f"FTCS discrete L2 norm: {discrete_l2_norm_ftcs}")

print(f"BTCS infinity norm: {infinity_norm_btcs}")
print(f"BTCS discrete L2 norm: {discrete_l2_norm_btcs}")


fig = plt.figure(figsize=(18, 6))
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.25)

# Subplots
ax0 = plt.subplot(gs[0])  # FTCS
ax1 = plt.subplot(gs[1])  # BTCS
ax2 = plt.subplot(gs[2])  # Analytical
cax = plt.subplot(gs[3])  # Colorbar

vmin = min(u0.data[idx][:].min(), u1.data[idx][:].min(), u_exact.data[:].min())
vmax = max(u0.data[idx][:].max(), u1.data[idx][:].max(), u_exact.data[:].max())

# FTCS Solution
c1 = ax0.contourf(X, Y, u0.data[idx][:], levels=100, cmap='viridis', vmin=vmin, vmax=vmax)
ax0.set_title(f'FTCS Solution at t={t_to_compare:.2f}')
ax0.set_xlabel('$x$')
ax0.set_ylabel('$y$')

# BTCS Solution
c2 = ax1.contourf(X, Y, u1.data[idx][:], levels=100, cmap='viridis', vmin=vmin, vmax=vmax)
ax1.set_title(f'BTCS Solution at t={t_to_compare:.2f}')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')

# Analytical Solution
c3 = ax2.contourf(X, Y, u_exact.data[:], levels=100, cmap='viridis', vmin=vmin, vmax=vmax)
ax2.set_title(f'Analytical Solution at t={t_to_compare:.2f}')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$y$')

cb = fig.colorbar(c3, cax=cax)
cb.set_label('$u$')

for ax in [ax0, ax1, ax2]:
    ax.tick_params(axis='x', pad=5)
    ax.tick_params(axis='y', pad=5)


plt.subplots_adjust(left=0.02, right=0.95, top=0.92, bottom=0.12, wspace=0.25)
plt.savefig("3_2_5_compare.png", dpi=200, bbox_inches='tight', pad_inches=0.2)
plt.show()
