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


# 1D test
# Solving utt = c^2 * uxx + f(x,t)
# ref - https://hplgit.github.io/num-methods-for-PDEs/doc/pub/wave/pdf/wave-4print-A4-2up.pdf?
# https://library.oapen.org/bitstream/handle/20.500.12657/27809/1/1002196.pdf

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


def exact(x, t, Lx=2.5):
    return x*(Lx - x)*(1 + 0.5*t)

def I(x):
    return exact(x, 0)

def V(x):
    return 0.5*exact(x, 0)

def f(x, t, c=1.5):
    return 2.*(1. + 0.5*t)*(c**2)

Lx = np.float64(2.5)
ti = 0.
tf = 18.0
c = 1.5
C = 0.75
n = 7  # Very coarse mesh for this exact test
dt = C*(Lx/(n-1))/c
nt = int((tf - ti) / dt)
dx = Lx/(n-1)


grid = Grid(
    shape=(n,), extent=(Lx,), subdomains=subdomains, dtype=np.float64
)

# For FTCS scheme
u0 = TimeFunction(name='u0', grid=grid, space_order=2, time_order=2, save=nt+1)
# For BTCS scheme
u1 = TimeFunction(name='u1', grid=grid, space_order=2, time_order=2, save=nt+1)

bc = Function(name='bc', grid=grid, space_order=2)

X = np.linspace(0, Lx, n).astype(np.float64)

u0.data[0] = I(X)  # I(x)
u0.data[0][0] = 0.
u0.data[0][-1] = 0.

u1.data[0] = I(X)  # I(x)
u1.data[0][0] = 0.
u1.data[0][-1] = 0.

lap0 = (u0.data[0][:-2] - 2.*u0.data[0][1:-1] + u0.data[0][2:])
lap1 = (u1.data[0][:-2] - 2.*u1.data[0][1:-1] + u1.data[0][2:])

# V(x) = 0.5x(Lx - x)
# FTCS
u0.data[1][1:-1] = u0.data[0][1:-1] + dt*V(X[1:-1]) + 0.5 * (C**2) * lap0 + 0.5* dt**2 * f(X[1:-1], 0, c)
u0.data[1][0] = 0.
u0.data[1][-1] = 0.

# BTCS
u1.data[1][1:-1] = u1.data[0][1:-1] + dt*V(X[1:-1]) + 0.5 * (C**2) * lap1 + 0.5* dt**2 * f(X[1:-1], 0, c)
u1.data[1][0] = 0.
u1.data[1][-1] = 0.

t = grid.time_dim

eqn_ftcs = Eq(u0.dt2, (c**2)*u0.laplace + 2.0 * (1. + 0.5*(t)*dt) * (c**2), subdomain=grid.interior)
eqn_btcs = Eq(u1.dt2, (c**2)*u1.forward.laplace + 2.0 * (1. + 0.5*(t+1)*dt) * (c**2), subdomain=grid.interior)

bc.data[:] = np.float64(0.0)

# Create boundary condition expressions using subdomains
bcs_ftcs = [EssentialBC(u0.forward, bc, subdomain=sub1)]
bcs_ftcs += [EssentialBC(u0.forward, bc, subdomain=sub2)]

bcs_btcs = [EssentialBC(u1.forward, bc, subdomain=sub1)]
bcs_btcs += [EssentialBC(u1.forward, bc, subdomain=sub2)]

exprs_ftcs = [eqn_ftcs] + bcs_ftcs
ftcs_solver = petscsolve(
    exprs_ftcs,
    target=u0.forward,
    solver_parameters={'ksp_rtol': 1e-10, 'ksp_type': 'gmres', 'pc_type': 'none'},
    options_prefix='wave_explicit'
)

exprs_btcs = [eqn_btcs] + bcs_btcs
btcs_solver = petscsolve(
    exprs_btcs,
    target=u1.forward,
    solver_parameters={'ksp_rtol': 1e-10, 'ksp_type': 'gmres', 'pc_type': 'none'},
    options_prefix='wave_implicit'
)

with switchconfig(log_level='DEBUG'):
    op = Operator([ftcs_solver, btcs_solver], language='petsc')
    summary = op.apply(dt=dt)


idx = 86
t_to_compare = idx*dt

u_exact = Function(name='u_exact', grid=grid, space_order=2)
u_exact.data[:] = exact(X, t_to_compare, Lx)

diff_ftcs = Function(name='diff_ftcs', grid=grid, space_order=2)
diff_ftcs.data[:] = u_exact.data[:] - u0.data[idx][:]

diff_btcs = Function(name='diff_btcs', grid=grid, space_order=2)
diff_btcs.data[:] = u_exact.data[:] - u1.data[idx][:]

# Compute infinity norm using numpy
infinity_norm_ftcs = np.linalg.norm(diff_ftcs.data[:].ravel(), ord=np.inf)
infinity_norm_btcs = np.linalg.norm(diff_btcs.data[:].ravel(), ord=np.inf)

# Compute discrete L2 norm (RMS error)
n_interior = np.prod([s - 1 for s in grid.shape])

discrete_l2_norm_ftcs = norm(diff_ftcs) / np.sqrt(n_interior)
discrete_l2_norm_btcs = norm(diff_btcs) / np.sqrt(n_interior)

print(f"Infinity norm (FTCS): {infinity_norm_ftcs}")
print(f"Discrete L2 norm (FTCS): {discrete_l2_norm_ftcs}")
print(f"Infinity norm (BTCS): {infinity_norm_btcs}")
print(f"Discrete L2 norm (BTCS): {discrete_l2_norm_btcs}")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.figure(figsize=(10.0, 5.0))
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.grid(False)
plt.plot(X, u0.data[0], color='C2', linewidth=2, label='Initial condition')
plt.plot(X, u0.data[idx], color='brown',linewidth=2, label=f'FTCS solution at $t={tf}$')
plt.plot(X, u1.data[idx], color='brown', linewidth=2, linestyle='dashed', label=f'BTCS solution at $t={tf}$')
plt.plot(X, u_exact.data[:], color='C1', linestyle='dotted', linewidth=2, label=f'Exact solution at $t={tf}$')
plt.legend(fontsize=8, loc='upper right')

# Save fig
fig_path = '3_2_4.png'
plt.savefig(fig_path, bbox_inches='tight', dpi=300)