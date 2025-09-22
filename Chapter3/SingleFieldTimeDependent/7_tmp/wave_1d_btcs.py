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
# tf = 1.0

c = 1.5
C = 0.75

n = 7  # Very coarse mesh for this exact test

dt = C*(Lx/(n-1))/c
nt = int((tf - ti) / dt)

dx = Lx/(n-1)

infinity_norms = []
discrete_l2_norms = []
ksp_iters = []


grid = Grid(
    shape=(n,), extent=(Lx,), subdomains=subdomains, dtype=np.float64
)

u = TimeFunction(name='u', grid=grid, space_order=2, time_order=2, save=nt+1)
bc = Function(name='bc', grid=grid, space_order=2)

X = np.linspace(0, Lx, n).astype(np.float64)

u.data[0] = I(X)  # I(x)
u.data[0][0] = 0.
u.data[0][-1] = 0.

lap = (u.data[0][:-2] - 2.*u.data[0][1:-1] + u.data[0][2:])

# V(x) = 0.5x(Lx - x)
u.data[1][1:-1] = u.data[0][1:-1] + dt*V(X[1:-1]) + 0.5 * (C**2) * lap + 0.5* dt**2 * f(X[1:-1], 0, c)
# u.data[1][1:-1] = dt*V(X[1:-1]) + 0.5 * (C**2) * lap + 0.5* dt**2 * f(X[1:-1], 0, c)
u.data[1][0] = 0.
u.data[1][-1] = 0.

t = grid.time_dim

# Should it be t or t+1? - i think t for explicit, t+1 for implicit
eqn = Eq(u.dt2, (c**2)*u.forward.laplace + 2.0 * (1. + 0.5*(t+1)*dt) * (c**2), subdomain=grid.interior)

bc.data[:] = np.float64(0.0)
# from IPython import embed; embed()

# Create boundary condition expressions using subdomains
bcs = [EssentialBC(u.forward, bc, subdomain=sub1)]
bcs += [EssentialBC(u.forward, bc, subdomain=sub2)]

exprs = [eqn] + bcs
petsc = petscsolve(
    exprs,
    target=u.forward,
    solver_parameters={'ksp_rtol': 1e-10, 'ksp_type': 'gmres', 'pc_type': 'none'},
    options_prefix='wave_btcs'
)

with switchconfig(log_level='DEBUG'):
    op = Operator(petsc, language='petsc')
    summary = op.apply(dt=dt)
    print(op.arguments(dt=dt))
    # print(op.ccode)

idx = 86
t_to_compare = idx*dt

u_exact = Function(name='u_exact', grid=grid, space_order=2)
u_exact.data[:] = exact(X, t_to_compare, Lx)

diff = Function(name='diff', grid=grid, space_order=2)
diff.data[:] = u_exact.data[:] - u.data[idx][:]

# Compute infinity norm using numpy
infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
infinity_norms.append(infinity_norm)

# Compute discrete L2 norm (RMS error)
n_interior = np.prod([s - 1 for s in grid.shape])
discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
discrete_l2_norms.append(discrete_l2_norm)

print(f"Infinity norm: {infinity_norm}")

print(f"Discrete L2 norm: {discrete_l2_norm}")

from matplotlib import pyplot
# from IPython import embed; embed()
# Set the font family and size to use for Matplotlib figures.
pyplot.rcParams['font.family'] = 'serif'
pyplot.rcParams['font.size'] = 16


pyplot.figure(figsize=(10.0, 5.0))
pyplot.xlabel('...')
pyplot.ylabel('...')
# add title
pyplot.title('1D Wave Equation', fontsize=13)
pyplot.grid(False)
pyplot.plot(X, u.data[0], color='C2', linewidth=2, label='Initial condition')
pyplot.plot(X, u.data[idx], color='brown',linewidth=2, label=f'$t={tf}$')
pyplot.plot(X, u_exact.data[:], color='C1', linestyle='dotted', linewidth=2, label='Exact solution at $t=...$')
# pyplot.xlim(0.0, 1.)
# pyplot.ylim(-1.2, 2.3)
pyplot.legend(fontsize=10)

# Save fig
fig_path = '1d_wave_btcs.png'
pyplot.savefig(fig_path, bbox_inches='tight', dpi=300)