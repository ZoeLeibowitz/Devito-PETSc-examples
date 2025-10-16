import os
import numpy as np

from devito import (Grid, Function, TimeFunction, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, mmax)

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# 2D test - explicit heat equation 
# ref -> https://www.scirp.org/pdf/jamp_1724227.pdf
# ref -> An Efficient Explicit Scheme for Solving the 2D
# Heat Equation with Stability and Convergence
# Analysis

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

# Compute exact solution
def exact(x, y, T=1., alpha=1.):
    return np.exp(-2.*np.pi**2*T*alpha)*np.sin(np.pi*x)*np.sin(np.pi*y)

Lx = np.float64(1.)
Ly = np.float64(1.)

alpha = 1.0
dt = 0.0005
nt = int(1. / dt)

n = 21
h = Lx/(n-1)
infinity_norms = []
discrete_l2_norms = []
ksp_iters = []


grid = Grid(
    shape=(n, n), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
)

u = TimeFunction(name='u', grid=grid, space_order=2, save=nt+1)
bc = Function(name='bc', grid=grid, space_order=2)

eqn = Eq(u.dt, (alpha**2)*u.laplace, subdomain=grid.interior)

tmpx = np.linspace(0, Lx, n).astype(np.float64)
tmpy = np.linspace(0, Ly, n).astype(np.float64)

Y, X = np.meshgrid(tmpx, tmpy)

u.data[0] = np.sin(np.pi * X) * np.sin(np.pi * Y)  # Initial condition

bc.data[0, :] = 0.
bc.data[-1, :] = 0.
bc.data[:, 0] = 0.
bc.data[:, -1] = 0.

# Create boundary condition expressions using subdomains
bcs = [EssentialBC(u.forward, bc, subdomain=sub1)]
bcs += [EssentialBC(u.forward, bc, subdomain=sub2)]
bcs += [EssentialBC(u.forward, bc, subdomain=sub3)]
bcs += [EssentialBC(u.forward, bc, subdomain=sub4)]

exprs = [eqn] + bcs
petsc = petscsolve(
    exprs, target=u.forward,
    solver_parameters={'ksp_rtol': 1e-10, 'ksp_type': 'gmres', 'pc_type': 'none'},
    options_prefix='heat_explicit_2d'
)

with switchconfig(log_level='DEBUG'):
    op = Operator(petsc, language='petsc')
    summary = op.apply(dt=dt)
    

u_exact = Function(name='u_exact', grid=grid, space_order=2)
u_exact.data[:] = exact(X, Y)


diff = Function(name='diff', grid=grid, space_order=2)
tmp = np.abs(u_exact.data[:, int((n-1)/2)] - u.data[-1, :, int((n-1)/2)])



plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16

n = 21

# Create a 1x2 plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('u(x,0.5,T)', fontsize=11)
ax.set_title('FTCS vs Exact at y=0.5 (T=1)', fontsize=13)
ax.grid(False)
ax.plot(tmpx, u.data[-1, :, int((n-1)/2)].squeeze(), color='blue', linewidth=2, label='FTCS')
ax.plot(tmpx, u_exact.data[:, int((n-1)/2)], color='red', linestyle='dotted', linewidth=2, label='Exact')
ax.set_xlim(0.0, 1., )
ax.legend(fontsize=11)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(11)
ax.set_ylim(0., 3.0e-9)

ax = axes[1]
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('Absolute Error', fontsize=11)
ax.set_title('Error |FTCS - Exact| at y=0.5 (T=1)', fontsize=13)

ax.plot(
    tmpx, tmp,
    color='k',
    linewidth=2,
    linestyle='-',
    marker='D',
    markersize=5,
    markerfacecolor='none',
    markeredgecolor='k',
    markeredgewidth=1.0,
)

ax.set_xlim(0.0, 1.)
ax.set_ylim(0., 1.5e-10)
ax.yaxis.set_major_locator(MultipleLocator(0.5e-10))
ax.yaxis.get_offset_text().set_fontsize(11)

for ax in axes:
    ax.tick_params(axis='both', labelsize=11)

fig.text(0.25, 0.008, '(a) FTCS and Exact Solution', ha='center', fontsize=12)
fig.text(0.75, 0.008, '(b) Error', ha='center', fontsize=12)


plt.tight_layout()
plt.savefig('3_2_2_ftcs.png', bbox_inches='tight', dpi=300)
plt.show()
