import os
import numpy as np

from devito import (Grid, Function, TimeFunction, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, mmax)

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

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

# just compute final time for now
def exact(x, y, T=1., alpha=1.):
    return np.exp(-2.*np.pi**2*T*alpha)*np.sin(np.pi*x)*np.sin(np.pi*y)

Lx = np.float64(1.)
Ly = np.float64(1.)

alpha = 1.0
dt = 0.0005
nt = int(1. / dt)

# n = 9, 17, 33, 65, 129, 257, 513, 1025
n_values = [2**k + 1 for k in range(3, 11)]
n_values = [21]
h = np.array([Lx/(n-1) for n in n_values])
infinity_norms = []
discrete_l2_norms = []
ksp_iters = []

for n in n_values:
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

    # iters = summary.petsc[('section0', 'poisson_2d')].KSPGetIterationNumber
    # ksp_iters.append(iters)

    # u_exact = Function(name='u_exact', grid=grid, space_order=2)
    # u_exact.data[:] = exact(X, Y)

    # diff = Function(name='diff', grid=grid, space_order=2)
    # diff.data[:] = u_exact.data[:] - u.data[:]

    # # Compute infinity norm using numpy
    # # TODO: Figure out how to compute the infinity norm using Devito
    # infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
    # infinity_norms.append(infinity_norm)

    # # Compute discrete L2 norm (RMS error)
    # n_interior = np.prod([s - 1 for s in grid.shape])
    # discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
    # discrete_l2_norms.append(discrete_l2_norm)
    



u_exact = Function(name='u_exact', grid=grid, space_order=2)
u_exact.data[:] = exact(X, Y)


diff = Function(name='diff', grid=grid, space_order=2)
tmp = np.abs(u_exact.data[:, int((n-1)/2)] - u.data[-1, :, int((n-1)/2)])


from matplotlib import pyplot

# Set the font family and size to use for Matplotlib figures.
pyplot.rcParams['font.family'] = 'serif'
pyplot.rcParams['font.size'] = 16

# from IPython import embed; embed()
n = 21
# Plot the temperature along the rod.
pyplot.figure(figsize=(10.0, 5.0))
pyplot.xlabel('x')
pyplot.ylabel('u(x,0.5,T)')
# add title
pyplot.title('FTCS vs Exact at y=0.5 (T=1)', fontsize=13)
pyplot.grid(False)
# plot cross section at y=0.5
pyplot.plot(tmpx, u.data[-1, :, int((n-1)/2)].squeeze(), color='C1', linewidth=2, label='FTCS')
# pyplot.plot(X, T.data[0], color='C2', linewidth=2, label='Initial condition')
# pyplot.plot(X, T.data[-1], color='brown',linewidth=2, label=f'$t={tf}$')
pyplot.plot(tmpx, u_exact.data[:, int((n-1)/2)], color='C1', linestyle='dotted', linewidth=2, label='Exact')
pyplot.xlim(0.0, 1.)
pyplot.ylim(0., 3.0e-9)
pyplot.legend(fontsize=10)

# Save fig
fig_path = '2d_heat_explicit.png'
pyplot.savefig(fig_path, bbox_inches='tight', dpi=300)



############ plot diff ###############
pyplot.figure(figsize=(10.0, 5.0))
pyplot.xlabel('x')
pyplot.ylabel('Absolute Error')
# add title
pyplot.title('Error |FTCS - Exact| at y=0.5 (T=1)', fontsize=13)
pyplot.grid(False)
# plot cross section at y=0.5
pyplot.plot(tmpx, tmp, color='C1', linewidth=2)
pyplot.xlim(0.0, 1.)
pyplot.ylim(0., 1.5e-10)
# pyplot.legend(fontsize=10)

# Save fig
fig_path = 'diff.png'
pyplot.savefig(fig_path, bbox_inches='tight', dpi=300)