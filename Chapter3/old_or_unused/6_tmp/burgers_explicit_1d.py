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
# Burgers' equation - explicit time stepping
# Considered one of the most common non-linear PDEs
# ref - file:///Users/zoeleibowitz/Downloads/axioms-12-00982.pdf
# name ref - A Comparative Study of the Explicit Finite Difference Method and Physics-Informed Neural Networks for Solving the Burgers’ Equation
# test problem 3 in paper


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


def exact(x, t, nu, m):
    tmp1 = 2.*nu*np.pi*np.exp(-(np.pi)**2*nu*t)*np.sin(np.pi*x)
    tmp2 = m + np.exp(-(np.pi)**2*nu*t)*np.cos(np.pi*x)
    return tmp1 / tmp2

Lx = np.float64(1.)

n = 101

dx = Lx/(n-1)

alpha = 0.1
ti = 0.
tf = 2.0
nu = 0.02
m = 2
dt = 0.0001
nt = int((tf - ti) / dt)

infinity_norms = []
discrete_l2_norms = []
ksp_iters = []


grid = Grid(
    shape=(n,), extent=(Lx,), subdomains=subdomains, dtype=np.float64
)

u = TimeFunction(name='u', grid=grid, space_order=2, save=nt+1)
bc = Function(name='bc', grid=grid, space_order=2)

X = np.linspace(0, Lx, n).astype(np.float64)

u.data[0] = 2.0*nu*np.pi*np.sin(np.pi * X) / (m + np.cos(np.pi*X))  # Initial condition

eqn = Eq(u.dt, nu*u.laplace - u*u.dx, subdomain=grid.interior)

bc.data[:] = np.float64(0.0)

# Create boundary condition expressions using subdomains
bcs = [EssentialBC(u.forward, bc, subdomain=sub1)]
bcs += [EssentialBC(u.forward, bc, subdomain=sub2)]

exprs = [eqn] + bcs
petsc = petscsolve(
    exprs,
    target=u.forward,
    solver_parameters={'ksp_rtol': 1e-10, 'ksp_type': 'gmres', 'pc_type': 'none'},
    options_prefix='burgers_explicit_1d'
)

with switchconfig(log_level='DEBUG'):
    op = Operator(petsc, language='petsc')
    summary = op.apply(dt=dt)
    print(op.ccode)


u_exact = TimeFunction(name='u_exact', grid=grid, space_order=2, save=nt+1)
u_exact.data[:] = exact(X, tf, nu, m)

# u_exact at t=0.5
u_exact.data[5000] = exact(X, dt*5000, nu, m)
# u_exact at t=1.0
u_exact.data[10000] = exact(X, dt*10000, nu, m)
# u_exact at t=2.0
u_exact.data[20000] = exact(X, dt*20000, nu, m)

from matplotlib import pyplot

# Set the font family and size to use for Matplotlib figures.
pyplot.rcParams['font.family'] = 'serif'
pyplot.rcParams['font.size'] = 16


# Plot the temperature along the rod.
pyplot.figure(figsize=(10.0, 7.0))
pyplot.xlabel('x')
pyplot.ylabel('u(x,t)')
# add title
pyplot.title("Burgers' equation in 1D - Forward Euler Scheme", fontsize=13)
pyplot.grid(False)
pyplot.plot(X, u.data[0], color='C2', linewidth=1.5, label='Initial condition')
# plot at t=0.5
pyplot.plot(X, u.data[5000], color='magenta', linewidth=1.5, label=f'FD $t={0.5}$')
pyplot.plot(X, u_exact.data[5000], color='magenta', marker='*', linestyle='none', markersize=5, label='Exact $t=0.5$')

# plot at t=1.0
pyplot.plot(X, u.data[10000], color='brown', linewidth=1.5, label=f'FD $t={1.0}$')
pyplot.plot(X, u_exact.data[10000], color='brown', marker='*', linestyle='none', markersize=5, label='Exact $t=1.0$')

# plot at t=2.0
pyplot.plot(X, u.data[20000], color='r', linewidth=1.5, label=f'FD $t={2.0}$')
pyplot.plot(X, u_exact.data[20000], color='r', marker='*', linestyle='none', markersize=5, label='Exact $t=2.0$')


pyplot.xlim(0.0, 1.)
pyplot.ylim(0.0, 0.1)
pyplot.legend(fontsize=10)

# Save fig
fig_path = '1d_burgers_explicit.png'
pyplot.savefig(fig_path, bbox_inches='tight', dpi=300)
