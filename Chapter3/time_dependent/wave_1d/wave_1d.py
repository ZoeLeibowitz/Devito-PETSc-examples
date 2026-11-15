import os
import numpy as np

from devito import (Grid, Function, TimeFunction, Eq, Operator, switchconfig,
                    configuration, SubDomain)

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# 1D test
# Implicit scheme (the Laplacian at the new time level). The exact solution is linear
# in t and quadratic in space, so the scheme reproduces it to machine precision
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

u = TimeFunction(name='u', grid=grid, space_order=2, time_order=2, save=nt+1)

bc = Function(name='bc', grid=grid, space_order=2)

X = np.linspace(0, Lx, n).astype(np.float64)

u.data[0] = I(X)  # I(x)
u.data[0][0] = 0.
u.data[0][-1] = 0.

lap = (u.data[0][:-2] - 2.*u.data[0][1:-1] + u.data[0][2:])

# First step, V(x) = 0.5x(Lx - x)
u.data[1][1:-1] = u.data[0][1:-1] + dt*V(X[1:-1]) + 0.5 * (C**2) * lap + 0.5* dt**2 * f(X[1:-1], 0, c)
u.data[1][0] = 0.
u.data[1][-1] = 0.

t = grid.time_dim

# Implicit scheme: the Laplacian and the source at the new time level
eqn = Eq(u.dt2, (c**2)*u.forward.laplace + 2.0 * (1. + 0.5*(t+1)*dt) * (c**2), subdomain=grid.interior)

bc.data[:] = np.float64(0.0)

# Create boundary condition expressions using subdomains
bcs = [EssentialBC(u.forward, bc, subdomain=sub1)]
bcs += [EssentialBC(u.forward, bc, subdomain=sub2)]

solver = petscsolve(
    [eqn] + bcs,
    target=u.forward,
    solver_parameters={'ksp_rtol': 1e-10, 'ksp_type': 'gmres', 'pc_type': 'none'},
    options_prefix='wave_implicit'
)

with switchconfig(log_level='DEBUG'):
    op = Operator(solver, language='petsc')
    summary = op.apply(dt=dt)

# Infinity norm of the error at every time level, largest over all of them
# (as in Langtangen and Linge 2017, Sec. 2.3.3)
u_exact = np.array([exact(X, k*dt, Lx) for k in range(nt+1)])
error = np.abs(u.data[:] - u_exact).max()
print(f"Largest infinity norm of the error over all time levels: {error:.2e}")
assert error < 1e-12
