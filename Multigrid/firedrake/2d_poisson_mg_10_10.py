import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)

from devito.petsc import petscsolve, EssentialBC, GridHierarchy
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# 2D test
# Solving -u_xx - u_yy = f(x,y)
# copying firedrake test

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

def exact(x, y):
    return np.sin(np.pi*x) * np.tan(np.pi*x*0.25) * np.sin(np.pi*y)


def rhs(x, y):
    return -0.5*np.pi**2*(4*np.cos(np.pi*x) - 5*np.cos(np.pi*x*0.5) + 2)*np.sin(np.pi*y)


Lx = np.float64(1.)
Ly = np.float64(1.)



n = 41
# 41->21->11
dx = Lx/(n-1)
infinity_norms = []
discrete_l2_norms = []
ksp_iters = []


so = 2


grid = Grid(
    shape=(n, n), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
)


hierarchy = GridHierarchy(grid, nlevels=3)

u = Function(name='u', grid=grid, space_order=so)
f = Function(name='f', grid=grid, space_order=so)
bc = Function(name='bc', grid=grid, space_order=so)

eqn = Eq(-u.laplace, f, subdomain=grid.interior)

tmpx = np.linspace(0, Lx, n).astype(np.float64)
tmpy = np.linspace(0, Ly, n).astype(np.float64)

X, Y = np.meshgrid(tmpx, tmpy, indexing='ij')

f.data[:] = rhs(X, Y)

bc.data[0, :] = 0.
bc.data[-1, :] = 0.
bc.data[:, 0] = 0.
bc.data[:, -1] = 0.

# Create boundary condition expressions using subdomains
bcs = [EssentialBC(u, bc, subdomain=sub1)]
bcs += [EssentialBC(u, bc, subdomain=sub2)]
bcs += [EssentialBC(u, bc, subdomain=sub3)]
bcs += [EssentialBC(u, bc, subdomain=sub4)]

exprs = [eqn] + bcs
petsc = petscsolve(
    exprs, target=u,
    hierarchy=hierarchy,
    solver_parameters={
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'mg',
        'pc_mg_type': 'full',
        'mg_levels_ksp_type': 'chebyshev',
        'mg_levels_ksp_max_it': 4,
        'mg_levels_pc_type': 'jacobi',

        'mg_coarse_ksp_type': 'gmres',
        'mg_coarse_pc_type': 'none',
    },
    options_prefix='poisson_2d_mg'
)

with switchconfig(log_level='DEBUG'):
    op = Operator(petsc, language='petsc')
    summary = op.apply()

iters = summary.petsc[('section0', 'poisson_2d_mg')].KSPGetIterationNumber
ksp_iters.append(iters)

u_exact = Function(name='u_exact', grid=grid, space_order=so)
u_exact.data[:] = exact(X, Y)

diff = Function(name='diff', grid=grid, space_order=so)
diff.data[:] = u_exact.data[:] - u.data[:]

# Compute infinity norm using numpy
# TODO: Figure out how to compute the infinity norm using Devito
infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
infinity_norms.append(infinity_norm)

# Compute discrete L2 norm (RMS error)
n_interior = np.prod([s - 1 for s in grid.shape])
discrete_l2_norm = norm(diff) / np.sqrt(n_interior)

print(f"n={n}, discrete L2 norm={discrete_l2_norm}, KSP iterations={iters}")



