"""
Solve the 2D manufactured-Poisson problem for a single grid size n via
Devito/PETSc geometric multigrid, printing PETSc's normal
-ksp_monitor_singular_value output to stdout.

Usage: python _solve_one_figure_7_1.py <n> [mg|none]
"""
import os
import sys

import numpy as np

from devito import Grid, Function, Eq, Operator, switchconfig, configuration, SubDomain
from devito.petsc import petscsolve, EssentialBC, GridHierarchy
from devito.petsc.initialize import PetscInitialize

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

PetscInitialize()


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

Lx = np.float64(1.)
Ly = np.float64(1.)

n = int(sys.argv[1])
solver_type = sys.argv[2] if len(sys.argv) > 2 else 'mg'

solver_params_mg = {
    'snes_type': 'ksponly',
    'ksp_type': 'cg',
    'pc_type': 'mg',
    'mg_levels_ksp_type': 'chebyshev',
    'mg_levels_pc_type': 'jacobi',
    'mg_levels_ksp_max_it': 3,
    'mg_coarse_ksp_type': 'gmres',
    'mg_coarse_ksp_rtol': 1e-12,
    'mg_coarse_pc_type': 'none',
    'ksp_monitor_singular_value': None,
}

solver_params_none = {
    'snes_type': 'ksponly',
    'ksp_type': 'cg',
    'pc_type': 'none',
    'ksp_monitor_singular_value': None,
}

solver_params = solver_params_mg if solver_type == 'mg' else solver_params_none

grid = Grid(shape=(n, n), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64)

nlevels = int(np.log2(n - 1))
hierarchy = GridHierarchy(grid, nlevels=nlevels) if solver_type == 'mg' else None

u = Function(name='u', grid=grid, space_order=2)
f = Function(name='f', grid=grid, space_order=2)
bc = Function(name='bc', grid=grid, space_order=2)

eqn = Eq(-u.laplace, f, subdomain=grid.interior)

tmpx = np.linspace(0, Lx, n).astype(np.float64)
tmpy = np.linspace(0, Ly, n).astype(np.float64)
Y, X = np.meshgrid(tmpx, tmpy)

f.data[:] = X*np.float64(np.exp(Y))

bc.data[0, :] = 0.
bc.data[-1, :] = -np.exp(tmpy)
bc.data[:, 0] = -tmpx
bc.data[:, -1] = -tmpx*np.exp(1)

bcs = [EssentialBC(u, bc, subdomain=sub1)]
bcs += [EssentialBC(u, bc, subdomain=sub2)]
bcs += [EssentialBC(u, bc, subdomain=sub3)]
bcs += [EssentialBC(u, bc, subdomain=sub4)]

exprs = [eqn] + bcs
petscsolve_kwargs = dict(
    solver_parameters=solver_params,
    options_prefix='poisson_2d',
)
if hierarchy is not None:
    petscsolve_kwargs['hierarchy'] = hierarchy

petsc = petscsolve(exprs, target=u, **petscsolve_kwargs)

with switchconfig(log_level='DEBUG'):
    op = Operator(petsc, language='petsc')
    summary = op.apply()

iters = summary.petsc[('section0', 'poisson_2d')].KSPGetIterationNumber
print(f"RESULT n={n} nlevels={nlevels} iters={iters}")
