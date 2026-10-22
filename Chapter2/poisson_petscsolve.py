"""
Steady-state Poisson equation solved with petscsolve (Section 2.5.1,
Listing petscsolve_poisson of the thesis).

Solves the same problem as prior_poisson.py and
solving_poisson_with_snes/poisson_matfree.c
(ref - https://github.com/bueler/p4pdes/blob/master/c/ch6/fish.c):
    -p_xx - p_yy = b(x,y) = x e^y on the unit square,
    Dirichlet BCs from the exact solution p(x,y) = -x e^y:
    p(0,y) = 0, p(1,y) = -e^y, p(x,0) = -x, p(x,1) = -x e.

The solver settings are equivalent to those of the second and fourth runs in
Listing petsc_poisson_run (CG, no preconditioner, ksp_rtol 1e-12). They are
chosen to demonstrate the interface and to match the hand written code, not
for efficiency. The solve has the options prefix poisson_, so its options can
be set on the command line. Running
    python poisson_petscsolve.py -poisson_ksp_converged_reason
gives the same 60 iterations and maximum error as
    solving_poisson_with_snes/poisson_matfree -ksp_converged_reason
"""
import os

import numpy as np

from devito import Grid, Function, Eq, Operator, Border, configuration, mmax
from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'
# Switch off Devito's timing output, so that a run prints only the solver output
configuration['log-level'] = 'WARNING'

# Initialise PETSc, passing on any command line options
PetscInitialize()

# Number of grid points and the unit square domain
nx, ny = 17, 17
Lx, Ly = 1.0, 1.0
# Create the grid with PETSc's precision, here double
grid = Grid(shape=(nx, ny), extent=(Lx, Ly), dtype=np.float64)
# Border selects the boundary layer, one grid point thick
border = Border(grid, 1)

# Grid point coordinates, for b and bc
x_coord = np.linspace(0, Lx, nx)
y_coord = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x_coord, y_coord, indexing='ij')

# The source term b
b = Function(name='b', grid=grid)
b.data[:] = X * np.exp(Y)

# Exact solution, for the boundary values and the error
bc = Function(name='bc', grid=grid)
bc.data[:] = -X * np.exp(Y)

# The solution field p
p = Function(name='p', grid=grid, space_order=2)

# The Poisson equation on the interior
eqn = Eq(-p.laplace, b, subdomain=grid.interior)
# Dirichlet values, taken from bc on the border only
bc_eq = EssentialBC(p, bc, subdomain=border)

# Specify solver parameters via a Python dictionary
params = {'ksp_type': 'cg', 'pc_type': 'none',
          'ksp_rtol': 1e-12}
# Solve for p, specifying a prefix for command line options
poisson = petscsolve([eqn, bc_eq], target=p,
                     solver_parameters=params,
                     options_prefix='poisson')

# Generate the code with the PETSc backend and run it
op = Operator(poisson, language='petsc')
op.apply()

# Maximum error against the exact solution, held in bc
err = Function(name='err', grid=grid)
err.data[:] = np.abs(p.data - bc.data)
print('%d x %d grid, maximum error %.3e'
      % (nx, ny, mmax(err)))
