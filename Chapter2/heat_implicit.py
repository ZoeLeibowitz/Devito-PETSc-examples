# The implicit (BTCS) heat equation with petscsolve
import os
import numpy as np
from devito import Grid, TimeFunction, Eq, Operator, Border, configuration
from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

# PETSc is initialised and the Grid is created in PETSc's precision
PetscInitialize()

nx, ny = 41, 41
grid = Grid(shape=(nx, ny), extent=(1.0, 1.0), dtype=np.float64)
u = TimeFunction(name='u', grid=grid, space_order=2, time_order=1)
alpha = 0.1

eqn = Eq(u.dt, alpha**2 * (u.forward.dx2 + u.forward.dy2),
         subdomain=grid.interior)
bc = EssentialBC(u.forward, 0., subdomain=Border(grid, 1))
solve_u = petscsolve([eqn, bc], target=u.forward)
op = Operator(solve_u, language='petsc')

x_coords = np.linspace(0, 1.0, nx)
y_coords = np.linspace(0, 1.0, ny)
X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
u.data[0] = np.sin(np.pi*X) * np.sin(np.pi*Y)

# dt is 3.2 times the explicit stability limit h^2/(4 alpha^2) = 0.0156
dt, nt = 0.05, 40
op.apply(dt=dt, time_M=nt-1)
print(op.ccode)

T = nt*dt
exact = np.exp(-2*np.pi**2*alpha**2*T)*np.sin(np.pi*X)*np.sin(np.pi*Y)
error = np.abs(u.data[nt % 2] - exact).max()
print('T = %g, max |u - u_exact| = %.3e, max |u_exact| = %.3e' % (T, error, exact.max()))
