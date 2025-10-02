import devito as dv
import numpy as np
import os

import sympy as sp
from schism import BoundaryConditions , BoundaryGeometry , Boundary

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

dv.configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


PetscInitialize()

grid = dv.Grid(shape =(101 , 101) , extent =(1. , 1.), dtype=np.float64)


sdf = dv. Function(name='sdf', grid=grid , space_order =2)
x_msh , y_msh = np.meshgrid (*[ np.linspace (-0.5, 0.5, 101)
for d in grid. dimensions ])
sdf.data [:] = 0.25 - np.sqrt(x_msh **2 + y_msh **2)


c = 1. # Diffusion constant
dt = 0.000025 # Timestep ( stability limit)
u = dv. TimeFunction (name='u', grid=grid , space_order =2)
u.data [0, 40: -40 , 40: -40] = 1.
#### Boundary setup ####
zero = sp.core.numbers.Zero ()
# Override default cutoff (eta =0)
bg = BoundaryGeometry (sdf , cutoff ={( zero , zero): 0.})
bcs = BoundaryConditions ([dv.Eq(u, 0),
dv.Eq(u.laplace , 0)])
boundary = Boundary(bcs , bg)
subs = boundary . substitutions ((u.dx2 , u.dy2))
# #######################
#### Substitute stencils into RHS ####

eq = dv.Eq(u.dt, c*u.laplace.subs(subs))
petsc = petscsolve(eq, u.forward)

# #####################################
# Run for 1000 timesteps
op = dv.Operator(petsc, language='petsc')
print(op.ccode)
op.apply(dt=dt, t_M =1000)



u2 = dv. TimeFunction (name='u2', grid=grid , space_order =2)
u2.data [0, 40: -40 , 40: -40] = 1.
#### Boundary setup ####
zero = sp.core.numbers.Zero ()
# Override default cutoff (eta =0)
bg = BoundaryGeometry (sdf , cutoff ={( zero , zero): 0.})
bcs = BoundaryConditions ([dv.Eq(u2, 0),
dv.Eq(u2.laplace , 0)])
boundary = Boundary(bcs , bg)
subs = boundary . substitutions ((u2.dx2 , u2.dy2))
# #######################
#### Substitute stencils into RHS ####


rhs = u2 + dt*c*u2.laplace

# ##################s###################
# Run for 1000 timesteps
eq = dv.Eq(u2.forward, rhs.subs(subs))

op = dv.Operator(eq)
# print(op.ccode)
op.apply(dt=dt, t_M =1000)



# print diff
diff = u.data[-1] - u2.data[-1]

import matplotlib.pyplot as plt
plt.imshow(diff[:].T, origin='lower')

plt.colorbar(fraction=0.046, pad=0.04) 
plt.contour(sdf.data.T, levels=[0], colors='k')
plt.title('Diff: petscsolve vs standard')

# move colourbar to rhs more


# save fig
plt.savefig('diff.png', dpi=300)

# print(np.linalg.norm(diff.data[:]))



