import devito as dv
import numpy as np
import os

import sympy as sp
from schism import BoundaryConditions , BoundaryGeometry , Boundary

import matplotlib.pyplot as plt

dv.configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


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


rhs = u + dt*c*u.laplace

# ##################s###################
# Run for 1000 timesteps
eq = dv.Eq(u.forward, rhs.subs(subs))

op = dv.Operator(eq)
# print(op.ccode)
op.apply(dt=dt, t_M =1000)


# Create plt image

import matplotlib.pyplot as plt
plt.imshow(u.data[-1].T, origin='lower')
plt.colorbar()
plt.contour(sdf.data.T, levels=[0], colors='k')
plt.title('Heat equation with immersed boundary conditions')

# save fig
plt.savefig('heat_equation_immersed_bc.png', dpi=300)


