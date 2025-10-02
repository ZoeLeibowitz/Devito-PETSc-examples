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



###################### with no immersed bcs ######################

grid = dv.Grid(shape =(101 , 101) , extent =(1. , 1.), dtype=np.float64) # FD grid
c = 1. # Diffusion constant
dt = 0.000025 # Timestep ( stability limit)
u0 = dv. TimeFunction (name='u0', grid=grid , space_order =2, save=1000)
u0.data [0, 40: -40 , 40: -40] = 1. # Initialise field
rhs = u0 + dt*c*u0.laplace
eq = dv.Eq(u0.forward , rhs) # Update equation
petsc = petscsolve(eq, u0.forward) # Solve with petsc
# Run for 1000 timesteps
op = dv.Operator(petsc, language='petsc')
op.apply(dt=dt)






###################### with immersed bcs ######################
grid = dv.Grid(shape =(101 , 101) , extent =(1. , 1.), dtype=np.float64)


sdf = dv. Function(name='sdf', grid=grid , space_order =2)
x_msh , y_msh = np.meshgrid (*[ np.linspace (-0.5, 0.5, 101)
for d in grid. dimensions ])
sdf.data [:] = 0.25 - np.sqrt(x_msh **2 + y_msh **2)


c = 1. # Diffusion constant
dt = 0.000025 # Timestep ( stability limit)
u1 = dv. TimeFunction (name='u1', grid=grid , space_order =2, save=1000)
u1.data [0, 40: -40 , 40: -40] = 1.
#### Boundary setup ####
zero = sp.core.numbers.Zero ()
# Override default cutoff (eta =0)
bg = BoundaryGeometry (sdf , cutoff ={( zero , zero): 0.})
bcs = BoundaryConditions ([dv.Eq(u1, 0),
dv.Eq(u1.laplace , 0)])
boundary = Boundary(bcs , bg)
subs = boundary . substitutions ((u1.dx2 , u1.dy2))
# #######################
#### Substitute stencils into RHS ####

eq = dv.Eq(u1.dt, c*u1.laplace.subs(subs))
petsc = petscsolve(eq, u1.forward)

# #####################################
# Run for 1000 timesteps
op = dv.Operator(petsc, language='petsc')
print(op.ccode)
op.apply(dt=dt)




import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Row 1: without immersed boundaries (u0)
im0 = axes[0, 0].imshow(u0.data[0].T, origin='lower',
                        extent=[-0.5, 0.5, -0.5, 0.5])
axes[0, 0].set_title('No IB – Initial condition')
axes[0, 0].set_xlim(-0.5, 0.5)
axes[0, 0].set_ylim(-0.5, 0.5)
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

im1 = axes[0, 1].imshow(u0.data[-1].T, origin='lower',
                        extent=[-0.5, 0.5, -0.5, 0.5], vmax=0.15)
axes[0, 1].set_title('No IB – Final field')
axes[0, 1].set_xlim(-0.5, 0.5)
axes[0, 1].set_ylim(-0.5, 0.5)
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

# Row 2: with immersed boundaries (u1)
im2 = axes[1, 0].imshow(u1.data[0].T, origin='lower',
                        extent=[-0.5, 0.5, -0.5, 0.5])
axes[1, 0].set_title('With IB – Initial condition')
axes[1, 0].contour(sdf.data.T, levels=[0], colors='k',
                   extent=[-0.5, 0.5, -0.5, 0.5])
axes[1, 0].set_xlim(-0.5, 0.5)
axes[1, 0].set_ylim(-0.5, 0.5)
plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

im3 = axes[1, 1].imshow(u1.data[-1].T, origin='lower',
                        extent=[-0.5, 0.5, -0.5, 0.5], vmax=0.15)
axes[1, 1].set_title('With IB – Final field')
axes[1, 1].contour(sdf.data.T, levels=[0], colors='k',
                   extent=[-0.5, 0.5, -0.5, 0.5])
axes[1, 1].set_xlim(-0.5, 0.5)
axes[1, 1].set_ylim(-0.5, 0.5)
plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)


axes[0, 0].set_xlabel("x(m)")
axes[0, 0].set_ylabel("y(m)")

axes[0, 1].set_xlabel("x(m)")
axes[0, 1].set_ylabel("y(m)")

axes[1, 0].set_xlabel("x(m)")
axes[1, 0].set_ylabel("y(m)")

axes[1, 1].set_xlabel("x(m)")
axes[1, 1].set_ylabel("y(m)")


# Adjust layout
fig.tight_layout()

# Save figure
plt.savefig('heat_equation_comparison_explicit.png', dpi=300)
plt.show()




