import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, TimeFunction)

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

PetscInitialize()

########### No petsc ###########
grid = Grid(shape=(4,4), dtype=np.float64)
u = TimeFunction(name='u', grid=grid)

eqn = Eq(u.forward, u + 1 + grid.time_dim)

op1 = Operator(eqn)
op1.apply(time_M=1)

assert np.all(u.data[1] == 1.)
assert np.all(u.data[0] == 3.)

# Reset u
u.data[:] = 0.
print(u.data)

########### with petsc ###########
petsc = PETScSolve(eqn, target=u.forward)

with switchconfig():
    op = Operator(petsc, language='petsc')
    # op.apply(time_M=1)

# print(u.data)







# petsc = PETScSolve([eqn], target=u.forward)

# with switchconfig(log_level='DEBUG'):
#     op = Operator(petsc, language='petsc')
#     summary = op.apply()

# from IPython import embed; embed()

# print(u.data[1])