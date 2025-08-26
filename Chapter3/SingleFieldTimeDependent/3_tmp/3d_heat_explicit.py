import os
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, mmax, TimeFunction)

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# 3D test
# Solving u.dt = u.laplace
# Dirichlet BCs
# ref -> file:///Users/zoeleibowitz/Downloads/IJM2C_Volume11_Issue1WINTER_Pages49-60.pdf


PetscInitialize()

# Subdomains to implement BCs

# Subdomain for z = 1 (top)
class SubTop(SubDomain):
    name = 'subtop'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('middle', 1, 1), y: ('middle', 1, 1), z: ('right', 1)}

# Subdomain for z = 0 (bottom)
class SubBottom(SubDomain):
    name = 'subbottom'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('middle', 1, 1), y: ('middle', 1, 1), z: ('left', 1)}

# Subdomain for y = 1 (back)
class SubBack(SubDomain):
    name = 'subback'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('middle', 1, 1), y: ('right', 1), z: z}

# Subdomain for y = 0 (front)
class SubFront(SubDomain):
    name = 'subfront'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('middle', 1, 1), y: ('left', 1), z: z}

# Subdomain for x = 0 (left)
class SubLeft(SubDomain):
    name = 'subleft'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('left', 1), y: y, z: z}

# Subdomain for x = 1 (right)
class SubRight(SubDomain):
    name = 'subright'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('right', 1), y: y, z: z}


sub1 = SubTop()
sub2 = SubBottom()
sub3 = SubLeft()
sub4 = SubRight()
sub5 = SubBack()
sub6 = SubFront()


subdomains = (sub1, sub2, sub3, sub4, sub5, sub6)

def exact(x, y, z, t):
    tmp1 = np.exp(-(np.pi**2)*t/3.)
    tmp2 = np.sin((np.pi/3.)*(x+y+z))
    tmp3 = x*y*z
    return tmp1*tmp2 + tmp3

Lx = np.float64(1.)
Ly = np.float64(1.)
Lz = np.float64(1.)

dt = 0.0001
n = 33

# n = 9, 17, 33, 65, 129, 257
n_values = [2**k + 1 for k in range(3, 9)]
n_values = [33]
h = np.array([Lx/(n-1) for n in n_values])
infinity_norms = []
discrete_l2_norms = []
ksp_iters = []


grid = Grid(
    shape=(n, n, n), extent=(Lx, Ly, Lz), subdomains=subdomains, dtype=np.float64
)

u = TimeFunction(name='u', grid=grid, space_order=2)
bc = TimeFunction(name='bc', grid=grid, space_order=2)

x, y, z = grid.dimensions
eqn = Eq(u.dt, u.laplace + u.time_dim, subdomain=grid.interior)

t = grid.time_dim

# from IPython import embed; embed()

tmpx = np.linspace(0, Lx, n).astype(np.float64)
tmpy = np.linspace(0, Ly, n).astype(np.float64)
tmpz = np.linspace(0, Lz, n).astype(np.float64)

X, Y, Z = np.meshgrid(tmpx, tmpy, tmpz, indexing="ij")

# Create the 2D meshes for BCs
X_Y, Z_Y = np.meshgrid(tmpx, tmpz, indexing="ij")  # For faces where y varies
X_Z, Y_Z = np.meshgrid(tmpx, tmpy, indexing="ij")  # For faces where z varies
Y_X, Z_X = np.meshgrid(tmpy, tmpz, indexing="ij")  # For faces where x varies

# u(0,y,z,t) = 
# bc.data[0, :, :] = 0.0


# # u(1,y,z) = -exp(y+z)
# bc.data[-1, :, :] = -np.exp(Y_X + Z_X)

# # u(x,0,z) = -x*exp(z)
# bc.data[:, 0, :] = -X_Y * np.exp(Z_Y)

# # u(x,1,z) = -x*exp(1+z)
# bc.data[:, -1, :] = -X_Y * np.exp(1.0 + Z_Y)

# # u(x,y,0) = -x*exp(y)
# bc.data[:, :, 0] = -X_Z * np.exp(Y_Z)

# # u(x,y,1) = -x*exp(y+1)
# bc.data[:, :, -1] = -X_Z * np.exp(Y_Z + 1.0)

# # Create boundary condition expressions using subdomains
bcs = []
# bcs += [EssentialBC(u.forward, grid.time_dim, subdomain=sub1)]  # top
# bcs += [EssentialBC(u.forward, bc, subdomain=sub2)]  # bottom
# bcs += [EssentialBC(u.forward, bc, subdomain=sub3)]  # subleft
# bcs += [EssentialBC(u.forward, bc, subdomain=sub4)]  # subright
# bcs += [EssentialBC(u.forward, bc, subdomain=sub5)]  # subback
# bcs += [EssentialBC(u.forward, bc, subdomain=sub6)]  # subfront

exprs = [eqn] + bcs
petsc = PETScSolve(
    exprs, target=u.forward,
    solver_parameters={'ksp_rtol': 1e-12, 'ksp_type': 'cg', 'pc_type': 'none'},
    options_prefix='heat_explicit_3d'
)

with switchconfig(log_level='DEBUG'):
    op = Operator(petsc, language='petsc')
    # summary = op.apply()
    print(op.ccode)

# iters = summary.petsc[('section0', 'heat_explicit_3d')].KSPGetIterationNumber
# ksp_iters.append(iters)

# u_exact = Function(name='u_exact', grid=grid, space_order=2)
# u_exact.data[:] = exact(X, Y, Z)

# diff = Function(name='diff', grid=grid, space_order=2)
# diff.data[:] = u_exact.data[:] - u.data[:]

# # # Compute infinity norm using numpy
# # # TODO: Figure out how to compute the infinity norm using Devito
# infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
# infinity_norms.append(infinity_norm)

# # Compute discrete L2 norm (RMS error)
# n_interior = np.prod([s - 1 for s in grid.shape])
# discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
# discrete_l2_norms.append(discrete_l2_norm)


