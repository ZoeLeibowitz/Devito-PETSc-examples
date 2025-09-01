import os
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, mmax, TimeFunction, sin)

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
nt = int(1.0 / dt)

# n = 9, 17, 33, 65, 129, 257
n_values = [2**k + 1 for k in range(3, 9)]
n_values = [n]
h = np.array([Lx/(n-1) for n in n_values])
infinity_norms = []
discrete_l2_norms = []
ksp_iters = []


grid = Grid(
    shape=(n, n, n), extent=(Lx, Ly, Lz), subdomains=subdomains, dtype=np.float64
)

# u = TimeFunction(name='u', grid=grid, space_order=2, save=nt+1)
u = TimeFunction(name='u', grid=grid, space_order=2, save=nt+1)
# bc = TimeFunction(name='bc', grid=grid, space_order=2)

x, y, z = grid.dimensions
eqn = Eq(u.dt, u.laplace, subdomain=grid.interior)

t = grid.time_dim

tmpx = np.linspace(0, Lx, n).astype(np.float64)
tmpy = np.linspace(0, Ly, n).astype(np.float64)
tmpz = np.linspace(0, Lz, n).astype(np.float64)

X, Y, Z = np.meshgrid(tmpx, tmpy, tmpz, indexing="ij")

# Create the 2D meshes for BCs
X_Y, Z_Y = np.meshgrid(tmpx, tmpz, indexing="ij")  # For faces where y varies
X_Z, Y_Z = np.meshgrid(tmpx, tmpy, indexing="ij")  # For faces where z varies
Y_X, Z_X = np.meshgrid(tmpy, tmpz, indexing="ij")  # For faces where x varies

u.data[0] = np.sin((np.pi/3.)*(X + Y + Z)) + X*Y*Z  # Initial condition


h_x, h_y, h_z = grid.spacing

# Create boundary condition expressions using subdomains
bcs = []


# TODO: CHECK.. IS IT DEFINITELY SUPPOSED TO BE T+1?
# left: u(0,y,z,t)
bcs += [EssentialBC(u.forward, sp.exp(-(sp.pi*sp.pi)*(t+1)*dt/3.)*sin(sp.pi*(y*h_y+z*h_z)/3.), subdomain=sub3)]

# # right: u(1,y,z,t)
bcs += [EssentialBC(u.forward, sp.exp(-(sp.pi*sp.pi)*(t+1)*dt/3.)*sin(sp.pi*(1.+y*h_y+z*h_z)/3.) + y*h_y*z*h_z, subdomain=sub4)]

# # front: u(x,0,z,t)
bcs += [EssentialBC(u.forward, sp.exp(-(sp.pi*sp.pi)*(t+1)*dt/3.)*sin(sp.pi*(x*h_x+z*h_z)/3.), subdomain=sub6)]

# # back: u(x,1,z,t)
bcs += [EssentialBC(u.forward, sp.exp(-(sp.pi*sp.pi)*(t+1)*dt/3.)*sin(sp.pi*(x*h_x+1.+z*h_z)/3.) + x*h_x*z*h_z, subdomain=sub5)]

# # bottom: u(x,y,0,t)
bcs += [EssentialBC(u.forward, sp.exp(-(sp.pi*sp.pi)*(t+1)*dt/3.)*sin(sp.pi*(x*h_x+y*h_y)/3.), subdomain=sub2)]

# # top: u(x,y,1,t)
bcs += [EssentialBC(u.forward, sp.exp(-(sp.pi*sp.pi)*(t+1)*dt/3.)*sin(sp.pi*(x*h_x+y*h_y+1.)/3.) + x*h_x*y*h_y, subdomain=sub1)]


exprs = [eqn] + bcs
petsc = PETScSolve(
    exprs, target=u.forward,
    solver_parameters={'ksp_rtol': 1e-7, 'ksp_type': 'gmres', 'pc_type': 'none'},
    options_prefix='heat_explicit_3d'
)

with switchconfig():
    op = Operator(petsc, language='petsc')
    summary = op.apply(dt=dt)
    # print(op.ccode)

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

# from IPython import embed; embed()


u_exact = Function(name='u_exact', grid=grid, space_order=2)
u_exact.data[:] = exact(X, Y, Z, dt*nt)

# print('exact is:', u_exact.data[0,:,:])


# print('numerical is', u.data[1,0,:,:])
# print('numerical is', u.data[1,0,:,:])


# from IPython import embed; embed()


from matplotlib import pyplot

# Set the font family and size to use for Matplotlib figures.
pyplot.rcParams['font.family'] = 'serif'
pyplot.rcParams['font.size'] = 16

# from IPython import embed; embed()
# n = 21
# Plot the temperature along the rod.
pyplot.figure(figsize=(10.0, 5.0))
pyplot.xlabel('x')
pyplot.ylabel('u(x,0.5,T)')
# add title
pyplot.title('FTCS vs Exact at y=0.5 (T=1)', fontsize=13)
pyplot.grid(False)
# plot cross section at y=0.5
# from IPython import embed; embed()
pyplot.plot(tmpz, u.data[-1, int((n-1)/2), int((n-1)/2), :].squeeze(), color='C1', linewidth=2, label='t=1.0')

pyplot.plot(tmpz, u.data[500, int((n-1)/2), int((n-1)/2), :].squeeze(), color='C1', linewidth=2, label='t=0.05')

# pyplot.plot(X, T.data[0], color='C2', linewidth=2, label='Initial condition')
# pyplot.plot(X, T.data[-1], color='brown',linewidth=2, label=f'$t={tf}$')
# pyplot.plot(tmpx, u_exact.data[:, int((n-1)/2)], color='C1', linestyle='dotted', linewidth=2, label='Exact')
pyplot.xlim(0.0, 1.)
pyplot.ylim(0., 1.7)
pyplot.legend(fontsize=10)

# Save fig
fig_path = '3d_heat_explicit.png'
pyplot.savefig(fig_path, bbox_inches='tight', dpi=300)


