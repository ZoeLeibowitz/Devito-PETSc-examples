import os
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from devito import (Grid, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, TimeFunction, sin)

from devito.petsc import petscsolve, EssentialBC
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

n_values = [n]


grid = Grid(
    shape=(n, n, n), extent=(Lx, Ly, Lz), subdomains=subdomains, dtype=np.float64
)

u = TimeFunction(name='u', grid=grid, space_order=2, save=nt+1)

x, y, z = grid.dimensions
eqn = Eq(u.dt, u.forward.laplace, subdomain=grid.interior)

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


# TODO: CHECK... IS IT DEFINITELY SUPPOSED TO BE T+1?
# I THINK IT should be for implicit but only t for explicit?
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
petsc = petscsolve(
    exprs, target=u.forward,
    solver_parameters={'ksp_rtol': 1e-7, 'ksp_type': 'gmres', 'pc_type': 'none'},
    options_prefix='heat_implicit_3d'
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


u_exact = TimeFunction(name='u_exact', grid=grid, space_order=2, save=nt+1)
# u_exact at t=0.05
u_exact.data[500] = exact(X, Y, Z, dt*500)
# u_exact at t=0.25
u_exact.data[2500] = exact(X, Y, Z, dt*2500)
# u_exact at t=0.5
u_exact.data[5000] = exact(X, Y, Z, dt*5000)
# u_exact at t=1.0
u_exact.data[10000] = exact(X, Y, Z, dt*10000)



from matplotlib import pyplot

# Set the font family and size to use for Matplotlib figures.
pyplot.rcParams['font.family'] = 'serif'
pyplot.rcParams['font.size'] = 16

pyplot.figure(figsize=(10.0, 7.0))
pyplot.xlabel('z')
pyplot.ylabel('Temperature')
pyplot.grid(False)

pyplot.plot(tmpz, u.data[500, int((n-1)/2), int((n-1)/2), :].squeeze(), color='r', linewidth=2, label=' FD t=0.05')
pyplot.plot(tmpz, u_exact.data[500, int((n-1)/2), int((n-1)/2), :].squeeze(), color='b', marker='*', linestyle='none', markersize=8, label='Exa t=0.05')

pyplot.plot(tmpz, u.data[2500, int((n-1)/2), int((n-1)/2), :].squeeze(), color='m', linewidth=2, label=' FD t=0.25')
pyplot.plot(tmpz, u_exact.data[2500, int((n-1)/2), int((n-1)/2), :].squeeze(), color='g', marker='*', linestyle='none', markersize=8, label='Exa t=0.25')

pyplot.plot(tmpz, u.data[5000, int((n-1)/2), int((n-1)/2), :].squeeze(), color='c', linewidth=2, label=' FD t=0.5')
pyplot.plot(tmpz, u_exact.data[5000, int((n-1)/2), int((n-1)/2), :].squeeze(), color='y', marker='*', linestyle='none', markersize=8, label='Exa t=0.5')

pyplot.plot(tmpz, u.data[10000, int((n-1)/2), int((n-1)/2), :].squeeze(), color='b', linewidth=2, label=' FD t=1.0')
pyplot.plot(tmpz, u_exact.data[10000, int((n-1)/2), int((n-1)/2), :].squeeze(), color='k', marker='*', linestyle='none', markersize=8, label='Exa t=1.0')


pyplot.xlim(0.0, 1.)
pyplot.ylim(0., 1.6)
pyplot.legend(fontsize=10, loc='upper left')

# Save fig
fig_path = '3d_heat_implicit.png'
pyplot.savefig(fig_path, bbox_inches='tight', dpi=300)




import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Example arrays (replace with your real data)
# u.shape = (time_steps, nx, ny, nz)
time_indices = [500, 2500, 5000, 10000]
time_labels = ['0.05', '0.25', '0.5', '1.0']

nrows, ncols = 4, 2
fig = plt.figure(figsize=(12, 16))

# Normalize across dataset
# vmin = min(u.data.min(), u_exact.data.min())
# vmax = max(u.data.max(), u_exact.data.max())
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
cmap = plt.cm.jet

X, Y, Z = np.meshgrid(tmpx, tmpy, tmpz, indexing='ij')

def plot_cube_faces(ax, data):
    # x=0 face
    ax.plot_surface(X[0,:,:], Y[0,:,:], Z[0,:,:],
                    facecolors=cmap(norm(data[0,:,:])),
                    rstride=1, cstride=1, shade=False, alpha=0.95)
    # x=max face
    ax.plot_surface(X[-1,:,:], Y[-1,:,:], Z[-1,:,:],
                    facecolors=cmap(norm(data[-1,:,:])),
                    rstride=1, cstride=1, shade=False, alpha=0.95)
    # y=0 face
    ax.plot_surface(X[:,0,:], Y[:,0,:], Z[:,0,:],
                    facecolors=cmap(norm(data[:,0,:])),
                    rstride=1, cstride=1, shade=False, alpha=0.95)
    # y=max face
    ax.plot_surface(X[:,-1,:], Y[:,-1,:], Z[:,-1,:],
                    facecolors=cmap(norm(data[:,-1,:])),
                    rstride=1, cstride=1, shade=False, alpha=0.95)
    # z=0 face
    ax.plot_surface(X[:,:,0], Y[:,:,0], Z[:,:,0],
                    facecolors=cmap(norm(data[:,:,0])),
                    rstride=1, cstride=1, shade=False, alpha=0.95)
    # z=max face
    ax.plot_surface(X[:,:,-1], Y[:,:,-1], Z[:,:,-1],
                    facecolors=cmap(norm(data[:,:,-1])),
                    rstride=1, cstride=1, shade=False, alpha=0.95)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)


for i, t_idx in enumerate(time_indices):
    # FD (left column)
    ax1 = fig.add_subplot(nrows, ncols, i*2 + 1, projection='3d')
    plot_cube_faces(ax1, u.data[t_idx])
    ax1.set_title(f'Finite Difference Solution (t={time_labels[i]})')
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    # ax1.view_init(elev=26, azim=25)
    mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(mappable, ax=ax1, shrink=0.6, aspect=10, pad=0.1, label="Temperature")

    # Exact (right column)
    ax2 = fig.add_subplot(nrows, ncols, i*2 + 2, projection='3d')
    plot_cube_faces(ax2, u_exact.data[t_idx])
    ax2.set_title(f'Exact Solution (t={time_labels[i]})')
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    # ax2.view_init(elev=26, azim=25)
    mappable2 = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(mappable2, ax=ax2, shrink=0.6, aspect=10, pad=0.1, label="Temperature")

plt.tight_layout()
plt.savefig("3d_contour_cube_faces_implicit.png", dpi=300)
# plt.show()
