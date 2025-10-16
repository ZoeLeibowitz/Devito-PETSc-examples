import os
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, mmax, TimeFunction, sin)

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# 3D test
# Solving 3D heat equation with time dependent BCS
# Utilising both a FTCS and BTCS scheme
# Dirichlet BCs (time dependent)
# ref -> file:///Users/zoeleibowitz/Downloads/IJM2C_Volume11_Issue1WINTER_Pages49-60.pdf
# ref - A Numerical Solution of Three-Dimensional Unsteady State Heat Equation, E. G. Tsega


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
    return np.exp(-(np.pi**2)*t/3.)*np.sin((np.pi/3.)*(x+y+z)) + x*y*z

Lx = np.float64(1.)
Ly = np.float64(1.)
Lz = np.float64(1.)

dt = 0.0001
n = 33
nt = int(1.0 / dt)

grid = Grid(
    shape=(n, n, n), extent=(Lx, Ly, Lz), subdomains=subdomains, dtype=np.float64
)

u0 = TimeFunction(name='u0', grid=grid, space_order=2, save=nt+1)  # for FTCS solve
u1 = TimeFunction(name='u1', grid=grid, space_order=2, save=nt+1)  # for BTCS solve

x, y, z = grid.dimensions

ftcs_eqn = Eq(u0.dt, u0.laplace, subdomain=grid.interior)
btcs_eqn = Eq(u1.dt, u1.forward.laplace, subdomain=grid.interior)

t = grid.time_dim

tmpx = np.linspace(0, Lx, n).astype(np.float64)
tmpy = np.linspace(0, Ly, n).astype(np.float64)
tmpz = np.linspace(0, Lz, n).astype(np.float64)

X, Y, Z = np.meshgrid(tmpx, tmpy, tmpz, indexing="ij")

# Create the 2D meshes for BCs
X_Y, Z_Y = np.meshgrid(tmpx, tmpz, indexing="ij")  # For faces where y varies
X_Z, Y_Z = np.meshgrid(tmpx, tmpy, indexing="ij")  # For faces where z varies
Y_X, Z_X = np.meshgrid(tmpy, tmpz, indexing="ij")  # For faces where x varies

# Initial condition for each scheme
u0.data[0] = np.sin((np.pi/3.)*(X + Y + Z)) + X*Y*Z
u1.data[0] = np.sin((np.pi/3.)*(X + Y + Z)) + X*Y*Z

h_x, h_y, h_z = grid.spacing

# Create boundary condition expressions using subdomains
bcs_ftcs = []
bcs_btcs = []

# TODO: CHECK.. pretty sure it's supposed to be t+1?
# left: u(0,y,z,t)
bcs_ftcs += [EssentialBC(u0.forward, sp.exp(-(sp.pi*sp.pi)*(t+1)*dt/3.)*sin(sp.pi*(y*h_y+z*h_z)/3.), subdomain=sub3)]
bcs_btcs += [EssentialBC(u1.forward, sp.exp(-(sp.pi*sp.pi)*(t+1)*dt/3.)*sin(sp.pi*(y*h_y+z*h_z)/3.), subdomain=sub3)]

# right: u(1,y,z,t)
bcs_ftcs += [EssentialBC(u0.forward, sp.exp(-(sp.pi*sp.pi)*(t+1)*dt/3.)*sin(sp.pi*(1.+y*h_y+z*h_z)/3.) + y*h_y*z*h_z, subdomain=sub4)]
bcs_btcs += [EssentialBC(u1.forward, sp.exp(-(sp.pi*sp.pi)*(t+1)*dt/3.)*sin(sp.pi*(1.+y*h_y+z*h_z)/3.) + y*h_y*z*h_z, subdomain=sub4)]

# front: u(x,0,z,t)
bcs_ftcs += [EssentialBC(u0.forward, sp.exp(-(sp.pi*sp.pi)*(t+1)*dt/3.)*sin(sp.pi*(x*h_x+z*h_z)/3.), subdomain=sub6)]
bcs_btcs += [EssentialBC(u1.forward, sp.exp(-(sp.pi*sp.pi)*(t+1)*dt/3.)*sin(sp.pi*(x*h_x+z*h_z)/3.), subdomain=sub6)]

# back: u(x,1,z,t)
bcs_ftcs += [EssentialBC(u0.forward, sp.exp(-(sp.pi*sp.pi)*(t+1)*dt/3.)*sin(sp.pi*(x*h_x+1.+z*h_z)/3.) + x*h_x*z*h_z, subdomain=sub5)]
bcs_btcs += [EssentialBC(u1.forward, sp.exp(-(sp.pi*sp.pi)*(t+1)*dt/3.)*sin(sp.pi*(x*h_x+1.+z*h_z)/3.) + x*h_x*z*h_z, subdomain=sub5)]

# bottom: u(x,y,0,t)
bcs_ftcs += [EssentialBC(u0.forward, sp.exp(-(sp.pi*sp.pi)*(t+1)*dt/3.)*sin(sp.pi*(x*h_x+y*h_y)/3.), subdomain=sub2)]
bcs_btcs += [EssentialBC(u1.forward, sp.exp(-(sp.pi*sp.pi)*(t+1)*dt/3.)*sin(sp.pi*(x*h_x+y*h_y)/3.), subdomain=sub2)]

# top: u(x,y,1,t)
bcs_ftcs += [EssentialBC(u0.forward, sp.exp(-(sp.pi*sp.pi)*(t+1)*dt/3.)*sin(sp.pi*(x*h_x+y*h_y+1.)/3.) + x*h_x*y*h_y, subdomain=sub1)]
bcs_btcs += [EssentialBC(u1.forward, sp.exp(-(sp.pi*sp.pi)*(t+1)*dt/3.)*sin(sp.pi*(x*h_x+y*h_y+1.)/3.) + x*h_x*y*h_y, subdomain=sub1)]

exprs_ftcs = [ftcs_eqn] + bcs_ftcs
ftcs_solver = petscsolve(
    exprs_ftcs, target=u0.forward,
    solver_parameters={'ksp_rtol': 1e-7, 'ksp_type': 'gmres', 'pc_type': 'none'},
    options_prefix='heat_explicit_3d'
)

exprs_btcs = [btcs_eqn] + bcs_btcs
btcs_solver = petscsolve(
    exprs_btcs, target=u1.forward,
    solver_parameters={'ksp_rtol': 1e-7, 'ksp_type': 'gmres', 'pc_type': 'none'},
    options_prefix='heat_btcs_3d'
)

with switchconfig(log_level='DEBUG'):
    op1 = Operator(ftcs_solver, language='petsc')
    op2 = Operator(btcs_solver, language='petsc')
    summary = op1.apply(dt=dt)
    summary = op2.apply(dt=dt)


u_exact = TimeFunction(name='u_exact', grid=grid, space_order=2, save=nt+1)

nt = u_exact.data.shape[0]  # total number of time steps

for t_idx in range(nt):
    u_exact.data[t_idx] = exact(X, Y, Z, dt * t_idx)


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16

plt.figure(figsize=(10.0, 7.0))
plt.xlabel('z')
plt.ylabel('Temperature')
plt.grid(False)

plt.plot(tmpz, u0.data[500, int((n-1)/2), int((n-1)/2), :].squeeze(), color='r', linewidth=2, label=' FTCS t=0.05')
plt.plot(tmpz, u1.data[500, int((n-1)/2), int((n-1)/2), :].squeeze(), color='r', linewidth=2, linestyle='--', label=' BTCS t=0.05')
plt.plot(tmpz, u_exact.data[500, int((n-1)/2), int((n-1)/2), :].squeeze(), color='b', marker='*', linestyle='none', markersize=8, label='Exa t=0.05')

plt.plot(tmpz, u0.data[2500, int((n-1)/2), int((n-1)/2), :].squeeze(), color='m', linewidth=2, label=' FTCS t=0.25')
plt.plot(tmpz, u1.data[2500, int((n-1)/2), int((n-1)/2), :].squeeze(), color='m', linewidth=2, linestyle='--', label=' BTCS t=0.25')
plt.plot(tmpz, u_exact.data[2500, int((n-1)/2), int((n-1)/2), :].squeeze(), color='g', marker='*', linestyle='none', markersize=8, label='Exa t=0.25')

plt.plot(tmpz, u0.data[5000, int((n-1)/2), int((n-1)/2), :].squeeze(), color='c', linewidth=2, label=' FTCS t=0.5')
plt.plot(tmpz, u1.data[5000, int((n-1)/2), int((n-1)/2), :].squeeze(), color='c', linewidth=2, linestyle='--', label=' BTCS t=0.5')
plt.plot(tmpz, u_exact.data[5000, int((n-1)/2), int((n-1)/2), :].squeeze(), color='y', marker='*', linestyle='none', markersize=8, label='Exa t=0.5')

plt.plot(tmpz, u0.data[10000, int((n-1)/2), int((n-1)/2), :].squeeze(), color='b', linewidth=2, label=' FTCS t=1.0')
plt.plot(tmpz, u1.data[10000, int((n-1)/2), int((n-1)/2), :].squeeze(), color='b', linewidth=2, linestyle='--', label=' BTCS t=1.0')
plt.plot(tmpz, u_exact.data[10000, int((n-1)/2), int((n-1)/2), :].squeeze(), color='k', marker='*', linestyle='none', markersize=8, label='Exa t=1.0')


plt.xlim(0.0, 1.)
plt.ylim(0., 1.6)
plt.legend(fontsize=8, loc='upper left')


# Save fig
fig_path = '3_2_3_centreline.png'
plt.savefig(fig_path, bbox_inches='tight', dpi=300)


# u.shape = (time_steps, nx, ny, nz)
time_indices = [500, 2500, 5000, 10000]
time_labels = ['0.05', '0.25', '0.5', '1.0']

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# (FTCS, BTCS, Exact)
nrows, ncols = 4, 3
fig = plt.figure(figsize=(18, 16))

# Normalization and colormap
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
cmap = plt.cm.jet

X, Y, Z = np.meshgrid(tmpx, tmpy, tmpz, indexing='ij')

def plot_cube_faces(ax, data):
    # Draw six faces of cube
    ax.plot_surface(X[0,:,:], Y[0,:,:], Z[0,:,:],
                    facecolors=cmap(norm(data[0,:,:])), rstride=1, cstride=1, shade=False, alpha=0.95)
    ax.plot_surface(X[-1,:,:], Y[-1,:,:], Z[-1,:,:],
                    facecolors=cmap(norm(data[-1,:,:])), rstride=1, cstride=1, shade=False, alpha=0.95)
    ax.plot_surface(X[:,0,:], Y[:,0,:], Z[:,0,:],
                    facecolors=cmap(norm(data[:,0,:])), rstride=1, cstride=1, shade=False, alpha=0.95)
    ax.plot_surface(X[:,-1,:], Y[:,-1,:], Z[:,-1,:],
                    facecolors=cmap(norm(data[:,-1,:])), rstride=1, cstride=1, shade=False, alpha=0.95)
    ax.plot_surface(X[:,:,0], Y[:,:,0], Z[:,:,0],
                    facecolors=cmap(norm(data[:,:,0])), rstride=1, cstride=1, shade=False, alpha=0.95)
    ax.plot_surface(X[:,:,-1], Y[:,:,-1], Z[:,:,-1],
                    facecolors=cmap(norm(data[:,:,-1])), rstride=1, cstride=1, shade=False, alpha=0.95)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

for i, t_idx in enumerate(time_indices):
    # FTCS
    ax1 = fig.add_subplot(nrows, ncols, i * ncols + 1, projection='3d')
    plot_cube_faces(ax1, u0.data[t_idx])
    ax1.set_title(f'FTCS Solution (t={time_labels[i]})')
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    mappable1 = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(mappable1, ax=ax1, shrink=0.6, aspect=10, pad=0.1, label="Temperature")

    # BTCS
    ax2 = fig.add_subplot(nrows, ncols, i * ncols + 2, projection='3d')
    plot_cube_faces(ax2, u1.data[t_idx])
    ax2.set_title(f'BTCS Solution (t={time_labels[i]})')
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    mappable2 = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(mappable2, ax=ax2, shrink=0.6, aspect=10, pad=0.1, label="Temperature")

    # Exact
    ax3 = fig.add_subplot(nrows, ncols, i * ncols + 3, projection='3d')
    plot_cube_faces(ax3, u_exact.data[t_idx])
    ax3.set_title(f'Exact Solution (t={time_labels[i]})')
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
    mappable3 = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(mappable3, ax=ax3, shrink=0.6, aspect=10, pad=0.1, label="Temperature")

plt.tight_layout()
plt.savefig("3_2_3_cubes.png", dpi=300)
plt.show()



############################################ for an animation only ############################################

# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import numpy as np
# from matplotlib import animation

# X, Y, Z = np.meshgrid(tmpx, tmpy, tmpz, indexing='ij')

# # Normalization
# norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
# cmap = plt.cm.jet

# fig = plt.figure(figsize=(12, 6))

# # Left: FD solution
# ax1 = fig.add_subplot(1, 2, 1, projection="3d")
# ax1.set_title("Finite Difference Solution", fontsize=10)
# ax1.set_xlabel("x", fontsize=10); ax1.set_ylabel("y", fontsize=10); ax1.set_zlabel("z", fontsize=10)
# ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_zlim(0, 1)
# ax1.tick_params(axis='both', which='major', labelsize=8)
# ax1.tick_params(axis='both', which='minor', labelsize=6)
# mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
# cbar1 = fig.colorbar(mappable, ax=ax1, shrink=0.6, aspect=10, pad=0.15)
# cbar1.set_label("Temperature", fontsize=10)
# cbar1.ax.tick_params(labelsize=8)

# # Right: Exact solution
# ax2 = fig.add_subplot(1, 2, 2, projection="3d")
# ax2.set_title("Exact Solution", fontsize=10)
# ax2.set_xlabel("x", fontsize=10); ax2.set_ylabel("y", fontsize=10); ax2.set_zlabel("z", fontsize=10)
# ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_zlim(0, 1)
# ax2.tick_params(axis='both', which='major', labelsize=8)
# ax2.tick_params(axis='both', which='minor', labelsize=6)
# mappable2 = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
# cbar2 = fig.colorbar(mappable2, ax=ax2, shrink=0.6, aspect=10, pad=0.15)
# cbar2.set_label("Temperature", fontsize=10)
# cbar2.ax.tick_params(labelsize=8)


# def plot_cube_faces(ax, data):
#     faces = []
#     # x=0 face
#     faces.append(ax.plot_surface(X[0,:,:], Y[0,:,:], Z[0,:,:],
#                                  facecolors=cmap(norm(data[0,:,:])),
#                                  rstride=1, cstride=1, shade=False, alpha=0.95))
#     # x=max face
#     faces.append(ax.plot_surface(X[-1,:,:], Y[-1,:,:], Z[-1,:,:],
#                                  facecolors=cmap(norm(data[-1,:,:])),
#                                  rstride=1, cstride=1, shade=False, alpha=0.95))
#     # y=0 face
#     faces.append(ax.plot_surface(X[:,0,:], Y[:,0,:], Z[:,0,:],
#                                  facecolors=cmap(norm(data[:,0,:])),
#                                  rstride=1, cstride=1, shade=False, alpha=0.95))
#     # y=max face
#     faces.append(ax.plot_surface(X[:,-1,:], Y[:,-1,:], Z[:,-1,:],
#                                  facecolors=cmap(norm(data[:,-1,:])),
#                                  rstride=1, cstride=1, shade=False, alpha=0.95))
#     # z=0 face
#     faces.append(ax.plot_surface(X[:,:,0], Y[:,:,0], Z[:,:,0],
#                                  facecolors=cmap(norm(data[:,:,0])),
#                                  rstride=1, cstride=1, shade=False, alpha=0.95))
#     # z=max face
#     faces.append(ax.plot_surface(X[:,:,-1], Y[:,:,-1], Z[:,:,-1],
#                                  facecolors=cmap(norm(data[:,:,-1])),
#                                  rstride=1, cstride=1, shade=False, alpha=0.95))
#     return faces

# # Initial plots
# fd_faces = plot_cube_faces(ax1, u.data[0])
# exact_faces = plot_cube_faces(ax2, u_exact.data[0])


# def update(frame):
#     # Clear old surfaces
#     for f in fd_faces: f.remove()
#     for f in exact_faces: f.remove()

#     # Redraw with new data
#     new_fd = plot_cube_faces(ax1, u.data[frame])
#     new_exact = plot_cube_faces(ax2, u_exact.data[frame])

#     # Replace global handles
#     fd_faces[:] = new_fd
#     exact_faces[:] = new_exact

#     ax1.set_title(f"Finite Difference Solution (t={frame*dt:.2f})", fontsize=10)
#     ax2.set_title(f"Exact Solution (t={frame*dt:.2f})", fontsize=10)
#     return fd_faces + exact_faces

# # Make animation
# ani = animation.FuncAnimation(fig, update, frames=range(0, u.data.shape[0], 500),
#                               blit=False, interval=200)

# ani.save("cube_faces_comparison.gif", writer="pillow", fps=5)

# plt.close(fig)
