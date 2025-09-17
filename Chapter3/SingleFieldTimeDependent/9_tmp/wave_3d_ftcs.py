import os
import numpy as np

from devito import (Grid, Function, TimeFunction, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)
from devito.symbolics import retrieve_functions, INT

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'



# 3D test
# Solving utt = c^2 * (uxx + yyy + uzz) + f(x,y,z,t)
# ref - https://hplgit.github.io/num-methods-for-PDEs/doc/pub/wave/pdf/wave-4print-A4-2up.pdf?
# ref - file:///Users/zoeleibowitz/Downloads/wave-4print-A4-2up.pdf


PetscInitialize()

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


def exact(x, y, z, t, Lx=2.5, Ly=2.5, Lz=2.5):
    return x*(Lx - x)*y*(Ly - y)*z*(Lz - z)*(1 + 0.5*t)

def I(x, y, z):
    return exact(x, y, z, 0)

def V(x, y, z):
    return 0.5*exact(x, y, z, 0)

def f(x, y, z, t, c=1.5, Lx=2.5, Ly=2.5, Lz=2.5):
    return 2.*(1. + 0.5*t)*(y*(Ly - y)*z*(Lz - z) + x*(Lx - x)*z*(Lz - z) + + x*(Lx - x)*y*(Ly - y))*(c**2)


Lx = np.float64(2.5)
Ly = np.float64(2.5)
Lz = np.float64(2.5)


n = 7  # Very coarse mesh for this exact test

dx = Lx/(n-1)
dy = Ly/(n-1)
dz = Lz/(n-1)


ti = 0.
tf = 18.0

c = 1.5
stability_limit = (1/np.float64(c))*(1/np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2))

dt = stability_limit

C = c*dt*(1/dx)  # Courant number
nt = int((tf - ti) / dt)

infinity_norms = []
discrete_l2_norms = []
ksp_iters = []


grid = Grid(
    shape=(n, n, n), extent=(Lx, Ly, Lz), subdomains=subdomains, dtype=np.float64
)

u = TimeFunction(name='u', grid=grid, space_order=2, time_order=2, save=nt+1)
bc = Function(name='bc', grid=grid, space_order=2)

tmpx = np.linspace(0, Lx, n).astype(np.float64)
tmpy = np.linspace(0, Ly, n).astype(np.float64)
tmpz = np.linspace(0, Lz, n).astype(np.float64)

X, Y, Z = np.meshgrid(tmpx, tmpy, tmpz, indexing="ij")

u.data[0] = I(X, Y, Z)

lap = (
    u.data[0][1:-1, :-2] +  # left
    u.data[0][1:-1, 2:]  +  # right
    u.data[0][:-2, 1:-1] +  # down
    u.data[0][2:, 1:-1]  -  # up
    4.0 * u.data[0][1:-1, 1:-1]  # center
)

lap = (
    u.data[0][1:-1, :-2, 1:-1] +  # left
    u.data[0][1:-1, 2:, 1:-1]  +  # right
    u.data[0][:-2, 1:-1, 1:-1] +  # down
    u.data[0][2:, 1:-1, 1:-1]  +  # up
    u.data[0][1:-1, 1:-1, :-2] +  # back
    u.data[0][1:-1, 1:-1, 2:]  -  # front
    6.0 * u.data[0][1:-1, 1:-1, 1:-1]  # center
)

u.data[1][1:-1, 1:-1, 1:-1] = (
    u.data[0][1:-1, 1:-1, 1:-1]
    + dt * V(X[1:-1, 1:-1, 1:-1], Y[1:-1, 1:-1, 1:-1], Z[1:-1, 1:-1, 1:-1])
    + 0.5 * (C**2) * lap
    + 0.5 * dt**2 * f(X[1:-1, 1:-1, 1:-1], Y[1:-1, 1:-1, 1:-1], Z[1:-1, 1:-1, 1:-1], 0, c)
)

# u.data[1][0] = 0.
# u.data[1][-1] = 0.

t = grid.time_dim
x, y, z = grid.dimensions

h_x, h_y, h_z = grid.spacing

# Should it be t or t+1? - i think t for explicit, t+1 for implicit
eqn = Eq(u.dt2, (c**2)*u.laplace + 2.*(1. + 0.5*(t*dt))*((z*h_z)*(Lz-(z*h_z))*(y*h_y)*(Ly-(y*h_y)) + (x*h_x)*(Lx-(x*h_x))*(z*h_z)*(Lz-(z*h_z)) + (x*h_x)*(Lx-(x*h_x))*(y*h_y)*(Ly-(y*h_y)))*(c**2), subdomain=grid.interior)

bc.data[:] = np.float64(0.0)

# Create boundary condition expressions using subdomains
bcs = [EssentialBC(u.forward, bc, subdomain=sub1)]
bcs += [EssentialBC(u.forward, bc, subdomain=sub2)]
bcs += [EssentialBC(u.forward, bc, subdomain=sub3)]
bcs += [EssentialBC(u.forward, bc, subdomain=sub4)]
bcs += [EssentialBC(u.forward, bc, subdomain=sub5)]
bcs += [EssentialBC(u.forward, bc, subdomain=sub6)]

exprs = [eqn] + bcs
petsc = PETScSolve(
    exprs,
    target=u.forward,
    solver_parameters={'ksp_rtol': 1e-10, 'ksp_type': 'gmres', 'pc_type': 'none'},
    options_prefix='wave_3d_explicit'
)

with switchconfig(log_level='DEBUG'):
    op = Operator(petsc, language='petsc')
    summary = op.apply(dt=dt)
    print(op.arguments(dt=dt))

# don't acc need idx
idx = nt
t_to_compare = idx*dt
print(f"t to compare: {t_to_compare}")

u_exact = Function(name='u_exact', grid=grid, space_order=2)
u_exact.data[:] = exact(X, Y, Z, t_to_compare, Lx, Ly, Lz)

diff = Function(name='diff', grid=grid, space_order=2)
# diff.data[:] = u_exact.data[:] - u.data[idx][:]
diff.data[:] = u.data[idx][:] - u_exact.data[:]

# Compute infinity norm using numpy
infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
infinity_norms.append(infinity_norm)

# Compute discrete L2 norm (RMS error)
n_interior = np.prod([s - 1 for s in grid.shape])
discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
discrete_l2_norms.append(discrete_l2_norm)

print(f"Infinity norm: {infinity_norm}")
print(f"Discrete L2 norm: {discrete_l2_norm}")


from matplotlib import pyplot

# Set the font family and size to use for Matplotlib figures.
pyplot.rcParams['font.family'] = 'serif'
pyplot.rcParams['font.size'] = 16

pyplot.figure(figsize=(10.0, 7.0))
pyplot.xlabel('z')
pyplot.ylabel('$u$')
pyplot.grid(False)

pyplot.plot(tmpz, u.data[idx, int((n-1)/2), int((n-1)/2), :].squeeze(), color='b', linewidth=2, label=f'FD t={t_to_compare:.2f}')
pyplot.plot(tmpz, u_exact.data[int((n-1)/2), int((n-1)/2), :].squeeze(), color='k', marker='*', linestyle='none', markersize=8, label=f'Exa t={t_to_compare:.2f}')


# pyplot.xlim(0.0, 1.)
# pyplot.ylim(0., 1.6)
pyplot.legend(fontsize=10, loc='upper left')

# Save fig
fig_path = 'wave_3d_ftcs_slice.png'
pyplot.savefig(fig_path, bbox_inches='tight', dpi=300)
