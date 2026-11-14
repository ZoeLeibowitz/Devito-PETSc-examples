# 3D heat equation with time dependent BCs, using
# petscsolve for two time stepping schemes (BTCS, CN)
# ref -> A Numerical Solution of Three-Dimensional Unsteady State Heat Equation, E. G. Tsega, IJM2C 11(1), 49-60, 2021

# Same as heat_3d_bcs_without_border.py but using Border
import os
import numpy as np
import sympy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

from devito import (Grid, TimeFunction, Eq, Operator, Border, configuration,
                    exp, sin)
from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

PetscInitialize()

n = 33
dt = 1e-4
T = 1.0
nt = int(round(T/dt))
times = (0.05, 0.25, 0.5, 1.0)
solver_parameters = {'ksp_type': 'cg', 'pc_type': 'none', 'ksp_rtol': 1e-12}

grid = Grid(shape=(n, n, n), extent=(1., 1., 1.), dtype=np.float64)

coords = np.linspace(0, 1., n)
X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')


def exact(X, Y, Z, time):
    return np.exp(-np.pi**2*time/3) * np.sin(np.pi*(X + Y + Z)/3) + X*Y*Z


def solve(scheme):
    u = TimeFunction(name='u', grid=grid, space_order=2, save=nt+1)

    # set time stepping scheme
    if scheme == 'BTCS':
        eqn = Eq(u.dt, u.forward.laplace, subdomain=grid.interior)
    elif scheme == 'CN':
        eqn = Eq(u.dt, (u.laplace + u.forward.laplace)/2, subdomain=grid.interior)

    # time dependent boundary values: the exact solution at the target's time level
    t = grid.time_dim
    x, y, z = (d*d.spacing for d in grid.dimensions)
    g = exp(-sp.pi**2*(t + 1)*t.spacing/3) * sin(sp.pi*(x + y + z)/3) + x*y*z
    bc = EssentialBC(u.forward, g, subdomain=Border(grid, 1))

    solve_u = petscsolve([eqn, bc], target=u.forward,
                         solver_parameters=solver_parameters)
    op = Operator(solve_u, language='petsc')

    u.data[0] = exact(X, Y, Z, 0.)
    op.apply(dt=dt, time_M=nt-1)

    # keep plotted times
    return {time: np.array(u.data[int(round(time/dt))]) for time in times}


solutions = {}
for scheme in ('BTCS', 'CN'):
    solutions[scheme] = solve(scheme)
    errors = [np.abs(solutions[scheme][time] - exact(X, Y, Z, time)).max()
              for time in times]
    print('%-4s ||e||_inf at t = %s: %s'
          % (scheme, ', '.join('%g' % time for time in times),
             ' '.join('%.3e' % e for e in errors)))


### plotting, laid out as Figures 4 and 5 of the reference
plt.rcParams['font.size'] = 16

# Solutions along the vertical centreline of the cube (x = 0.5, y = 0.5, 0 <= z <= 1)
m = (n - 1)//2
line_colours = ('r', 'm', 'c', 'b')
exact_colours = ('b', 'g', 'y', 'k')
plt.figure(figsize=(10.0, 7.0))
for time, lc, ec in zip(times, line_colours, exact_colours):
    plt.plot(coords, solutions['BTCS'][time][m, m, :], color=lc, linewidth=2,
             label=f'BTCS t={time}')
    plt.plot(coords, solutions['CN'][time][m, m, :], color=lc, linewidth=2,
             linestyle='--', label=f'CN t={time}')
    plt.plot(coords, exact(0.5, 0.5, coords, time), color=ec, marker='*',
             linestyle='none', markersize=8, label=f'Exact t={time}')
plt.xlabel('z')
plt.ylabel('Temperature')
plt.xlim(0., 1.)
plt.ylim(0., 1.6)
plt.legend(fontsize=8, loc='upper left')
plt.savefig('heat_3d_bcs_centreline.png', bbox_inches='tight', dpi=200)

# Solution on the six faces of the cube, BTCS, CN and exact at each time
fields = {'BTCS': solutions['BTCS'], 'CN': solutions['CN'],
          'Exact': {time: exact(X, Y, Z, time) for time in times}}
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
cmap = plt.cm.jet


def plot_cube_faces(ax, data):
    for face in (np.s_[0, :, :], np.s_[-1, :, :], np.s_[:, 0, :],
                 np.s_[:, -1, :], np.s_[:, :, 0], np.s_[:, :, -1]):
        ax.plot_surface(X[face], Y[face], Z[face], facecolors=cmap(norm(data[face])),
                        rstride=1, cstride=1, shade=False, alpha=0.95)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


fig = plt.figure(figsize=(18, 16))
for i, time in enumerate(times):
    for j, (name, field) in enumerate(fields.items()):
        ax = fig.add_subplot(len(times), len(fields), i*len(fields) + j + 1,
                             projection='3d')
        plot_cube_faces(ax, field[time])
        ax.set_title(f'{name} Solution (t={time})')
        fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, shrink=0.6,
                     aspect=10, pad=0.1, label='Temperature')
plt.tight_layout()
plt.savefig('heat_3d_bcs_cubes.png', bbox_inches='tight', dpi=200)
