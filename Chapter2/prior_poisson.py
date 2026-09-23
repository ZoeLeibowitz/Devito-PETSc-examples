"""
Steady-state Poisson equation, adapted from Devito's tutorial notebook
(devito/examples/cfd/06_poisson.ipynb, "Example 6: Poisson equation").
"""
import numpy as np

from devito import Grid, Function, TimeFunction, Eq, Operator, solve, configuration
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

configuration['log-level'] = 'ERROR'

nx, ny = 50, 50
nt = 100
Lx, Ly = 2.0, 1.0

grid = Grid(shape=(nx, ny), extent=(Lx, Ly))

b = Function(name='b', grid=grid)
b.data[:] = 0.
b.data[int(nx / 4), int(ny / 4)] = -100
b.data[int(3 * nx / 4), int(3 * ny / 4)] = 100


p = TimeFunction(name='p', grid=grid, space_order=2)
p.data[:] = 0.

# Create Laplace equation base on `p`
eq = Eq(-p.laplace, b)
# Solve for the central stencil point
stencil = solve(eq, p)
# Let stencil populate the buffer `p.forward`, restricted to grid.interior
eq_stencil = Eq(p.forward, stencil, subdomain=grid.interior)

# p is initialised to 0 everywhere and grid.interior never updates the
# boundary layer, so it stays fixed at 0 - already satisfying the
# homogeneous Dirichlet BC, with no separate BC equations needed.
op = Operator([eq_stencil])
op(time=nt)

buffer_size = p.time_order + 1
final_idx = nt % buffer_size
p_final = p.data[final_idx]


x_coord = np.linspace(0, Lx, nx)
y_coord = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x_coord, y_coord, indexing='ij')

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, p_final, cmap=cm.viridis, rstride=1, cstride=1,
                 linewidth=0, antialiased=False)
ax.set_xlim(0.0, Lx)
ax.set_ylim(0.0, Ly)
ax.set_zlim(np.min(p_final), np.max(p_final))
ax.view_init(30, 225)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Jacobi sweep Poisson solution')

plt.tight_layout()
plt.savefig('prior_poisson.png', bbox_inches='tight', dpi=300)
plt.show()
