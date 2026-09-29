"""
Steady-state Poisson equation, adapted from Devito's tutorial notebook
(devito/examples/cfd/06_poisson.ipynb, "Example 6: Poisson equation").

Solves the same problem as solving_poisson_with_snes/snes_poisson_2d.c
(ref - https://github.com/bueler/p4pdes/blob/master/c/ch6/fish.c):
    -p_xx - p_yy = b(x,y) = x e^y on the unit square,
    Dirichlet BCs from the exact solution p(x,y) = -x e^y:
    p(0,y) = 0, p(1,y) = -e^y, p(x,0) = -x, p(x,1) = -x e.
"""
import numpy as np

from devito import (Grid, Function, TimeFunction, Eq, Operator, solve,
                    configuration, Border)
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

configuration['log-level'] = 'ERROR'


nx, ny = 17, 17
nt = 2000
Lx, Ly = 1.0, 1.0

grid = Grid(shape=(nx, ny), extent=(Lx, Ly), dtype=np.float64)

# Boundary layer of thickness 1 on all sides, to implement BCs
border = Border(grid, 1)

x_coord = np.linspace(0, Lx, nx)
y_coord = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x_coord, y_coord, indexing='ij')

b = Function(name='b', grid=grid)
b.data[:] = X * np.exp(Y)

p_exact = -X * np.exp(Y)

# Only the boundary values are used, via the Border BC
bc = Function(name='bc', grid=grid)
bc.data[:] = p_exact

p = TimeFunction(name='p', grid=grid, space_order=2)
p.data[:] = 0.

# Define the Poisson equation for `p`
eqn = Eq(-p.laplace, b)
# Solve for the central stencil point
stencil = solve(eqn, p)
# Let stencil populate the buffer `p.forward`, restricted to grid.interior
stencil = Eq(p.forward, stencil, subdomain=grid.interior)

# Dirichlet BCs, set on the boundary
bc_eq = Eq(p.forward, bc, subdomain=border)

op = Operator([stencil, bc_eq])
op.apply(time=nt)

buffer_size = p.time_order + 1
final_idx = (nt + 1) % buffer_size  # op.apply(time=nt) runs nt + 1 sweeps
p_final = p.data[final_idx]

errinf = np.abs(p_final - p_exact).max()
err2h = np.linalg.norm(p_final - p_exact) / np.sqrt((nx - 1) * (ny - 1))
print(f"problem on {nx} x {ny} point 2D grid:\n"
      f"  error |p-pexact|_inf = {errinf:.3e}, |p-pexact|_h = {err2h:.3e}")

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
