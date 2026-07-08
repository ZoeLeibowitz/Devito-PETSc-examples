import numpy as np
import matplotlib.pyplot as plt
from devito import Grid, Function, Eq, Operator, NODE

implicit_diffusion = True

if implicit_diffusion:
    string = 'implicit_diffusion'
else:
    string = 'explicit_diffusion'

def compute_vorticity(rank):
    u_arr = np.loadtxt(f'u_original_{rank}.txt')
    v_arr = np.loadtxt(f'v_original_{rank}.txt')

    nx, ny = u_arr.shape
    grid = Grid(shape=(nx, ny), extent=(1., 1.))
    x, y = grid.dimensions

    u_fn = Function(name='u_fn', grid=grid, space_order=2, staggered=y)
    v_fn = Function(name='v_fn', grid=grid, space_order=2, staggered=x)
    u_fn.data[:] = u_arr
    v_fn.data[:] = v_arr

    # compute vorticity at pressure nodes
    vorticity = Function(name='vorticity', grid=grid, space_order=2, staggered=(x,y))
    op = Operator([Eq(vorticity, v_fn.dx - u_fn.dy, subdomain=grid.interior)])
    op.apply()

    omega = vorticity.data[:]

    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)

    plt.figure(figsize=(6, 5))
    plt.contourf(x_coords, y_coords, omega.T, levels=50, cmap='RdBu_r')
    plt.colorbar(label='vorticity')
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(f'Vorticity (serial recompute, rank={rank})')
    plt.tight_layout()
    plt.savefig(f'vorticity_serial_{string}_{rank}.png', dpi=150, bbox_inches='tight')
    plt.show()

    return omega


if __name__ == '__main__':
    compute_vorticity(2)
