import sys, os

import numpy as np
from matplotlib import pyplot
from solver import make_solver
from devito.mpi.distributed import MPI


rank = MPI.COMM_WORLD.Get_size()

# Build solver: dx=dy=0.05 over the 22 x 4.1 channel
nx, ny = 441, 83
run_solver = make_solver(nx=nx, ny=ny, ab2=False, implicit_diffusion=True)


t_end = 10

# _original is before interpolation back to node
x, y, U_data, V_data, Omega_data, Stream_data, my_rank, u_original, v_original = run_solver(20, tol=1e-4, t_end=t_end, fixed=True)

if my_rank == 0:
    # staggered u and v
    np.savetxt(f'u_original_{rank}.txt', u_original, fmt='%12.6f')
    np.savetxt(f'v_original_{rank}.txt', v_original, fmt='%12.6f')

if my_rank == 0:

    np.savetxt(f'omega_data_{rank}.txt', Omega_data, fmt='%12.6f')

    np.savetxt(f'u_data_{rank}.txt', U_data, fmt='%12.6f')

    np.savetxt(f'v_data_{rank}.txt', V_data, fmt='%12.6f')


    pyplot.figure(figsize=(10, 5))
    pyplot.subplot(1, 2, 1)
    pyplot.contourf(x, y, U_data.T, levels=20)
    pyplot.colorbar()
    pyplot.xlabel('x')
    pyplot.ylabel('y')
    pyplot.title('u-velocity')

    pyplot.subplot(1, 2, 2)
    pyplot.contourf(x, y, V_data.T, levels=20)
    pyplot.colorbar()
    pyplot.xlabel('x')
    pyplot.ylabel('y')
    pyplot.title('v-velocity')
    pyplot.show()
    pyplot.savefig(f'velocity_fields_{rank}.png', dpi=150, bbox_inches='tight')

    # vorticity
    pyplot.figure(figsize=(6, 5))
    pyplot.contourf(x, y, Omega_data.T, levels=20)
    pyplot.colorbar()
    pyplot.xlabel('x')
    pyplot.ylabel('y')
    pyplot.title('vorticity')
    pyplot.show()
    pyplot.savefig(f'vorticity_field_{rank}.png', dpi=150, bbox_inches='tight')


    # stream function
    pyplot.figure(figsize=(6, 5))
    pyplot.contourf(x, y, Stream_data.T, levels=20)
    pyplot.colorbar()
    pyplot.xlabel('x')
    pyplot.ylabel('y')
    pyplot.title('stream function')
    pyplot.show()
    pyplot.savefig(f'stream_function_{rank}.png', dpi=150, bbox_inches='tight')
