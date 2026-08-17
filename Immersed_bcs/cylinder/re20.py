import numpy as np
from solver_cylinder import make_solver
from devito.mpi.distributed import MPI


# Build solver in the DFG benchmark's own physical units: dx=dy=0.01 over the
# 2.2 x 0.41 channel (D=0.1 cylinder), same relative resolution (dx/D=0.1) as
# the previous x10-scaled grid, so cost/step-count is unchanged.
nx, ny = 331, 62
# nx, ny = 441, 83
# nx, ny = 221, 42
run_solver = make_solver(nx=nx, ny=ny, ab2=False, implicit_diffusion=False, u_max=0.3)

t_end = 20

suffix = '_331_62_04'

# _original is before interpolation back to node
# fixed=False: run in chunks, checking ||u(t)-u(t-dt)|| every check_every steps
# and stopping once the relative change drops below tol, rather than blindly
# running to a guessed t_end.
x, y, U_data, V_data, Omega_data, Stream_data, my_rank, u_original, v_original, p_original, P_data, node_mask, x_mask, y_mask, x_y_mask = run_solver(20, tol=1e-5, t_end=t_end, check_every=400, fixed=True)

# All plotting lives in generate_plots.py -- this script just runs the solve
# and saves every field needed to reproduce the plots from disk.
if my_rank == 0:
    # staggered u, v and p (pre-interpolation, raw solve data)
    np.savetxt(f'u_original{suffix}.txt', u_original, fmt='%12.6f')
    np.savetxt(f'v_original{suffix}.txt', v_original, fmt='%12.6f')
    np.savetxt(f'p_original{suffix}.txt', p_original, fmt='%12.6f')

    # node-interpolated fields
    np.savetxt(f'u_data{suffix}.txt', U_data, fmt='%12.6f')
    np.savetxt(f'v_data{suffix}.txt', V_data, fmt='%12.6f')
    np.savetxt(f'p_data{suffix}.txt', P_data, fmt='%12.6f')
    np.savetxt(f'omega_data{suffix}.txt', Omega_data, fmt='%12.6f')
    np.savetxt(f'psi_data{suffix}.txt', Stream_data, fmt='%12.6f')

    np.savetxt(f'node_mask{suffix}.txt', node_mask, fmt='%d')
    # masks matching v's/u's/p's own staggering, for masking the raw
    # pre-interpolation u_original/v_original/p_original fields
    np.savetxt(f'x_mask{suffix}.txt', x_mask, fmt='%d')
    np.savetxt(f'y_mask{suffix}.txt', y_mask, fmt='%d')
    np.savetxt(f'x_y_mask{suffix}.txt', x_y_mask, fmt='%d')
