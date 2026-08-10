import sys, os

import numpy as np
from matplotlib import pyplot
from solver_cylinder import make_solver
from devito.mpi.distributed import MPI


rank = MPI.COMM_WORLD.Get_size()

# Build solver in the DFG benchmark's own physical units: dx=dy=0.01 over the
# 2.2 x 0.41 channel (D=0.1 cylinder), same relative resolution (dx/D=0.1) as
# the previous x10-scaled grid, so cost/step-count is unchanged.
nx, ny = 221, 42
# TODO: switch to implicit etc..
run_solver = make_solver(nx=nx, ny=ny, ab2=False, implicit_diffusion=False, u_max=0.3)

t_end = 50.0

# _original is before interpolation back to node
# fixed=False: run in chunks, checking ||u(t)-u(t-dt)|| every check_every steps
# and stopping once the relative change drops below tol, rather than blindly
# running to a guessed t_end.
x, y, U_data, V_data, Omega_data, Stream_data, my_rank, u_original, v_original, delta_p, P_data = run_solver(20, tol=1e-5, t_end=t_end, check_every=400, fixed=True)

if my_rank == 0:
    # staggered u and v
    np.savetxt(f'u_original_{rank}.txt', u_original, fmt='%12.6f')
    np.savetxt(f'v_original_{rank}.txt', v_original, fmt='%12.6f')

if my_rank == 0:

    np.savetxt(f'omega_data_{rank}.txt', Omega_data, fmt='%12.6f')

    np.savetxt(f'u_data_{rank}.txt', U_data, fmt='%12.6f')

    np.savetxt(f'v_data_{rank}.txt', V_data, fmt='%12.6f')


    fig, (ax_u, ax_v) = pyplot.subplots(2, 1, figsize=(10, 6))

    im_u = ax_u.contourf(x, y, U_data.T, levels=20)
    pyplot.colorbar(im_u, ax=ax_u)
    ax_u.set_xlabel('x')
    ax_u.set_ylabel('y')
    ax_u.set_title('u-velocity')
    ax_u.set_aspect('equal')

    im_v = ax_v.contourf(x, y, V_data.T, levels=20)
    pyplot.colorbar(im_v, ax=ax_v)
    ax_v.set_xlabel('x')
    ax_v.set_ylabel('y')
    ax_v.set_title('v-velocity')
    ax_v.set_aspect('equal')

    pyplot.tight_layout()
    pyplot.savefig(f'velocity_fields_{rank}.png', dpi=150, bbox_inches='tight')
    pyplot.show()

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

    # DFG benchmark's own 3 result plots (velocity magnitude, pressure,
    # streamfunction), see
    # https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html
    centre_x, centre_y, radius = 0.2, 0.2, 0.05
    speed = np.sqrt(U_data**2 + V_data**2)

    def draw_cylinder(ax):
        ax.add_patch(pyplot.Circle((centre_x, centre_y), radius,
                                    facecolor='white', edgecolor='black', linewidth=0.75, zorder=5))

    fig, (ax_speed, ax_p, ax_psi, ax_text) = pyplot.subplots(4, 1, figsize=(10, 10.5))

    # vmin/vmax match the DFG reference plots' own colorbars (read off the
    # reference images at https://wwwold.mathematik.tu-dortmund.de/~featflow/
    # en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html) so colors
    # are directly comparable to the benchmark page's figures.
    im_speed = ax_speed.pcolormesh(x, y, speed.T, shading='gouraud', cmap='jet', vmin=0, vmax=0.405)
    pyplot.colorbar(im_speed, ax=ax_speed)
    draw_cylinder(ax_speed)
    ax_speed.set_title('Velocity magnitude')
    ax_speed.set_aspect('equal')

    im_p = ax_p.pcolormesh(x, y, P_data.T, shading='gouraud', cmap='jet', vmin=-0.0115, vmax=0.131)
    pyplot.colorbar(im_p, ax=ax_p)
    draw_cylinder(ax_p)
    ax_p.set_title('Pressure')
    ax_p.set_aspect('equal')
    
    im_psi = ax_psi.pcolormesh(x, y, Stream_data.T, shading='gouraud', cmap='jet')
    pyplot.colorbar(im_psi, ax=ax_psi)
    draw_cylinder(ax_psi)
    ax_psi.set_title('Streamfunction')
    ax_psi.set_aspect('equal')

    dp_target = 0.11752016697
    dp_error_pct = abs(delta_p - dp_target) / dp_target * 100

    summary_text = (
        f"DFG 2D-1 benchmark (Re=20, steady)\n\n"
        f"Grid: {nx} x {ny}  (dx=dy=0.01)\n"
        f"t_end: {t_end}\n\n"
        f"$\\Delta p$ computed: {delta_p:.11f}\n"
        f"$\\Delta p$ target:   {dp_target:.11f}\n"
        f"error: {dp_error_pct:.2f}%"
    )
    ax_text.axis('off')
    ax_text.text(0.05, 0.5, summary_text, transform=ax_text.transAxes,
                  fontsize=11, va='center', ha='left', family='monospace')

    for ax in (ax_speed, ax_p, ax_psi):
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    pyplot.tight_layout()
    pyplot.savefig(f'dfg_benchmark_plots_{rank}.png', dpi=150, bbox_inches='tight')
    pyplot.show()
