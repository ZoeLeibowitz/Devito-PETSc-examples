import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from matplotlib import pyplot
from solver import make_solver


rank = 1

# Build solver
h = 1       # step height
Re = 50

grid_size = 41
run_solver = make_solver(ny=grid_size, nx=None, ab2=True, implicit_diffusion=False)

t_end = 55

# _original is before interpolation back to node
x, y, U_data, V_data, Omega_data, Stream_data, my_rank, u_original, v_original = run_solver(Re, tol=1e-4, t_end=t_end, fixed=True)

if my_rank == 0:
    np.savetxt(f'u_original_{rank}.txt', u_original, fmt='%12.6f')
    np.savetxt(f'v_original_{rank}.txt', v_original, fmt='%12.6f')
    np.savetxt(f'omega_data_{rank}.txt', Omega_data, fmt='%12.6f')
    np.savetxt(f'u_data_{rank}.txt', U_data, fmt='%12.6f')
    np.savetxt(f'v_data_{rank}.txt', V_data, fmt='%12.6f')

    gs = f'{len(x)}x{grid_size}'

    # --- velocity fields (interpolated to nodes) ---
    pyplot.figure(figsize=(14, 4))
    pyplot.subplot(1, 2, 1)
    pyplot.contourf(x, y, U_data.T, levels=20)
    pyplot.colorbar()
    pyplot.xlabel('x'); pyplot.ylabel('y'); pyplot.title('u-velocity')
    pyplot.subplot(1, 2, 2)
    pyplot.contourf(x, y, V_data.T, levels=20)
    pyplot.colorbar()
    pyplot.xlabel('x'); pyplot.ylabel('y'); pyplot.title('v-velocity')
    pyplot.tight_layout()
    pyplot.savefig(f'velocity_fields_{rank}.png', dpi=150, bbox_inches='tight')
    pyplot.show()

    # --- velocity fields (original staggered) ---
    pyplot.figure(figsize=(14, 4))
    pyplot.subplot(1, 2, 1)
    pyplot.contourf(x, y, u_original.T, levels=20)
    pyplot.colorbar()
    pyplot.xlabel('x'); pyplot.ylabel('y'); pyplot.title('u-velocity (original staggered)')
    pyplot.subplot(1, 2, 2)
    pyplot.contourf(x, y, v_original.T, levels=20)
    pyplot.colorbar()
    pyplot.xlabel('x'); pyplot.ylabel('y'); pyplot.title('v-velocity (original staggered)')
    pyplot.tight_layout()
    pyplot.savefig(f'velocity_fields_original_{rank}.png', dpi=150, bbox_inches='tight')
    pyplot.show()

    # vorticity
    pyplot.figure(figsize=(14, 3))
    pyplot.contourf(x, y, Omega_data.T, levels=20)
    pyplot.colorbar()
    pyplot.xlabel('x'); pyplot.ylabel('y'); pyplot.title('vorticity')
    pyplot.tight_layout()
    pyplot.savefig(f'vorticity_field_{rank}.png', dpi=150, bbox_inches='tight')
    pyplot.show()

    # stream function
    pyplot.figure(figsize=(14, 3))
    pyplot.contourf(x, y, Stream_data.T, levels=20)
    pyplot.colorbar()
    pyplot.xlabel('x'); pyplot.ylabel('y'); pyplot.title('stream function')
    pyplot.tight_layout()
    pyplot.savefig(f'stream_function_{rank}.png', dpi=150, bbox_inches='tight')
    pyplot.show()

    # reattachment length
    # U_data is indexed [x_idx, y_idx], so U_data[:, 0] is near-wall u at y≈dy/2 for all x.
    u_bottom = U_data[:, 0]   # near-wall u along y=0, shape (nx,)
    sign_changes = np.where(np.diff(np.sign(u_bottom)))[0]
    x_r = None
    for idx in sign_changes:
        if x[idx] > h:
            x0, x1 = x[idx], x[idx + 1]
            u0, u1 = u_bottom[idx], u_bottom[idx + 1]
            x_r = x0 - u0 * (x1 - x0) / (u1 - u0)
            break

    with open(f'reattachment_re100_{rank}.txt', 'w') as f:
        f.write(f'Re={Re}, grid={gs}\n')
        if x_r is not None:
            f.write(f'Reattachment length x_r/h = {x_r/h:.4f}  (x_r = {x_r:.4f})\n')
            f.write('Kim & Moin (1985) Re=100: x_r/h ~ 3.0\n')
        else:
            f.write('No reattachment detected (flow may not have reached steady state)\n')

    print(f'Reattachment x_r/h = {x_r/h:.4f}' if x_r else 'No reattachment detected')

    # --- u profiles at several x-stations ---
    x_stations = [1.0, 3.0, 5.0, 7.0, 10.0, 15.0]
    fig, axes = pyplot.subplots(1, len(x_stations), figsize=(14, 4), sharey=True)
    for ax, xs in zip(axes, x_stations):
        i_s = np.argmin(np.abs(x - xs))
        ax.plot(U_data[i_s, :], y, 'k-', linewidth=1.2)
        ax.axvline(0, color='k', linewidth=0.4, linestyle='--')
        ax.set_title(f'x={xs}')
        ax.set_xlabel('u')
    axes[0].set_ylabel('y')
    pyplot.suptitle(f'u-velocity profiles, Re={Re}, {gs}')
    pyplot.tight_layout()
    pyplot.savefig(f'u_profiles_{rank}.png', dpi=150, bbox_inches='tight')
    pyplot.show()
