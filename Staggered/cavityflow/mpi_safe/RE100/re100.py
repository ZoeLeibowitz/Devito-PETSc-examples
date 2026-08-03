import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from matplotlib import pyplot
from solver import make_solver
from devito.mpi.distributed import MPI


rank = MPI.COMM_WORLD.Get_size()

# Build solver
grid_size = 97
run_solver = make_solver(nx=grid_size, ny=grid_size, ab2=True, implicit_diffusion=True)


t_end = 21

# _original is before interpolation back to node
x, y, U_data, V_data, Omega_data, Stream_data, my_rank, u_original, v_original = run_solver(100, tol=1e-4, t_end=t_end, fixed=True)

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
    pyplot.contourf(x, y, U_data, levels=20)
    pyplot.colorbar()
    pyplot.xlabel('x')
    pyplot.ylabel('y')
    pyplot.title('u-velocity')

    pyplot.subplot(1, 2, 2)
    pyplot.contourf(x, y, V_data, levels=20)
    pyplot.colorbar()
    pyplot.xlabel('x')
    pyplot.ylabel('y')
    pyplot.title('v-velocity')
    pyplot.show()
    pyplot.savefig(f'velocity_fields_{rank}.png', dpi=150, bbox_inches='tight')

    # vorticity
    pyplot.figure(figsize=(6, 5))
    pyplot.contourf(x, y, Omega_data, levels=20)
    pyplot.colorbar()
    pyplot.xlabel('x')
    pyplot.ylabel('y')
    pyplot.title('vorticity')
    pyplot.show()
    pyplot.savefig(f'vorticity_field_{rank}.png', dpi=150, bbox_inches='tight')


    # stream function
    pyplot.figure(figsize=(6, 5))
    pyplot.contourf(x, y, Stream_data, levels=20)
    pyplot.colorbar()
    pyplot.xlabel('x')
    pyplot.ylabel('y')
    pyplot.title('stream function')
    pyplot.show()
    pyplot.savefig(f'stream_function_{rank}.png', dpi=150, bbox_inches='tight')


    # Devito Poisson stream function
    # Need to find primary vortext center location
    # psi_d = np.array(Stream_data)
    # Only look at interior to avoid boundary ... The streamfunction is minimum at the vortex center
    # np.argmin returns index (as a single flat number) of the smallest value
    flat_idx = np.argmin(Stream_data[1:-1, 1:-1])
    # convert flat index into 2D coordinates
    i, j = np.unravel_index(flat_idx, Stream_data[1:-1, 1:-1].shape)
    # shift back to original indexing because we just inspected the interior
    i += 1; j += 1
    

    # Instead of taking the this point as the minimum, to make it more accurate fit a smooth curve through nearby points
    # and take minimum 


    # To get more accurate minimum??, fit parabola to the 3 points around the minimum in x and y directions, and find the vertex of the parabola
    # Takes 3 points vertically and fits a quadratic f(x) = ax**2+bx+c then cx = [a,b,c]
    cx = np.polyfit(x[i-1:i+2], Stream_data[i-1:i+2, j], 2)
    # xmin = -b/(2a)
    x_vortex = -cx[1] / (2 * cx[0])
    cy = np.polyfit(y[j-1:j+2], Stream_data[i, j-1:j+2], 2)
    y_vortex = -cy[1] / (2 * cy[0])
    # put refined x_vortex back into the x dir parabola to give estimate of psi at vortex center
    psi_min = np.polyval(cx, x_vortex)

    # Then we find omega at (x_vortex, y_vortex) using bilinear interpolation
    # Bilinear interpolation of omega to sub-grid vortex location
    from scipy.interpolate import RegularGridInterpolator
    omega_interp = RegularGridInterpolator((x, y), Omega_data, method='linear')
    omega_at_center = float(omega_interp([[x_vortex, y_vortex]]))

    gs = f'{grid_size}x{grid_size}'
    with open(f'vortex_center_re100_{rank}.txt', 'w') as f:
        f.write(f'{"quantity":>12}  {f"devito ({gs})":>17}  {"Kim & Moin (65x65)":>18}  {"Ghia et al. (129x129)":>21}  {"Schreiber & Keller (121x121)":>28}  {"Vanka (321x321)":>15}  {"Wright (1024x1024)":>18}\n')
        f.write(f'{"x_center":>12}  {x_vortex:17.6f}  {"-":>18}  {"0.6172":>21}  {"0.61667":>28}  {"0.6188":>15}  {"0.6157":>18}\n')
        f.write(f'{"y_center":>12}  {y_vortex:17.6f}  {"-":>18}  {"0.7344":>21}  {"0.74167":>28}  {"0.7375":>15}  {"0.7378":>18}\n')
        f.write(f'{"psi":>12}  {psi_min:17.6f}  {"-0.103":>18}  {"-0.103423":>21}  {"-0.10330":>28}  {"0.1034":>15}  {"0.103516":>18}\n')
        f.write(f'{"omega":>12}  {omega_at_center:17.6f}  {"-3.177":>18}  {"3.16646":>21}  {"-3.18200":>28}  {"-":>15}  {"3.17044":>18}\n')

    # Ghia, u vel, Re=100
    y_ghia = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344,
                       0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703,
                       0.0625, 0.0547, 0.0000])
    u_ghia = np.array([ 1.00000,  0.84123,  0.78871,  0.73722,  0.68717,  0.23151,
                        0.00332, -0.13641, -0.20581, -0.21090, -0.15662, -0.10150,
                       -0.06434, -0.04775, -0.04192, -0.03717,  0.00000])

    # GPU paper (same y positions as Ghia) — please verify against original table
    u_gpu = np.array([ 1.0000,  0.8500,  0.8057,  0.7512,  0.6982,  0.2398,
                       0.0070, -0.1379, -0.2091, -0.2140, -0.1580, -0.1019,
                      -0.0648, -0.0469, -0.0419, -0.0368,  0.0000])

    # Interpolate my u at x=0.5 to the Ghia y positions (interpolation is just used for the table, not the plot)
    from scipy.interpolate import interp1d
    i_mid = np.argmin(np.abs(x - 0.5))
    u_centreline = U_data[i_mid, :]
    u_interp_fn = interp1d(y, u_centreline, kind='linear')
    u_mine_at_ghia = u_interp_fn(y_ghia)

    # Ghia, v-vel, Re=100
    x_ghia = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594,
                       0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781,
                       0.0703, 0.0625, 0.0000])
    v_ghia = np.array([ 0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914,
                       -0.22445, -0.24533,  0.05454,  0.17527,  0.17507,  0.16077,
                        0.12317,  0.10890,  0.10091,  0.09233,  0.00000])

    # GPU paper (it used same positiions as Ghia)
    v_gpu = np.array([ 0.0000, -0.0580, -0.0747, -0.0911, -0.1041, -0.1750,
                      -0.2323, -0.2536,  0.0575,  0.1796,  0.1794,  0.1652,
                       0.1266,  0.1127,  0.1040,  0.0947,  0.0000])

    # Interpolate my v at y=0.5 to the Ghia x positions (interpolation is just used for the table, not the plot)
    j_mid = np.argmin(np.abs(y - 0.5))
    v_centreline = V_data[:, j_mid]
    v_interp_fn = interp1d(x, v_centreline, kind='linear')
    v_mine_at_ghia = v_interp_fn(x_ghia)

    # Comparison tables
    with open(f'u_centreline_re100_{rank}.txt', 'w') as f:
        f.write(f'{"y":>8}  {"u_present":>12}  {"u_ghia":>12}\n')
        for yg, um, ug in zip(y_ghia, u_mine_at_ghia, u_ghia):
            f.write(f'{yg:8.4f}  {um:12.5f}  {ug:12.5f}\n')
    with open(f'v_centreline_re100_{rank}.txt', 'w') as f:
        f.write(f'{"x":>8}  {"v_present":>12}  {"v_ghia":>12}\n')
        for xg, vm, vg in zip(x_ghia, v_mine_at_ghia, v_ghia):
            f.write(f'{xg:8.4f}  {vm:12.5f}  {vg:12.5f}\n')

    vort_data = -np.array(Omega_data).T  # matches Ghia sign convention
    psi_devito = np.array(Stream_data).T

    # vorticity contour levels — Ghia Table III
    vort_levels = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    vort_fmt = {-3.0: '-4', -2.0: '-3', -1.0: '-2', -0.5: '-1', 0.0: '0',
                 0.5: '+1', 1.0: '2', 2.0: '3', 3.0: '+4', 4.0: '5', 5.0: '6'}

    # Stream func contour levels - Ghia table III
    psi_levels = sorted([-0.1175, -0.115, -0.11, -0.1, -0.09, -0.07, -0.05, -0.03,
                         -0.01, -1e-4, -1e-5, -1e-7, -1e-10,
                          1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 1.5e-3, 3e-3])

    fig, (ax_psi_devito, ax_vort) = pyplot.subplots(1, 2, figsize=(12, 6))

    ax_psi_devito.contour(x, y, psi_devito, levels=psi_levels, colors='k', linewidths=0.6)
    ax_psi_devito.set_aspect('equal')
    ax_psi_devito.set_xlim(0, 1); ax_psi_devito.set_ylim(0, 1)
    ax_psi_devito.set_xticks([]); ax_psi_devito.set_yticks([])
    ax_psi_devito.set_title('Stream function')

    cs_vort = ax_vort.contour(x, y, vort_data, levels=vort_levels, colors='k', linewidths=0.6)
    ax_vort.clabel(cs_vort, inline=True, fontsize=7, fmt=vort_fmt)
    ax_vort.set_aspect('equal')
    ax_vort.set_xlim(0, 1); ax_vort.set_ylim(0, 1)
    ax_vort.set_xticks([]); ax_vort.set_yticks([])
    ax_vort.set_title('Vorticity')

    pyplot.suptitle(f'Re=100, {gs}')
    pyplot.tight_layout()
    pyplot.savefig(f're100_{rank}.png', dpi=150, bbox_inches='tight')

    # u and v centrelines
    fig_uv, (ax_u, ax_v) = pyplot.subplots(1, 2, figsize=(10, 6))

    ax_u.plot(u_centreline, y, 'k-', linewidth=1.2, label=f'Present ({gs})')
    ax_u.plot(u_ghia, y_ghia, 'ko', markersize=5, fillstyle='none', label='Ghia et al. (129x129)')
    ax_u.plot(u_gpu, y_ghia, 'b^', markersize=4, fillstyle='none', label='GPU paper')
    ax_u.axvline(0, color='k', linewidth=0.4, linestyle='--')
    ax_u.set_xlabel('u')
    ax_u.set_ylabel('y')
    ax_u.set_title('u along vertical centreline')
    ax_u.legend()

    ax_v.plot(x, v_centreline, 'k-', linewidth=1.2, label=f'Present ({gs})')
    ax_v.plot(x_ghia, v_ghia, 'ko', markersize=5, fillstyle='none', label='Ghia et al.')
    ax_v.plot(x_ghia, v_gpu, 'b^', markersize=4, fillstyle='none', label='GPU paper')
    ax_v.axhline(0, color='k', linewidth=0.4, linestyle='--')
    ax_v.set_xlabel('x')
    ax_v.set_ylabel('v')
    ax_v.set_title('v along horizontal centreline')
    ax_v.legend()

    pyplot.suptitle(f'Re=100, {gs}')
    pyplot.tight_layout()
    pyplot.savefig(f'centreline_re100_{rank}.png', dpi=150, bbox_inches='tight')

    pyplot.show()
