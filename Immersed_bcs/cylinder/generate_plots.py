import numpy as np
import matplotlib.colors as mcolors
from matplotlib import pyplot

# DFG benchmark's own physical units, same as solver_cylinder.make_solver /
# re20.py. Reads the fields re20.py saved to disk and reproduces the plots
# that used to be generated inline there.
x_extent, y_extent = 2.2, 0.41
centre_x, centre_y, radius = 0.2, 0.2, 0.05

# DFG reference plots use ParaView's default "Blue to Red Rainbow" preset, not
# matplotlib's jet -- jet has extra dark navy/maroon segments at each end, so
# its high/low extremes read much darker than the reference. This is the
# plain 5-stop blue-cyan-green-yellow-red ramp ParaView uses instead.
paraview_rainbow = mcolors.LinearSegmentedColormap.from_list(
    'paraview_rainbow', ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000'])

U_data = np.loadtxt(f'u_data.txt')
V_data = np.loadtxt(f'v_data.txt')
P_data = np.loadtxt(f'p_data.txt')
Omega_data = np.loadtxt(f'omega_data.txt')
Stream_data = np.loadtxt(f'psi_data.txt')
u_original = np.loadtxt(f'u_original.txt')
v_original = np.loadtxt(f'v_original.txt')
p_original = np.loadtxt(f'p_original.txt')

node_mask = np.loadtxt(f'node_mask.txt').astype(bool)
# masks matching v's/u's/p's own staggering (sdf_x/sdf_y/sdf_x_y), for
# masking the raw pre-interpolation u_original/v_original/p_original fields,
# which live on those subgrids, not the NODE one node_mask covers
x_mask = np.loadtxt(f'x_mask.txt').astype(bool)
y_mask = np.loadtxt(f'y_mask.txt').astype(bool)
x_y_mask = np.loadtxt(f'x_y_mask.txt').astype(bool)

nx, ny = U_data.shape
x = np.linspace(0, x_extent, nx)
y = np.linspace(0, y_extent, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]

imshow_extent = (x[0] - dx/2, x[-1] + dx/2, y[0] - dy/2, y[-1] + dy/2)


def draw_cylinder(ax):
    ax.add_patch(pyplot.Circle((centre_x, centre_y), radius,
                                facecolor='white', edgecolor='black', linewidth=0.75, zorder=5))


def masked(field, mask=node_mask):
    field = field.copy()
    field[~mask] = np.nan
    return field


def nearest_valid(field, px, py):
    X, Y = np.meshgrid(x, y, indexing='ij')
    dist = np.sqrt((X - px)**2 + (Y - py)**2)
    dist = np.where(node_mask, dist, np.inf)
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    return field[i, j], x[i], y[j]


# DFG 2D-1 benchmark points sit exactly on the cylinder surface, which schism
# excludes -- so read pressure from the nearest point that's actually part of
# the solve instead (see project notes: nearest literal point lands in the
# excluded band and reads a meaningless exact 0). Coordinates actually used
# are reported alongside delta_p since they're not the literal benchmark
# points (0.15,0.2)/(0.25,0.2) -- this is not a one-to-one comparison.
dp_target = 0.11752016697
p_front, xf, yf = nearest_valid(P_data, 0.15, 0.2)
p_rear, xr, yr = nearest_valid(P_data, 0.25, 0.2)
delta_p = p_front - p_rear
dp_error_pct = abs(delta_p - dp_target) / dp_target * 100


fig, ((ax_u_orig, ax_v_orig), (ax_u, ax_v)) = pyplot.subplots(2, 2, figsize=(16, 6))

# raw, pre-interpolation staggered data (u/v_original), for comparison against
# the node-interpolated fields below. imshow (not contourf) so individual grid
# points are visible, not smoothed away. Masked with each field's own
# staggering (y_mask/x_mask match u's/v's subgrids, not the NODE one
# node_mask covers) so the never-updated points read as blank, not stale IC.
im_u_orig = ax_u_orig.imshow(masked(u_original, y_mask).T, origin='lower', extent=imshow_extent)
pyplot.colorbar(im_u_orig, ax=ax_u_orig)
draw_cylinder(ax_u_orig)
ax_u_orig.set_xlabel('x')
ax_u_orig.set_ylabel('y')
ax_u_orig.set_title('u-velocity (original, pre-interpolation)')
ax_u_orig.set_aspect('equal')

im_v_orig = ax_v_orig.imshow(masked(v_original, x_mask).T, origin='lower', extent=imshow_extent)
pyplot.colorbar(im_v_orig, ax=ax_v_orig)
draw_cylinder(ax_v_orig)
ax_v_orig.set_xlabel('x')
ax_v_orig.set_ylabel('y')
ax_v_orig.set_title('v-velocity (original, pre-interpolation)')
ax_v_orig.set_aspect('equal')

im_u = ax_u.imshow(masked(U_data).T, origin='lower', extent=imshow_extent)
pyplot.colorbar(im_u, ax=ax_u)
draw_cylinder(ax_u)
ax_u.set_xlabel('x')
ax_u.set_ylabel('y')
ax_u.set_title('u-velocity')
ax_u.set_aspect('equal')

im_v = ax_v.imshow(masked(V_data).T, origin='lower', extent=imshow_extent)
pyplot.colorbar(im_v, ax=ax_v)
draw_cylinder(ax_v)
ax_v.set_xlabel('x')
ax_v.set_ylabel('y')
ax_v.set_title('v-velocity')
ax_v.set_aspect('equal')

pyplot.tight_layout()
pyplot.savefig(f'velocity_fields.png', dpi=150, bbox_inches='tight')

# pressure, pre-interpolation, masked with x_y_mask (p's own staggered
# subgrid -- already the tightest of the three, no fluid-side exclusion at
# its current cutoff, unlike node_mask which P_data/plotfunc_p use elsewhere)
fig_p_orig, ax_p_orig = pyplot.subplots(figsize=(11, 3))
im_p_orig = ax_p_orig.imshow(masked(p_original, x_y_mask).T, origin='lower', extent=imshow_extent)
pyplot.colorbar(im_p_orig, ax=ax_p_orig)
draw_cylinder(ax_p_orig)
ax_p_orig.set_xlabel('x')
ax_p_orig.set_ylabel('y')
ax_p_orig.set_title('pressure (original, pre-interpolation)')
ax_p_orig.set_aspect('equal')
pyplot.savefig(f'pressure_original.png', dpi=150, bbox_inches='tight')

# vorticity
fig_omega, ax_omega = pyplot.subplots(figsize=(6, 5))
im_omega = ax_omega.contourf(x, y, Omega_data.T, levels=20)
pyplot.colorbar(im_omega, ax=ax_omega)
draw_cylinder(ax_omega)
ax_omega.set_xlabel('x')
ax_omega.set_ylabel('y')
ax_omega.set_title('vorticity')
ax_omega.set_aspect('equal')
pyplot.savefig(f'vorticity_field.png', dpi=150, bbox_inches='tight')

# stream function
fig_psi, ax_psi_solo = pyplot.subplots(figsize=(6, 5))
im_psi_solo = ax_psi_solo.contourf(x, y, Stream_data.T, levels=20)
pyplot.colorbar(im_psi_solo, ax=ax_psi_solo)
draw_cylinder(ax_psi_solo)
ax_psi_solo.set_xlabel('x')
ax_psi_solo.set_ylabel('y')
ax_psi_solo.set_title('stream function')
ax_psi_solo.set_aspect('equal')
pyplot.savefig(f'stream_function.png', dpi=150, bbox_inches='tight')

# DFG benchmark's own 3 result plots (velocity magnitude, pressure,
# streamfunction), see
# https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html
speed = np.sqrt(U_data**2 + V_data**2)

fig, (ax_speed, ax_p, ax_psi) = pyplot.subplots(3, 1, figsize=(10, 8))

# vmin/vmax match the DFG reference plots' own colorbars (read off the
# reference images at https://wwwold.mathematik.tu-dortmund.de/~featflow/
# en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html) so colors
# are directly comparable to the benchmark page's figures.
im_speed = ax_speed.pcolormesh(x, y, masked(speed).T, shading='gouraud', cmap=paraview_rainbow, vmin=0, vmax=0.405)
pyplot.colorbar(im_speed, ax=ax_speed)
draw_cylinder(ax_speed)
ax_speed.set_title('Velocity magnitude')
ax_speed.set_aspect('equal')

im_p = ax_p.pcolormesh(x, y, masked(P_data).T, shading='gouraud', cmap=paraview_rainbow, vmin=-0.0115, vmax=0.131)
pyplot.colorbar(im_p, ax=ax_p)
draw_cylinder(ax_p)
ax_p.set_title('Pressure')
ax_p.set_aspect('equal')
ax_p.text(0.98, 0.05,
          f'$\\Delta p$ (nearest valid pt) = {delta_p:.4f}\n'
          f'target = {dp_target:.4f}  (error {dp_error_pct:.2f}%)\n'
          f'front pt used: ({xf:.2f},{yf:.2f})  vs target (0.15,0.20)\n'
          f'rear pt used:  ({xr:.2f},{yr:.2f})  vs target (0.25,0.20)',
          transform=ax_p.transAxes, ha='right', va='bottom', fontsize=8,
          bbox=dict(facecolor='white', edgecolor='black', linewidth=0.5, alpha=0.85))

im_psi = ax_psi.pcolormesh(x, y, Stream_data.T, shading='gouraud', cmap=paraview_rainbow)
pyplot.colorbar(im_psi, ax=ax_psi)
draw_cylinder(ax_psi)
ax_psi.set_title('Streamfunction')
ax_psi.set_aspect('equal')

for ax in (ax_speed, ax_p, ax_psi):
    ax.set_xlabel('x')
    ax.set_ylabel('y')

pyplot.tight_layout()
pyplot.savefig(f'dfg_benchmark_plots.png', dpi=150, bbox_inches='tight')
