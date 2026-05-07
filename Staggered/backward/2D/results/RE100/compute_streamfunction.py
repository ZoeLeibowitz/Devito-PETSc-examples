import numpy as np
import matplotlib.pyplot as plt

rank = 1

u_arr = np.loadtxt(f'u_original_{rank}.txt')   # shape (nx, ny): u staggered in y
nx, ny = u_arr.shape                             # ny=41 includes ghost cell at top

# Grid parameters (uniform, matching solver.py with ny=41)
ny_nodes = ny - 1                                # 40 cells in y → 41 nodes
dy = 2.0 / ny_nodes                              # = 0.05
dx = dy
x_extent = (nx - 1) * dx                        # = 30.0

x_coords = np.linspace(0, x_extent, nx)
# Stream function on cell faces: y = 0, dy, 2dy, ..., 2h  (ny values)
y_faces = np.arange(ny_nodes + 1) * dy          # shape (41,), from 0 to 2.0

# psi(x, y) = integral_0^y u(x, y') dy'
# u_arr[:, j] is at y = (j + 0.5)*dy (staggered midpoint of cell j)
# Skip ghost cell at j=ny-1=40; use j=0..39 for the 40 interior cells
psi = np.zeros((nx, ny_nodes + 1))              # shape (601, 41)
psi[:, 1:] = np.cumsum(u_arr[:, :ny_nodes] * dy, axis=1)

# Contour levels: evenly spaced between min and max, always including 0
psi_min, psi_max = psi.min(), psi.max()
neg_levels = np.linspace(psi_min, 0, 12)[:-1]   # negative levels (recirculation)
pos_levels = np.linspace(0, psi_max, 20)          # positive levels (main flow)
levels = np.concatenate([neg_levels, pos_levels])

fig, ax = plt.subplots(figsize=(14, 3))
cs = ax.contour(x_coords, y_faces, psi.T, levels=levels, colors='k', linewidths=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Stream function (rank={rank})')
ax.set_xlim(0, 15)
ax.set_ylim(0, 2)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(f'streamfunction_{rank}.png', dpi=150, bbox_inches='tight')
plt.show()

# Total flow rate at inlet (should be 1.0 for parabolic inlet with U_avg=1, h=1)
Q = psi[0, -1]
print(f'Total flow rate Q = {Q:.6f}  (expected 1.0)')
print(f'psi range: [{psi_min:.4f}, {psi_max:.4f}]')
print(f'Saved streamfunction_{rank}.png')
