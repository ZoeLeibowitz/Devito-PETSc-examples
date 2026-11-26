# The MAC grid of the lid driven cavity

from pathlib import Path

import matplotlib.pyplot as plt

U_COLOUR = "#1B5E20"
V_COLOUR = "C1"
P_COLOUR = "C0"
NODE = "#555555"
GRID = "#B0B0B0"
FS = 12
L = 0.17
FADE = 0.25
N = 3


def arrow(ax, x, y, dx, dy, colour, alpha):
    ax.annotate('', xy=(x + dx, y + dy), xytext=(x - dx, y - dy),
                arrowprops=dict(arrowstyle='-|>,head_length=0.45,head_width=0.22',
                                ec=colour, fc=colour, lw=1.8, shrinkA=0, shrinkB=0,
                                alpha=alpha), zorder=4)


def in_centre(x, y):
    return 1 <= x <= 2 and 1 <= y <= 2


fig, ax = plt.subplots(figsize=(6.2, 5.6))

for k in range(N + 1):
    ax.plot([k, k], [0, N], color=GRID, lw=1.0, zorder=1)
    ax.plot([0, N], [k, k], color=GRID, lw=1.0, zorder=1)
for a in range(N + 1):
    for b in range(N + 1):
        ax.plot(a, b, '+', color=NODE, ms=10, mew=1.6, zorder=3,
                alpha=1 if in_centre(a, b) else FADE)
for a in range(N):
    for b in range(N):
        ax.plot(a + 0.5, b + 0.5, 'o', color=P_COLOUR, ms=8, zorder=4,
                alpha=1 if in_centre(a + 0.5, b + 0.5) else FADE)
for a in range(N + 1):
    for b in range(N):
        arrow(ax, a, b + 0.5, L, 0, U_COLOUR, 1 if in_centre(a, b + 0.5) else FADE)
for a in range(N):
    for b in range(N + 1):
        arrow(ax, a + 0.5, b, 0, L, V_COLOUR, 1 if in_centre(a + 0.5, b) else FADE)

lab = dict(fontsize=FS, zorder=6,
           bbox=dict(boxstyle='round,pad=0.06', fc='white', ec='none', alpha=0.85))
ax.text(1.5, 1.5 - 0.09, r'$\phi_{i+\frac{1}{2},\,j+\frac{1}{2}}$', color=P_COLOUR,
        ha='center', va='top', **lab)
ax.text(1, 1.5 + 0.08, r'$u_{i,\,j+\frac{1}{2}}$', color=U_COLOUR, ha='center',
        va='bottom', **lab)
ax.text(2, 1.5 + 0.08, r'$u_{i+1,\,j+\frac{1}{2}}$', color=U_COLOUR, ha='center',
        va='bottom', **lab)
ax.text(1.5, 1 - L - 0.03, r'$v_{i+\frac{1}{2},\,j}$', color=V_COLOUR, ha='center',
        va='top', **lab)
ax.text(1.5, 2 + L + 0.03, r'$v_{i+\frac{1}{2},\,j+1}$', color=V_COLOUR, ha='center',
        va='bottom', **lab)
for (a, b), s, ha, va in (((1, 1), r'$(i,\,j)$', 'right', 'top'),
                          ((2, 1), r'$(i+1,\,j)$', 'left', 'top'),
                          ((1, 2), r'$(i,\,j+1)$', 'right', 'bottom'),
                          ((2, 2), r'$(i+1,\,j+1)$', 'left', 'bottom')):
    dx = -0.06 if ha == 'right' else 0.06
    dy = -0.06 if va == 'top' else 0.06
    ax.text(a + dx, b + dy, s, color=NODE, ha=ha, va=va, fontsize=0.7*FS, zorder=6,
            bbox=lab['bbox'])

ax.set_xlim(-0.3, N + 0.3)
ax.set_ylim(-0.3, N + 0.3)
ax.set_aspect('equal')
ax.axis('off')

out = Path(__file__).parent / "mac_grid.png"
plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.15, facecolor="white")
print(f"Saved {out}")
