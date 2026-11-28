# The five SubDomains of u in the lid driven cavity

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from devito import Grid, Function, Eq, Operator, SubDomain

DARK_GREEN = "#1B5E20"
MID_GREEN = "#2E7D32"
FILL_GREEN = "#E8F5E9"
TEXT_DARK = "#1B3A1F"
DOT_IN = "#1B5E20"
DOT_OUT = "#B0B0B0"
WALL = "#333333"

N = 9
nx, ny = N, N


# The SubDomains of u, as in solver.py
class UInterior(SubDomain):
    name = 'u_interior'
    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('middle', 1, 2)}

class ULeftWall(SubDomain):
    name = 'u_left_wall'
    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: ('left', ny-1)}

class URightWall(SubDomain):
    name = 'u_right_wall'
    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: ('left', ny-1)}

class UBottomRow(SubDomain):
    name = 'u_bottom_row'
    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('left', 1)}

class UTopRow(SubDomain):
    name = 'u_top_row'
    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('middle', ny-2, 1)}


grid = Grid(shape=(nx, ny), extent=(1., 1.))
x, y = grid.dimensions
subdomains = [cls(grid=grid) for cls in
              (UInterior, ULeftWall, URightWall, UBottomRow, UTopRow)]
masks, shapes = {}, {}
for sd in subdomains:
    mask = Function(name='mask', grid=grid, staggered=y)
    Operator(Eq(mask, 1, subdomain=sd)).apply()
    masks[sd.name] = mask.data.astype(bool).copy()
    shapes[sd.name] = sd.shape

outside = np.zeros((N, N), dtype=bool)
outside[:, N - 1] = True
count = sum(m.astype(int) for m in masks.values())
assert (count[~outside] == 1).all() and (count[outside] == 0).all(), \
    "the SubDomains of u should cover every u point inside the cavity exactly once"


def region_box(ax, mask, color=DARK_GREEN, linestyle='-'):
    i, j = np.nonzero(mask)
    x0, x1, y0, y1 = i.min(), i.max(), j.min() + 0.5, j.max() + 0.5
    ax.add_patch(FancyBboxPatch(
        (x0 - 0.38, y0 - 0.38), (x1 - x0) + 0.76, (y1 - y0) + 0.76,
        boxstyle="round,pad=0.02,rounding_size=0.3", linestyle=linestyle,
        linewidth=2.0, edgecolor=color, facecolor="none", zorder=4))


def cavity(ax):
    # Cell lines between the nodes, and the walls of the cavity
    for k in range(N):
        ax.plot([k, k], [0, N - 1], color='#D0D0D0', lw=0.8, zorder=1)
        ax.plot([0, N - 1], [k, k], color='#D0D0D0', lw=0.8, zorder=1)
    ax.add_patch(Rectangle((0, 0), N - 1, N - 1, fill=False, lw=2.2, edgecolor=WALL,
                           zorder=2))


def arrows(ax, highlight):
    # u points at (x_i, y_j + h/2), drawn as arrows across the vertical cell faces
    for i in range(N):
        for j in range(N):
            color = DOT_IN if highlight[i, j] else DOT_OUT
            head = 'white' if outside[i, j] else color
            ax.annotate('', xy=(i + 0.27, j + 0.5), xytext=(i - 0.27, j + 0.5),
                        arrowprops=dict(arrowstyle='-|>,head_length=0.45,head_width=0.22',
                                        ec=color, fc=head, lw=1.3,
                                        shrinkA=0, shrinkB=0),
                        zorder=3, annotation_clip=False)


def finish(ax, title, label):
    ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT_DARK, pad=6)
    ax.text(0.5, -0.01, label, transform=ax.transAxes, ha="center", va="top",
            fontsize=12, family="monospace", color=MID_GREEN)
    ax.set_xlim(-1.1, N)
    ax.set_ylim(-0.6, N + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")


fig = plt.figure(figsize=(9.5, 10.6))
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.0], hspace=0.34, wspace=0.08)

ax_code = fig.add_subplot(gs[0, :])
ax_code.set_xlim(0, 1)
ax_code.set_ylim(0, 1)
ax_code.axis("off")
ax_code.add_patch(FancyBboxPatch(
    (0.01, 0.03), 0.98, 0.94, boxstyle="round,pad=0.02,rounding_size=0.03",
    linewidth=1.6, edgecolor=MID_GREEN, facecolor=FILL_GREEN,
    transform=ax_code.transAxes, zorder=1, clip_on=False))
code_lines = [
    (f">>> nx, ny = {N}, {N}", TEXT_DARK, "bold", 12),
    (">>> class UTopRow(SubDomain):", TEXT_DARK, "bold", 12),
    ("...     name = 'u_top_row'", TEXT_DARK, "bold", 12),
    ("...", TEXT_DARK, "bold", 12),
    ("...     def define(self, dimensions):", TEXT_DARK, "bold", 12),
    ("...         x, y = dimensions", TEXT_DARK, "bold", 12),
    ("...         return {x: ('middle', 1, 1), y: ('middle', ny-2, 1)}", TEXT_DARK, "bold", 12),
    ("...", TEXT_DARK, "bold", 12),
    (">>> grid = Grid(shape=(nx, ny), extent=(1., 1.))", TEXT_DARK, "bold", 12),
    (">>> u_top_row = UTopRow(grid=grid)", TEXT_DARK, "bold", 12),
    (">>> u_top_row.shape", TEXT_DARK, "bold", 12),
    (str(shapes['u_top_row']), "#333333", "normal", 10.5),
]
y0, dy = 0.915, 0.0755
for i, (line, color, weight, fontsize) in enumerate(code_lines):
    ax_code.text(0.05, y0 - i * dy, line, ha="left", va="center", fontsize=fontsize,
                 family="monospace", color=color, fontweight=weight,
                 transform=ax_code.transAxes, zorder=2)

order = ['u_interior', 'u_left_wall', 'u_right_wall', 'u_bottom_row', 'u_top_row']
for k, name in enumerate(order):
    ax = fig.add_subplot(gs[1 + k // 3, k % 3])
    cavity(ax)
    arrows(ax, masks[name])
    region_box(ax, masks[name])
    finish(ax, f"'{name}'", f"shape = {shapes[name]}")

# The row above the lid, handled automatically by petscsolve
ax = fig.add_subplot(gs[2, 2])
cavity(ax)
arrows(ax, np.zeros((N, N), dtype=bool))
region_box(ax, outside, color=DOT_OUT, linestyle='--')
finish(ax, "outside the cavity", "removed by petscsolve")

out = Path(__file__).parent / "cavity_u_subdomains.png"
plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.3, facecolor="white")
print(f"Saved {out}")
