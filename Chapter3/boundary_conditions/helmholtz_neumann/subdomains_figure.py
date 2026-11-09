from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DARK_GREEN = "#1B5E20"
MID_GREEN = "#2E7D32"
FILL_GREEN = "#E8F5E9"
TEXT_DARK = "#1B3A1F"
DOT_IN = "#1B5E20"
DOT_OUT = "#B0B0B0"

N = 9

fig = plt.figure(figsize=(9.5, 8.7))
gs = fig.add_gridspec(2, 2, height_ratios=[1.28, 1.55], hspace=0.18, wspace=0.18)

ax_code = fig.add_subplot(gs[0, :])
ax_code.set_xlim(0, 1)
ax_code.set_ylim(0, 1)
ax_code.axis("off")
ax_code.add_patch(FancyBboxPatch(
    (0.01, 0.03), 0.98, 0.94,
    boxstyle="round,pad=0.02,rounding_size=0.03",
    linewidth=1.6, edgecolor=MID_GREEN, facecolor=FILL_GREEN,
    transform=ax_code.transAxes, zorder=1,
))

code_lines = [
    (">>> class SubLeft(SubDomain):", TEXT_DARK, "bold", 13),
    ("...     name = 'subleft'", TEXT_DARK, "bold", 13),
    ("...", TEXT_DARK, "bold", 13),
    ("...     def define(self, dimensions):", TEXT_DARK, "bold", 13),
    ("...         x, y = dimensions", TEXT_DARK, "bold", 13),
    ("...         return {x: ('left', 1), y: ('middle', 1, 1)}", TEXT_DARK, "bold", 13),
    ("", TEXT_DARK, "normal", 13),
    (">>> grid = Grid(shape=(9, 9))", TEXT_DARK, "bold", 13),
    (">>> left = SubLeft(grid=grid)", TEXT_DARK, "bold", 13),
    (">>> left.shape", TEXT_DARK, "bold", 13),
    ("(1, 7)", "#333333", "normal", 10.5),
]
y0, dy = 0.925, 0.081
for i, (line, color, weight, fontsize) in enumerate(code_lines):
    ax_code.text(0.05, y0 - i * dy, line, ha="left", va="center",
                 fontsize=fontsize, family="monospace", color=color,
                 fontweight=weight, transform=ax_code.transAxes, zorder=2)


def box(ax, x0, x1, y0, y1):
    ax.add_patch(FancyBboxPatch(
        (x0 - 0.4, y0 - 0.4), (x1 - x0) + 0.8, (y1 - y0) + 0.8,
        boxstyle="round,pad=0.02,rounding_size=0.3",
        linewidth=2.0, edgecolor=DARK_GREEN, facecolor="none", zorder=2))


def panel(ax, title, regions, label):
    for x in range(N):
        for y in range(N):
            inside = any(x0 <= x <= x1 and y0 <= y <= y1
                         for (x0, x1, y0, y1) in regions)
            ax.scatter(x, y, s=42, color=DOT_IN if inside else DOT_OUT,
                       zorder=3, clip_on=False)
    for r in regions:
        box(ax, *r)
    ax.set_title(title, fontsize=15, fontweight="bold", color=TEXT_DARK, pad=10)
    ax.text(0.5, -0.02, label, transform=ax.transAxes, ha="center", va="top",
            fontsize=12, family="monospace", color=MID_GREEN)
    ax.set_xlim(-1.1, N)
    ax.set_ylim(-1.1, N)
    ax.set_aspect("equal")
    ax.axis("off")


left = [(0, 0, 1, N - 2)]
sides_and_corners = [
    (0, 0, 1, N - 2), (N - 1, N - 1, 1, N - 2),          # left, right
    (1, N - 2, 0, 0), (1, N - 2, N - 1, N - 1),          # bottom, top
    (0, 0, 0, 0), (N - 1, N - 1, 0, 0),                  # corners
    (0, 0, N - 1, N - 1), (N - 1, N - 1, N - 1, N - 1),
]
panel(fig.add_subplot(gs[1, 0]), "'subleft'", left, "shape = (1, 7)")
panel(fig.add_subplot(gs[1, 1]), "all eight SubDomains", sides_and_corners,
      "four sides, four corners")

out = Path(__file__).with_name("helmholtz_neumann_subdomains.png")
plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.3, facecolor="white")
print(f"Saved {out}")
