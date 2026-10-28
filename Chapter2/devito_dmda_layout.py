"""
How Devito's array of the solution maps onto the DMDA local vector (Section
2.6.4), drawn for the 5 x 5 grid of Section 2.4.5 with space_order=2.

Devito's array of p holds the domain and a halo of width space_order on every
side. The DMDA is created with a stencil width s = space_order and with
DM_BOUNDARY_GHOSTED, which keeps ghost points beyond the physical boundary as
well, so its local vectors have exactly the same layout. The local vector of the
solution is therefore created around the data of p itself, in which the access
p[x, y] is offset to p[x + 2, y + 2], as in Section 2.2.1. For the 17 x 17 grid of the
example, the array has 21 x 21 = 441 entries.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# Colours of Figure fig:compiler:petscsolve_expressions (TikZ orange!80!black
# and blue!55!black), so that boundary and interior points match it
ORANGE = "#CC6600"
BLUE = "#00008C"
NAV = "#607D8B"
TEXT = "#222222"

N = 5        # points in each direction of the domain
HALO = 2     # space_order, and the stencil width s of the DMDA
d = 0.165    # distance between points, in inches
r = 0.040    # radius of a point

W, H = 4.96, 2.52
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(-H, 0)
ax.axis("off")


def point(x, y, kind):
    if kind == "halo":
        face, edge = "white", NAV
    elif kind == "boundary":
        face, edge = ORANGE, ORANGE
    else:
        face, edge = BLUE, BLUE
    ax.add_patch(Circle((x, y), r, facecolor=face, edgecolor=edge, lw=0.8, zorder=3))


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4, arrowprops=dict(
        arrowstyle="-|>", color=TEXT, lw=0.8, mutation_scale=7, shrinkA=0, shrinkB=0))


def lines(x, y_top, rows, ha):
    """A block of text, one entry per line, with the first in bold."""
    for k, s in enumerate(rows):
        ax.text(x, y_top - k * 0.19, s, ha=ha, va="center", color=TEXT,
                fontsize=8.5 if k == 0 else 7.5, fontweight="bold" if k == 0 else "normal")


# The array, with x to the right and y upwards, so that (0, 0) is the bottom
# left point of the domain
n = N + 2 * HALO
span = (n - 1) * d
x0 = W / 2 - span / 2
y_bot = -0.38 - span


def pos(i, j):
    """Position of the point with Devito index (i, j), halo points included."""
    return x0 + (i + HALO) * d, y_bot + (j + HALO) * d


for i in range(-HALO, N + HALO):
    for j in range(-HALO, N + HALO):
        if not (0 <= i < N and 0 <= j < N):
            kind = "halo"
        elif i in (0, N - 1) or j in (0, N - 1):
            kind = "boundary"
        else:
            kind = "interior"
        point(*pos(i, j), kind)

# The domain, and the width of the halo below it
xa, ya = pos(0, 0)
ax.add_patch(Rectangle((xa - d / 2, ya - d / 2), N * d, N * d, fill=False, ls="--",
                       lw=0.7, edgecolor=NAV, zorder=2))
yb = y_bot - 0.13
xh0, xh1 = pos(-HALO, 0)[0], pos(-1, 0)[0]
ax.plot([xh0, xh0, xh1, xh1], [yb + 0.04, yb, yb, yb + 0.04], color=NAV, lw=0.8)
ax.text((xh0 + xh1) / 2, yb - 0.11, "2", ha="center", va="center", fontsize=7.5,
        color=TEXT)

# What Devito calls each part, on the left, and what PETSc calls it, on the right
y_top = -0.55
lines(x0 - 0.22, y_top, ["Devito", "array of $p$", f"domain, {N} $\\times$ {N}",
                         "halo of width", r"$\mathtt{space\_order} = 2$"], ha="right")
lines(x0 + span + 0.22, y_top, ["PETSc", "local vector of the solution",
                                f"owned points, {N} $\\times$ {N}",
                                r"ghost points of width $\mathtt{s} = 2$,",
                                "kept at the physical boundary",
                                r"by $\mathtt{DM\_BOUNDARY\_GHOSTED}$"], ha="left")

# The first point of the domain, and where it is stored in the array
y_idx = y_bot - 0.30
arrow(xa + 0.26, y_idx + 0.06, xa + 0.75 * r, ya - 0.75 * r)
ax.text(xa + 0.28, y_idx, r"the access $\mathtt{p[0, 0]}$ is offset to $\mathtt{p[0 + 2, 0 + 2]}$",
        ha="left", va="center", fontsize=7.5, color=TEXT)

# Key
y_key = -H + 0.14
entries = [("halo", "halo and ghost point"), ("boundary", "boundary point"),
           ("interior", "interior point")]
labels = []
for kind, label in entries:
    labels.append(ax.text(0, y_key, label, ha="left", va="center", fontsize=7.5, color=TEXT))
fig.canvas.draw()
inv = ax.transData.inverted()
widths = []
for t in labels:
    (a0, _), (a1, _) = inv.transform(t.get_window_extent().get_points())
    widths.append(a1 - a0)
gap, sw = 0.30, 0.10
total = sum(sw + w for w in widths) + gap * (len(entries) - 1)
x = W / 2 - total / 2
for (kind, _), t, w in zip(entries, labels, widths):
    point(x, y_key, kind)
    t.set_x(x + sw)
    x += sw + w + gap

# Nothing may leave the figure
fig.canvas.draw()
for t in ax.texts:
    (a0, b0), (a1, b1) = inv.transform(t.get_window_extent().get_points())
    assert a0 > 0.02 and a1 < W - 0.02 and b0 > -H + 0.02 and b1 < -0.02, t.get_text()

out_path = Path(__file__).with_suffix(".png")
plt.savefig(out_path, dpi=300, facecolor="white")
print(f"Saved {out_path}")
