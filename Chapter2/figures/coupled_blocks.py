"""
The block Jacobian of a coupled solve (Section 2.7.2)
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Colours of devito_compiler_petsc.py
DARK_BLUE = "#0D47A1"
FILL_BLUE = "#E3F2FD"
NAV = "#607D8B"
TEXT = "#222222"

W, H = 4.96, 2.66
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(-H, 0)
ax.axis("off")

b = 0.82          # size of a block, in inches
vw = 0.40         # width of a vector
y_top = -0.40     # top of the matrix and the vectors

# The Jacobian, with the row field on the left and the column field on top
xm = 1.05
fields = ["v", "u"]
blocks = {(0, 0): "J00", (1, 0): "J10", (1, 1): "J11"}
for i in range(2):
    for j in range(2):
        x0, y0 = xm + j * b, y_top - (i + 1) * b
        name = blocks.get((i, j))
        if name:
            ax.add_patch(Rectangle((x0, y0), b, b, facecolor=FILL_BLUE, edgecolor=DARK_BLUE,
                                   lw=1.0, zorder=2))
            ax.text(x0 + b / 2, y0 + b / 2 + 0.11, f"$J_{{{i}{j}}}$", ha="center", va="center",
                    fontsize=9, color=DARK_BLUE, zorder=3)
            ax.text(x0 + b / 2, y0 + b / 2 - 0.13, f"$\\mathtt{{{name}\\_}}$\n"
                    r"$\mathtt{JacobianMult0}$", ha="center", va="center", fontsize=5.8,
                    color=TEXT, zorder=3, linespacing=1.3)
        else:
            ax.add_patch(Rectangle((x0, y0), b, b, facecolor="white", edgecolor=NAV, lw=0.8,
                                   zorder=2))
            ax.text(x0 + b / 2, y0 + b / 2 + 0.06, "0", ha="center", va="center", fontsize=9,
                    color=NAV, zorder=3)
            ax.text(x0 + b / 2, y0 + b / 2 - 0.13, "no callback", ha="center", va="center",
                    fontsize=6.5, color=NAV, zorder=3)
for k, fname in enumerate(fields):
    ax.text(xm - 0.10, y_top - (k + 0.5) * b, f"${fname}$", ha="right", va="center", fontsize=8.5,
            color=TEXT)
    ax.text(xm + (k + 0.5) * b, y_top + 0.10, f"${fname}$", ha="center", va="center",
            fontsize=8.5, color=TEXT)


def vector(x0, name):
    for k, fname in enumerate(fields):
        y0 = y_top - (k + 1) * b
        ax.add_patch(Rectangle((x0, y0), vw, b, facecolor="white", edgecolor=TEXT, lw=0.8,
                               zorder=2))
        ax.text(x0 + vw / 2, y0 + b / 2, f"$\\mathtt{{{name}\\_{fname}}}$", ha="center",
                va="center", fontsize=8, color=TEXT, zorder=3)


xx = xm + 2 * b + 0.30
vector(xx, "x")
ax.text(xx + vw + 0.18, y_top - b, "=", ha="center", va="center", fontsize=11, color=TEXT)
xy = xx + vw + 0.36
vector(xy, "y")

# What the product of the whole Jacobian does, below
y_text = y_top - 2 * b - 0.25
ax.text(W / 2, y_text, r"$\mathtt{WholeJacobianMult0}$ zeroes $\mathtt{y}$, then applies each nonzero "
        r"block to its part of $\mathtt{x}$", ha="center", va="center", fontsize=7.5, color=TEXT)
ax.text(W / 2, y_text - 0.18, r"and adds the result into its part of $\mathtt{y}$", ha="center",
        va="center", fontsize=7.5, color=TEXT)

# Nothing may leave the figure
fig.canvas.draw()
inv = ax.transData.inverted()
for t in ax.texts:
    (a0, b0), (a1, b1) = inv.transform(t.get_window_extent().get_points())
    assert a0 > 0.02 and a1 < W - 0.02 and b0 > -H + 0.02 and b1 < -0.02, t.get_text()

out_path = Path(__file__).with_suffix(".png")
plt.savefig(out_path, dpi=300, facecolor="white")
print(f"Saved {out_path}")
