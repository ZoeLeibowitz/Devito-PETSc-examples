from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Colours of devito_compiler_petsc.py
DARK_GREEN = "#1B5E20"
FILL_GREEN = "#E8F5E9"
DARK_BLUE = "#0D47A1"
FILL_BLUE = "#E3F2FD"
TEXT = "#222222"

W, H = 4.96, 1.58
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(-H, 0)
ax.axis("off")

UF = r"$\mathtt{u.\!forward}$"   # the negative space removes the gap mathtext leaves after the dot
steps = [
    (DARK_GREEN, FILL_GREEN, "Explicit update", ["a loop nest"]),
    (DARK_BLUE, FILL_BLUE, "Solve", ["for " + UF + ",", r"prefix $\mathtt{u\_}$"]),
    (DARK_BLUE, FILL_BLUE, "Solve", [r"for $\mathtt{p}$, which reads", UF + ",",
                                     r"prefix $\mathtt{p\_}$"]),
    (DARK_GREEN, FILL_GREEN, "Explicit update", ["of " + UF]),
]

bw, bh, gap = 1.07, 0.74, 0.14
x_left = (W - (len(steps) * bw + (len(steps) - 1) * gap)) / 2
y_box = -0.36 - bh       # bottom of the boxes

# The time loop around the steps
pad = 0.08
ax.add_patch(FancyBboxPatch((x_left - pad, y_box - 0.40), len(steps) * bw + (len(steps) - 1) * gap
                            + 2 * pad, bh + 0.40 + 0.28, boxstyle="round,pad=0,rounding_size=0.04",
                            linewidth=1.1, edgecolor=DARK_GREEN, facecolor="white", zorder=1))
ax.text(x_left - pad + 0.08, y_box + bh + 0.14, "Time loop", ha="left", va="center", fontsize=8,
        fontweight="bold", color=DARK_GREEN)

centres = []
for k, (edge, face, title, lines) in enumerate(steps):
    x0 = x_left + k * (bw + gap)
    ax.add_patch(FancyBboxPatch((x0, y_box), bw, bh, boxstyle="round,pad=0,rounding_size=0.03",
                                linewidth=1.0, edgecolor=edge, facecolor=face, zorder=2))
    ax.text(x0 + bw / 2, y_box + bh - 0.14, title, ha="center", va="center", fontsize=7.2,
            fontweight="bold", color=edge, zorder=3)
    for j, s in enumerate(lines):
        ax.text(x0 + bw / 2, y_box + bh - 0.32 - j * 0.15, s, ha="center", va="center",
                fontsize=7, color=TEXT, zorder=3)
    centres.append(x0 + bw / 2)
    if k:
        ax.annotate("", xy=(x0, y_box + bh / 2), xytext=(x0 - gap, y_box + bh / 2), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=0.9, mutation_scale=8,
                                    shrinkA=0, shrinkB=0))

# Back to the first step, for the next time step
y_back = y_box - 0.20
ax.add_patch(FancyArrowPatch((centres[-1], y_box), (centres[0], y_box),
                             connectionstyle="bar,fraction=-0.06", arrowstyle="-|>",
                             mutation_scale=8, color=TEXT, lw=0.9, zorder=3))
ax.text((centres[0] + centres[-1]) / 2, y_back - 0.02, "next time step", ha="center", va="center",
        fontsize=7, color=TEXT, zorder=3,
        bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))

# Nothing may leave the figure, and each title must fit inside its box
fig.canvas.draw()
inv = ax.transData.inverted()
for k in range(len(steps)):
    x0 = x_left + k * (bw + gap)
    for t in ax.texts:
        (a0, b0), (a1, b1) = inv.transform(t.get_window_extent().get_points())
        if y_box < (b0 + b1) / 2 < y_box + bh and x0 < (a0 + a1) / 2 < x0 + bw:
            assert a0 > x0 + 0.03 and a1 < x0 + bw - 0.03, t.get_text()
for t in ax.texts:
    (a0, b0), (a1, b1) = inv.transform(t.get_window_extent().get_points())
    assert a0 > 0.02 and a1 < W - 0.02 and b0 > -H + 0.02 and b1 < -0.02, t.get_text()

out_path = Path(__file__).with_suffix(".png")
plt.savefig(out_path, dpi=300, facecolor="white")
print(f"Saved {out_path}")
