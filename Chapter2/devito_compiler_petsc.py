from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

# Existing stages, as in devito_compiler.py
DARK_GREEN = "#1B5E20"
FILL_GREEN = "#E8F5E9"
TEXT_DARK = "#1B3A1F"
# What the PETSc code generation adds
DARK_BLUE = "#0D47A1"
FILL_BLUE = "#E3F2FD"
NAV = "#37474F"

STAGES = [
    "Equations lowering",
    "Clustering",
    "Symbolic optimisation",
    "IET construction",
    "IET optimisation",
    "Synthesis",
    "JIT Compilation",
]
IET_OPT = 4

W = 4.96
card_x, card_r = 0.42, 4.47
card_w = card_r - card_x

m = 0.07
h_add, add_w = 0.20, 1.84
add_x = card_r - m - add_w
name_cx = (card_x + add_x) / 2
h_card, h_tall, gap = h_add + 2 * m, 3 * h_add + 4 * m, 0.12
h_in, h_out = 0.60, 0.55
pad = 0.07

mx = W - 0.17
bx = card_r + 0.10

# Vertical layout, top to bottom, with y = 0 at the top
y_in = -0.04 - h_in
rows = []
y = y_in - gap
for i, _ in enumerate(STAGES):
    h = h_tall if i == IET_OPT else h_card
    rows.append((y - h, h))
    y -= h + gap
y_out = y - h_out

h_sw = 0.13
y_key = y_out - 0.12 - h_sw / 2
H = -(y_key - h_sw / 2 - 0.04)

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(-H, 0)
ax.axis("off")

contained = []


def box(x, y0, w, h, edge, face, lw=1.0, z=3):
    ax.add_patch(FancyBboxPatch(
        (x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.03",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=z,
    ))


def text(x, y0, s, bounds, **kwargs):
    t = ax.text(x, y0, s, va="center", zorder=5, **kwargs)
    contained.append((t, bounds))
    return t


def marker(y0, n):
    ax.add_patch(Circle((mx, y0), 0.085, facecolor=NAV, edgecolor="none", zorder=6))
    ax.text(mx, y0, str(n), ha="center", va="center", fontsize=7, fontweight="bold",
            color="white", zorder=7)


def bracket(y_top, y_bot):
    ax.plot([bx - 0.05, bx, bx, bx - 0.05], [y_top, y_top, y_bot, y_bot],
            color=NAV, lw=0.9, zorder=2)
    return (y_top + y_bot) / 2


def arrow(x0, y0, x1, y1, color=DARK_GREEN, lw=1.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2, arrowprops=dict(
        arrowstyle="-|>", color=color, lw=lw, mutation_scale=8, shrinkA=0, shrinkB=0))


def addition(y0, h, lines):
    """What the PETSc code generation adds to a stage, inside its card."""
    box(add_x, y0, add_w, h, DARK_BLUE, FILL_BLUE, lw=0.9, z=4)
    bounds = (add_x, y0, add_x + add_w, y0 + h)
    n = len(lines)
    ys = [y0 + h / 2] if n == 1 else [y0 + h - 0.11 - k * (h - 0.22) / (n - 1) for k in range(n)]
    for s, yk in zip(lines, ys):
        text(add_x + pad, yk, s, bounds, ha="left", fontsize=7.5, color=DARK_BLUE)
    return ys


# Input, the equation returned by petscsolve (marker 1)
cx = card_x + card_w / 2
b_in = (card_x, y_in, card_r, y_in + h_in)
box(card_x, y_in, card_w, h_in, DARK_BLUE, FILL_BLUE)
text(cx, y_in + h_in - 0.13, r"Equation returned by $\mathtt{petscsolve}$", b_in, ha="center",
     fontsize=8.5, fontweight="bold", color=DARK_BLUE)
text(cx, y_in + h_in / 2 - 0.01, "PetscEq(p(x, y), LinearSolverMetaData(...))", b_in,
     ha="center", fontsize=7.5, family="monospace", color=DARK_BLUE)
text(cx, y_in + 0.12, "metadata (residual, action of the Jacobian, solver options, ...)", b_in,
     ha="center", fontsize=7.5, style="italic", color="#333333")
marker(y_in + h_in / 2, 1)

# The stages of the Operator, as in devito_compiler.py
for i, (title, (y0, h)) in enumerate(zip(STAGES, rows)):
    box(card_x, y0, card_w, h, DARK_GREEN, FILL_GREEN, lw=1.1)
    text(name_cx, y0 + h / 2, title, (card_x, y0, add_x, y0 + h), ha="center",
         fontsize=8, fontweight="bold", color=TEXT_DARK)
    top = y_in if i == 0 else rows[i - 1][0]
    arrow(name_cx, top, name_cx, y0 + h)
arrow(name_cx, rows[-1][0], name_cx, y_out + h_out)

# What the PETSc code generation adds to each stage
(y1, h1), (y2, h2), (y3, h3), (y4, h4), (y5, h5), (y6, h6), (y7, h7) = rows
addition(y2 + m, h2 - 2 * m, ["solve kept in its own loop nest"])
addition(y4 + m, h4 - 2 * m, ["solve becomes a single IET node"])
# IET optimisation has one addition per marker
hB = h_add
yA = y5 + h5 - m - hB
marker(addition(yA, hB, ["callbacks generated"])[0], 3)
marker(addition(yA - m - hB, hB, ["setup and solve calls generated"])[0], 4)
yB = y5 + m
marker(addition(yB, hB, ["user contexts completed"])[0], 5)
assert abs((yA - m - hB) - (yB + hB + m)) < 1e-9, "IET optimisation additions"
marker(addition(y6 + m, h6 - 2 * m, ["PETSc types and headers"])[0], 6)
marker(addition(y7 + m, h7 - 2 * m, ["linked against PETSc"])[0], 7)

# Marker 2 spans equations lowering to IET construction
marker(bracket(y1 + h1 / 2, y4 + h4 / 2), 2)

# Recursive compilation of the callbacks
lx = 0.20
y_start = y5 + 0.12
y_end = y1 + h1 - 0.08
ax.plot([card_x, lx, lx], [y_start, y_start, y_end], color=DARK_BLUE, lw=1.3, zorder=2)
arrow(lx, y_end, card_x, y_end, color=DARK_BLUE, lw=1.3)

y_mid = (y_start + y_end) / 2
loop_label = ax.text(lx, y_mid, "recursive compilation of each callback", rotation=90,
                     ha="center", va="center", fontsize=7, color=DARK_BLUE, zorder=5)
fig.canvas.draw()
(_, b0), (_, b1) = ax.transData.inverted().transform(
    loop_label.get_window_extent().get_points())
lb_w, lb_h = 0.17, (b1 - b0) + 0.12
box(lx - lb_w / 2, y_mid - lb_h / 2, lb_w, lb_h, DARK_BLUE, FILL_BLUE, lw=0.9)

# Output, the generated code (marker 8)
b_out = (card_x, y_out, card_r, y_out + h_out)
box(card_x, y_out, card_w, h_out, "#333333", "white")
text(cx, y_out + h_out - 0.12, "Generated code", b_out, ha="center", fontsize=8.5,
     fontweight="bold", color="#222222")
text(cx, y_out + h_out / 2 - 0.03, r"$\mathtt{Kernel}$, with the setup and solve calls", b_out,
     ha="center", fontsize=7.5, color="#222222")
text(cx, y_out + 0.11, r"callbacks ($\mathtt{FormFunction0}$, $\mathtt{MatMult0}$, ...)", b_out,
     ha="center", fontsize=7.5, color="#222222")
marker(y_out + h_out / 2, 8)

# Key
entries = [(DARK_GREEN, FILL_GREEN, "existing compilation stage"),
           (DARK_BLUE, FILL_BLUE, "added for PETSc")]
sw_w, sw_gap, entry_gap = 0.22, 0.07, 0.30
labels = [ax.text(0, y_key, s, ha="left", va="center", fontsize=7.5, color="#222222", zorder=5)
          for _, _, s in entries]
fig.canvas.draw()
widths = []
for t in labels:
    (a0, _), (a1, _) = ax.transData.inverted().transform(t.get_window_extent().get_points())
    widths.append(a1 - a0)
total = sum(sw_w + sw_gap + w for w in widths) + entry_gap * (len(entries) - 1)
x = cx - total / 2
for (edge, face, _), t, w in zip(entries, labels, widths):
    box(x, y_key - h_sw / 2, sw_w, h_sw, edge, face, lw=0.9)
    t.set_x(x + sw_w + sw_gap)
    x += sw_w + sw_gap + w + entry_gap
assert cx - total / 2 > card_x and cx + total / 2 < card_r, "key width"

fig.canvas.draw()
inv = ax.transData.inverted()
for t, (x0, y0, x1, y1) in contained:
    (a0, b0), (a1, b1) = inv.transform(t.get_window_extent().get_points())
    assert a0 >= x0 + 0.02 and a1 <= x1 - 0.02 and b0 >= y0 and b1 <= y1, t.get_text()
assert y_mid - lb_h / 2 > y_start + 0.05 and y_mid + lb_h / 2 < y_end - 0.05, "loop label box"

out_path = Path(__file__).with_suffix(".png")
plt.savefig(out_path, dpi=300, facecolor="white")
print(f"Saved {out_path}")
