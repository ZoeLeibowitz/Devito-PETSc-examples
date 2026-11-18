from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Colours of the Chapter 2 figures
DARK_BLUE = "#0D47A1"
FILL_BLUE = "#E3F2FD"
NUMBER = "#78909C"
TEXT = "#222222"

SECTION_COLOURS = {
    "3.3": ("#1B5E20", "#E8F5E9", "Boundary conditions beyond Dirichlet"),
    "3.5": ("#E65100", "#FFF3E0", "Higher order spatial discretisations"),
    "3.6": ("#4A148C", "#F3E5F5", "Time dependent problems"),
}

CELLS = [
    ("3.2", "Problems in one,\ntwo and three\ndimensions",
     r"$\mathtt{DMDA}$ and callback" "\n" r"loop nests for any $\mathtt{Grid}$," "\n"
     "Dirichlet conditions"),
    ("3.3.1", "Neumann\nconditions", "user modified stencils\nin the residual\nand Jacobian"),
    ("3.3.2", "Mixed conditions\non a rectangular\ngrid",
     r"$\mathtt{EssentialBC}$s with" "\n" "modified stencils,\n"
     r"$\mathtt{DMDA}$ with $n_x \neq n_y$"),
    ("3.3.3", "Robin conditions", None),
    ("3.4", "Singular systems\nand null spaces", "null spaces attached\nto the Jacobian"),
    ("3.5.1", "Modified stencils\nat higher orders",
     r"callbacks and $\mathtt{DMDA}$" "\n" r"follow the $\mathtt{space\_order}$," "\n"
     "orders 2 to 8"),
    ("3.5.2", "Custom finite\ndifference weights",
     "compact schemes,\nuser's weights\nin callbacks"),
    ("3.6.1", "Implicit time\nstepping",
     r"$\mathtt{SNESSolve}$ in time loop," "\n" "time data via user\ncontext, residual and\n"
     r"Jacobian of the scheme" "\n" r"set by the $\mathtt{Eq}$"),
    ("3.6.2", "Time dependent\nboundary\nconditions",
     r"loop variable $\mathtt{time}$" "\n" "updated in user context\n"
     r"before each $\mathtt{SNESSolve}$"),
    ("3.6.3", "Second\nderivatives\nin time", "multiple time levels\nin the residual"),
    ("3.6.4", "IMEX schemes",
     "explicit terms only in\nthe residual, implicit\nterms in the Jacobian"),
    ("3.7", "Nonlinear\nproblems", "nonlinear residuals,\nJFNK through\nsolver options"),
    ("3.8", "Staggered grids", None),
    ("3.9", "Coupled systems",
     r"$\mathtt{DMDA}$ with dof > 1," "\n" "one callback per\nJacobian block"),
    ("3.10", "Solves\nalongside other\ncomputations in\na simulation",
     "solve, explicit updates\nand sparse operations\n" r"in one $\mathtt{Operator}$"),
    ("3.11", "Immersed\nboundaries\nwith Schism",
     "compatibility with other\nlibraries, immersed\nboundary stencils in\nan implicit solve"),
]
NCOL = 4
NROW = (len(CELLS) + NCOL - 1) // NCOL

W = 4.96          # figure width, in inches (the text width of the thesis)
margin = 0.02     # space around the grid
gap = 0.08        # space between squares
bw = (W - 2 * margin - (NCOL - 1) * gap) / NCOL
bh = 1.12         # height of a square
key_h = 0.42      # height of the key below the grid
H = 2 * margin + NROW * bh + (NROW - 1) * gap + key_h

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(-H, 0)
ax.axis("off")


def colours(number):
    """Edge and fill of a square, from its section if that has subsections."""
    section = number.rsplit(".", 1)[0] if number.count(".") == 2 else None
    edge, fill, _ = SECTION_COLOURS.get(section, (DARK_BLUE, FILL_BLUE, None))
    return edge, fill


boxes = []
stacks = []
for k, (number, title, note) in enumerate(CELLS):
    row, col = divmod(k, NCOL)
    x0 = margin + col * (bw + gap)
    y1 = -margin - row * (bh + gap)
    edge, fill = colours(number)
    ax.add_patch(FancyBboxPatch((x0, y1 - bh), bw, bh,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                linewidth=0.9, edgecolor=edge, facecolor=fill))
    boxes.append((x0, y1 - bh, x0 + bw, y1))
    cx = x0 + bw / 2
    texts = [ax.text(cx, 0, number, ha="center", va="top", fontsize=6.5, color=NUMBER),
             ax.text(cx, 0, title, ha="center", va="top", fontsize=7.5, fontweight="bold",
                     color=TEXT, linespacing=1.15)]
    if note is None:
        texts.append(ax.text(cx, 0, "planned", ha="center", va="top", fontsize=6.5,
                             style="italic", color=NUMBER))
    else:
        texts.append(ax.text(cx, 0, note, ha="center", va="top", fontsize=6.5, color=TEXT,
                             linespacing=1.2))
    stacks.append((texts, (y1 - bh / 2)))

# Centre each stack of number, title and note vertically in its square
fig.canvas.draw()
inv = ax.transData.inverted()


def extent(t):
    (a0, b0), (a1, b1) = inv.transform(t.get_window_extent().get_points())
    return a0, b0, a1, b1


spacing = [0.05, 0.07]   # after the number, after the title
for texts, yc in stacks:
    heights = [extent(t)[3] - extent(t)[1] for t in texts]
    y = yc + (sum(heights) + sum(spacing)) / 2
    for t, h, s in zip(texts, heights, spacing + [0]):
        t.set_y(y)
        y -= h + s

# Key, centred below the grid, for the sections with subsections
y_key = -H + margin + key_h / 2
sw, sw_gap, entry_gap = 0.13, 0.05, 0.12   # swatch width, swatch to label, between entries
key_texts = [ax.text(0, y_key, label, ha="left", va="center", fontsize=5.4, color=TEXT)
             for _, _, label in SECTION_COLOURS.values()]
fig.canvas.draw()
widths = [extent(t)[2] - extent(t)[0] for t in key_texts]
total = sum(sw + sw_gap + w for w in widths) + entry_gap * (len(widths) - 1)
xk = (W - total) / 2
for (edge, fill, _), t, w in zip(SECTION_COLOURS.values(), key_texts, widths):
    ax.add_patch(FancyBboxPatch((xk, y_key - 0.045), sw, 0.09,
                                boxstyle="round,pad=0,rounding_size=0.02",
                                linewidth=0.8, edgecolor=edge, facecolor=fill))
    t.set_x(xk + sw + sw_gap)
    xk += sw + sw_gap + w + entry_gap

# Every text must stay inside its square, and the key inside the figure
fig.canvas.draw()
for (texts, _), (bx0, by0, bx1, by1) in zip(stacks, boxes):
    for t in texts:
        a0, b0, a1, b1 = extent(t)
        assert a0 > bx0 + 0.03 and a1 < bx1 - 0.03, t.get_text()
        assert b0 > by0 + 0.03 and b1 < by1 - 0.03, t.get_text()
for t in key_texts:
    a0, b0, a1, b1 = extent(t)
    assert a1 < W - 0.02 and b0 > -H + 0.02, t.get_text()

out_path = Path(__file__).with_suffix(".png")
plt.savefig(out_path, dpi=300, facecolor="white")
print(f"Saved {out_path}")
