from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


DARK_GREEN = "#1B5E20"
MID_GREEN = "#2E7D32"
FILL_GREEN = "#E8F5E9"
HEADER_GREEN = "#A5D6A7"
TEXT_DARK = "#1B3A1F"

fig, ax = plt.subplots(figsize=(6.4, 8.6))
ax.set_xlim(0, 7)
ax.axis("off")

card_x = 0.7
card_w = 5.6
card_cx = card_x + card_w / 2

def draw_card(y_bottom, h_card, header_h, num, title, code_line):
    # card body
    body = FancyBboxPatch(
        (card_x, y_bottom), card_w, h_card,
        boxstyle="round,pad=0.02,rounding_size=0.07",
        linewidth=1.6, edgecolor=MID_GREEN, facecolor=FILL_GREEN, zorder=3,
    )
    ax.add_patch(body)

    inset = 0.05
    header = Rectangle(
        (card_x + inset, y_bottom + h_card - header_h),
        card_w - 2 * inset, header_h - inset,
        facecolor=HEADER_GREEN, edgecolor="none", zorder=4,
    )
    ax.add_patch(header)

    ax.plot(
        [card_x + 0.04, card_x + card_w - 0.04],
        [y_bottom + h_card - header_h, y_bottom + h_card - header_h],
        color=MID_GREEN, linewidth=1.1, zorder=5,
    )

    ax.text(
        card_x + 0.22, y_bottom + h_card - header_h / 2 - 0.02,
        f"{num}  {title}", ha="left", va="center",
        fontsize=12, fontweight="bold", color=TEXT_DARK, zorder=6,
    )
    code_y = y_bottom + (h_card - header_h) / 2
    ax.text(
        card_x + 0.22, code_y, code_line, ha="left", va="center",
        fontsize=10.3, family="monospace", color="#333333", zorder=6,
    )

def draw_arrow(y_top, y_bottom, color=DARK_GREEN):
    ax.annotate(
        "", xy=(card_cx, y_bottom), xytext=(card_cx, y_top),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=18),
        zorder=2,
    )

h_card = 1.35
header_h = 0.5
gap = 0.55

y_grid = 11.7
y_symfunc = y_grid - gap - h_card
y_symeq = y_symfunc - gap - h_card
y_op = y_symeq - gap - h_card

h_so = 1.0
y_so = y_op - (gap + 0.55) - h_so

draw_card(y_grid, h_card, header_h, "1", "Grid", "grid = Grid(shape=(...))")
draw_arrow(y_grid, y_symfunc + h_card)

draw_card(y_symfunc, h_card, header_h, "2", "Symbolic Functions",
          "u = TimeFunction(name='u', grid=grid)")
draw_arrow(y_symfunc, y_symeq + h_card)

draw_card(y_symeq, h_card, header_h, "3", "Symbolic Equations",
          "eqn = u.dt - alpha*u.laplace")
draw_arrow(y_symeq, y_op + h_card)

draw_card(y_op, h_card, header_h, "4", "Operator", "op = Operator(eqn)")

boundary_pad = 0.28
boundary_pad_bottom = 0.48
boundary_y_bottom = y_op - boundary_pad_bottom
boundary_y_top = y_grid + h_card + boundary_pad
boundary_box = FancyBboxPatch(
    (card_x - boundary_pad, boundary_y_bottom),
    card_w + 2 * boundary_pad,
    boundary_y_top - boundary_y_bottom,
    boxstyle="round,pad=0.02,rounding_size=0.14",
    linestyle=(0, (6, 3, 1, 3)), linewidth=1.4,
    edgecolor=DARK_GREEN, facecolor="none", zorder=1,
)
ax.add_patch(boundary_box)
ax.text(
    card_x + card_w + boundary_pad - 0.15, boundary_y_bottom + 0.1, "Python",
    fontsize=9.5, style="italic", fontweight="bold", color=DARK_GREEN,
    ha="right", va="bottom", zorder=1,
)

draw_arrow(y_op, y_so + h_so)

so_box = FancyBboxPatch(
    (card_x, y_so), card_w, h_so,
    boxstyle="round,pad=0.02,rounding_size=0.1",
    linewidth=1.8, edgecolor=DARK_GREEN, facecolor=DARK_GREEN, zorder=3,
)
ax.add_patch(so_box)
ax.text(card_cx, y_so + h_so / 2, "JIT-compiled .so", ha="center", va="center",
         fontsize=13, fontweight="bold", color="white", zorder=4)

ax.set_ylim(y_so - 0.4, boundary_y_top + 0.3)

plt.tight_layout()
out_path = Path(__file__).with_suffix(".png")
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")
