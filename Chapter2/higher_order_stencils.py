from pathlib import Path

import matplotlib.pyplot as plt
import sympy as sp
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from devito import Function, Grid

DARK_GREEN = "#1B5E20"
MID_GREEN = "#2E7D32"
FILL_GREEN = "#E8F5E9"
TEXT_DARK = "#1B3A1F"
NEG_COLOR = "#B71C1C"

SPACE_ORDERS = [2, 6, 12]


def get_stencil_lines(so):
    grid = Grid(shape=(11,))
    x = grid.dimensions[0]
    h = x.spacing

    f = Function(name="f", grid=grid, space_order=so)
    expr = f.dx2.evaluate
    terms = sp.Add.make_args(sp.expand(expr))

    ordered = []
    for t in terms:
        call = list(t.atoms(sp.Function))[0]
        offset = int(sp.expand(call.args[0] - x).coeff(h))
        ordered.append((offset, t))
    ordered.sort(key=lambda p: p[0])

    lines = []
    for offset, t in ordered:
        s = sp.sstr(t)
        if not s.startswith("-"):
            s = "+ " + s
        else:
            s = "- " + s[1:]
        lines.append(s)
    return lines


def draw_code_card(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    card = FancyBboxPatch(
        (0.01, 0.05), 0.98, 0.9,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6, edgecolor=MID_GREEN, facecolor=FILL_GREEN,
        transform=ax.transAxes, zorder=1,
    )
    ax.add_patch(card)

    code_lines = [
        ">>> grid = Grid(shape=(11,))          # 1D grid",
        ">>> f = Function(name='f', grid=grid, space_order=so)",
    ]
    y0, dy = 0.68, 0.4
    for i, line in enumerate(code_lines):
        ax.text(0.5, y0 - i * dy, line, ha="center", va="center",
                 fontsize=19, family="monospace", fontweight="bold",
                 color=TEXT_DARK, transform=ax.transAxes, zorder=2)


def draw_panel(ax, so):
    lines = get_stencil_lines(so)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(0.5, 9.85, f"so = {so}", ha="center", va="top",
             fontsize=21, fontweight="bold", color=DARK_GREEN,
             family="monospace")

    ax.add_patch(FancyArrowPatch((0.5, 9.0), (0.5, 8.5),
                                  arrowstyle="-|>", mutation_scale=22,
                                  linewidth=2.6, color=MID_GREEN))
    ax.text(0.5, 8.2, ">>> f.dx2.evaluate", ha="center", va="top",
             fontsize=16.5, fontweight="bold", color=TEXT_DARK,
             family="monospace")

    n = len(lines)
    card_top, card_bot = 7.55, 0.55
    box = FancyBboxPatch((0.015, card_bot), 0.97, card_top - card_bot,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          linewidth=1.4, edgecolor=MID_GREEN,
                          facecolor=FILL_GREEN, zorder=1)
    ax.add_patch(box)

    pad = 0.32
    line_h = min(0.95, (card_top - card_bot - 2 * pad) / max(n - 1, 1))
    fontsize = 18 if line_h > 0.6 else (14.5 if line_h > 0.42 else 12.5)

    y = card_top - pad
    text_objs = []
    for line in lines:
        color = NEG_COLOR if line.startswith("-") else DARK_GREEN
        t = ax.text(0.5, y, line, ha="center", va="center", fontsize=fontsize,
                     color=color, family="monospace", zorder=2)
        text_objs.append(t)
        y -= line_h

    ax.text(0.5, card_bot - 0.3, f"stencil: {n} points",
             ha="center", va="top", fontsize=16, color=TEXT_DARK,
             fontweight="bold")

    return text_objs


fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, len(SPACE_ORDERS), height_ratios=[0.24, 1],
                       hspace=0.0, wspace=0.05)

ax_code = fig.add_subplot(gs[0, :])
draw_code_card(ax_code)

panel_text_objs = []
for col, so in enumerate(SPACE_ORDERS):
    ax = fig.add_subplot(gs[1, col])
    panel_text_objs.append(draw_panel(ax, so))


fig.canvas.draw()
renderer = fig.canvas.get_renderer()
fitted_fontsize = float("inf")
for text_objs in panel_text_objs:
    ax = text_objs[0].axes
    ax_width_px = ax.get_window_extent(renderer=renderer).width
    box_width_px = ax_width_px * 0.90
    base_fontsize = text_objs[0].get_fontsize()
    max_text_px = max(t.get_window_extent(renderer=renderer).width for t in text_objs)
    scale = min(1.0, box_width_px / max_text_px)
    fitted_fontsize = min(fitted_fontsize, base_fontsize * scale)

for text_objs in panel_text_objs:
    for t in text_objs:
        t.set_fontsize(fitted_fontsize)

out_path = Path(__file__).with_name("higher_order_stencils.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15,
            facecolor="white")
print(f"Saved {out_path}")
