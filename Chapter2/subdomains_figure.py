from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DARK_GREEN = "#1B5E20"
MID_GREEN = "#2E7D32"
FILL_GREEN = "#E8F5E9"
HEADER_GREEN = "#A5D6A7"
TEXT_DARK = "#1B3A1F"
DOT_IN = "#1B5E20"
DOT_OUT = "#B0B0B0"

N = 9
PAD = 1 

fig = plt.figure(figsize=(9.5, 8.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.55], hspace=0.18, wspace=0.18)

ax_code = fig.add_subplot(gs[0, :])
ax_code.set_xlim(0, 1)
ax_code.set_ylim(0, 1)
ax_code.axis("off")

card = FancyBboxPatch(
    (0.01, 0.03), 0.98, 0.94,
    boxstyle="round,pad=0.02,rounding_size=0.03",
    linewidth=1.6, edgecolor=MID_GREEN, facecolor=FILL_GREEN,
    transform=ax_code.transAxes, zorder=1,
)
ax_code.add_patch(card)

code_lines = [
    (">>> from devito import Grid", TEXT_DARK, "bold", 13),
    ("", TEXT_DARK, "normal", 13),
    (">>> grid = Grid(shape=(9, 9))", TEXT_DARK, "bold", 13),
    ("", TEXT_DARK, "normal", 13),
    (">>> grid.subdomains", TEXT_DARK, "bold", 13),
    ("{'domain': Domain[domain(x, y)], 'interior': Interior[interior(x, y)]}", "#333333", "normal", 10.5),
    ("", TEXT_DARK, "normal", 13),
    (">>> grid.subdomains['domain'].shape", TEXT_DARK, "bold", 13),
    ("(9, 9)", "#333333", "normal", 10.5),
    ("", TEXT_DARK, "normal", 13),
    (">>> grid.subdomains['interior'].shape", TEXT_DARK, "bold", 13),
    ("(7, 7)", "#333333", "normal", 10.5),
]

y0 = 0.93
dy = 0.072
for i, (line, color, weight, fontsize) in enumerate(code_lines):
    ax_code.text(
        0.05, y0 - i * dy, line, ha="left", va="center",
        fontsize=fontsize, family="monospace", color=color, fontweight=weight,
        transform=ax_code.transAxes, zorder=2,
    )

def draw_panel(ax, title, lo, hi, shape_label):
    coords = range(N)
    for x in coords:
        for y in coords:
            inside = lo <= x <= hi and lo <= y <= hi
            ax.scatter(
                x, y,
                s=42 if inside else 22,
                color=DOT_IN if inside else DOT_OUT,
                zorder=3,
                clip_on=False,
            )

    box = FancyBboxPatch(
        (lo - 0.6, lo - 0.6), (hi - lo) + 1.2, (hi - lo) + 1.2,
        boxstyle="round,pad=0.02,rounding_size=0.35",
        linewidth=2.0, edgecolor=DARK_GREEN, facecolor="none", zorder=2,
    )
    ax.add_patch(box)

    ax.set_title(f"'{title}'", fontsize=15, fontweight="bold",
                 color=TEXT_DARK, pad=10)
    ax.text(0.5, -0.02, f"shape = {shape_label}", transform=ax.transAxes,
            ha="center", va="top", fontsize=12, family="monospace",
            color=MID_GREEN)

    ax.set_xlim(-1.1, N)
    ax.set_ylim(-1.1, N)
    ax.set_aspect("equal")
    ax.axis("off")


ax_domain = fig.add_subplot(gs[1, 0])
draw_panel(ax_domain, "domain", lo=0, hi=N - 1, shape_label="(9, 9)")

ax_interior = fig.add_subplot(gs[1, 1])
draw_panel(ax_interior, "interior", lo=PAD, hi=N - 1 - PAD, shape_label="(7, 7)")

out_path = Path(__file__).with_name("subdomains.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.3, facecolor="white")
print(f"Saved {out_path}")
