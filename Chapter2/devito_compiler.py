from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DARK_GREEN = "#1B5E20"
MID_GREEN = "#2E7D32"
FILL_GREEN = "#E8F5E9"
HEADER_GREEN = "#A5D6A7"
TEXT_DARK = "#1B3A1F"

STAGES = [
    ("Equations lowering", "Input Equations → Lowered Equations", False),
    ("Local analysis", "", False),
    ("Clustering", "Lowered Equations → Clusters", True),
    ("Symbolic optimisation", "Clusters → Clusters", True),
    ("IET construction", "Clusters → IET", False),
    ("IET analysis", "IET → IET", False),
    ("IET optimisation", "IET → IET", True),
    ("Synthesis", "IET → CGen AST → C string", True),
    ("JIT Compilation", "C string → kernel.c → kernel.so", False),
]
STAGES = [(f"{i}.  {title}", subtitle, highlighted)
          for i, (title, subtitle, highlighted) in enumerate(STAGES, start=1)]

fig, ax = plt.subplots(figsize=(5.4, 8.1))
ax.axis("off")

card_w = 4.4
card_x = 0.4
card_cx = card_x + card_w / 2
h_card = 0.68
gap = 0.28

n = len(STAGES)
y_positions = []
y = (n - 1) * (h_card + gap)
for _ in STAGES:
    y_positions.append(y)
    y -= (h_card + gap)

ax.set_xlim(0, card_x * 2 + card_w)


def draw_card(y_bottom, title, subtitle, highlighted):
    facecolor = FILL_GREEN
    edgecolor = DARK_GREEN
    linewidth = 1.7

    body = FancyBboxPatch(
        (card_x, y_bottom), card_w, h_card,
        boxstyle="round,pad=0.02,rounding_size=0.07",
        linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor, zorder=3,
    )
    ax.add_patch(body)

    cy = y_bottom + h_card / 2
    if subtitle:
        ax.text(
            card_cx, cy + 0.12, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=TEXT_DARK, zorder=4,
        )
        ax.text(
            card_cx, cy - 0.15, subtitle, ha="center", va="center",
            fontsize=8, style="italic", color="#333333", zorder=4,
        )
    else:
        ax.text(
            card_cx, cy, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=TEXT_DARK, zorder=4,
        )


def draw_arrow(y_top, y_bottom):
    ax.annotate(
        "", xy=(card_cx, y_bottom), xytext=(card_cx, y_top),
        arrowprops=dict(arrowstyle="-|>", color=DARK_GREEN, lw=1.5, mutation_scale=13),
        zorder=2,
    )


for i, ((title, subtitle, highlighted), y_bottom) in enumerate(zip(STAGES, y_positions)):
    draw_card(y_bottom, title, subtitle, highlighted)
    if i > 0:
        prev_y_bottom = y_positions[i - 1]
        draw_arrow(prev_y_bottom, y_bottom + h_card)

ax.set_ylim(y_positions[-1] - 0.3, y_positions[0] + h_card + 0.3)

plt.tight_layout()
out_path = Path(__file__).with_suffix(".png")
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")
