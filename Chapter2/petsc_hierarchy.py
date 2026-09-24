from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


# File to generate a diagram of the PETSc API hierarchy. Similar to the one in the PETSc manual,
# see Figure 1.1 in https://petsc.org/release/manual/manual.pdf

SOLVER = dict(edge="#1E4F8C", fill="#EEF4FB", tab="#C9DDF2", text="#12304F")
DOMAIN = dict(edge="#5B2A7A", fill="#F6F0F9", tab="#E1CFEA", text="#3A1A4F")
DATA = dict(edge="#9A4A12", fill="#FDF4EB", tab="#F6D9BD", text="#4F260A")
BODY_TEXT = "#333333"

fig, ax = plt.subplots(figsize=(10.5, 8.6))
ax.set_xlim(1.05, 11.35)
ax.set_aspect("equal")
ax.axis("off")


def draw_card(x, y, w, h, abbr, name, examples, style, tab_w=0.95,
              fontsize=10.5):
    box = dict(boxstyle="round,pad=0,rounding_size=0.1")
    ax.add_patch(FancyBboxPatch((x, y), w, h, **box, linewidth=0,
                                facecolor=style["fill"], zorder=3))
    # Tinted tab on the left holding the acronym
    ax.add_patch(FancyBboxPatch((x, y), tab_w, h, **box, linewidth=0,
                                facecolor=style["tab"], zorder=4))
    ax.add_patch(Rectangle((x + tab_w - 0.15, y), 0.15, h,
                           facecolor=style["tab"], edgecolor="none", zorder=4))
    ax.plot([x + tab_w] * 2, [y, y + h], color=style["edge"], linewidth=1.2,
            zorder=5)
    ax.add_patch(FancyBboxPatch((x, y), w, h, **box, linewidth=1.6,
                                edgecolor=style["edge"], facecolor="none",
                                zorder=6))
    ax.text(x + tab_w / 2, y + h / 2, abbr, ha="center", va="center",
            fontsize=14, fontweight="bold", color=style["text"], zorder=7)

    tx = x + tab_w + 0.2
    lines = [name] + examples
    n = len(lines)
    spacing = 0.36
    for i, line in enumerate(lines):
        ly = y + h / 2 + spacing * ((n - 1) / 2 - i)
        if i == 0:
            ax.text(tx, ly, line, ha="left", va="center", fontsize=12.5,
                    fontweight="bold", color=style["text"], zorder=5)
        else:
            ax.text(tx, ly, line, ha="left", va="center", fontsize=fontsize,
                    color=BODY_TEXT, zorder=5)



left, right = 1.2, 11.2
main_w = 5.5
gap = 0.3
side_x = left + main_w + gap
side_w = right - side_x

h_card = 1.1
full_w = right - left
rows = [11.0 - i * (h_card + gap) for i in range(6)]
y_ts, y_snes, y_ksp, y_pc, y_mat, y_vec = rows
y_data = y_vec


draw_card(left, y_ts, main_w, h_card, "TS", "Time Steppers",
          ["Forward Euler  ·  Backward Euler  ·  RK  ·  …"], SOLVER)
draw_card(side_x, y_ts, side_w, h_card, "DM", "Domain Management",
          ["DMDA · DMPlex · DMShell · …"], DOMAIN)

draw_card(left, y_snes, main_w, h_card, "SNES", "Nonlinear Solvers",
          ["Newton line search · Trust region · NGMRES · …"], SOLVER)
draw_card(side_x, y_snes, side_w, h_card, "TAO", "Optimisation",
          ["Newton · Levenberg–Marquardt · …"], SOLVER)

draw_card(left, y_ksp, full_w, h_card, "KSP", "Krylov Subspace Methods",
          ["CG  ·  GMRES  ·  BiCGStab  ·  MINRES  ·  Richardson  ·  Chebyshev  ·  …"], SOLVER)
draw_card(left, y_pc, full_w, h_card, "PC", "Preconditioners",
          ["Jacobi  ·  SOR  ·  ILU  ·  LU  ·  MG  ·  AMG  ·  Shell  ·  …"], SOLVER)

draw_card(left, y_mat, full_w, h_card, "Mat", "Operators",
          ["CSR  ·  Block CSR  ·  Dense  ·  Shell  ·  …"], DATA)
draw_card(left, y_vec, main_w, h_card, "Vec", "Vectors",
          ["Standard  ·  CUDA  ·  …"], DATA)
draw_card(side_x, y_vec, side_w, h_card, "IS", "Index Sets",
          ["General · Block · Stride"], DATA)

ax.set_ylim(y_data - 0.15, y_ts + h_card + 0.15)

plt.tight_layout()
out_path = Path(__file__).with_suffix(".png")
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
