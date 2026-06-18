#!/usr/bin/env python3
"""
Waffle chart showing the 400:1 ratio of masculine to feminine
patronymic suffixes in the extracted entity population.

Each cell represents ~500 entity strings. 399 cells are masculine,
1 cell is feminine.

Outputs:
  - fig_rq2_gender_patronymic.png (300 DPI)
  - fig_rq2_gender_patronymic.pdf

Setup:
    Edit OUTPUT_DIR below, then run:

        python3 plot_gender_patronymic.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

# ── CONFIGURATION ────────────────────────────────────────────────────

OUTPUT_DIR = Path(".")

# ── END CONFIGURATION ────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
    })

    grid = np.ones((20, 20))
    grid[19, 0] = 0

    fig, ax = plt.subplots(figsize=(5.5, 6.2))

    cell_size = 0.9
    gap = 0.1

    masc_color = "blue"
    fem_color = "pink"

    for row in range(20):
        for col in range(20):
            x = col * (cell_size + gap)
            y = (19 - row) * (cell_size + gap)
            color = masc_color if grid[row, col] == 1 else fem_color
            lw = 0.3 if grid[row, col] == 1 else 1.5
            ec = "#666" if grid[row, col] == 1 else "#333"
            rect = mpatches.FancyBboxPatch(
                (x, y), cell_size, cell_size,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor=ec, linewidth=lw,
            )
            ax.add_patch(rect)

    total_w = 20 * (cell_size + gap)
    ax.set_xlim(-0.5, total_w)
    ax.set_ylim(-4.5, total_w + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(total_w / 2, total_w + 0.8,
            "1 cell \u2248 500 entity strings with patronymic suffix",
            ha="center", fontsize=10, color="#555")

    lx = 1.0
    ly1 = -1.5
    ly2 = -2.8

    ax.add_patch(mpatches.FancyBboxPatch(
        (lx, ly1), cell_size * 0.8, cell_size * 0.8,
        boxstyle="round,pad=0.02",
        facecolor=masc_color, edgecolor="#666", linewidth=0.3))
    ax.text(lx + 1.3, ly1 + 0.35,
            "Masculine suffix (-sz, -zoon, -sen): 202,553  (99.8%)",
            fontsize=9, va="center")

    ax.add_patch(mpatches.FancyBboxPatch(
        (lx, ly2), cell_size * 0.8, cell_size * 0.8,
        boxstyle="round,pad=0.02",
        facecolor=fem_color, edgecolor="#333", linewidth=1.5))
    ax.text(lx + 1.3, ly2 + 0.35,
            "Feminine suffix (-dochter, -dr): 490  (0.2%)",
            fontsize=9, va="center")

    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"fig_rq2_gender_patronymic.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
