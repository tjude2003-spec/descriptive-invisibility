#!/usr/bin/env python3
"""
Plot distribution of extracted relation types as a bar chart.

Reads relation type counts from rq3_network_stats_v3.json and produces
a log-scale bar chart showing the five relation types with counts
and percentages.

Outputs:
  - fig_rq3_relation_types.png (300 DPI)
  - fig_rq3_relation_types.pdf

Setup:
    Edit the paths in the CONFIGURATION block below, then run:

        python3 plot_relation_types.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# ── CONFIGURATION ────────────────────────────────────────────────────

OUTPUT_DIR = Path("output")

# ── END CONFIGURATION ────────────────────────────────────────────────

LABELS = ["Widow", "Spouse", "Child", "Sibling", "Widower"]
COUNTS = [28946, 13454, 1655, 192, 2]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = sum(COUNTS)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.linewidth": 0.7,
    })

    fig, ax = plt.subplots(figsize=(7, 4))

    colors = cm.OrRd([(i + 2) / (len(LABELS) + 3)
                       for i in range(len(LABELS))])

    bars = ax.bar(LABELS, COUNTS, color=colors, edgecolor="black",
                  linewidth=0.5, width=0.6)

    ax.set_yscale("log")
    ax.set_ylabel("Number of extracted relations")
    ax.set_title(f"Typed relations extracted (n = {total:,})",
                 loc="left", fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, count in zip(bars, COUNTS):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                count * 1.4,
                f"{count:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylim(top=COUNTS[0] * 4)

    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"fig_rq3_relation_types.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
