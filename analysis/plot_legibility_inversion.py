#!/usr/bin/env python3
"""
Plot the legibility inversion: NER recall vs authority match rate
across name structure types, shown as a butterfly chart.

The figure shows that NER recall and authority matchability run in
opposite directions across the same name structures.

Outputs:
  - fig_rq2_legibility_inversion.png (300 DPI)
  - fig_rq2_legibility_inversion.pdf

Setup:
    Edit OUTPUT_DIR below, then run:

        python3 plot_legibility_inversion.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

# ── CONFIGURATION ────────────────────────────────────────────────────

OUTPUT_DIR = Path(".")

# ── END CONFIGURATION ────────────────────────────────────────────────

TYPES = [
    "Given + patronymic",
    "Given + family",
    "Given + prefix + family",
    "Given + patronymic\n+ family",
]

NER_RECALL =    [0.691, 0.599, 0.541, 0.445]
ECARTICO_RATE = [0.0,   2.1,   2.8,   8.2]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.linewidth": 0.7,
    })

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(9, 3.5),
        gridspec_kw={"wspace": 0.02},
        sharey=True,
    )

    y_pos = np.arange(len(TYPES))

    # Left: NER recall (reversed x-axis)
    colors_ner = cm.RdPu([0.7, 0.6, 0.5, 0.4])
    ax_left.barh(y_pos, NER_RECALL, height=0.55,
                 color=colors_ner, edgecolor="black", linewidth=0.5)
    ax_left.set_xlim(0.8, 0)
    ax_left.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax_left.set_xlabel("NER recall")
    ax_left.set_title("NER extraction", fontsize=11, loc="center")
    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(TYPES, fontsize=9.5, ha="center")
    ax_left.tick_params(axis="y", length=0, pad=55)
    ax_left.spines["right"].set_visible(False)
    ax_left.spines["top"].set_visible(False)

    for i, v in enumerate(NER_RECALL):
        ax_left.text(v + 0.1, i, f"{v:.3f}",
                     va="center", ha="left", fontsize=9)

    # Right: ECARTICO match rate
    colors_ec = cm.RdPu([0.2, 0.35, 0.45, 0.75])
    ax_right.barh(y_pos, ECARTICO_RATE, height=0.55,
                  color=colors_ec, edgecolor="black", linewidth=0.5)
    ax_right.set_xlim(0, 10)
    ax_right.set_xlabel("ECARTICO match rate (%)")
    ax_right.set_title("Authority recognition", fontsize=11, loc="center")
    ax_right.tick_params(axis="y", left=False)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["top"].set_visible(False)

    for i, v in enumerate(ECARTICO_RATE):
        if v == 0:
            ax_right.text(0.15, i, "0.0%",
                          va="center", ha="left", fontsize=9)
        else:
            ax_right.text(v + 0.15, i, f"{v:.1f}%",
                          va="center", ha="left", fontsize=9)

    # Direction arrows
    ax_left.annotate("", xy=(0.1, -3.0), xytext=(0.7, -0.7),
                     arrowprops=dict(arrowstyle="->", color="#666", lw=1.2),
                     clip_on=False)
    ax_left.text(0.4, -2.0, "decreasing recall", ha="center",
                 fontsize=8, color="#666", fontstyle="italic", clip_on=False)

    ax_right.annotate("", xy=(7.5, -3.0), xytext=(0.5, -0.7),
                      arrowprops=dict(arrowstyle="->", color="#666", lw=1.2),
                      clip_on=False)
    ax_right.text(4.0, -2.0, "increasing match rate", ha="center",
                  fontsize=8, color="#666", fontstyle="italic", clip_on=False)

    fig.subplots_adjust(bottom=0.50)

    for ext in ("png", "pdf"):
        path = OUTPUT_DIR / f"fig_rq2_legibility_inversion.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
