#!/usr/bin/env python3
"""
Plot name structure type distribution as a horizontal bar chart.

Reads name_decomposition_v2_stats.json and produces a horizontal bar chart
showing the distribution of name structure types across extracted entities.

Outputs:
  - fig_rq2_name_structures.png (300 DPI)
  - fig_rq2_name_structures.pdf

Setup:
    Edit the paths in the CONFIGURATION block below, then run:

        python3 plot_name_structures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── CONFIGURATION ────────────────────────────────────────────────────
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

INPUT_JSON = DATA_DIR / "name_decomposition_v2_stats.json"
OUTPUT_DIR = Path(".")

# ── END CONFIGURATION ────────────────────────────────────────────────

STRUCTURE_ORDER = [
    ("given_family",            "Given + family"),
    ("given_prefix_family",     "Given + prefix + family"),
    ("given_patronymic",        "Given + patronymic"),
    ("given_patronymic_family", "Given + patronymic + family"),
    ("given_pat_prefix_family", "Given + pat. + prefix + family"),
    ("given_prefix",            "Given + prefix (truncated)"),
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_JSON) as f:
        data = json.load(f)

    total = data["total_entities"]
    structs = data["structure_types"]

    keys =   [k for k, _ in STRUCTURE_ORDER]
    labels = [l for _, l in STRUCTURE_ORDER]
    counts = [structs[k]["count"] for k in keys]
    pcts =   [structs[k]["pct"] for k in keys]

    labels = labels[::-1]
    counts = counts[::-1]
    pcts = pcts[::-1]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    })

    fig, ax = plt.subplots(figsize=(7, 3.5))

    n = len(labels)
    colors = cm.RdPu([(i + 2) / (n + 3) for i in range(n)])

    ax.barh(range(n), counts, color=colors, edgecolor="black",
            linewidth=0.5, height=0.6)

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("Number of entity strings")
    ax.set_title(f"Name structure types (n = {total:,})",
                 loc="left", fontsize=11)

    for i, (count, pct) in enumerate(zip(counts, pcts)):
        ax.text(count * 1.3, i, f"{count:,} ({pct:.1f}%)",
                va="center", fontsize=9)

    ax.set_xlim(right=counts[-1] * 8)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = OUTPUT_DIR / f"fig_rq2_name_structures.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
