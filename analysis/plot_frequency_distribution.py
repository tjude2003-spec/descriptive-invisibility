#!/usr/bin/env python3
"""
Plot frequency distribution of extracted entity strings as a two-panel figure.

Reads rq2_frequency_distribution.csv and produces:
  (a) Bar chart for entity strings appearing in 1-10 inventories with cumulative %
  (b) Log-scale scatter of the full distribution including long tail

Outputs:
  - fig_rq2_frequency_distribution.png (300 DPI)
  - fig_rq2_frequency_distribution.pdf

Setup:
    Edit the paths in the CONFIGURATION block below, then run:

        python3 plot_frequency_distribution.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm 

# ── CONFIGURATION ────────────────────────────────────────────────────
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

INPUT_CSV  = DATA_DIR / "rq2_frequency_distribution.csv"
OUTPUT_DIR = Path(".")

# ── END CONFIGURATION ────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_CSV) as f:
        rows = list(csv.DictReader(f))

    inv_counts = [int(r["inventory_count"]) for r in rows]
    num_ents = [int(r["num_entities"]) for r in rows]
    total = sum(num_ents)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    })

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 5.5),
        gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.6},
    )

    # -- Panel (a): bar chart, frequencies 1-10 --
    show_n = 10
    x_vals = inv_counts[:show_n]
    y_vals = num_ents[:show_n]

    colors = cm.RdPu([(i + 3) / (show_n + 3) for i in range(show_n)])
    ax_top.bar(x_vals, y_vals, color=colors, edgecolor="black",
               linewidth=0.5, width=0.7)

    for x, y in zip(x_vals, y_vals):
        label = f"{y/1000:.0f}k" if y > 50000 else f"{y/1000:.1f}k"
        ax_top.text(x, y + total * 0.01, label,
                    ha="center", va="bottom", fontsize=8)
        ax_top.set_ylim(0, max(y_vals) * 1.15)

    # Cumulative % on secondary axis
    ax_top_r = ax_top.twinx()
    cumul = []
    running = 0
    for y in y_vals:
        running += y
        cumul.append(running / total * 100)

    ax_top_r.plot(x_vals, cumul, color="black", linewidth=1,
                  marker="o", markersize=3, zorder=5)
    ax_top_r.set_ylim(75, 101)
    ax_top_r.set_yticks([])
    ax_top_r.set_ylabel("")

    for i, label_idx in enumerate([0, 4, 9]):
        ax_top_r.annotate(
        f"{cumul[label_idx]:.1f}%",
        xy=(x_vals[label_idx], cumul[label_idx]),
        xytext=(0, -14), textcoords="offset points",
        ha="center", fontsize=8, fontstyle="italic",
        arrowprops=dict(arrowstyle="-", linewidth=0.4, color="gray"),
    )

    ax_top.set_xlabel(
        "Number of inventories in which a name string appears")
    ax_top.set_ylabel("Unique entity strings")
    ax_top.set_xticks(x_vals)
    ax_top.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, p: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))
    ax_top.set_title(
        "(a) Entity strings appearing in 1\u201310 inventories"
        " (98.5% of population)",
        loc="left", fontsize=11)

    # -- Panel (b): log-scale scatter, full distribution --
    ax_bot.scatter(inv_counts, num_ents, s=8, color="blue",
                   edgecolors="none", zorder=3)
    ax_bot.set_yscale("log")
    ax_bot.set_xlabel(
        "Number of inventories in which a name string appears")
    ax_bot.set_ylabel("Unique entity strings (log)")
    ax_bot.set_title("(b) Full distribution including long tail",
                     loc="left", fontsize=11)
    ax_bot.set_xlim(-10, max(inv_counts) + 30)

    ax_bot.annotate(
        '\u201cjan jansz\u201d\n1,529 inventories',
        xy=(1529, 1), xytext=(1200, 50),
        fontsize=8, fontstyle="italic",
        arrowprops=dict(arrowstyle="-", linewidth=0.5, color="black"),
        ha="center",
    )

    for ext in ("png", "pdf"):
        path = OUTPUT_DIR / f"fig_rq2_frequency_distribution.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
