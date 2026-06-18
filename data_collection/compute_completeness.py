#!/usr/bin/env python3
"""
Compute HTR completeness per inventory and produce histogram figure.

For each inventory in the HTR cache, counts non-empty .txt files
against the expected page count from the Transkribus manifest.

Outputs:
  - completeness_per_inventory.csv
  - fig_completeness_histogram.png (300 DPI)
  - fig_completeness_histogram.pdf

Setup:
    Edit the paths in the CONFIGURATION block below to match your
    local directory structure, then run:

        python3 compute_completeness.py
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CONFIGURATION ────────────────────────────────────────────────────
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, HTR_CACHE

CACHE_DIR = HTR_CACHE                                                  # directory of per-inventory .txt files
MANIFEST  = DATA_DIR / "_manifest_merged.json"                         # Transkribus manifest with page counts
OUTPUT_DIR = Path(".")                    # where CSV and figures are written
THRESHOLD  = 0.98                              # completeness cutoff

# ── END CONFIGURATION ────────────────────────────────────────────────


def compute_completeness(cache_dir: Path, manifest: dict) -> list:
    """
    For each inventory in the manifest, compute:
      - expected pages (from manifest)
      - actual non-empty .txt files in cache
      - completeness fraction
    """
    results = []

    for inv_nr, status in manifest.items():
        expected = status.get("page_count", 0)
        if expected == 0:
            continue

        inv_dir = cache_dir / inv_nr
        non_empty = 0
        if inv_dir.exists():
            for f in inv_dir.iterdir():
                if f.suffix == ".txt" and f.stat().st_size > 0:
                    non_empty += 1

        completeness = non_empty / expected
        results.append({
            "inventory_number": inv_nr,
            "expected_pages": expected,
            "non_empty_pages": non_empty,
            "empty_or_missing": expected - non_empty,
            "completeness": round(completeness, 4),
        })

    results.sort(key=lambda r: r["completeness"])
    return results


def make_histogram(results: list, threshold: float, out_dir: Path):
    """Histogram of completeness fractions with threshold line."""
    fractions = [r["completeness"] for r in results]
    n_above = sum(1 for f in fractions if f >= threshold)
    n_below = sum(1 for f in fractions if f < threshold)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 12,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    fig, ax = plt.subplots(figsize=(7, 4))

    bins = [i / 20 for i in range(21)]
    ax.hist(fractions, bins=bins, color="white", edgecolor="black", linewidth=0.8)

    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=0.8)
    ax.text(
        threshold - 0.01, ax.get_ylim()[1] * 0.92,
        f"{threshold:.0%} threshold",
        ha="right", va="top", fontsize=10, fontstyle="italic",
    )

    ax.text(
        0.99, 0.95,
        f"n \u2265 {threshold:.0%}: {n_above}\nn < {threshold:.0%}: {n_below}",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
    )

    ax.set_xlabel("Completeness (non-empty pages / expected pages)")
    ax.set_ylabel("Number of inventories")
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = out_dir / f"fig_completeness_histogram.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figures to {out_dir}/")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"Manifest: {len(manifest)} inventories")

    results = compute_completeness(CACHE_DIR, manifest)
    print(f"Computed completeness for {len(results)} inventories")

    # ── CSV ──
    csv_path = OUTPUT_DIR / "completeness_per_inventory.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "inventory_number", "expected_pages", "non_empty_pages",
            "empty_or_missing", "completeness",
        ])
        w.writeheader()
        w.writerows(results)
    print(f"Saved {csv_path}")

    # ── Summary ──
    fractions = [r["completeness"] for r in results]
    n_above = sum(1 for f in fractions if f >= THRESHOLD)
    n_below = sum(1 for f in fractions if f < THRESHOLD)
    print(f"\nThreshold: {THRESHOLD:.0%}")
    print(f"  Above: {n_above}")
    print(f"  Below: {n_below}")

    if n_below > 0:
        below = [r for r in results if r["completeness"] < THRESHOLD]
        print(f"  Below-threshold range: "
              f"{below[0]['completeness']:.1%} – {below[-1]['completeness']:.1%}")

    # ── Figure ──
    make_histogram(results, THRESHOLD, OUTPUT_DIR)


if __name__ == "__main__":
    main()
