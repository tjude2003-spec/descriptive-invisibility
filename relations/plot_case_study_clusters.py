#!/usr/bin/env python3
"""
Plot case study network clusters for Philip Zweerts and Jan Verleij.

Nodes are rendered as text boxes containing the entity string, colored
by given name. Edges are styled by relation type (solid=widow/spouse,
dashed=child, dash-dot=sibling).

Outputs:
  - fig_rq3_zweerts_cluster.png / .pdf
  - fig_rq3_verleij_cluster.png / .pdf

Requires: networkx, matplotlib
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from pathlib import Path

# ── CONFIGURATION ────────────────────────────────────────────────────
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

ZWEERTS_JSON = DATA_DIR / "rq3_case_philip_zweerts_v2.json"
VERLEIJ_JSON = DATA_DIR / "rq3_case_jan_verleij_v2.json"
OUTPUT_DIR   = Path(".")

# ── END CONFIGURATION ────────────────────────────────────────────────

REL_STYLES = {
    "widow":   {"color": "#8B0000", "style": "-"},
    "spouse":  {"color": "#2E5090", "style": "-"},
    "child":   {"color": "#2E7D32", "style": "--"},
    "sibling": {"color": "#F57F17", "style": "-."},
}

GIVEN_COLORS = ["#7b1fa2", "#1565c0", "#2e7d32", "#c62828",
                "#ef6c00", "#00838f"]
GIVEN_FILLS  = ["#e1bee7", "#bbdefb", "#c8e6c9", "#ffcdd2",
                "#ffe0b2", "#b2ebf2"]


def draw_cluster(ax, cluster, title):
    G = nx.Graph()
    members = {m["entity"]: m for m in cluster["members"]}

    for m in cluster["members"]:
        G.add_node(m["entity"])
    for r in cluster["relations"]:
        G.add_edge(r["person_1"], r["person_2"], type=r["type"])

    pos = nx.spring_layout(G, seed=42, k=3.0, iterations=100)

    given_groups = cluster["given_name_groups"]
    given_to_color = {gn: GIVEN_COLORS[i % len(GIVEN_COLORS)]
                      for i, gn in enumerate(given_groups)}
    given_to_fill = {gn: GIVEN_FILLS[i % len(GIVEN_FILLS)]
                     for i, gn in enumerate(given_groups)}

    for u, v, data in G.edges(data=True):
        rtype = data["type"]
        style = REL_STYLES.get(rtype, {"color": "#999", "style": "-"})
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=style["color"], linestyle=style["style"],
                linewidth=1.2, zorder=1)
        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2
        ax.text(mid_x, mid_y, rtype, fontsize=6,
                ha="center", va="center", color=style["color"],
                fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.9),
                zorder=3)

    for node in G.nodes():
        x, y = pos[node]
        given = members[node]["given"]
        ax.text(x, y, node, fontsize=8, ha="center", va="center",
                fontfamily="serif", zorder=5,
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor=given_to_fill.get(given, "#eee"),
                          edgecolor=given_to_color.get(given, "#666"),
                          linewidth=1.0))

    ax.set_title(title, fontsize=11, pad=12)
    ax.axis("off")
    ax.set_aspect("equal")

    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    margin = 0.4
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    legend_handles = [
        mpatches.Patch(facecolor=given_to_fill[gn],
                       edgecolor=given_to_color[gn],
                       linewidth=1.0, label=f'"{gn}"')
        for gn in given_groups
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8,
              framealpha=0.9, title="Given name", title_fontsize=8)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
    })

    with open(ZWEERTS_JSON) as f:
        zweerts = json.load(f)
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_cluster(ax, zweerts["top_clusters"][0],
                 "Philip Zweerts: orthographic variation cluster\n"
                 "(9 nodes, 2 given names, collapsible)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"fig_rq3_zweerts_cluster.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    with open(VERLEIJ_JSON) as f:
        verleij = json.load(f)
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_cluster(ax, verleij["top_clusters"][0],
                 "Jan Verleij: family structure cluster\n"
                 "(6 nodes, 4 given names, not collapsible)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"fig_rq3_verleij_cluster.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
