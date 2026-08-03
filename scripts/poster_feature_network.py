#!/usr/bin/env python3
"""Create a poster-ready correlation network from the promoted model's top features.

The chart intentionally separates three ideas:
* node size: relative feature importance stored with the promoted model;
* edge colour: sign of the Pearson correlation in the feature table;
* edge width: absolute correlation strength.

This is an exploratory association graphic, not a causal diagram.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = ROOT / "outputs/features/full_features.parquet"
DEFAULT_METADATA = ROOT / "outputs/metadata/dataset_info.json"
DEFAULT_OUTPUT = ROOT / "outputs/figures/poster_feature_correlation_network"

POSITIVE = "#B2182B"
NEGATIVE = "#2166AC"
BACKGROUND = "#FBFAF7"
INK = "#252525"

GROUPS = {
    "season": ("Seasonality / calendar", "#F3C969"),
    "spatial": ("Neighbouring fire activity", "#9B8AC4"),
    "history": ("Cell fire history", "#E88973"),
    "land": ("Location / vegetation", "#79B791"),
}

LABELS = {
    "doy_sin": "Day-of-year (sin)",
    "doy_cos": "Day-of-year (cos)",
    "week_sin": "Week-of-year (sin)",
    "week_cos": "Week-of-year (cos)",
    "month_sin": "Month (sin)",
    "day_of_week": "Day of week",
    "neighbor_frp_today": "Neighbour FRP (today)",
    "neighbor_fire_today": "Neighbour fires (today)",
    "neighbor_frp_sum_14d": "Neighbour FRP (14-day)",
    "wide_neighbor_frp_sum_14d": "Wide-area FRP (14-day)",
    "fire_sum_14d": "Cell fires (14-day)",
    "fire_sum_21d": "Cell fires (21-day)",
    "fire_sum_30d": "Cell fires (30-day)",
    "days_since_last_fire": "Days since last fire",
    "days_since_last_fire_norm": "Normalised fire interval",
    "fire_count_same_month_hist": "Same-month fire history",
    "tree_cover_pct_2000": "Tree cover (2000)",
    "tree_loss_pct_recent": "Recent tree-cover loss",
    "lat_grid": "Latitude",
    "lon_grid": "Longitude",
}

# A semantic layout makes the default top-20 chart stable and poster-readable.
# Calendar features run left-to-right into fire-history and spatial-fire clusters;
# weakly connected land variables sit on the lower row.
POSTER_POSITION = {
    # Left panel: seasonal cycles and fire recency.
    "doy_sin": (-1.37, 0.58),
    "week_sin": (-0.98, 0.76),
    "month_sin": (-0.88, 0.34),
    "doy_cos": (-1.36, -0.12),
    "week_cos": (-0.92, -0.12),
    "days_since_last_fire": (-0.30, 0.54),
    "days_since_last_fire_norm": (-0.27, 0.08),
    # Right panel: present, neighbouring and rolling fire activity.
    "neighbor_frp_today": (0.32, 0.70),
    "neighbor_fire_today": (1.12, 0.70),
    "neighbor_frp_sum_14d": (0.48, 0.27),
    "wide_neighbor_frp_sum_14d": (1.20, 0.25),
    "fire_sum_14d": (0.27, -0.18),
    "fire_sum_21d": (0.78, -0.33),
    "fire_sum_30d": (1.34, -0.17),
    "fire_count_same_month_hist": (0.32, -0.53),
    # Bottom strip: top-ranked variables without an edge at the chosen cutoff.
    "day_of_week": (-1.30, -1.00),
    "tree_cover_pct_2000": (-0.66, -1.00),
    "tree_loss_pct_recent": (-0.05, -1.00),
    "lat_grid": (0.58, -1.00),
    "lon_grid": (1.21, -1.00),
}


def group_for(feature: str) -> str:
    if feature.startswith(("doy_", "week_", "month_")) or feature == "day_of_week":
        return "season"
    if feature.startswith(("neighbor_", "wide_neighbor_")):
        return "spatial"
    if feature.startswith("tree_") or feature in {"lat_grid", "lon_grid"}:
        return "land"
    return "history"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output path without extension")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top model features to show")
    parser.add_argument("--edge-min", type=float, default=0.40,
                        help="Show edges with |Pearson r| at or above this value")
    parser.add_argument("--label-min", type=float, default=0.85,
                        help="Label edges with |Pearson r| at or above this value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.metadata.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    ranked = metadata.get("feature_importance_top", [])[: args.top]
    if not ranked:
        raise ValueError("metadata has no feature_importance_top entries")

    features = [item["feature"] for item in ranked]
    importance = {item["feature"]: float(item["importance"]) for item in ranked}
    frame = pd.read_parquet(args.features, columns=features)
    corr = frame.corr(method="pearson", min_periods=2)

    graph = nx.Graph()
    for feature in features:
        graph.add_node(feature, importance=importance[feature], group=group_for(feature))
    for i, left in enumerate(features):
        for right in features[i + 1 :]:
            value = float(corr.loc[left, right])
            if abs(value) >= args.edge_min:
                graph.add_edge(left, right, correlation=value, strength=abs(value))

    if all(feature in POSTER_POSITION for feature in features):
        position = {feature: POSTER_POSITION[feature] for feature in features}
    else:
        # Absolute correlation controls attraction; the sign is only visual.
        position = nx.spring_layout(
            graph, seed=19, k=1.10, iterations=500, weight="strength", scale=1.0
        )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.unicode_minus": False,
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(17, 10), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    panel_style = {
        "boxstyle": "round,pad=0.025,rounding_size=0.04",
        "linewidth": 1.1,
        "edgecolor": "#D7D3CC",
        "zorder": 0,
    }
    ax.add_patch(FancyBboxPatch(
        (-1.57, -0.48), 1.48, 1.42, facecolor="#FFF9EA", **panel_style
    ))
    ax.add_patch(FancyBboxPatch(
        (0.04, -0.67), 1.50, 1.61, facecolor="#F8F4FA", **panel_style
    ))
    ax.add_patch(FancyBboxPatch(
        (-1.57, -1.18), 3.11, 0.39, facecolor="#F2F6F3", **panel_style
    ))
    ax.text(-0.83, 0.865, "SEASONALITY & FIRE RECENCY", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#74663A", zorder=1)
    ax.text(0.79, 0.865, "NEIGHBOURING & HISTORICAL FIRE ACTIVITY", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#65587B", zorder=1)
    ax.text(-1.48, -0.835, "OTHER TOP-RANKED FEATURES — no edges at |r| ≥ 0.40",
            ha="left", va="center", fontsize=9.5, fontweight="bold",
            color="#52705E", zorder=1)

    edges = list(graph.edges(data=True))
    edge_colours = [POSITIVE if data["correlation"] > 0 else NEGATIVE for _, _, data in edges]
    edge_widths = [0.8 + data["strength"] * 4.5 for _, _, data in edges]
    nx.draw_networkx_edges(
        graph,
        position,
        ax=ax,
        edgelist=[(left, right) for left, right, _ in edges],
        width=edge_widths,
        edge_color=edge_colours,
        alpha=0.58,
    )

    # Opaque, fixed-padding labels keep every name readable. Edges are drawn
    # first, so they disappear cleanly beneath each box instead of crossing text.
    for node, (x, y) in position.items():
        group = graph.nodes[node]["group"]
        ax.text(
            x,
            y,
            f"{LABELS.get(node, node)}\n{importance[node]:.1f}% importance",
            ha="center",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color=INK,
            linespacing=1.35,
            zorder=3,
            bbox={
                "boxstyle": "round,pad=0.58,rounding_size=0.22",
                "facecolor": GROUPS[group][1],
                "edgecolor": INK,
                "linewidth": 1.25,
            },
        )

    strong_labels = {
        (left, right): f"{data['correlation']:+.2f}"
        for left, right, data in edges
        if data["strength"] >= args.label_min
    }
    nx.draw_networkx_edge_labels(
        graph,
        position,
        ax=ax,
        edge_labels=strong_labels,
        font_family="DejaVu Sans",
        font_size=8.2,
        rotate=False,
        bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.88},
    )

    model_date = str(metadata.get("trained_at", ""))[:10]
    data_start = metadata.get("earliest_date", "ไม่ระบุ")
    data_end = metadata.get("latest_date", "ไม่ระบุ")
    ax.set_title(
        "Correlation Network of Key Wildfire-Prediction Features",
        fontsize=23,
        fontweight="bold",
        color=INK,
        pad=34,
    )
    ax.text(
        0.5,
        1.015,
        f"Top 20 model features  ·  Daily observations: {data_start}–{data_end}  ·  Model: {model_date}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12.5,
        color="#555555",
    )

    legend_items = [
        Line2D([0], [0], color=POSITIVE, lw=4, label="Positive correlation"),
        Line2D([0], [0], color=NEGATIVE, lw=4, label="Negative correlation"),
        *[
            Patch(facecolor=colour, edgecolor=INK, label=title)
            for title, colour in GROUPS.values()
        ],
    ]
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.025),
        ncol=6,
        frameon=False,
        fontsize=10.5,
        handlelength=2.3,
        columnspacing=1.4,
    )
    ax.text(
        0.5,
        -0.105,
        f"Edges: |r| ≥ {args.edge_min:.2f}  ·  Labels: |r| ≥ {args.label_min:.2f}  ·  "
        "Edge width scales with |r|  ·  Model importance is shown inside each node",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        color="#4A4A4A",
    )
    ax.text(
        0.5,
        -0.142,
        "Exploratory analysis: correlation and feature importance do not establish causality.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#6A6A6A",
        style="italic",
    )

    ax.set_xlim(-1.64, 1.62)
    ax.set_ylim(-1.23, 1.02)
    ax.axis("off")
    fig.subplots_adjust(left=0.035, right=0.965, top=0.88, bottom=0.18)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        target = args.output.with_suffix(f".{suffix}")
        fig.savefig(target, dpi=300 if suffix == "png" else None,
                    facecolor=BACKGROUND, bbox_inches="tight")
        print(target)
    corr.to_csv(args.output.with_name(args.output.name + "_correlations.csv"))
    plt.close(fig)


if __name__ == "__main__":
    main()
