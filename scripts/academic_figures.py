#!/usr/bin/env python3
"""Generate publication-ready figures for the Thailand wildfire model.

All model-evaluation figures use the promoted model's untouched chronological
test split. Experimental ablation and rolling-origin figures are explicitly
labelled and kept separate from the frozen test evaluation.

Run from the repository root:
    .venv/bin/python scripts/academic_figures.py

Outputs:
    outputs/figures/academic/*.png, *.svg, *.pdf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logging.getLogger("fontTools").setLevel(logging.WARNING)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs/models/lgbm_fire_date_model.pkl"
FEATURES_PATH = ROOT / "outputs/features/full_features.parquet"
METADATA_PATH = ROOT / "outputs/metadata/dataset_info.json"
ROLLING_PATH = ROOT / "outputs/metadata/rolling_backtest.json"
ABLATION_PATH = ROOT / "outputs/metadata/ablation_weather_features.json"
OUTPUT_DIR = ROOT / "outputs/figures/academic"
CACHE_PATH = OUTPUT_DIR / "current_test_predictions.npz"
CACHE_META_PATH = OUTPUT_DIR / "current_test_predictions.json"

BLUE = "#24557A"
BLUE_LIGHT = "#91B6CF"
GOLD = "#D8A33E"
ORANGE = "#C5683A"
PURPLE = "#7665A6"
TEAL = "#3B8178"
INK = "#25282B"
MID = "#62676C"
GRID = "#DDE1E4"
PALE = "#F5F7F8"
RED = "#A84A4A"

FEATURE_LABELS = {
    "doy_sin": "Day of year (sin)",
    "doy_cos": "Day of year (cos)",
    "neighbor_frp_today": "Neighbour FRP today",
    "tree_cover_pct_2000": "Tree cover (2000)",
    "lat_grid": "Latitude",
    "week_sin": "Week of year (sin)",
    "tree_loss_pct_recent": "Recent tree-cover loss",
    "day_of_week": "Day of week",
    "week_cos": "Week of year (cos)",
    "lon_grid": "Longitude",
    "days_since_last_fire": "Days since last fire",
    "fire_count_same_month_hist": "Same-month fire history",
    "month_sin": "Month (sin)",
    "neighbor_fire_today": "Neighbour fires today",
    "days_since_last_fire_norm": "Normalised fire interval",
    "neighbor_frp_sum_14d": "Neighbour FRP (14 d)",
    "fire_sum_14d": "Cell fires (14 d)",
    "fire_sum_21d": "Cell fires (21 d)",
    "wide_neighbor_frp_sum_14d": "Wide-area FRP (14 d)",
    "fire_sum_30d": "Cell fires (30 d)",
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#AEB4B8",
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "xtick.color": MID,
        "ytick.color": MID,
        "text.color": INK,
        "axes.labelcolor": INK,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def style_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def figure_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.06, y=0.985, ha="left", fontsize=18,
                 fontweight="bold", color=INK)
    fig.text(0.06, 0.945, subtitle, ha="left", va="top", fontsize=10, color=MID)


def source_note(fig: plt.Figure, text: str) -> None:
    fig.text(0.06, 0.018, text, ha="left", va="bottom", fontsize=8.2, color=MID)


def save_figure(fig: plt.Figure, stem: str, output_dir: Path,
                formats: Iterable[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=300 if fmt == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def short_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def load_metadata_and_model() -> tuple[dict, object]:
    sys.path.insert(0, str(ROOT / "src"))
    import train  # noqa: PLC0415

    model, bundle_meta = train.load_model_bundle(str(MODEL_PATH))
    external = load_json(METADATA_PATH)
    metadata = bundle_meta or external
    if metadata.get("trained_at") != external.get("trained_at"):
        raise RuntimeError("Model bundle and dataset_info.json refer to different model versions")
    return metadata, model


def load_test_predictions(metadata: dict, model: object,
                          force: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict the current untouched test split, with provenance-aware caching."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    provenance = {
        "model_sha256_16": short_hash(MODEL_PATH),
        "feature_size": FEATURES_PATH.stat().st_size,
        "feature_mtime_ns": FEATURES_PATH.stat().st_mtime_ns,
        "trained_at": metadata["trained_at"],
        "test_range": metadata["evaluation_protocol"]["split_date_ranges"]["test"],
        "feature_count": len(metadata["features"]),
    }
    if not force and CACHE_PATH.exists() and CACHE_META_PATH.exists():
        cached_meta = load_json(CACHE_META_PATH)
        if cached_meta == provenance:
            cached = np.load(CACHE_PATH)
            return cached["y"], cached["p"], cached["dates"]

    features = list(metadata["features"])
    start, end = pd.to_datetime(provenance["test_range"])
    columns = features + ["date", "days_until_fire"]
    parquet = pq.ParquetFile(FEATURES_PATH)
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    dates: list[np.ndarray] = []
    for batch in parquet.iter_batches(batch_size=180_000, columns=columns):
        frame = batch.to_pandas()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.loc[frame["date"].between(start, end)]
        if frame.empty:
            continue
        X = frame[features]
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 2:
            proba = proba[:, 1]
        days = pd.to_numeric(frame["days_until_fire"], errors="coerce").to_numpy()
        label = ((days >= 1) & (days <= int(metadata.get("imminent_days", 3)))).astype(np.int8)
        ys.append(label)
        ps.append(proba.astype(np.float32))
        dates.append(frame["date"].to_numpy(dtype="datetime64[D]"))
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    date_values = np.concatenate(dates)
    expected = metadata["model"]["test_metrics"]["high_test_metrics"]["n"]
    if len(y) != expected:
        raise RuntimeError(f"Test prediction count {len(y):,} does not match metadata {expected:,}")
    np.savez_compressed(CACHE_PATH, y=y, p=p, dates=date_values)
    CACHE_META_PATH.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return y, p, date_values


def test_context(metadata: dict, y: np.ndarray) -> str:
    start, end = metadata["evaluation_protocol"]["split_date_ranges"]["test"]
    return (f"Promoted model recomputed on the current feature snapshot · test dates: {start}–{end} · "
            f"n = {len(y):,} cell-days · prevalence = {y.mean():.2%}")


def operating_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alert_rate": float(predicted.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def plot_discrimination(metadata: dict, y: np.ndarray, p: np.ndarray) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    prevalence = y.mean()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    figure_title(fig, "Model discrimination on the held-out test set", test_context(metadata, y))

    ax = axes[0]
    ax.plot(fpr, tpr, color=BLUE, linewidth=2.5, label=f"LightGBM (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=MID, linestyle="--", linewidth=1.2, label="Random classifier")
    ax.fill_between(fpr, tpr, fpr, color=BLUE_LIGHT, alpha=0.18)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1),
           title="A. Receiver operating characteristic")
    ax.legend(loc="lower right")
    style_axis(ax)

    ax = axes[1]
    ax.plot(recall, precision, color=ORANGE, linewidth=2.5, label=f"LightGBM (AP = {ap:.3f})")
    ax.axhline(prevalence, color=MID, linestyle="--", linewidth=1.2,
               label=f"No-skill baseline ({prevalence:.3f})")
    ax.fill_between(recall, precision, prevalence, where=precision >= prevalence,
                    color=GOLD, alpha=0.13)
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1),
           title="B. Precision–recall curve")
    ax.legend(loc="upper right")
    style_axis(ax)
    fig.subplots_adjust(top=0.82, bottom=0.15, left=0.08, right=0.97, wspace=0.24)
    source_note(fig, "Source: promoted model predictions; thresholds were selected on calibration data, never on this test set.")
    return fig


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    bin_id = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for idx in range(n_bins):
        mask = bin_id == idx
        if not mask.any():
            continue
        rows.append({
            "bin": idx,
            "lower": edges[idx],
            "upper": edges[idx + 1],
            "mean_predicted": float(p[mask].mean()),
            "observed_rate": float(y[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def plot_calibration(metadata: dict, y: np.ndarray, p: np.ndarray) -> plt.Figure:
    rel = reliability_table(y, p)
    ece = float(np.sum(rel["count"] * np.abs(rel["observed_rate"] - rel["mean_predicted"])) / len(y))
    brier = float(np.mean((p - y) ** 2))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    figure_title(fig, "Probability calibration and score distribution",
                 test_context(metadata, y) + f" · Brier = {brier:.3f} · ECE = {ece:.3f}")

    ax = axes[0]
    sizes = 35 + 220 * np.sqrt(rel["count"] / rel["count"].max())
    ax.plot([0, 0.65], [0, 0.65], color=MID, linestyle="--", linewidth=1.2, label="Perfect calibration")
    ax.plot(rel["mean_predicted"], rel["observed_rate"], color=BLUE, linewidth=2,
            marker="o", markersize=5, label="Observed")
    ax.scatter(rel["mean_predicted"], rel["observed_rate"], s=sizes,
               color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed event rate",
           xlim=(0, 0.65), ylim=(0, 0.65), title="A. Reliability diagram")
    ax.legend(loc="upper left")
    style_axis(ax)

    ax = axes[1]
    bins = np.linspace(0, max(0.62, float(p.max()) + 0.01), 35)
    ax.hist(p[y == 0], bins=bins, color=BLUE_LIGHT, alpha=0.72, label="No fire within 3 days")
    ax.hist(p[y == 1], bins=bins, color=ORANGE, alpha=0.72, label="Fire within 3 days")
    thresholds = metadata["probability_tier_thresholds"]
    ax.axvline(thresholds["WATCH"], color=PURPLE, linestyle="--", linewidth=1.6, label="WATCH threshold")
    ax.axvline(thresholds["HIGH"], color=RED, linestyle="-.", linewidth=1.6, label="HIGH threshold")
    ax.set_yscale("log")
    ax.set(xlabel="Predicted probability", ylabel="Cell-days (log scale)",
           title="B. Prediction-score distribution")
    ax.legend(loc="upper right", fontsize=8.5)
    style_axis(ax)
    fig.subplots_adjust(top=0.82, bottom=0.15, left=0.08, right=0.97, wspace=0.25)
    source_note(fig, "Bubble area reflects bin sample size. Equal-width probability bins; empty bins omitted.")
    return fig


def plot_confusion_matrices(metadata: dict, y: np.ndarray, p: np.ndarray) -> plt.Figure:
    thresholds = metadata["probability_tier_thresholds"]
    items = [(name, operating_metrics(y, p, thresholds[name])) for name in ("HIGH", "WATCH")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
    start, end = metadata["evaluation_protocol"]["split_date_ranges"]["test"]
    figure_title(fig, "Operational confusion matrices",
                 f"Held-out test set: {start}–{end} · rows normalized within each actual class")
    cmap = LinearSegmentedColormap.from_list("academic_blue", ["#F7FAFC", BLUE])
    for ax, (name, values) in zip(axes, items):
        cm_data = values["confusion_matrix"]
        matrix = np.array([[cm_data["tn"], cm_data["fp"]], [cm_data["fn"], cm_data["tp"]]], dtype=float)
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(normalized, cmap=cmap, vmin=0, vmax=1)
        for row in range(2):
            for col in range(2):
                color = "white" if normalized[row, col] > 0.55 else INK
                ax.text(col, row, f"{normalized[row, col]:.1%}\n({int(matrix[row, col]):,})",
                        ha="center", va="center", fontsize=12, fontweight="bold", color=color)
        ax.set_xticks([0, 1], ["Predicted negative", "Predicted positive"])
        ax.set_yticks([0, 1], ["Actual negative", "Actual positive"])
        ax.set_title(f"{name}: threshold = {values['threshold']:.4f}\n"
                     f"Precision {values['precision']:.3f} · Recall {values['recall']:.3f} · Alert rate {values['alert_rate']:.2%}")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.08, right=0.96, wspace=0.28)
    source_note(fig, "Counts in parentheses. Values are recomputed from the same current feature snapshot used for the ROC/PR curves.")
    return fig


def plot_feature_importance(metadata: dict) -> plt.Figure:
    values = pd.DataFrame(metadata["feature_importance_top"]).sort_values("importance")
    labels = [FEATURE_LABELS.get(name, name) for name in values["feature"]]
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    figure_title(fig, "Top model features", "Relative LightGBM split importance for the promoted five-model ensemble")
    colours = [GOLD if any(key in name for key in ("doy", "week", "month")) else
               PURPLE if "neighbor" in name else
               TEAL if name in {"lat_grid", "lon_grid", "tree_cover_pct_2000", "tree_loss_pct_recent"} else ORANGE
               for name in values["feature"]]
    bars = ax.barh(labels, values["importance"], color=colours, edgecolor="white", linewidth=0.7)
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values["importance"]],
                 padding=4, fontsize=8.5, color=INK)
    ax.set(xlabel="Relative importance (%)", xlim=(0, values["importance"].max() * 1.18))
    ax.set_title("Feature ranking", loc="left")
    style_axis(ax, "x")
    legend = [Patch(facecolor=GOLD, label="Seasonality"), Patch(facecolor=PURPLE, label="Spatial fire"),
              Patch(facecolor=ORANGE, label="Cell history"), Patch(facecolor=TEAL, label="Geography / vegetation")]
    ax.legend(handles=legend, loc="lower right", fontsize=8.5)
    fig.subplots_adjust(top=0.88, bottom=0.1, left=0.28, right=0.94)
    source_note(fig, "Importance measures model usage, not causality; correlated features may share or redistribute importance.")
    return fig


def plot_baseline_comparison(metadata: dict) -> plt.Figure:
    test = metadata["model"]["test_metrics"]
    baselines = test["baselines"]
    rows = [{"model": "LightGBM", "ROC-AUC": test["roc_auc"], "Average precision": test["average_precision"]}]
    names = {
        "persistence_fired_last_3d": "Persistence (3 d)",
        "persistence_fired_last_7d": "Persistence (7 d)",
        "cell_climatology_train_rate": "Cell climatology",
    }
    for key, values in baselines.items():
        rows.append({"model": names[key], "ROC-AUC": values["roc_auc"], "Average precision": values["average_precision"]})
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))
    figure_title(fig, "Model performance versus non-ML baselines",
                 "Training-time frozen-test metrics stored with the promoted model; higher is better")
    for ax, metric, colour, baseline in zip(axes, ["ROC-AUC", "Average precision"], [BLUE, ORANGE], [0.5, test["test_positive_rate"]]):
        ordered = frame.sort_values(metric)
        bars = ax.barh(ordered["model"], ordered[metric], color=[colour if x == "LightGBM" else "#BFC6CB" for x in ordered["model"]])
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in ordered[metric]], padding=4, fontsize=9)
        ax.axvline(baseline, color=MID, linestyle="--", linewidth=1.1,
                   label="Random / prevalence baseline")
        ax.set(xlabel=metric, xlim=(0, 1), title=metric)
        ax.legend(loc="lower right", fontsize=8)
        style_axis(ax, "x")
    fig.subplots_adjust(top=0.8, bottom=0.15, left=0.15, right=0.96, wspace=0.3)
    source_note(fig, "Archived training-time metrics. Persistence uses recent fire activity; climatology uses each cell's training-period fire rate.")
    return fig


def threshold_table(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    order = np.argsort(p)[::-1]
    sorted_p = p[order]
    sorted_y = y[order]
    cum_tp = np.cumsum(sorted_y)
    total_pos = sorted_y.sum()
    rows = []
    for threshold in thresholds:
        predicted = int(np.searchsorted(-sorted_p, -threshold, side="right"))
        tp = int(cum_tp[predicted - 1]) if predicted else 0
        fp = predicted - tp
        fn = int(total_pos - tp)
        precision = tp / predicted if predicted else 0.0
        recall = tp / total_pos if total_pos else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": threshold, "precision": precision, "recall": recall,
                     "f1": f1, "alert_rate": predicted / len(y), "fp": fp, "fn": fn})
    return pd.DataFrame(rows)


def plot_threshold_tradeoff(metadata: dict, y: np.ndarray, p: np.ndarray) -> plt.Figure:
    grid = np.linspace(0, max(0.62, float(p.max())), 180)
    table = threshold_table(y, p, grid)
    limits = metadata["probability_tier_thresholds"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.7))
    figure_title(fig, "Operational threshold trade-offs", test_context(metadata, y))
    ax = axes[0]
    for metric, colour, style in [("precision", BLUE, "-"), ("recall", ORANGE, "-"),
                                   ("f1", TEAL, "-"), ("alert_rate", MID, "--")]:
        ax.plot(table["threshold"], table[metric], color=colour, linestyle=style,
                linewidth=2, label=metric.replace("_", " ").title())
    for name, colour in [("WATCH", PURPLE), ("HIGH", RED)]:
        ax.axvline(limits[name], color=colour, linestyle=":" if name == "WATCH" else "-.",
                   linewidth=1.8, label=f"{name} = {limits[name]:.4f}")
    ax.set(xlabel="Probability threshold", ylabel="Metric value", ylim=(0, 1),
           title="A. Test performance across frozen thresholds")
    ax.legend(fontsize=8, ncol=2)
    style_axis(ax)

    ax = axes[1]
    names = ["WATCH", "HIGH"]
    op = {name: operating_metrics(y, p, limits[name]) for name in names}
    precision = [op[name]["precision"] for name in names]
    recall = [op[name]["recall"] for name in names]
    alert = [op[name]["alert_rate"] for name in names]
    x = np.arange(2)
    width = 0.23
    for offset, values, label, colour in [(-width, precision, "Precision", BLUE),
                                           (0, recall, "Recall", ORANGE),
                                           (width, alert, "Alert rate", MID)]:
        bars = ax.bar(x + offset, values, width, label=label, color=colour)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=3, fontsize=8)
    ax.set_xticks(x, names)
    ax.set(ylabel="Proportion", ylim=(0, max(0.45, max(precision + recall + alert) * 1.25)),
           title="B. Deployed operating points on test data")
    ax.legend(loc="upper right", fontsize=8)
    style_axis(ax, "y")
    fig.subplots_adjust(top=0.81, bottom=0.15, left=0.08, right=0.97, wspace=0.25)
    source_note(fig, "WATCH and HIGH were selected on calibration data. Curves shown here are descriptive test evaluations only.")
    return fig


def topk_table(y: np.ndarray, p: np.ndarray, percentages: Iterable[int]) -> pd.DataFrame:
    order = np.argsort(p)[::-1]
    ranked = y[order]
    total = ranked.sum()
    prevalence = y.mean()
    rows = []
    for pct in percentages:
        count = max(1, int(np.ceil(len(y) * pct / 100)))
        positives = int(ranked[:count].sum())
        precision = positives / count
        rows.append({"top_pct": pct, "precision": precision,
                     "recall_captured": positives / total, "lift": precision / prevalence})
    return pd.DataFrame(rows)


def plot_topk(metadata: dict, y: np.ndarray, p: np.ndarray) -> plt.Figure:
    table = topk_table(y, p, range(1, 21))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    figure_title(fig, "Operational prioritization by ranked risk", test_context(metadata, y))
    ax = axes[0]
    ax.plot(table["top_pct"], table["precision"], color=BLUE, linewidth=2.4, marker="o", markevery=[4, 9, 19], label="Precision at top K%")
    ax.axhline(y.mean(), color=MID, linestyle="--", label=f"Test prevalence ({y.mean():.2%})")
    ax.set(xlabel="Highest-risk cell-days reviewed (%)", ylabel="Precision", xlim=(1, 20),
           ylim=(0, max(table["precision"]) * 1.1), title="A. Event yield in the priority list")
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.legend(fontsize=8.5)
    style_axis(ax)

    ax = axes[1]
    ax.plot(table["top_pct"], table["recall_captured"], color=ORANGE, linewidth=2.4,
            marker="o", markevery=[4, 9, 19], label="Recall captured")
    for pct in (5, 10, 20):
        row = table.loc[table["top_pct"] == pct].iloc[0]
        ax.annotate(f"{row['recall_captured']:.1%}", (pct, row["recall_captured"]),
                    xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5)
    ax.set(xlabel="Highest-risk cell-days reviewed (%)", ylabel="Fraction of all events captured",
           xlim=(1, 20), ylim=(0, 1), title="B. Event coverage")
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    style_axis(ax)
    fig.subplots_adjust(top=0.8, bottom=0.15, left=0.08, right=0.97, wspace=0.25)
    source_note(fig, "Top-K ranking is a relative prioritization view and is separate from absolute WATCH/HIGH alert thresholds.")
    return fig


def plot_split_protocol(metadata: dict) -> plt.Figure:
    protocol = metadata["evaluation_protocol"]
    ranges = protocol["split_date_ranges"]
    prevalence = protocol["split_prevalence"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), gridspec_kw={"width_ratios": [0.8, 1.45]})
    figure_title(fig, "Chronological evaluation protocol and prevalence drift",
                 f"Whole-date splits · {protocol['purge_days']}-day purge before each boundary · test evaluated once")
    order = ["train", "val", "calib", "test"]
    colours = [BLUE, GOLD, PURPLE, ORANGE]
    ax = axes[0]
    bars = ax.bar([name.upper() for name in order], [prevalence[name] for name in order], color=colours)
    ax.bar_label(bars, labels=[f"{prevalence[name]:.1%}" for name in order], padding=3, fontsize=9)
    ax.set(ylabel="Positive prevalence", ylim=(0, max(prevalence.values()) * 1.2),
           title="A. Natural class prevalence by split")
    style_axis(ax, "y")

    ax = axes[1]
    base = pd.Timestamp(ranges["train"][0])
    for idx, (name, colour) in enumerate(zip(order, colours)):
        start, end = map(pd.Timestamp, ranges[name])
        left = (start - base).days
        width = (end - start).days + 1
        ax.barh(idx, width, left=left, color=colour, height=0.55, edgecolor="white")
        ax.text(left + width / 2, idx, f"{start:%d %b %Y}\n{end:%d %b %Y}",
                ha="center", va="center", fontsize=7.5, color="white" if name in {"train", "calib"} else INK,
                fontweight="bold")
    ax.set_yticks(range(4), [name.upper() for name in order])
    ticks = pd.date_range(base, pd.Timestamp(ranges["test"][1]), freq="2MS")
    ax.set_xticks([(date - base).days for date in ticks], [date.strftime("%b\n%Y") for date in ticks])
    ax.set_title("B. Date-grouped split timeline")
    ax.invert_yaxis()
    style_axis(ax, "x")
    fig.subplots_adjust(top=0.79, bottom=0.17, left=0.08, right=0.97, wspace=0.3)
    source_note(fig, "Training was undersampled only after splitting; validation, calibration, and test retain natural prevalence.")
    return fig


def plot_rolling_backtest(rolling: dict) -> plt.Figure:
    frame = pd.DataFrame(rolling["folds"]).sort_values("year")
    deployment = pd.json_normalize(frame["deployment"])
    watch = pd.json_normalize(frame["watch_list"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    aggregate = rolling["aggregate"]
    figure_title(fig, "Rolling-origin seasonal backtest",
                 f"{aggregate['n_folds']} annual folds · January–April test windows · mean AUC = {aggregate['auc_mean']:.3f} ± {aggregate['auc_std']:.3f}")
    ax = axes[0]
    ax.plot(frame["year"], frame["roc_auc"], color=BLUE, marker="o", linewidth=2.2, label="LightGBM ROC-AUC")
    ax.plot(frame["year"], frame["persistence_auc"], color=MID, marker="s", linestyle="--", linewidth=1.7, label="Persistence ROC-AUC")
    ax.fill_between(frame["year"], frame["roc_auc"], frame["persistence_auc"], color=BLUE_LIGHT, alpha=0.2)
    ax.set(xlabel="Test year", ylabel="ROC-AUC", ylim=(0.5, 0.9), title="A. Ranking performance across years")
    ax.legend(fontsize=8.5)
    style_axis(ax)

    ax = axes[1]
    ax.plot(frame["year"], deployment["recall"], color=ORANGE, marker="o", linewidth=2.2, label="HIGH recall")
    ax.plot(frame["year"], watch["recall"], color=PURPLE, marker="s", linewidth=2.2, label="WATCH recall")
    ax.plot(frame["year"], watch["precision"], color=TEAL, marker="^", linestyle="--", linewidth=1.7, label="WATCH precision")
    ax.set(xlabel="Test year", ylabel="Metric value", ylim=(0, 1.05), title="B. Operational performance across years")
    ax.legend(fontsize=8.5)
    style_axis(ax)
    fig.subplots_adjust(top=0.8, bottom=0.15, left=0.08, right=0.97, wspace=0.25)
    source_note(fig, "Each fold fits on prior years, calibrates on the preceding 30 days, then tests the next fire season.")
    return fig


def load_daily_activity() -> pd.DataFrame:
    frame = pd.read_parquet(FEATURES_PATH, columns=["date", "fire_count_today"])
    frame["date"] = pd.to_datetime(frame["date"])
    daily = frame.groupby("date", as_index=False).agg(
        hotspot_count=("fire_count_today", "sum"),
        active_cells=("fire_count_today", lambda x: int((x > 0).sum())),
    )
    daily["hotspot_ma7"] = daily["hotspot_count"].rolling(7, min_periods=1).mean()
    daily["cells_ma7"] = daily["active_cells"].rolling(7, min_periods=1).mean()
    return daily


def plot_daily_activity(metadata: dict, daily: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.0), sharex=True)
    figure_title(fig, "Observed daily fire activity in the modelling dataset",
                 f"NASA FIRMS VIIRS · {daily['date'].min():%Y-%m-%d}–{daily['date'].max():%Y-%m-%d} · raw daily counts and 7-day moving averages")
    ax = axes[0]
    ax.bar(daily["date"], daily["hotspot_count"], width=1, color=BLUE_LIGHT, alpha=0.55, label="Daily hotspots")
    ax.plot(daily["date"], daily["hotspot_ma7"], color=BLUE, linewidth=2, label="7-day mean")
    ax.set(ylabel="Hotspot detections", title="A. Hotspots per day")
    ax.legend(loc="upper right")
    style_axis(ax)
    ax = axes[1]
    ax.bar(daily["date"], daily["active_cells"], width=1, color="#E7BE8E", alpha=0.6, label="Daily active cells")
    ax.plot(daily["date"], daily["cells_ma7"], color=ORANGE, linewidth=2, label="7-day mean")
    ax.set(xlabel="Date", ylabel="Active 0.1° cells", title="B. Cells with at least one hotspot")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.legend(loc="upper right")
    style_axis(ax)
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.08, right=0.97, hspace=0.25)
    source_note(fig, "Counts are observations, not model predictions. One cell represents a 0.1° grid location.")
    return fig


def plot_correlation_heatmap(metadata: dict) -> plt.Figure:
    features = [row["feature"] for row in metadata["feature_importance_top"][:15]]
    frame = pd.read_parquet(FEATURES_PATH, columns=features)
    corr = frame.corr().to_numpy()
    labels = [FEATURE_LABELS.get(name, name) for name in features]
    cmap = LinearSegmentedColormap.from_list("correlation", ["#37699A", "#F7F7F4", "#A64747"])
    fig, ax = plt.subplots(figsize=(10.5, 9.2))
    figure_title(fig, "Correlation among the 15 highest-ranked model features",
                 "Pearson correlation across all available cell-days; annotations shown when |r| ≥ 0.50")
    image = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    for row in range(len(features)):
        for col in range(len(features)):
            if row == col:
                continue
            value = corr[row, col]
            if abs(value) >= 0.5:
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(value) > 0.7 else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=48, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colourbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.03)
    colourbar.set_label("Pearson r")
    fig.subplots_adjust(top=0.85, bottom=0.25, left=0.26, right=0.93)
    source_note(fig, "Correlation indicates association, not causal influence or model importance.")
    return fig


def plot_weather_ablation(ablation: dict) -> plt.Figure:
    results = ablation["results"]
    names = ["Baseline", "Missingness flags", "Drought features", "Both additions"]
    keys = ["baseline", "missingness", "drought", "missingness_plus_drought"]
    auc = [results[key]["raw_ranking"]["roc_auc"] for key in keys]
    ap = [results[key]["raw_ranking"]["average_precision"] for key in keys]
    brier = [results[key]["isotonic_calibrated"]["brier"] for key in keys]
    ece = [results[key]["isotonic_calibrated"]["ece"] for key in keys]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    date_range = ablation["split"]["date_ranges"]
    figure_title(fig, "Weather-feature ablation study",
                 f"Separate pre-test experiment · selection metric: {ablation['selection_metric']} · frozen test set was not used")
    x = np.arange(len(names))
    width = 0.35
    ax = axes[0]
    bars1 = ax.bar(x - width / 2, auc, width, color=BLUE, label="ROC-AUC")
    bars2 = ax.bar(x + width / 2, ap, width, color=ORANGE, label="Average precision")
    ax.bar_label(bars1, labels=[f"{v:.3f}" for v in auc], padding=3, fontsize=7.5)
    ax.bar_label(bars2, labels=[f"{v:.3f}" for v in ap], padding=3, fontsize=7.5)
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set(ylabel="Ranking metric", ylim=(0, 1), title="A. Discrimination")
    ax.legend(fontsize=8.5)
    style_axis(ax, "y")
    ax = axes[1]
    bars1 = ax.bar(x - width / 2, brier, width, color=PURPLE, label="Brier score")
    bars2 = ax.bar(x + width / 2, ece, width, color=TEAL, label="ECE")
    ax.bar_label(bars1, labels=[f"{v:.3f}" for v in brier], padding=3, fontsize=7.5)
    ax.bar_label(bars2, labels=[f"{v:.3f}" for v in ece], padding=3, fontsize=7.5)
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set(ylabel="Error (lower is better)", ylim=(0, max(brier) * 1.25), title="B. Calibration error")
    ax.legend(fontsize=8.5)
    style_axis(ax, "y")
    fig.subplots_adjust(top=0.8, bottom=0.22, left=0.08, right=0.97, wspace=0.25)
    source_note(
        fig,
        f"Experimental evaluation: {date_range['test'][0]}–{date_range['test'][1]}; "
        "the official frozen test beginning 2026-05-15 was not used. These are not promoted-model test metrics.",
    )
    return fig


def plot_summary(metadata: dict, y: np.ndarray, p: np.ndarray) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y, p)
    precision_curve, recall_curve, _ = precision_recall_curve(y, p)
    rel = reliability_table(y, p)
    importance = pd.DataFrame(metadata["feature_importance_top"][:10]).sort_values("importance")
    protocol = metadata["evaluation_protocol"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    figure_title(fig, "Academic evaluation summary — wildfire risk within 3 days", test_context(metadata, y))

    ax = axes[0, 0]
    ax.plot(fpr, tpr, color=BLUE, linewidth=2.2)
    ax.plot([0, 1], [0, 1], color=MID, linestyle="--")
    ax.text(0.96, 0.08, f"AUC = {roc_auc_score(y, p):.3f}", transform=ax.transAxes,
            ha="right", fontweight="bold", color=BLUE)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="A. ROC curve", xlim=(0, 1), ylim=(0, 1))
    style_axis(ax)

    ax = axes[0, 1]
    ax.plot(recall_curve, precision_curve, color=ORANGE, linewidth=2.2)
    ax.axhline(y.mean(), color=MID, linestyle="--")
    ax.text(0.96, 0.9, f"AP = {average_precision_score(y, p):.3f}", transform=ax.transAxes,
            ha="right", fontweight="bold", color=ORANGE)
    ax.set(xlabel="Recall", ylabel="Precision", title="B. Precision–recall", xlim=(0, 1), ylim=(0, 1))
    style_axis(ax)

    ax = axes[0, 2]
    ax.plot([0, 0.65], [0, 0.65], color=MID, linestyle="--")
    ax.plot(rel["mean_predicted"], rel["observed_rate"], color=PURPLE, marker="o", linewidth=2)
    ax.set(xlabel="Predicted probability", ylabel="Observed rate", title="C. Calibration", xlim=(0, 0.65), ylim=(0, 0.65))
    style_axis(ax)

    ax = axes[1, 0]
    labels = [FEATURE_LABELS.get(name, name) for name in importance["feature"]]
    ax.barh(labels, importance["importance"], color=BLUE_LIGHT)
    ax.set(xlabel="Importance (%)", title="D. Top 10 features")
    ax.tick_params(axis="y", labelsize=7.5)
    style_axis(ax, "x")

    ax = axes[1, 1]
    order = ["train", "val", "calib", "test"]
    values = [protocol["split_prevalence"][name] for name in order]
    bars = ax.bar([name.upper() for name in order], values, color=[BLUE, GOLD, PURPLE, ORANGE])
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=2, fontsize=8)
    ax.set(ylabel="Positive prevalence", title="E. Prevalence by split", ylim=(0, max(values) * 1.2))
    style_axis(ax, "y")

    ax = axes[1, 2]
    names = ["WATCH", "HIGH"]
    thresholds = metadata["probability_tier_thresholds"]
    op = {name: operating_metrics(y, p, thresholds[name]) for name in names}
    precision = [op[name]["precision"] for name in names]
    recall = [op[name]["recall"] for name in names]
    x = np.arange(2)
    width = 0.34
    ax.bar(x - width / 2, precision, width, color=BLUE, label="Precision")
    ax.bar(x + width / 2, recall, width, color=ORANGE, label="Recall")
    ax.set_xticks(x, names)
    ax.set(ylabel="Metric value", title="F. Operating points", ylim=(0, 0.5))
    ax.legend(fontsize=8)
    style_axis(ax, "y")

    fig.subplots_adjust(top=0.86, bottom=0.09, left=0.07, right=0.98, wspace=0.34, hspace=0.38)
    source_note(fig, "Source: NASA FIRMS VIIRS, ERA5/Open-Meteo, Hansen GFC. Evaluation uses whole-date splits with a 7-day purge.")
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--formats", default="png,svg,pdf",
                        help="Comma-separated output formats")
    parser.add_argument("--force-predictions", action="store_true",
                        help="Ignore the provenance-aware prediction cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = tuple(item.strip().lower() for item in args.formats.split(",") if item.strip())
    configure_style()
    metadata, model = load_metadata_and_model()
    y, p, _ = load_test_predictions(metadata, model, force=args.force_predictions)
    daily = load_daily_activity()
    rolling = load_json(ROLLING_PATH)
    ablation = load_json(ABLATION_PATH)

    figures = [
        ("00_academic_evaluation_summary", plot_summary(metadata, y, p)),
        ("01_roc_precision_recall", plot_discrimination(metadata, y, p)),
        ("02_probability_calibration", plot_calibration(metadata, y, p)),
        ("03_operational_confusion_matrices", plot_confusion_matrices(metadata, y, p)),
        ("04_feature_importance", plot_feature_importance(metadata)),
        ("05_baseline_comparison", plot_baseline_comparison(metadata)),
        ("06_threshold_tradeoff", plot_threshold_tradeoff(metadata, y, p)),
        ("07_topk_prioritization", plot_topk(metadata, y, p)),
        ("08_split_protocol_prevalence", plot_split_protocol(metadata)),
        ("09_rolling_origin_backtest", plot_rolling_backtest(rolling)),
        ("10_daily_fire_activity", plot_daily_activity(metadata, daily)),
        ("11_top_feature_correlation", plot_correlation_heatmap(metadata)),
        ("12_weather_feature_ablation", plot_weather_ablation(ablation)),
    ]
    written = []
    for stem, figure in figures:
        written.extend(save_figure(figure, stem, args.output_dir, formats))
    print(f"Wrote {len(written)} figure files to {args.output_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
