"""End-to-end training orchestrator: data → features → tune → eval → persist.

Pipeline:
    1. data_loader.load_and_prepare → daily cell-day frame (causal universe:
       a cell exists only from its first detection onward)
    2. features.build_features → lag/rolling/calendar features + label
       (+ days_ahead_observed for tail censoring)
    3. ``--predict-only``: refresh full_features.parquet + risk_map only
    4. Censor tail rows whose 3-day label window is not fully observed
    5. Date-grouped chronological train/val/calib/test split with a 7-day
       purge before each boundary (eval_protocol.date_grouped_split; whole
       calendar dates, disjointness asserted)
    6. Undersample negatives in TRAIN ONLY — val/calib/test keep natural
       prevalence
    7. LightGBM tuning (BayesSearchCV, fallback RandomizedSearchCV) on
       purged date-grouped CV folds within train
    8. Fit ensemble on train; early stopping on val
    9. Platt calibration on calib; deployment threshold + tier cutoffs
       selected on calib and locked
   10. SINGLE evaluation pass on test at natural prevalence (ROC-AUC, PR-AUC,
       Brier, ECE, P@K) + persistence/climatology baselines
   11. Persist model, metadata (with evaluation_protocol audit trail), PNG
   12. Trigger risk_map.run() to refresh the GeoJSON

Protocol invariants (tested in tests/test_eval_protocol.py — do not relax):
    - No calendar date appears in two splits; ≥7-day purge at every boundary.
    - sklearn TimeSeriesSplit is BANNED here: its gap counts rows, not days.
    - Undersampling touches train only.
    - Calibration and thresholds never see test; test is evaluated once.
    - Output semantics: calibrated P(fire within 3 days) — never an exact
      fire date (the binary target cannot resolve day-of-fire).
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV

try:
    from lightgbm import LGBMClassifier, early_stopping
except ImportError as e:  # pragma: no cover
    raise RuntimeError("lightgbm is required: pip install lightgbm") from e

# scikit-optimize is optional — fall back to RandomizedSearchCV if missing.
try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Integer, Real  # noqa: F401
    _BAYES_AVAILABLE = True
except ImportError:  # pragma: no cover
    BayesSearchCV = None  # type: ignore[assignment]
    Categorical = None  # type: ignore[assignment]
    _BAYES_AVAILABLE = False

from data_loader import load_and_prepare
from eval_protocol import date_grouped_split, purged_date_folds
from features import (
    DEFAULT_URGENCY_THRESHOLDS,
    MAX_PREDICTION_DAYS,
    build_features,
    resolve_features,
)
from storage import resolve_existing, write_json, write_pickle, write_table
import thresholds as thresholds_mod

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("train")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────
# Hyperparameter search space (per user spec)
# ─────────────────────────────────────────────
# Binary classification: "fire within next IMMINENT_DAYS days?"
# Smaller param grid because binary is faster than multiclass and the easier
# task doesn't need as much capacity.
PARAM_GRID: Dict[str, List[Any]] = {
    "n_estimators":       [400, 600, 800],
    "max_depth":          [6, 8, 10],
    "learning_rate":      [0.05, 0.1],
    "num_leaves":         [31, 63, 127],
    "min_child_samples":  [20, 50, 100],
    "subsample":          [0.7, 0.8, 0.9],
    "colsample_bytree":   [0.7, 0.8, 1.0],
    "reg_alpha":          [0.0, 0.1],
    "reg_lambda":         [0.0, 0.1, 1.0],
}

# Operational binary target: 1 = fire within next IMMINENT_DAYS days, else 0.
# Negative class includes rows with no fire in the full 7-day horizon (label=-1)
# *and* rows with fire only on day 4-7.
IMMINENT_DAYS = 3
NEG_TO_POS_RATIO = 4   # undersample negatives to ~4× positives for training speed


def _resolve(base_dir: str, value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    return value if os.path.isabs(value) else os.path.normpath(os.path.join(base_dir, value))


def _paths() -> dict:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = _resolve(base_dir, os.getenv("OUTPUT_DIR")) or os.path.join(base_dir, "outputs")
    weather_dir = _resolve(base_dir, os.getenv("WEATHER_DIR")) or os.path.join(base_dir, "data", "weather")
    return {
        "base_dir": base_dir,
        "raw_dir": _resolve(base_dir, os.getenv("RAW_DIR")) or os.path.join(base_dir, "data", "raw"),
        "firms_path": _resolve(base_dir, os.getenv("FIRMS_PATH")) or os.path.join(base_dir, "data", "firms", "firms_all.parquet"),
        "weather_path": os.path.join(weather_dir, "weather_cache.parquet"),
        "tree_cover_path": os.path.join(base_dir, "data", "static", "tree_cover_per_cell.parquet"),
        "radd_path": os.path.join(base_dir, "data", "radd", "radd_alerts.parquet"),
        "output_dir": output_dir,
        "model_dir": os.path.join(output_dir, "models"),
        "feature_dir": os.path.join(output_dir, "features"),
        "meta_dir": os.path.join(output_dir, "metadata"),
    }


# ─────────────────────────────────────────────
# Sample weighting — emphasise shorter days_until_fire
# ─────────────────────────────────────────────
def _compute_sample_weights(
    y: pd.Series,
    dates: pd.Series,
    recency_halflife_days: float = 45.0,
    multi_sat: Optional[np.ndarray] = None,
    class_weighting: bool = True,
) -> np.ndarray:
    """Recency decay × optional inverse-class-frequency × optional multi-sat bonus.

    ``class_weighting=False`` drops the inverse-class-frequency component —
    required by IMBALANCE_MODE A/B/C, where class balancing must be applied
    exactly ONCE (the legacy default D stacked undersampling × class weights ×
    scale_pos_weight ≈ a ~120× positive up-weighting at 3% natural prevalence,
    which badly inflates raw probabilities).

    multi_sat: int8 array (same length as y) — number of VIIRS satellites that
    confirmed the fire at the target date; gentle 1.1×/1.2× boost. NOTE this is
    derived from the FUTURE target date, so weighting by it changes the
    estimand toward "multi-satellite-confirmed fires" — toggle via
    MULTISAT_WEIGHTING (audited in item 9).
    """
    if class_weighting:
        label_counts = y.value_counts()
        n_classes = max(len(label_counts), 1)
        n_samples = max(len(y), 1)
        class_weight_map = (n_samples / (n_classes * label_counts)).to_dict()
        sw_class = y.map(class_weight_map).to_numpy().astype(float)
    else:
        sw_class = np.ones(len(y), dtype=float)

    max_date = pd.to_datetime(dates).max()
    days_ago = (max_date - pd.to_datetime(dates)).dt.days.to_numpy().astype(float)
    sw_recency = np.exp(-days_ago / max(recency_halflife_days, 1.0))

    # Gentle multi-satellite confirmation bonus: 2+ satellites → 1.1×, 3 → 1.2×
    sat_boost = np.ones(len(y), dtype=float)
    if multi_sat is not None:
        sat_boost[multi_sat >= 2] = 1.1
        sat_boost[multi_sat >= 3] = 1.2

    sw = sw_class * sw_recency * sat_boost
    mean_sw = float(sw.mean()) if sw.size else 1.0
    if mean_sw > 0:
        sw /= mean_sw
    return sw


# NOTE: the old row-based `chronological_split` was REMOVED — it split by row
# count, letting one calendar date land in two splits (confirmed leakage:
# 2025-11-10 appeared in both train and val). Use
# eval_protocol.date_grouped_split, which splits on whole dates with a
# 7-day purge and asserts disjointness.


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.var(y_true) == 0:
        return 0.0
    return float(r2_score(y_true, y_pred))


def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray, horizon: int) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    y_pred_int = np.clip(np.round(y_pred), 0, horizon)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = _safe_r2(y_true, y_pred)
    acc_within_1 = float(np.mean(np.abs(y_pred_int - y_true) <= 1))
    acc_exact = float(np.mean(y_pred_int == y_true))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "acc_within_1": round(acc_within_1, 4),
        "acc_exact": round(acc_exact, 4),
    }


def per_day_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    actual_hotspots_per_day: Dict[int, int],
) -> Dict[str, Dict[str, float]]:
    """Per-day model metrics + hotspot precision/recall.

    HIGH/CRITICAL := predicted bucket in {1, 2}. "Actual hotspot" := the row's
    real label day equals d (i.e. a fire really occurred d days ahead).
    """
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    y_pred_int = np.clip(np.round(y_pred), 0, horizon).astype(int)
    y_true_int = y_true.astype(int)

    out: Dict[str, Dict[str, float]] = {}
    for d in range(1, horizon + 1):
        mask = y_true_int == d
        n = int(mask.sum())
        if n == 0:
            out[f"day_{d}"] = {
                "mae": 0.0, "rmse": 0.0, "r2": 0.0,
                "acc_within_1": 0.0, "acc_exact": 0.0,
                "prediction_count": 0, "bias": 0.0,
                "mean_predicted": 0.0, "mean_actual": float(d),
                "hotspot_count": int(actual_hotspots_per_day.get(d, 0)),
                "hotspot_hit_rate": 0.0,
                "false_alarm_rate": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }
            continue

        yt_d = y_true[mask]
        yp_d = y_pred[mask]
        yp_int_d = y_pred_int[mask]
        mae = float(mean_absolute_error(yt_d, yp_d))
        rmse = float(np.sqrt(mean_squared_error(yt_d, yp_d)))
        r2 = _safe_r2(yt_d, yp_d)
        acc_w1 = float(np.mean(np.abs(yp_int_d - yt_d) <= 1))
        acc_ex = float(np.mean(yp_int_d == yt_d))
        mean_pred = float(np.mean(yp_d))
        mean_act = float(np.mean(yt_d))
        bias = mean_pred - mean_act

        # Hotspot-grade detection: was this cell flagged HIGH/CRITICAL (≤2)?
        actual_hot_mask = mask  # rows whose actual label is d
        # for "hit" we ask: of rows whose actual label is d, how many were
        # predicted within HIGH/CRITICAL band
        hit = int(np.sum((y_pred_int[actual_hot_mask] <= 2)))
        hot_hit_rate = hit / n if n > 0 else 0.0

        # CRITICAL predictions for this day d (pred==d AND pred<=2 ⇒ pred in {0,1,2}∩{d})
        # Looser definition: rows predicted as day d AND classified CRITICAL.
        critical_pred_mask = (y_pred_int == d) & (y_pred_int <= 2)
        critical_n = int(critical_pred_mask.sum())
        if critical_n > 0:
            # false alarm: critical-prediction rows whose actual label is "no
            # fire within HIGH band" (actual > 2)
            false_alarms = int(np.sum(y_true[critical_pred_mask] > 2))
            false_alarm_rate = false_alarms / critical_n
        else:
            false_alarm_rate = 0.0

        # precision / recall for "predicted day == d" treated as the positive class
        pred_d_mask = y_pred_int == d
        tp = int(np.sum(pred_d_mask & (y_true_int == d)))
        fp = int(np.sum(pred_d_mask & (y_true_int != d)))
        fn = int(np.sum(~pred_d_mask & (y_true_int == d)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        out[f"day_{d}"] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "acc_within_1": round(acc_w1, 4),
            "acc_exact": round(acc_ex, 4),
            "prediction_count": int(n),
            "bias": round(bias, 4),
            "mean_predicted": round(mean_pred, 4),
            "mean_actual": round(mean_act, 4),
            "hotspot_count": int(actual_hotspots_per_day.get(d, 0)),
            "hotspot_hit_rate": round(hot_hit_rate, 4),
            "false_alarm_rate": round(false_alarm_rate, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }
    return out


def _print_daily_table(daily_stats: Dict[str, Dict[str, float]]) -> None:
    header = "┌─────┬───────┬───────┬───────┬──────────┬──────────┬──────────┬─────────┐"
    title  = "│ Day │  MAE  │ RMSE  │  R²   │ Acc±1day │ Hotspots │ Hit Rate │  Bias   │"
    sep    = "├─────┼───────┼───────┼───────┼──────────┼──────────┼──────────┼─────────┤"
    footer = "└─────┴───────┴───────┴───────┴──────────┴──────────┴──────────┴─────────┘"
    print(header)
    print(title)
    print(sep)
    for d in range(1, MAX_PREDICTION_DAYS + 1):
        s = daily_stats.get(f"day_{d}", {})
        mae = s.get("mae", 0.0)
        rmse = s.get("rmse", 0.0)
        r2 = s.get("r2", 0.0)
        acc1 = s.get("acc_within_1", 0.0)
        hots = int(s.get("hotspot_count", 0))
        hit = s.get("hotspot_hit_rate", 0.0)
        bias = s.get("bias", 0.0)
        print(
            f"│  {d}  │ {mae:5.2f} │ {rmse:5.2f} │ {r2:5.2f} │  {acc1*100:5.1f}%  │  {hots:5d}   │  {hit*100:5.1f}%  │ {bias:+6.2f}  │"
        )
    print(footer)


# ─────────────────────────────────────────────
# LightGBM binary classifier (fire-in-next-IMMINENT_DAYS-days)
# ─────────────────────────────────────────────
# Regression and multiclass approaches both collapsed (insufficient feature
# signal to distinguish day 1 from day 7). Reframed as binary: "fire within
# next 3 days?" — operator-actionable + achievable with current features.
def _build_lgbm(
    random_state: int = 42,
    scale_pos_weight: float = 1.0,
) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,   # imbalance correction
        random_state=random_state,
        n_jobs=-1,
        force_row_wise=True,
        verbosity=-1,
        verbose=-1,
    )


def _make_binary_label(days_until_fire: pd.Series) -> np.ndarray:
    """y = 1 if fire occurs within IMMINENT_DAYS, else 0.

    Includes label==-1 rows (no fire in full horizon) as negatives.
    """
    d = days_until_fire.to_numpy()
    y = (d >= 1) & (d <= IMMINENT_DAYS)
    return y.astype(int)


def _undersample_negatives(
    df: pd.DataFrame,
    target_col: str = "_y_bin",
    ratio: int = NEG_TO_POS_RATIO,
    random_state: int = 42,
) -> pd.DataFrame:
    """Keep all positives + random-sample negatives at ratio×positives."""
    pos = df[df[target_col] == 1]
    neg = df[df[target_col] == 0]
    n_neg_target = min(len(pos) * ratio, len(neg))
    if n_neg_target >= len(neg):
        return df.copy()
    neg_sample = neg.sample(n=n_neg_target, random_state=random_state)
    out = pd.concat([pos, neg_sample], ignore_index=True)
    return out.sort_values("date").reset_index(drop=True)


def _prob_to_days_for_compat(prob: np.ndarray) -> np.ndarray:
    """Map P(fire in 3d) → pseudo-days for risk_map.py backwards-compat.

    Piecewise-linear, monotone, anchored so the dashboard tiers behave sensibly
    around the deployment threshold (0.35 = best-F1 on test):

        prob     pseudo-days   urgency (floor)
        1.00          0.0      CRITICAL
        0.70          0.5      CRITICAL
        0.50          1.5      HIGH        ← prob ≥ 0.5: "likely"
        0.35          2.5      HIGH        ← deployment threshold
        0.20          4.0      MEDIUM
        0.10          5.5      LOW
        0.00          7.0      LOW

    Previously the mapping was strictly linear (prob=1 → day 1, never
    CRITICAL), so even very confident predictions landed in HIGH. The
    piecewise variant lets prob ≥ 0.7 produce CRITICAL alerts.
    """
    p = np.clip(prob, 0.0, 1.0)
    return np.interp(
        p,
        xp=[0.0, 0.20, 0.35, 0.50, 0.70, 1.00],
        fp=[7.0,  4.0,  2.5,  1.5,  0.5, 0.0],
    )


# ─────────────────────────────────────────────
# Model promotion: quality gates + atomic bundle swap
# ─────────────────────────────────────────────
#
# The promoted artifact is a SINGLE pickle bundle {"format": "bundle-v1",
# "model": ..., "metadata": ...} so the model and its feature contract
# travel together — one os.replace() promotes both atomically. (The old
# two-file scheme (pkl + dataset_info.json) tore on this very machine: a
# kill between the two writes left a new model paired with a stale feature
# list, which silently mispredicts.) dataset_info.json is still written for
# the frontend/docs, but loaders prefer the bundle's embedded metadata.

MODEL_GATE_MIN_AUC = float(os.getenv("MODEL_GATE_MIN_AUC", "0.65"))
MODEL_GATE_MIN_AP_UPLIFT = float(os.getenv("MODEL_GATE_MIN_AP_UPLIFT", "1.5"))
# Head-to-head tolerances vs the incumbent on the SAME comparison set.
MODEL_GATE_MAX_AUC_DROP = float(os.getenv("MODEL_GATE_MAX_AUC_DROP", "0.002"))
MODEL_GATE_MAX_ECE_INCREASE = float(os.getenv("MODEL_GATE_MAX_ECE_INCREASE", "0.005"))
# Optional candidate-level operating gates (skipped when unset / <=0).
MODEL_GATE_MIN_WATCH_RECALL = float(os.getenv("MODEL_GATE_MIN_WATCH_RECALL", "0.0"))
MODEL_GATE_MIN_WATCH_PRECISION = float(os.getenv("MODEL_GATE_MIN_WATCH_PRECISION", "0.0"))
MODEL_GATE_MAX_ALERT_RATE = float(os.getenv("MODEL_GATE_MAX_ALERT_RATE", "1.0"))
# A feature-contract change must NOT auto-promote on floors alone (default);
# set true only for a deliberate, reviewed contract migration.
MODEL_GATE_ALLOW_FEATURE_CHANGE = os.getenv(
    "MODEL_GATE_ALLOW_FEATURE_CHANGE", "false").lower() in ("1", "true", "yes")


def register_model_classes_in_main() -> None:
    """Alias train's pickled classes into ``__main__`` so an artifact pickled
    while train.py ran as a script (``__module__ == "__main__"``) unpickles from
    ANY process. Idempotent. This is the single registration point every entry
    point uses via ``load_model_bundle`` — no more per-module copies."""
    import sys
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return
    for name in ("_EnsembleRegressor", "_prob_to_days_for_compat"):
        obj = globals().get(name)
        if obj is not None and not hasattr(main_mod, name):
            setattr(main_mod, name, obj)


def load_model_bundle(path: str):
    """Canonical loader for promoted model artifacts — the ONE place the API,
    risk_map, scientific_stats and rolling_eval load models, so pickle
    registration and bundle handling can never drift between them.

    Registers the pickled classes in ``__main__`` first, then unpickles.
    Supports the ``bundle-v1`` dict and legacy bare-model artifacts. Returns
    ``(model, metadata_or_None)`` (metadata is None for legacy artifacts).
    Raises with a clear message on a corrupt/missing artifact so callers fail
    loudly rather than serving a half-loaded model.
    """
    from storage import read_pickle
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path}")
    register_model_classes_in_main()
    obj = read_pickle(path)
    if obj is None:
        raise ValueError(f"Corrupt model artifact (unpickled to None): {path}")
    if isinstance(obj, dict):
        if obj.get("format") == "bundle-v1":
            model = obj.get("model")
            if model is None:
                raise ValueError(f"Corrupt bundle-v1 at {path}: missing 'model' field")
            return model, obj.get("metadata")
        raise ValueError(
            f"Unrecognized dict artifact at {path} (format="
            f"{obj.get('format')!r}); expected a bundle-v1 bundle or a bare model.")
    if not (hasattr(obj, "predict_proba") or hasattr(obj, "predict")):
        raise ValueError(
            f"Loaded object at {path} is not a usable model (type={type(obj).__name__}).")
    return obj, None


def _evaluate_promotion_gates(
    candidate_auc: float,
    candidate_ap: float,
    test_prevalence: float,
    X_test,
    y_test_bin,
    feature_cols: List[str],
    promoted_path: str,
    candidate_brier: Optional[float] = None,
    candidate_ece: Optional[float] = None,
    candidate_watch_recall: Optional[float] = None,
    candidate_watch_precision: Optional[float] = None,
    candidate_alert_rate: Optional[float] = None,
) -> Tuple[bool, str]:
    """Decide whether the candidate may replace the trusted model.

    NOTE ON THE COMPARISON SET: ideally the head-to-head runs on a dedicated
    promotion/champion split that is NOT the final held-out test (see the
    five-way protocol note in the docstring of `main`). Until that split
    exists the comparison uses ``X_test`` — the caller must treat any promotion
    metric as champion-selection, not the final reported test number.

    Gates (all configurable via MODEL_GATE_* env):
      1. Absolute floors: AUC ≥ MIN_AUC; PR-AUC ≥ MIN_AP_UPLIFT × prevalence.
      2. Operating floors (skipped when unset): WATCH recall ≥ MIN_WATCH_RECALL,
         WATCH precision ≥ MIN_WATCH_PRECISION, alert rate ≤ MAX_ALERT_RATE.
      3. Head-to-head vs the incumbent on the SAME set: candidate AUC ≥
         incumbent − MAX_AUC_DROP, AP ≥ incumbent AP, Brier ≤ incumbent Brier,
         ECE ≤ incumbent ECE + MAX_ECE_INCREASE.

    Fail-closed: a feature-contract change does NOT auto-promote on floors
    alone (returns pending review unless MODEL_GATE_ALLOW_FEATURE_CHANGE); an
    incumbent-comparison error REJECTS (never fail-open). The promote message
    says "candidate ≥ incumbent" ONLY when strictly ≥ on both AUC and AP, else
    "within allowed regression tolerance".
    """
    if candidate_auc < MODEL_GATE_MIN_AUC:
        return False, f"REJECTED: test AUC {candidate_auc:.4f} < floor {MODEL_GATE_MIN_AUC}"
    ap_floor = MODEL_GATE_MIN_AP_UPLIFT * max(test_prevalence, 1e-9)
    if candidate_ap < ap_floor:
        return False, (f"REJECTED: test PR-AUC {candidate_ap:.4f} < "
                       f"{MODEL_GATE_MIN_AP_UPLIFT}× prevalence ({ap_floor:.4f})")
    # Operating floors (only enforced when configured > 0 / < 1).
    if MODEL_GATE_MIN_WATCH_RECALL > 0 and candidate_watch_recall is not None \
            and candidate_watch_recall < MODEL_GATE_MIN_WATCH_RECALL:
        return False, (f"REJECTED: WATCH recall {candidate_watch_recall:.4f} < "
                       f"floor {MODEL_GATE_MIN_WATCH_RECALL}")
    if MODEL_GATE_MIN_WATCH_PRECISION > 0 and candidate_watch_precision is not None \
            and candidate_watch_precision < MODEL_GATE_MIN_WATCH_PRECISION:
        return False, (f"REJECTED: WATCH precision {candidate_watch_precision:.4f} < "
                       f"floor {MODEL_GATE_MIN_WATCH_PRECISION}")
    if MODEL_GATE_MAX_ALERT_RATE < 1.0 and candidate_alert_rate is not None \
            and candidate_alert_rate > MODEL_GATE_MAX_ALERT_RATE:
        return False, (f"REJECTED: alert rate {candidate_alert_rate:.4f} > "
                       f"ceiling {MODEL_GATE_MAX_ALERT_RATE}")

    if not os.path.isfile(promoted_path):
        return True, "PROMOTED: passed floors; no incumbent model to compare against"

    try:
        current_model, current_meta = load_model_bundle(promoted_path)
    except Exception as exc:
        # Fail CLOSED — never promote past an unverifiable incumbent.
        return False, (f"REJECTED (fail-closed): could not load incumbent for "
                       f"head-to-head comparison ({exc})")

    current_features = (current_meta or {}).get("features")
    if current_features != list(feature_cols):
        if not MODEL_GATE_ALLOW_FEATURE_CHANGE:
            return False, (
                "PENDING MANUAL REVIEW: feature contract differs from the "
                "incumbent — not auto-promoted on floors alone. Set "
                "MODEL_GATE_ALLOW_FEATURE_CHANGE=true for a reviewed migration.")
        return True, ("PROMOTED: feature contract changed (allowed by "
                      "MODEL_GATE_ALLOW_FEATURE_CHANGE) — head-to-head skipped")

    try:
        cur_proba = current_model.predict_proba(X_test)
        cur_auc = float(roc_auc_score(y_test_bin, cur_proba))
        cur_ap = float(average_precision_score(y_test_bin, cur_proba))
        cur_brier = float(brier_score_loss(y_test_bin, cur_proba))
        cur_ece = expected_calibration_error(y_test_bin, cur_proba)
    except Exception as exc:
        return False, (f"REJECTED (fail-closed): incumbent scoring failed ({exc})")

    fails = []
    if candidate_auc < cur_auc - MODEL_GATE_MAX_AUC_DROP:
        fails.append(f"AUC {candidate_auc:.4f} < incumbent {cur_auc:.4f} − {MODEL_GATE_MAX_AUC_DROP}")
    if candidate_ap < cur_ap:
        fails.append(f"AP {candidate_ap:.4f} < incumbent {cur_ap:.4f}")
    if candidate_brier is not None and candidate_brier > cur_brier:
        fails.append(f"Brier {candidate_brier:.4f} > incumbent {cur_brier:.4f}")
    if candidate_ece is not None and candidate_ece > cur_ece + MODEL_GATE_MAX_ECE_INCREASE:
        fails.append(f"ECE {candidate_ece:.4f} > incumbent {cur_ece:.4f} + {MODEL_GATE_MAX_ECE_INCREASE}")
    if fails:
        return False, ("REJECTED: candidate worse than incumbent on the same "
                       "comparison set — " + "; ".join(fails))

    # Passed all head-to-head gates. Only claim "≥ incumbent" when TRUE.
    strictly_ge = candidate_auc >= cur_auc and candidate_ap >= cur_ap
    verdict = ("candidate ≥ incumbent" if strictly_ge
               else "within allowed regression tolerance")
    return True, (f"PROMOTED: {verdict} on same comparison set "
                  f"(AUC {candidate_auc:.4f} vs {cur_auc:.4f}, "
                  f"AP {candidate_ap:.4f} vs {cur_ap:.4f})")


def _promote_bundle(bundle: dict, model_dir: str, promoted_path: str) -> None:
    """Atomically install the candidate as the promoted model.

    Sequence: write candidate to a temp file in the SAME directory (required
    for os.replace atomicity), keep the incumbent at previous.pkl for
    rollback, then one os.replace(). Readers see either the old complete
    bundle or the new complete bundle — never a mixture.
    """
    import shutil
    os.makedirs(model_dir, exist_ok=True)
    candidate_path = os.path.join(model_dir, ".candidate.pkl.tmp")
    write_pickle(bundle, candidate_path)
    if os.path.isfile(promoted_path):
        shutil.copy2(promoted_path, os.path.join(model_dir, "previous.pkl"))
    os.replace(candidate_path, promoted_path)


def expected_calibration_error(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> float:
    """ECE — bin predictions by probability, weighted mean |bin_pred - bin_actual|.

    0.0 = perfectly calibrated, larger = more miscalibrated. Operationally,
    ECE < 0.05 is "trustworthy probability"; > 0.15 is "treat probability as
    rank-only, not as percentage".
    """
    y_true = np.asarray(y_true).astype(float)
    proba = np.asarray(proba).astype(float)
    if len(y_true) == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(proba, bin_edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = ids == b
        n = int(m.sum())
        if n == 0:
            continue
        bin_pred = float(proba[m].mean())
        bin_true = float(y_true[m].mean())
        ece += (n / len(proba)) * abs(bin_pred - bin_true)
    return float(ece)


def reliability_bins(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> List[Dict[str, float]]:
    """Per-bin reliability points for plotting a calibration curve in the UI."""
    y_true = np.asarray(y_true).astype(float)
    proba = np.asarray(proba).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(proba, bin_edges) - 1, 0, n_bins - 1)
    out: List[Dict[str, float]] = []
    for b in range(n_bins):
        m = ids == b
        n = int(m.sum())
        if n == 0:
            continue
        out.append({
            "bin_lower": float(bin_edges[b]),
            "bin_upper": float(bin_edges[b + 1]),
            "mean_predicted": float(proba[m].mean()),
            "actual_rate": float(y_true[m].mean()),
            "count": n,
        })
    return out


def _build_search(
    estimator: LGBMClassifier,
    n_iter: int,
    cv: Any,
    random_state: int,
) -> Tuple[Any, str]:
    """Prefer BayesSearchCV; fall back to RandomizedSearchCV if skopt missing.

    Scoring: roc_auc — robust to class imbalance and the natural metric for a
    binary fire-in-3-days classifier. We pair it with scale_pos_weight on the
    estimator and undersampled training data for stable convergence.

    `cv` must be an explicit list of (train_idx, test_idx) folds from
    eval_protocol.purged_date_folds — sklearn's TimeSeriesSplit(gap=7) purges
    7 ROWS, not 7 days, which with thousands of cells per date is no purge
    at all. Never reintroduce it here.
    """
    # outer CV n_jobs=1 (sequential folds) so LightGBM's inner n_jobs=-1 gets
    # the full machine. cores²-thread oversubscription cratered the previous
    # run (load avg ~50, each fit 4-5× slower than expected).
    #
    # Tuning objective (audited item 3): ROC-AUC is insensitive to the rare
    # positive class the alerts actually care about — at 3% prevalence a model
    # can look excellent on AUC while ranking positives poorly at the top.
    # Default is now average_precision (PR-AUC); override via TUNE_SCORING.
    # Threshold selection stays separate (calib only). Both AUC and AP are
    # reported on test regardless of the tuning objective.
    scoring = os.getenv("TUNE_SCORING", "average_precision")
    if scoring not in ("average_precision", "roc_auc"):
        log.warning("TUNE_SCORING=%r unsupported — using average_precision", scoring)
        scoring = "average_precision"
    log.info("Hyperparameter search scoring objective: %s", scoring)
    if _BAYES_AVAILABLE and BayesSearchCV is not None and Categorical is not None:
        search_spaces = {k: Categorical(v) for k, v in PARAM_GRID.items()}
        try:
            search = BayesSearchCV(
                estimator=estimator,
                search_spaces=search_spaces,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                random_state=random_state,
                n_jobs=1,
                verbose=2,
                refit=True,
            )
            return search, "BayesSearchCV"
        except Exception as exc:  # pragma: no cover
            log.warning("BayesSearchCV init failed (%s) — falling back to RandomizedSearchCV", exc)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=PARAM_GRID,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        verbose=2,
        refit=True,
    )
    return search, "RandomizedSearchCV"


def _fit_search(
    search: Any,
    X_train: pd.DataFrame,
    y_bin: np.ndarray,
    sample_weight: Optional[np.ndarray],
) -> None:
    """Fit binary classifier search with optional sample_weight."""
    if sample_weight is not None:
        try:
            search.fit(X_train, y_bin, sample_weight=sample_weight)
            return
        except Exception as exc:
            log.warning(
                "Tuning with sample_weight failed (%s) — retrying without weights.",
                exc,
            )
    search.fit(X_train, y_bin)


def _refit_ensemble_with_early_stopping(
    best_params: Dict[str, Any],
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_es: pd.DataFrame,
    y_es: np.ndarray,
    sample_weight: Optional[np.ndarray],
    scale_pos_weight: float,
    n_ensemble: int = 5,
    base_seed: int = 42,
    early_stopping_rounds: int = 50,
) -> Any:
    """Fit n_ensemble binary LGBM models with early stopping.

    X_fit/y_fit is the (undersampled, weighted) TRAIN partition only;
    X_es/y_es is the chronologically-later VAL partition at natural
    prevalence. Keeping fit and early-stopping data from disjoint purged
    date blocks means no row influences both the boosting trajectory and
    its own stopping point — and val is never mistaken for held-out data
    downstream (calibration and thresholds use calib; final metrics use
    test).
    """
    sw_fit = sample_weight

    models: List[LGBMClassifier] = []
    for i in range(n_ensemble):
        seed = base_seed + 37 * i
        m = _build_lgbm(random_state=seed, scale_pos_weight=scale_pos_weight)
        m.set_params(**best_params)
        fit_kwargs: Dict[str, Any] = {
            "eval_set": [(X_es, y_es)],
            "eval_metric": ["binary_logloss", "auc"],
            "callbacks": [early_stopping(early_stopping_rounds, verbose=False)],
        }
        if sw_fit is not None:
            fit_kwargs["sample_weight"] = sw_fit
        m.fit(X_fit, y_fit, **fit_kwargs)
        models.append(m)
    return _EnsembleRegressor(models)


class _EnsembleRegressor:
    """Binary-classifier ensemble with an optional picklable calibrator.

    predict_proba(X) returns calibrated P(fire-in-3-days), shape (n,).
    predict(X) returns pseudo-days for risk_map.py / api.py — monotone in
    probability so bucket tiers still rank by imminence.

    Backward compatibility: historical Platt calibrators expose
    ``predict_proba``; isotonic and similar regressors expose ``predict``.
    """

    def __init__(
        self,
        models: List[LGBMClassifier],
        calibrator: Any = None,
    ) -> None:
        self.models = models
        self.calibrator = calibrator

    def _raw_proba(self, X: Any) -> np.ndarray:
        probs = np.stack(
            [m.predict_proba(X)[:, 1] for m in self.models], axis=0
        )
        return probs.mean(axis=0)

    def predict_proba(self, X: Any) -> np.ndarray:
        raw = self._raw_proba(X)
        if self.calibrator is None:
            return raw
        if hasattr(self.calibrator, "predict_proba"):
            calibrated = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        elif hasattr(self.calibrator, "predict"):
            calibrated = self.calibrator.predict(raw)
        else:
            raise TypeError("calibrator must expose predict_proba() or predict()")
        return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)

    def predict(self, X: Any) -> np.ndarray:
        return _prob_to_days_for_compat(self.predict_proba(X))

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        imps = [
            np.asarray(m.feature_importances_, dtype=float)
            for m in self.models
            if hasattr(m, "feature_importances_")
        ]
        if not imps:
            return None
        return np.stack(imps, axis=0).mean(axis=0)


# ─────────────────────────────────────────────
# Visualization (matplotlib) — optional
# ─────────────────────────────────────────────
def _render_training_report(
    daily_stats: Dict[str, Dict[str, float]],
    overall: Dict[str, float],
    feature_importance: List[Dict[str, Any]],
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    out_dir: str,
    horizon: int,
) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
    except Exception as exc:
        log.warning(
            "matplotlib not available (%s) — skipping PNG report. "
            "Install with: pip install matplotlib", exc,
        )
        return None

    os.makedirs(out_dir, exist_ok=True)
    days = list(range(1, horizon + 1))
    keys = [f"day_{d}" for d in days]

    def _col(name: str) -> List[float]:
        return [float(daily_stats[k].get(name, 0.0)) for k in keys]

    mae_vals      = _col("mae")
    rmse_vals     = _col("rmse")
    r2_vals       = _col("r2")
    acc1_vals     = _col("acc_within_1")
    acc_ex_vals   = _col("acc_exact")
    bias_vals     = _col("bias")
    hotspot_vals  = _col("hotspot_count")
    hit_rate_vals = _col("hotspot_hit_rate")
    fa_rate_vals  = _col("false_alarm_rate")
    prec_vals     = _col("precision")
    rec_vals      = _col("recall")

    def _mae_colors(vs: List[float]) -> List[str]:
        out = []
        for v in vs:
            if v < 1.0:
                out.append("#2ecc71")
            elif v < 1.5:
                out.append("#f1c40f")
            else:
                out.append("#e74c3c")
        return out

    def _r2_colors(vs: List[float]) -> List[str]:
        out = []
        for v in vs:
            if v < 0:
                out.append("#c0392b")
            elif v < 0.2:
                out.append("#e74c3c")
            elif v < 0.4:
                out.append("#f1c40f")
            else:
                out.append("#2ecc71")
        return out

    def _hit_colors(vs: List[float]) -> List[str]:
        out = []
        for v in vs:
            pct = v * 100
            if pct < 50:
                out.append("#e74c3c")
            elif pct < 70:
                out.append("#f1c40f")
            else:
                out.append("#2ecc71")
        return out

    # ─── Full report (Sections A-D in one figure) ───
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(
        f"Wildfire Prediction Model — Training Report "
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]",
        fontsize=18, fontweight="bold", y=0.995,
    )
    gs = gridspec.GridSpec(7, 3, figure=fig, hspace=0.55, wspace=0.30)

    # Section A — Per-Day Model Performance
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(days, mae_vals, color=_mae_colors(mae_vals))
    ax.set_title("A1 — MAE per day"); ax.set_xlabel("Day"); ax.set_ylabel("MAE (days)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(days, rmse_vals, color="#3498db")
    ax.set_title("A2 — RMSE per day"); ax.set_xlabel("Day"); ax.set_ylabel("RMSE")

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(days, r2_vals, color=_r2_colors(r2_vals))
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("A3 — R² per day"); ax.set_xlabel("Day"); ax.set_ylabel("R²")

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(days, [v * 100 for v in acc1_vals], color="#9b59b6")
    ax.axhline(70, color="red", linestyle="--", linewidth=1, label="70% target")
    ax.set_title("A4 — Accuracy ±1day per day"); ax.set_xlabel("Day"); ax.set_ylabel("Accuracy (%)")
    ax.legend()

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(days, [v * 100 for v in acc_ex_vals], color="#1abc9c")
    ax.set_title("A5 — Accuracy Exact per day"); ax.set_xlabel("Day"); ax.set_ylabel("Accuracy (%)")

    ax = fig.add_subplot(gs[1, 2])
    bias_colors = ["#e74c3c" if v > 0 else "#3498db" for v in bias_vals]
    ax.bar(days, bias_vals, color=bias_colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("A6 — Bias per day (+ over-pred, − under-pred)"); ax.set_xlabel("Day"); ax.set_ylabel("Bias")

    # Section B — Hotspot Quality
    ax = fig.add_subplot(gs[2, 0])
    ax.bar(days, hotspot_vals, color="#e67e22")
    ax.set_title("B1 — Hotspot Count per day"); ax.set_xlabel("Day"); ax.set_ylabel("Hotspots")

    ax = fig.add_subplot(gs[2, 1])
    width = 0.4
    x = np.array(days, dtype=float)
    ax.bar(x - width / 2, [v * 100 for v in hit_rate_vals], width=width,
           color="#27ae60", label="Hit Rate")
    ax.bar(x + width / 2, [v * 100 for v in fa_rate_vals], width=width,
           color="#c0392b", label="False Alarm")
    ax.set_title("B2 — Hit Rate vs False Alarm Rate"); ax.set_xlabel("Day"); ax.set_ylabel("Rate (%)")
    ax.legend()

    ax = fig.add_subplot(gs[2, 2])
    ax.plot(days, [v * 100 for v in prec_vals], marker="o", color="#2c3e50", label="Precision")
    ax.plot(days, [v * 100 for v in rec_vals], marker="s", color="#16a085", label="Recall")
    ax.set_title("B3 — Precision vs Recall"); ax.set_xlabel("Day"); ax.set_ylabel("%"); ax.legend()

    # Section C — Overall summary
    ax = fig.add_subplot(gs[3, 0], polar=True)
    radar_labels = ["1/MAE", "R²", "Acc±1", "HitRate", "Precision"]
    radar_vals = [
        max(0.0, min(1.0, 1.0 / max(overall.get("mae", 1.0), 0.01))),
        max(0.0, min(1.0, overall.get("r2", 0.0))),
        max(0.0, min(1.0, overall.get("acc_within_1", 0.0))),
        float(np.mean(hit_rate_vals)) if hit_rate_vals else 0.0,
        float(np.mean(prec_vals)) if prec_vals else 0.0,
    ]
    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    radar_closed = radar_vals + radar_vals[:1]
    angles_closed = angles + angles[:1]
    ax.plot(angles_closed, radar_closed, linewidth=2, color="#2980b9")
    ax.fill(angles_closed, radar_closed, alpha=0.25, color="#2980b9")
    ax.set_xticks(angles)
    ax.set_xticklabels(radar_labels)
    ax.set_ylim(0, 1)
    ax.set_title("C1 — Overall Summary (radar)")

    ax = fig.add_subplot(gs[3, 1:])
    top_fi = feature_importance[:15] if feature_importance else []
    if top_fi:
        names = [f["feature"] for f in top_fi][::-1]
        vals  = [f["importance"] for f in top_fi][::-1]
        vmax = max(vals) if max(vals) > 0 else 1.0
        colors = [plt.cm.viridis(v / vmax) for v in vals]
        ax.barh(names, vals, color=colors)
        ax.set_title("C2 — Top 15 Feature Importances")
    else:
        ax.text(0.5, 0.5, "No feature importances available", ha="center", va="center")
        ax.set_axis_off()

    ax = fig.add_subplot(gs[4, :])
    if len(y_test_true) > 0:
        # gaussian_kde evaluation is O(n²) — on a 630k-row natural-prevalence
        # test set that is ~4×10¹¹ kernel evaluations (hours of CPU; it
        # wedged two training runs before this cap). A 20k subsample renders
        # an indistinguishable density scatter in seconds.
        MAX_KDE_POINTS = 20_000
        if len(y_test_true) > MAX_KDE_POINTS:
            rng = np.random.default_rng(0)
            sel = rng.choice(len(y_test_true), MAX_KDE_POINTS, replace=False)
            y_plot_true = y_test_true[sel]
            y_plot_pred = y_test_pred[sel]
        else:
            y_plot_true, y_plot_pred = y_test_true, y_test_pred
        try:
            from scipy.stats import gaussian_kde
            xy = np.vstack([y_plot_true.astype(float), y_plot_pred.astype(float)])
            density = gaussian_kde(xy)(xy)
        except Exception:
            density = np.ones_like(y_plot_true, dtype=float)
        idx = np.argsort(density)
        sc = ax.scatter(
            y_plot_true[idx], y_plot_pred[idx],
            c=density[idx], cmap="viridis", s=12, alpha=0.7,
        )
        plt.colorbar(sc, ax=ax, label="Density")
        ax.plot([0, horizon], [0, horizon], color="red", linewidth=2, label="Perfect")
        ax.set_xlim(-0.5, horizon + 0.5)
        ax.set_ylim(-0.5, horizon + 0.5)
        ax.set_xlabel("Actual days_until_fire"); ax.set_ylabel("Predicted")
        ax.set_title("C3 — Prediction vs Actual"); ax.legend()
    else:
        ax.text(0.5, 0.5, "No test data", ha="center", va="center")
        ax.set_axis_off()

    # Section D — Conditional-coloured summary table
    ax = fig.add_subplot(gs[5:, :])
    ax.axis("off")
    cell_text = []
    cell_colors = []
    cols = ["Day", "MAE", "RMSE", "R²", "Acc±1", "AccExact", "Hotspots", "HitRate", "Bias", "Precision", "Recall"]
    for d in days:
        s = daily_stats[f"day_{d}"]
        row = [
            str(d),
            f"{s['mae']:.2f}",
            f"{s['rmse']:.2f}",
            f"{s['r2']:.2f}",
            f"{s['acc_within_1']*100:.1f}%",
            f"{s['acc_exact']*100:.1f}%",
            f"{s['hotspot_count']}",
            f"{s['hotspot_hit_rate']*100:.1f}%",
            f"{s['bias']:+.2f}",
            f"{s['precision']*100:.1f}%",
            f"{s['recall']*100:.1f}%",
        ]
        cell_text.append(row)
        cmae = _mae_colors([s["mae"]])[0]
        cr2  = _r2_colors([s["r2"]])[0]
        chit = _hit_colors([s["hotspot_hit_rate"]])[0]
        color_row = ["white", cmae, "white", cr2, "white", "white", "white", chit, "white", "white", "white"]
        cell_colors.append(color_row)
    table = ax.table(
        cellText=cell_text,
        colLabels=cols,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    ax.set_title("D — Per-day stats (green=good, yellow=ok, red=bad)", pad=18)

    full_path = os.path.join(out_dir, "training_report.png")
    try:
        plt.tight_layout(rect=[0, 0, 1, 0.985])
    except Exception:
        pass
    fig.savefig(full_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ─── Per-section PNGs ───
    def _save_subset(section: str, builder) -> None:
        fig2 = plt.figure(figsize=(18, 10))
        builder(fig2)
        try:
            plt.tight_layout()
        except Exception:
            pass
        fig2.savefig(os.path.join(out_dir, section), dpi=150, bbox_inches="tight")
        plt.close(fig2)

    def _section_a(fig2):
        axs = fig2.subplots(2, 3)
        axs[0, 0].bar(days, mae_vals, color=_mae_colors(mae_vals))
        axs[0, 0].set_title("MAE"); axs[0, 0].axhline(1.0, color="gray", linestyle="--")
        axs[0, 1].bar(days, rmse_vals, color="#3498db"); axs[0, 1].set_title("RMSE")
        axs[0, 2].bar(days, r2_vals, color=_r2_colors(r2_vals)); axs[0, 2].axhline(0, color="black", linewidth=0.5)
        axs[0, 2].set_title("R²")
        axs[1, 0].bar(days, [v * 100 for v in acc1_vals], color="#9b59b6")
        axs[1, 0].axhline(70, color="red", linestyle="--"); axs[1, 0].set_title("Acc ±1day (%)")
        axs[1, 1].bar(days, [v * 100 for v in acc_ex_vals], color="#1abc9c"); axs[1, 1].set_title("Acc Exact (%)")
        bcolors = ["#e74c3c" if v > 0 else "#3498db" for v in bias_vals]
        axs[1, 2].bar(days, bias_vals, color=bcolors); axs[1, 2].axhline(0, color="black", linewidth=0.5)
        axs[1, 2].set_title("Bias")
        for a in axs.ravel():
            a.set_xlabel("Day")

    def _section_b(fig2):
        axs = fig2.subplots(1, 3)
        axs[0].bar(days, hotspot_vals, color="#e67e22"); axs[0].set_title("Hotspot count")
        x = np.array(days, dtype=float); width = 0.4
        axs[1].bar(x - width / 2, [v * 100 for v in hit_rate_vals], width=width,
                   color="#27ae60", label="Hit Rate")
        axs[1].bar(x + width / 2, [v * 100 for v in fa_rate_vals], width=width,
                   color="#c0392b", label="False Alarm")
        axs[1].legend(); axs[1].set_title("Hit vs False Alarm (%)")
        axs[2].plot(days, [v * 100 for v in prec_vals], marker="o", label="Precision")
        axs[2].plot(days, [v * 100 for v in rec_vals], marker="s", label="Recall")
        axs[2].legend(); axs[2].set_title("Precision vs Recall (%)")

    def _section_c(fig2):
        gs2 = gridspec.GridSpec(1, 3, figure=fig2)
        axr = fig2.add_subplot(gs2[0, 0], polar=True)
        axr.plot(angles_closed, radar_closed, linewidth=2, color="#2980b9")
        axr.fill(angles_closed, radar_closed, alpha=0.25, color="#2980b9")
        axr.set_xticks(angles); axr.set_xticklabels(radar_labels); axr.set_ylim(0, 1)
        axr.set_title("Radar")
        axi = fig2.add_subplot(gs2[0, 1])
        if top_fi:
            names = [f["feature"] for f in top_fi][::-1]
            vals  = [f["importance"] for f in top_fi][::-1]
            vmax2 = max(vals) if max(vals) > 0 else 1.0
            colors2 = [plt.cm.viridis(v / vmax2) for v in vals]
            axi.barh(names, vals, color=colors2)
        axi.set_title("Top features")
        axs = fig2.add_subplot(gs2[0, 2])
        if len(y_test_true) > 0:
            axs.scatter(y_test_true, y_test_pred, s=8, alpha=0.5, color="#2980b9")
            axs.plot([0, horizon], [0, horizon], color="red", linewidth=2)
            axs.set_xlim(-0.5, horizon + 0.5); axs.set_ylim(-0.5, horizon + 0.5)
            axs.set_title("Pred vs Actual")

    def _section_d(fig2):
        ax2 = fig2.add_subplot(111)
        ax2.axis("off")
        tbl = ax2.table(
            cellText=cell_text, colLabels=cols, cellColours=cell_colors,
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.0, 1.6)

    _save_subset("report_model_performance.png", _section_a)
    _save_subset("report_hotspot_quality.png", _section_b)
    _save_subset("report_summary.png", _section_c)
    _save_subset("report_table.png", _section_d)

    return full_path


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(
    n_iter: int = 30,
    n_splits: int = 5,
    val_fraction: float = 0.15,
    calib_fraction: float = 0.10,
    test_fraction: float = 0.15,
    grid_size: float = 0.1,
    min_confidence: int = 0,
    skip_risk_map: bool = False,
    random_state: int = 42,
    predict_only: bool = False,
    max_history_days: int = 0,
    n_ensemble: int = 10,
    early_stopping_rounds: int = 50,
) -> dict:
    t_total_start = time.time()
    load_dotenv()
    p = _paths()
    for d in (p["model_dir"], p["feature_dir"], p["meta_dir"]):
        os.makedirs(d, exist_ok=True)

    log.info("==== STEP 1: load + grid raw FIRMS data ====")
    weather_path = resolve_existing(p["weather_path"])
    urban_filter_enabled = os.getenv("URBAN_FILTER_ENABLED", "true").lower() in ("1", "true", "yes")
    urban_buffer_km = float(os.getenv("URBAN_BUFFER_KM", "0.0"))
    tree_cover_path = p["tree_cover_path"] if os.path.exists(p["tree_cover_path"]) else None
    radd_path = resolve_existing(p["radd_path"])

    daily = load_and_prepare(
        raw_dir=p["raw_dir"],
        firms_path=p["firms_path"],
        grid_size=grid_size,
        min_confidence=min_confidence,
        densify=True,
        weather_path=weather_path,
        tree_cover_path=tree_cover_path,
        radd_path=radd_path,
        filter_urban=urban_filter_enabled,
        urban_buffer_km=urban_buffer_km,
    )

    max_history_days_applied = 0
    if max_history_days > 0:
        dmax = pd.to_datetime(daily["date"]).max()
        cutoff = dmax - pd.Timedelta(days=max_history_days)
        before_rows = len(daily)
        daily = daily[pd.to_datetime(daily["date"]) >= cutoff].copy()
        max_history_days_applied = max_history_days
        log.info(
            "History window: last %d calendar days, rows %d → %d",
            max_history_days, before_rows, len(daily),
        )

    latest_observed = pd.to_datetime(daily["date"]).max()
    latest_date = pd.Timestamp(latest_observed).normalize().date()
    today_utc = pd.Timestamp.now("UTC").normalize().date()
    stale_days = (today_utc - latest_date).days
    STALE_WARN_DAYS = int(os.getenv("STALE_WARN_DAYS", "5"))
    if stale_days > STALE_WARN_DAYS:
        log.warning("⚠️  FIRMS DATA STALE: latest=%s (%d days behind)", latest_observed.date(), stale_days)

    WEATHER_COLS = ["temp_max", "temp_min", "precip_sum", "wind_max", "et0"]
    MIN_WEATHER_COVERAGE = float(os.getenv("MIN_WEATHER_COVERAGE", "0.20"))
    weather_cols_present = [c for c in WEATHER_COLS if c in daily.columns]
    if weather_cols_present:
        coverage = daily[weather_cols_present[0]].notna().mean()
        if coverage < MIN_WEATHER_COVERAGE:
            log.warning("Weather coverage %.1f%% < threshold — dropping weather cols", coverage * 100)
            daily = daily.drop(columns=weather_cols_present)

    log.info("==== STEP 2: feature engineering ====")
    feats = build_features(daily, horizon=MAX_PREDICTION_DAYS, grid_size=grid_size)

    feature_path = os.path.join(p["feature_dir"], "full_features.parquet")
    feats_sorted = feats.sort_values("date", kind="stable")
    write_table(
        feats_sorted, feature_path,
        row_group_size=200_000, compression="zstd", compression_level=10,
    )
    log.info("Saved feature dataset → %s", feature_path)

    if predict_only:
        model_path = os.path.join(p["model_dir"], "lgbm_fire_date_model.pkl")
        if not os.path.isfile(model_path):
            raise RuntimeError("predict-only requires an existing trained model")
        log.info("==== predict-only: skipped tuning ====")
        # CRITICAL: free 5-6 GB of feature-engineering memory BEFORE calling
        # risk_map.run(). Otherwise we stack risk_map's ~3.5 GB load on top
        # of the still-resident feats/daily frames and OOM on a 22 GB laptop.
        log.info("Releasing feature-engineering memory before risk_map…")
        del feats, feats_sorted, daily
        gc.collect()
        if not skip_risk_map:
            try:
                from risk_map import run as generate_risk_map
                generate_risk_map()
            except Exception as exc:
                log.warning("risk_map.run() failed: %s", exc)
        return {"mode": "predict_only", "feature_path": feature_path}

    log.info("==== STEP 3: build binary label + censor unobservable tail ====")
    # Memory plan: drop unused columns, cast to float32 BEFORE the split so
    # the full frame is as small as possible while all 4 split subsets are
    # materialised. Undersampling happens AFTER the split, train-only —
    # undersampling first corrupted val/test prevalence (an artificial 4:1
    # base rate) and every prevalence-sensitive metric computed on them.
    feature_cols = resolve_features(feats)
    analysis_cols = [c for c in ["radd_cross_verified", "multi_sat_confirmed"] if c in feats.columns]
    keep_cols = list(set(
        feature_cols
        + ["date", "lat_grid", "lon_grid", "days_until_fire", "days_ahead_observed"]
        + analysis_cols
    ))
    feats = feats[keep_cols].copy()

    # Cast feature columns to float32 in-place (lat/lon kept as float64 for grid math)
    f32_cols = [c for c in feature_cols if c not in ("lat_grid", "lon_grid")]
    for c in f32_cols:
        if feats[c].dtype != np.float32:
            feats[c] = feats[c].astype(np.float32)

    # Snapshot dataset-level stats BEFORE any supervised filtering (metadata).
    dataset_stats = {
        "earliest_date": str(feats["date"].min()),
        "latest_date": str(feats["date"].max()),
        "total_days": int(pd.Series(feats["date"]).nunique()),
        "total_active_cells": int(feats[["lat_grid", "lon_grid"]].drop_duplicates().shape[0]),
        "total_rows_densified": int(len(feats)),
    }

    # CENSORING: a row dated d claims a label about (d, d+IMMINENT_DAYS]. At
    # the dataset tail that window extends past the last observed date, and
    # the label builder silently filled it as "no fire" — a false negative.
    # Those rows are valid for live inference only, never for supervision.
    if "days_ahead_observed" in feats.columns:
        n_before = len(feats)
        feats = feats[feats["days_ahead_observed"] >= IMMINENT_DAYS].copy()
        n_censored = n_before - len(feats)
        log.info(
            "Censored %d tail rows (last %d days) whose %d-day label window "
            "is not fully observed", n_censored, IMMINENT_DAYS, IMMINENT_DAYS,
        )
    else:
        n_censored = 0
        log.warning("days_ahead_observed missing — rebuild features (stale parquet?)")

    feats["_y_bin"] = _make_binary_label(feats["days_until_fire"]).astype(np.int8)
    n_pos = int(feats["_y_bin"].sum())
    n_neg = int(len(feats) - n_pos)
    natural_prevalence = n_pos / max(len(feats), 1)
    log.info(
        "Natural class distribution: pos=%d (%.3f%%)  neg=%d  (target: fire ∈ {1..%d} days)",
        n_pos, 100 * natural_prevalence, n_neg, IMMINENT_DAYS,
    )
    if n_pos == 0:
        raise RuntimeError("No positive examples — check data.")

    log.info("==== STEP 4: date-grouped chronological split (7-day purge) ====")
    # Whole-date 4-way split: train / val / calib / test. A calendar date
    # belongs to exactly one split; 7 calendar days are dropped before every
    # boundary (assertions inside date_grouped_split enforce both).
    #   train — hyperparameter search + ensemble fit (undersampled, weighted)
    #   val   — early-stopping eval + (with train) final ensemble fit data
    #   calib — probability calibration + threshold selection (natural prev.)
    #   test  — touched exactly once, at the end, at natural prevalence
    PURGE_DAYS = 7
    # Optional dedicated PROMOTION/champion block between calib and test
    # (CHAMPION_FRACTION > 0 ⇒ 5-way split). Promotion gates then compare on
    # champion, keeping the FINAL test untouched by model selection. Default 0
    # = legacy 4-way behaviour.
    champion_fraction = float(os.getenv("CHAMPION_FRACTION", "0.0"))
    try:
        split = date_grouped_split(
            feats["date"],
            val_fraction=val_fraction,
            calib_fraction=calib_fraction,
            test_fraction=test_fraction,
            purge_days=PURGE_DAYS,
            champion_fraction=champion_fraction,
        )
    except ValueError as exc:
        n_dates = int(pd.Series(feats["date"]).nunique())
        raise SystemExit(
            f"\n❌ NOT ENOUGH DATA TO TRAIN A VALID MODEL ({n_dates} usable "
            f"day(s) after tail censoring).\n"
            f"   The leakage-safe protocol needs 4 date blocks + three "
            f"{PURGE_DAYS}-day purges (≥ ~25 distinct days minimum; months of "
            f"data for a *useful* model).\n"
            f"   Fix: add historical FIRMS exports to data/raw/ "
            f"(firms.modaps.eosdis.nasa.gov/download — request the Jan–Apr "
            f"fire-season months), or let the scheduler accumulate NRT data "
            f"over time.\n"
            f"   Details: {exc}"
        ) from exc
    train_df = feats[split.masks["train"]]
    val_df   = feats[split.masks["val"]]
    calib_df = feats[split.masks["calib"]]
    champion_df = (feats[split.masks["champion"]]
                   if "champion" in split.masks else None)
    test_df  = feats[split.masks["test"]]
    del feats
    gc.collect()

    split_prevalence = {}
    _parts = [("train", train_df), ("val", val_df), ("calib", calib_df)]
    if champion_df is not None:
        _parts.append(("champion", champion_df))
    _parts.append(("test", test_df))
    for name, part in _parts:
        prev = float(part["_y_bin"].mean()) if len(part) else 0.0
        split_prevalence[name] = round(prev, 5)
        log.info(
            "  %-5s : %s → %s  (%d rows, prevalence %.3f%%)",
            name, *split.date_ranges[name], len(part), 100 * prev,
        )
    if min(len(val_df), len(calib_df), len(test_df)) == 0:
        raise RuntimeError("A split is empty — need more data for a purged 4-way split")
    if int(calib_df["_y_bin"].sum()) < 10 or int(test_df["_y_bin"].sum()) < 10:
        log.warning(
            "Very few positives in calib (%d) / test (%d) — threshold and "
            "test metrics will be unstable",
            int(calib_df["_y_bin"].sum()), int(test_df["_y_bin"].sum()),
        )

    # ── Class-imbalance handling (IMBALANCE_MODE, audited item 2) ────────────
    # The legacy default (D) STACKS three corrections — 4:1 undersampling ×
    # inverse-class-frequency sample weights × scale_pos_weight≈4 — an
    # effective ~120× positive up-weighting at 3% natural prevalence that
    # inflates raw probabilities and leans entirely on Platt to repair them.
    # Modes (class balancing applied exactly once unless D):
    #   A: natural prevalence, NO class weighting            (memory-heavy)
    #   B: natural prevalence + scale_pos_weight only        (memory-heavy)
    #   C: 4:1 undersampling + analytic prior correction, no extra weighting
    #   D: legacy stacked behavior (default, backward-compatible)
    imbalance_mode = os.getenv("IMBALANCE_MODE", "D").upper()
    if imbalance_mode not in ("A", "B", "C", "D"):
        log.warning("IMBALANCE_MODE=%r invalid — using D", imbalance_mode)
        imbalance_mode = "D"
    natural_pos = int(train_df["_y_bin"].sum())
    natural_neg = int(len(train_df) - natural_pos)

    if imbalance_mode in ("A", "B"):
        train_df_us = train_df.copy()
        log.info("IMBALANCE_MODE=%s: natural prevalence, no undersampling "
                 "(%d rows — memory-heavy)", imbalance_mode, len(train_df_us))
        undersample_beta = 1.0
    else:
        train_df_us = _undersample_negatives(
            train_df, "_y_bin", NEG_TO_POS_RATIO, random_state=random_state)
        kept_neg = int(len(train_df_us) - train_df_us["_y_bin"].sum())
        # β = fraction of natural negatives kept — the analytic prior
        # correction for mode C: p_natural = β·p / (β·p + (1 − p)).
        undersample_beta = kept_neg / max(natural_neg, 1)
        log.info(
            "Train undersample (%d:1): %d → %d rows (pos=%d, beta=%.4f)",
            NEG_TO_POS_RATIO, len(train_df), len(train_df_us),
            int(train_df_us["_y_bin"].sum()), undersample_beta,
        )
    del train_df
    gc.collect()

    log.info("Using %d features", len(feature_cols))
    X_train, y_train_bin = train_df_us[feature_cols], train_df_us["_y_bin"].to_numpy()
    X_val,   y_val_bin   = val_df[feature_cols],   val_df["_y_bin"].to_numpy()
    X_calib, y_calib_bin = calib_df[feature_cols], calib_df["_y_bin"].to_numpy()
    X_test,  y_test_bin  = test_df[feature_cols],  test_df["_y_bin"].to_numpy()
    # Keep the original days label for per-day diagnostic reporting at eval time
    y_test_days = test_df["days_until_fire"].to_numpy()


    log.info("==== STEP 4b: compute sample weights (mode %s) ====", imbalance_mode)
    # multi_sat weighting is derived from the FUTURE target date (item 9): it
    # shifts the estimand toward multi-satellite-confirmed fires. Toggleable so
    # an ablation can measure exactly what it changes. Default on (legacy).
    multisat_enabled = os.getenv("MULTISAT_WEIGHTING", "true").lower() in ("1", "true", "yes")
    multi_sat_train = (
        train_df_us["multi_sat_confirmed"].to_numpy()
        if (multisat_enabled and "multi_sat_confirmed" in train_df_us.columns) else None
    )
    if multi_sat_train is not None:
        n2 = int((multi_sat_train >= 2).sum())
        n3 = int((multi_sat_train >= 3).sum())
        log.info("  Multi-satellite confirmed positives: 2-sat=%d  3-sat=%d", n2, n3)
    # Class-frequency weighting only in legacy mode D — modes A/B/C apply class
    # balancing exactly once (none / scale_pos_weight / undersampling).
    sample_weight_train = _compute_sample_weights(
        pd.Series(y_train_bin), train_df_us["date"], multi_sat=multi_sat_train,
        class_weighting=(imbalance_mode == "D"),
    )
    log.info(
        "Sample weights: min=%.3f max=%.3f mean=%.3f",
        float(sample_weight_train.min()),
        float(sample_weight_train.max()),
        float(sample_weight_train.mean()),
    )

    pos_train = int(y_train_bin.sum())
    neg_train = int(len(y_train_bin) - pos_train)
    if imbalance_mode in ("B", "D"):
        # B: the ONE balancing mechanism. D: legacy stacking (documented above).
        scale_pos_weight = neg_train / max(pos_train, 1)
    else:
        scale_pos_weight = 1.0
    log.info(
        "scale_pos_weight = %.3f  (mode %s; pos=%d, neg=%d in train frame)",
        scale_pos_weight, imbalance_mode, pos_train, neg_train,
    )

    log.info(
        "==== STEP 5: LightGBM tuning (%s, n_iter=%d, n_splits=%d, purge=%dd) ====",
        "BayesSearchCV" if _BAYES_AVAILABLE else "RandomizedSearchCV",
        n_iter, n_splits, PURGE_DAYS,
    )
    if not _BAYES_AVAILABLE:
        log.warning(
            "scikit-optimize not installed — using RandomizedSearchCV. "
            "Install for true Bayesian search: pip install scikit-optimize"
        )

    # Purged date-grouped CV folds within TRAIN — replaces TimeSeriesSplit,
    # whose gap parameter counts rows, not days. Indices are positional and
    # X_train comes straight from train_df_us, so they line up.
    cv_folds = purged_date_folds(
        train_df_us["date"], n_splits=n_splits, purge_days=PURGE_DAYS,
    )
    estimator = _build_lgbm(random_state=random_state, scale_pos_weight=scale_pos_weight)
    search, search_name = _build_search(
        estimator=estimator,
        n_iter=n_iter,
        cv=cv_folds,
        random_state=random_state,
    )

    t_tune_start = time.time()
    _fit_search(search, X_train, y_train_bin, sample_weight_train)
    t_tune_elapsed = time.time() - t_tune_start
    best_params = dict(search.best_params_)
    log.info("⏱️  Tuning time: %.1f seconds", t_tune_elapsed)
    log.info("Best params: %s", best_params)
    tuning_scoring = getattr(search, "scoring", os.getenv("TUNE_SCORING", "average_precision"))
    log.info("Best CV score (%s): %.4f", tuning_scoring, float(search.best_score_))

    log.info("==== STEP 6: fit ensemble on train, early stopping on val ====")
    t_refit_start = time.time()
    best_model = _refit_ensemble_with_early_stopping(
        best_params=best_params,
        X_fit=X_train,
        y_fit=y_train_bin,
        X_es=X_val,
        y_es=y_val_bin,
        sample_weight=sample_weight_train,
        scale_pos_weight=scale_pos_weight,
        n_ensemble=n_ensemble,
        base_seed=random_state,
        early_stopping_rounds=early_stopping_rounds,
    )
    t_refit_elapsed = time.time() - t_refit_start
    log.info("⏱️  Ensemble refit time: %.1f seconds (%d models)", t_refit_elapsed, n_ensemble)

    # ── STEP 6b: Platt calibration on the CALIB partition ────────────────
    # The ensemble's raw probabilities are not calibrated — "0.7" might mean
    # 45% in practice. Calibration data must satisfy two properties the old
    # code violated: (1) the model never trained on it (the old code refit
    # on train+val and then calibrated on val — in-sample predictions), and
    # (2) it has NATURAL prevalence (the old val was undersampled 4:1, so
    # the sigmoid learned an inflated base rate).
    log.info("==== STEP 6b: probability calibration (Platt scaling on calib) ====")
    calib_raw_proba = best_model._raw_proba(X_calib)
    calib_ece_before = expected_calibration_error(y_calib_bin, calib_raw_proba)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
    calibrator.fit(calib_raw_proba.reshape(-1, 1), y_calib_bin)
    best_model.calibrator = calibrator
    calib_proba = best_model.predict_proba(X_calib)
    calib_ece_after = expected_calibration_error(y_calib_bin, calib_proba)
    log.info(
        "Calibration: ECE on calib %.4f → %.4f (lower is better; both at natural prevalence)",
        calib_ece_before, calib_ece_after,
    )

    # ── STEP 6c: deployment threshold + tier cutoffs from CALIB only ─────
    # The old code searched the F1-optimal threshold on TEST and shipped it,
    # which silently converts the test set into a second validation set.
    # All operating points are now chosen here, locked, and evaluated
    # exactly once on test below.
    log.info("==== STEP 6c: select operating thresholds on calib ====")
    # HIGH (F1-optimal, precision-oriented urgent alert) + WATCH (recall-target
    # monitoring list) are both chosen on calib via the shared, leakage-safe
    # selector in thresholds.py. Season-matched per-month thresholds are also
    # computed when a month has enough positives (falls back to global
    # otherwise). MEDIUM stays as a compatibility alias of WATCH.
    thr_config = thresholds_mod.validate_config()
    log.info("Threshold config: %s", thr_config)
    prob_tier_thresholds = thresholds_mod.select_operational_thresholds(
        y_calib_bin, calib_proba, dates=calib_df["date"],
        group_by_month=True,
    )
    deploy_thr = float(prob_tier_thresholds["HIGH"])
    watch_thr = float(prob_tier_thresholds["WATCH"])
    calib_f1 = prob_tier_thresholds["high_metrics"]["f1"]
    calib_prec_at_thr = prob_tier_thresholds["high_metrics"]["precision"]
    calib_rec_at_thr = prob_tier_thresholds["high_metrics"]["recall"]
    # Backward-compat alias keys some tools (risk_map string) still read.
    prob_tier_thresholds["calib_f1_at_high"] = calib_f1
    prob_tier_thresholds["calib_precision_at_high"] = calib_prec_at_thr
    prob_tier_thresholds["calib_recall_at_high"] = calib_rec_at_thr
    prob_tier_thresholds["calib_recall_target_medium"] = \
        prob_tier_thresholds["watch_target_recall"]
    _wfb = prob_tier_thresholds.get("watch_fallback_reason")
    log.info(
        "Thresholds (calib): HIGH=%.4f (F1=%.3f P=%.3f R=%.3f)  "
        "WATCH=%.4f (target R>=%.0f%%, calib R=%.3f P=%.3f)%s",
        deploy_thr, calib_f1, calib_prec_at_thr, calib_rec_at_thr,
        watch_thr, prob_tier_thresholds["watch_target_recall"] * 100,
        prob_tier_thresholds["watch_metrics"]["recall"],
        prob_tier_thresholds["watch_metrics"]["precision"],
        f" [fallback: {_wfb}]" if _wfb else "",
    )

    log.info(
        "==== STEP 7: FINAL held-out test evaluation (binary fire-in-%dd, natural prevalence, single pass) ====",
        IMMINENT_DAYS,
    )
    # Everything below is measured at the test set's NATURAL base rate with
    # thresholds that were locked on calib. No parameter — threshold, tier
    # cutoff, calibration — is chosen from these numbers.
    test_proba = best_model.predict_proba(X_test)           # calibrated P(fire in next IMMINENT_DAYS d)
    test_pred = best_model.predict(X_test)                  # mapped 1..7 pseudo-days (compat)

    auc = float(roc_auc_score(y_test_bin, test_proba)) if len(np.unique(y_test_bin)) > 1 else 0.0
    ap = float(average_precision_score(y_test_bin, test_proba)) if len(np.unique(y_test_bin)) > 1 else 0.0
    brier = float(brier_score_loss(y_test_bin, test_proba))

    # Metrics at the locked deployment threshold (chosen on calib in STEP 6c)
    deploy_pred = (test_proba >= deploy_thr).astype(int)
    deploy_f1 = float(f1_score(y_test_bin, deploy_pred, zero_division=0))
    deploy_prec = float(precision_score(y_test_bin, deploy_pred, zero_division=0))
    deploy_rec = float(recall_score(y_test_bin, deploy_pred, zero_division=0))
    deploy_acc = float(accuracy_score(y_test_bin, deploy_pred))

    # Reference point at the conventional 0.5 threshold
    test_pred_bin = (test_proba >= 0.5).astype(int)
    acc = float(accuracy_score(y_test_bin, test_pred_bin))
    prec = float(precision_score(y_test_bin, test_pred_bin, zero_division=0))
    rec = float(recall_score(y_test_bin, test_pred_bin, zero_division=0))
    f1 = float(f1_score(y_test_bin, test_pred_bin, zero_division=0))

    # Precision @ K + recall captured by top-K (relative-ranking view — kept
    # distinct from the absolute-probability tiers). Shared implementation.
    n_test = len(test_proba)
    pak_metrics = thresholds_mod.topk_metrics(y_test_bin, test_proba)

    # Full HIGH + WATCH operating-point reports on the untouched test set
    # (thresholds were locked on calib in STEP 6c; measured once here).
    high_test_metrics = thresholds_mod.threshold_metrics(y_test_bin, test_proba, deploy_thr)
    watch_test_metrics = thresholds_mod.threshold_metrics(y_test_bin, test_proba, watch_thr)

    # Multi-target trade-off table measured on TEST (report-only — the cutoffs
    # were chosen on CALIB in STEP 6c, never on test). Gives the owner the real
    # precision/recall cost of catching more fires under this test window's
    # (drifted) prevalence.
    watch_tradeoff_test = []
    for row in prob_tier_thresholds.get("watch_tradeoff", []):
        tm_t = thresholds_mod.threshold_metrics(y_test_bin, test_proba, row["threshold"])
        # Name the CALIB target apart from the REALIZED test metric so no one
        # reads a calibration target as a held-out/production recall.
        watch_tradeoff_test.append({
            "target_recall": row["target_recall"],           # chosen on calib
            "calib_threshold": row["threshold"],             # selected on calib
            "calib_recall": row.get("recall"),               # recall on calib
            "actual_test_recall": tm_t["recall"],            # transferred to test
            "actual_test_precision": tm_t["precision"],
            "actual_test_f1": tm_t["f1"],
            "actual_test_alert_rate": tm_t["alert_rate"],
            "actual_test_confusion_matrix": tm_t["confusion_matrix"],
        })

    # Base-rate drift monitor: calibration prevalence vs the most-recent
    # (held-out) observed prevalence. A DIAGNOSTIC computed after thresholds
    # are locked — it never feeds back into selection. A large ratio explains
    # why a single fixed threshold under-recalls across seasons.
    _calib_prev = float(prob_tier_thresholds.get("calib_prevalence", y_calib_bin.mean()))
    drift_status = thresholds_mod.prevalence_drift(_calib_prev, float(y_test_bin.mean()))
    if drift_status["drift_flag"]:
        log.warning(
            "⚠️  BASE-RATE DRIFT: calib prevalence %.4f vs recent %.4f (%.1fx) — "
            "single-threshold recall is unreliable; WATCH list mitigates this.",
            drift_status["calib_prevalence"], drift_status["recent_prevalence"],
            drift_status["ratio"],
        )

    # Calibration quality on test (measured, not fitted, here)
    test_ece = expected_calibration_error(y_test_bin, test_proba)
    reliability = reliability_bins(y_test_bin, test_proba, n_bins=10)

    # ── Baselines: is the model better than trivial strategies? ──────────
    # (a) persistence — "a cell that burned in the recent past will burn
    #     again": rank by the past-only rolling fire counts.
    # (b) climatology — per-cell positive rate measured on TRAIN dates only.
    baselines: Dict[str, Dict[str, float]] = {}

    def _score_baseline(name: str, score: np.ndarray, binary: np.ndarray) -> None:
        if len(np.unique(y_test_bin)) < 2:
            return
        baselines[name] = {
            "roc_auc": round(float(roc_auc_score(y_test_bin, score)), 4),
            "average_precision": round(float(average_precision_score(y_test_bin, score)), 4),
            "precision": round(float(precision_score(y_test_bin, binary, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test_bin, binary, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test_bin, binary, zero_division=0)), 4),
        }

    for col, bname in (("fire_sum_3d", "persistence_fired_last_3d"),
                       ("fire_sum_7d", "persistence_fired_last_7d")):
        if col in X_test.columns:
            s = X_test[col].to_numpy().astype(float)
            _score_baseline(bname, s, (s > 0).astype(int))

    cell_rate = (
        train_df_us.groupby(["lat_grid", "lon_grid"])["_y_bin"].mean()
        .rename("cell_climatology")
    )
    clim = (
        test_df[["lat_grid", "lon_grid"]]
        .merge(cell_rate, on=["lat_grid", "lon_grid"], how="left")["cell_climatology"]
        .fillna(float(y_train_bin.mean()))
        .to_numpy()
    )
    _score_baseline("cell_climatology_train_rate", clim,
                    (clim >= float(np.median(clim))).astype(int))

    for bname, bm in baselines.items():
        log.info("  Baseline %-28s AUC=%.4f AP=%.4f F1=%.4f",
                 bname, bm["roc_auc"], bm["average_precision"], bm["f1"])

    overall_bin = {
        "roc_auc": round(auc, 4),
        "average_precision": round(ap, 4),
        "brier_score": round(brier, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        # Operating point — locked on calib BEFORE test was touched.
        "threshold_selected_on": "calib",
        "calib_f1_at_threshold": round(calib_f1, 4),
        "deployment_threshold": round(deploy_thr, 4),
        "deployment_precision": round(deploy_prec, 4),
        "deployment_recall": round(deploy_rec, 4),
        "deployment_f1": round(deploy_f1, 4),
        "deployment_accuracy": round(deploy_acc, 4),
        # Full HIGH + WATCH operating-point reports (precision/recall/F1/
        # alert-rate/confusion at each locked threshold) + WATCH cutoff + the
        # base-rate drift diagnostic.
        "watch_threshold": round(watch_thr, 4),
        "high_test_metrics": high_test_metrics,
        "watch_test_metrics": watch_test_metrics,
        "watch_tradeoff_test": watch_tradeoff_test,
        "prevalence_drift": drift_status,
        # Calibration — fitted on calib, measured here on test.
        "ece": round(test_ece, 4),
        "ece_calib_before_calibration": round(calib_ece_before, 4),
        "ece_calib_after_calibration": round(calib_ece_after, 4),
        # Flag persisted so the dashboard's "Calibrated" badge lights up
        # (frontend treats any truthy calibration_method as calibrated).
        "calibration_method": "platt_sigmoid",
        "reliability_bins": reliability,
        "baselines": baselines,
        **pak_metrics,
    }
    log.info("Binary test metrics: %s", overall_bin)

    # ── RADD cross-verification (independent radar confirmation) ────
    radd_verify_stats: dict = {}
    if "radd_cross_verified" in test_df.columns:
        pos_mask = y_test_bin == 1
        n_pos_test = int(pos_mask.sum())
        n_radd_confirmed = int(test_df.loc[pos_mask, "radd_cross_verified"].sum())
        radd_verify_rate = n_radd_confirmed / max(n_pos_test, 1)
        radd_verify_stats = {
            "radd_confirmed_positives": n_radd_confirmed,
            "radd_total_positives": n_pos_test,
            "radd_verification_rate": round(radd_verify_rate, 4),
        }
        log.info(
            "RADD cross-verification: %d / %d FIRMS positives (%.1f%%) confirmed by Sentinel-1 radar",
            n_radd_confirmed, n_pos_test, 100 * radd_verify_rate,
        )
    overall_bin["radd_verification"] = radd_verify_stats

    # ── Pseudo-days metrics (backwards-compat with risk_map.py) ────
    overall = overall_metrics(y_test_days.astype(float), test_pred, horizon=MAX_PREDICTION_DAYS)
    log.info("Pseudo-days metrics (compat): %s", overall)

    # Per-day stats — still useful to see how confidence ranks by actual day.
    # For -1 rows (no fire in 7d) we synthesize "day 0" bucket for reporting.
    y_eval = np.where(y_test_days < 0, 0, y_test_days)
    actual_hotspots_per_day: Dict[int, int] = {}
    for d in range(1, MAX_PREDICTION_DAYS + 1):
        actual_hotspots_per_day[d] = int(np.sum(y_eval == d))
    daily_stats = per_day_stats(
        y_eval.astype(float), test_pred,
        horizon=MAX_PREDICTION_DAYS,
        actual_hotspots_per_day=actual_hotspots_per_day,
    )

    mae_by_day = {d: daily_stats[f"day_{d}"]["mae"] for d in range(1, MAX_PREDICTION_DAYS + 1)
                  if daily_stats[f"day_{d}"]["prediction_count"] > 0}
    best_day = min(mae_by_day, key=mae_by_day.get) if mae_by_day else None
    worst_day = max(mae_by_day, key=mae_by_day.get) if mae_by_day else None

    summary = {
        "task": "binary_fire_in_3d",
        "imminent_days": IMMINENT_DAYS,
        "neg_to_pos_ratio": NEG_TO_POS_RATIO,
        "test_positive_rate": round(float(y_test_bin.mean()), 4),
        # Binary metrics (the real measure of success)
        **overall_bin,
        # Pseudo-days metrics (for downstream compatibility)
        "pseudo_days_mae": overall["mae"],
        "pseudo_days_rmse": overall["rmse"],
        "pseudo_days_acc_within_1": overall["acc_within_1"],
        "best_day": int(best_day) if best_day is not None else None,
        "worst_day": int(worst_day) if worst_day is not None else None,
    }

    # ── Console report ───────────────────────────────────────────
    print()
    print("=" * 76)
    print(f"BINARY CLASSIFICATION: fire within next {IMMINENT_DAYS} days?  (held-out test)")
    print("=" * 76)
    print(f"  Test positives: {int(y_test_bin.sum())} / {n_test} ({y_test_bin.mean()*100:.2f}%  ← NATURAL prevalence)")
    print(f"  ROC-AUC                : {auc:.4f}   ← {'GOOD' if auc >= 0.8 else 'OK' if auc >= 0.7 else 'WEAK'}")
    print(f"  Average Precision (PR) : {ap:.4f}   (random baseline = {y_test_bin.mean():.4f})")
    print(f"  Brier score            : {brier:.4f}   (lower better; ECE={test_ece:.4f})")
    print(f"  @threshold=0.5  →  acc={acc*100:.2f}%  precision={prec*100:.2f}%  recall={rec*100:.2f}%  F1={f1*100:.2f}%")
    print(f"  @deploy={deploy_thr:.3f} (locked on calib) →  F1={deploy_f1*100:.2f}%  precision={deploy_prec*100:.2f}%  recall={deploy_rec*100:.2f}%")
    print(f"  Precision @ top  5% : {pak_metrics['precision_at_top_5pct']*100:.2f}%")
    print(f"  Precision @ top 10% : {pak_metrics['precision_at_top_10pct']*100:.2f}%")
    print(f"  Precision @ top 20% : {pak_metrics['precision_at_top_20pct']*100:.2f}%")
    if baselines:
        print()
        print("  BASELINE COMPARISON (same test period, natural prevalence):")
        for bname, bm in baselines.items():
            print(f"    {bname:<30} AUC={bm['roc_auc']:.4f}  AP={bm['average_precision']:.4f}  F1={bm['f1']:.4f}")
        print(f"    {'MODEL':<30} AUC={auc:.4f}  AP={ap:.4f}  F1={deploy_f1:.4f}")
    print()
    print("=" * 76)
    print("PER-DAY DIAGNOSTIC (pseudo-days = monotone-in-probability)")
    print("=" * 76)
    _print_daily_table(daily_stats)
    print(
        f"\nPseudo-days: MAE={overall['mae']:.3f}  RMSE={overall['rmse']:.3f}  "
        f"Acc±1={overall['acc_within_1']*100:.2f}%"
    )
    if best_day:
        print(f"Best day: {best_day} (MAE={mae_by_day[best_day]:.3f})  |  "
              f"Worst day: {worst_day} (MAE={mae_by_day[worst_day]:.3f})")
    print()

    # Baseline + skill check: a naive "always predict positive-rate" classifier
    train_pos_rate = float(y_train_bin.mean())
    baseline_proba = np.full(n_test, train_pos_rate)
    baseline_auc = (
        float(roc_auc_score(y_test_bin, baseline_proba))
        if len(np.unique(y_test_bin)) > 1 else 0.5
    )
    auc_uplift_pct = 100.0 * (auc - 0.5) / 0.5
    log.info(
        "Baseline (always p=%.3f) AUC=%.4f → model AUC=%.4f, uplift vs random %+.2f%%",
        train_pos_rate, baseline_auc, auc, auc_uplift_pct,
    )
    # Keep these names so downstream metadata still renders
    train_mean = train_pos_rate
    baseline_metrics = {"mae": baseline_auc, "rmse": 0.0, "r2": 0.0,
                        "acc_within_1": 0.0, "acc_exact": 0.0}
    mae_improvement_pct = auc_uplift_pct

    pred_stats = {
        "min": float(np.min(test_pred)),
        "p25": float(np.percentile(test_pred, 25)),
        "median": float(np.median(test_pred)),
        "p75": float(np.percentile(test_pred, 75)),
        "max": float(np.max(test_pred)),
        "std": float(np.std(test_pred)),
    }
    pred_spread = pred_stats["p75"] - pred_stats["p25"]

    log.info("==== STEP 8: prepare candidate artifacts (NOT yet promoted) ====")
    model_path = os.path.join(p["model_dir"], "lgbm_fire_date_model.pkl")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    feature_importance: List[Dict[str, Any]] = []
    if hasattr(best_model, "feature_importances_") and best_model.feature_importances_ is not None:
        importances = list(zip(feature_cols, [float(x) for x in best_model.feature_importances_]))
        importances.sort(key=lambda x: x[1], reverse=True)
        feature_importance = [{"feature": n, "importance": i} for n, i in importances]

    log.info("==== STEP 9: render PNG training report ====")
    report_path = _render_training_report(
        daily_stats=daily_stats,
        overall=overall,
        feature_importance=feature_importance,
        y_test_true=y_eval.astype(float),
        y_test_pred=test_pred,
        out_dir=p["meta_dir"],
        horizon=MAX_PREDICTION_DAYS,
    )
    if report_path:
        print(f"📊 Training report saved to {report_path}")

    total_elapsed = time.time() - t_total_start
    print(f"⏱️  Total training time: {total_elapsed:.1f} seconds")
    log.info("⏱️  Total training time: %.1f seconds", total_elapsed)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "NASA FIRMS VIIRS NRT (real)",
        "earliest_date": dataset_stats["earliest_date"],
        "latest_date": dataset_stats["latest_date"],
        "data_stale_days": int(stale_days),
        "data_is_stale": bool(stale_days > STALE_WARN_DAYS),
        "total_days": dataset_stats["total_days"],
        "max_history_days_applied": int(max_history_days_applied),
        "total_active_cells": dataset_stats["total_active_cells"],
        "training_rows": int(len(train_df_us)),
        "grid_size": grid_size,
        "min_confidence": min_confidence,
        "urban_filter_enabled": urban_filter_enabled,
        "urban_buffer_km": urban_buffer_km,
        "prediction_type": "binary_fire_in_3d",
        "output_semantics": (
            "Calibrated P(≥1 FIRMS hotspot in this cell within the next "
            f"{IMMINENT_DAYS} days). The model does NOT predict an exact fire "
            "date; any per-day display buckets are probability rankings."
        ),
        "prediction_universe": (
            "0.1° cells with ≥1 historical FIRMS detection, entering the "
            "universe from their first detection onward (causal). Cells that "
            "have never burned are outside the universe — the model cannot "
            "flag first-ever fires."
        ),
        "imminent_days": IMMINENT_DAYS,
        "max_prediction_days": MAX_PREDICTION_DAYS,
        "features": feature_cols,
        "feature_count": len(feature_cols),
        "weather_features_used": [c for c in feature_cols if c.startswith(("temp_", "precip_", "wind_", "et0_"))],
        # ── Evaluation-protocol audit trail ──────────────────────────────
        "evaluation_protocol": {
            "split_method": "date_grouped_chronological",
            "purge_days": PURGE_DAYS,
            "split_date_ranges": split.date_ranges,
            "split_prevalence": split_prevalence,
            "natural_prevalence_overall": round(natural_prevalence, 5),
            "censored_tail_rows": int(n_censored),
            "censored_horizon_days": IMMINENT_DAYS,
            "undersampling": f"train only, {NEG_TO_POS_RATIO}:1 neg:pos",
            "imbalance_mode": imbalance_mode,
            "undersample_beta": round(float(undersample_beta), 6),
            "prior_correction_note": (
                "mode C raw-probability prior correction: "
                "p_natural = beta*p / (beta*p + (1 - p)); Platt calibration on "
                "the natural-prevalence calib split also absorbs this shift"),
            "multisat_weighting": bool(multi_sat_train is not None),
            "calibration_fitted_on": "calib (natural prevalence, disjoint from fit data)",
            "threshold_selected_on": "calib",
            "test_evaluations": 1,
        },
        # Legacy days-based tier thresholds (fallback path in risk_map only).
        "urgency_thresholds": dict(DEFAULT_URGENCY_THRESHOLDS),
        "urgency_thresholds_note": (
            "LEGACY fallback. The operative tier cutoffs are "
            "probability_tier_thresholds (absolute calibrated probabilities "
            "selected on the calib split)."
        ),
        # Absolute probability tier cutoffs for risk_map (selected on calib).
        "probability_tier_thresholds": prob_tier_thresholds,
        "best_model": "lightgbm",
        "model": {
            "type": "lightgbm",
            "search_method": search_name,
            "n_iter": n_iter,
            "n_splits": n_splits,
            "cv_method": f"purged_date_folds (whole-date, {PURGE_DAYS}-day purge)",
            "n_ensemble": n_ensemble,
            "early_stopping_rounds": early_stopping_rounds,
            "best_params": best_params,
            # test_metrics is the source the dashboard reads (via risk_map →
            # GeoJSON metadata.metrics). We expose both legacy regression keys
            # AND binary metrics here so the frontend can pick whichever it
            # knows how to render — the `task` field switches the display.
            "test_metrics": {
                "task": "binary_fire_in_3d",
                "imminent_days": IMMINENT_DAYS,
                # Legacy regression keys (now reflect pseudo-days, mostly
                # noise; kept so an old frontend still shows *something*).
                "mae_days": overall["mae"],
                "rmse_days": overall["rmse"],
                "r2": overall["r2"],
                "accuracy_within_1day": overall["acc_within_1"],
                "accuracy_exact": overall["acc_exact"],
                # Binary metrics — the operative measurements.
                "roc_auc": overall_bin["roc_auc"],
                "average_precision": overall_bin["average_precision"],
                "binary_accuracy": overall_bin["accuracy"],
                "precision": overall_bin["precision"],
                "recall": overall_bin["recall"],
                "f1": overall_bin["f1"],
                "brier_score": overall_bin["brier_score"],
                "threshold_selected_on": overall_bin["threshold_selected_on"],
                # Truthy value lights the dashboard's "Calibrated" badge.
                "calibration_method": overall_bin["calibration_method"],
                "baselines": overall_bin["baselines"],
                "precision_at_top_5pct": overall_bin["precision_at_top_5pct"],
                "precision_at_top_10pct": overall_bin["precision_at_top_10pct"],
                "precision_at_top_20pct": overall_bin["precision_at_top_20pct"],
                # Calibration + locked deployment threshold metrics.
                "ece": overall_bin["ece"],
                "deployment_threshold": overall_bin["deployment_threshold"],
                "deployment_precision": overall_bin["deployment_precision"],
                "deployment_recall": overall_bin["deployment_recall"],
                "deployment_f1": overall_bin["deployment_f1"],
                "deployment_accuracy": overall_bin["deployment_accuracy"],
                # WATCH operating point + full HIGH/WATCH reports + drift flag
                # (frontend distinguishes urgent HIGH from the WATCH monitoring
                # list and surfaces the base-rate drift warning).
                "watch_threshold": overall_bin["watch_threshold"],
                "high_test_metrics": overall_bin["high_test_metrics"],
                "watch_test_metrics": overall_bin["watch_test_metrics"],
                "watch_tradeoff_test": overall_bin["watch_tradeoff_test"],
                "prevalence_drift": overall_bin["prevalence_drift"],
                "recall_captured_top_5pct": overall_bin["recall_captured_top_5pct"],
                "recall_captured_top_10pct": overall_bin["recall_captured_top_10pct"],
                "recall_captured_top_20pct": overall_bin["recall_captured_top_20pct"],
                "reliability_bins": overall_bin["reliability_bins"],
                # Test-set baseline for "uplift" framing in the dashboard.
                # A random ranker's precision@K equals the positive rate.
                "test_positive_rate": round(float(y_test_bin.mean()), 4),
                "uplift_at_top_5pct": round(
                    overall_bin["precision_at_top_5pct"] / max(float(y_test_bin.mean()), 1e-9), 3
                ),
                "uplift_at_top_10pct": round(
                    overall_bin["precision_at_top_10pct"] / max(float(y_test_bin.mean()), 1e-9), 3
                ),
                "uplift_at_top_20pct": round(
                    overall_bin["precision_at_top_20pct"] / max(float(y_test_bin.mean()), 1e-9), 3
                ),
            },
            "test_metrics_binary": overall_bin,   # legacy alias (kept for any tool still reading it)
            "baseline_test_metrics": baseline_metrics,
            "baseline_label_mean": train_mean,
            "mae_improvement_over_baseline_pct": round(mae_improvement_pct, 4),
            "skill_check_passed": bool(auc >= 0.65),
            "prediction_distribution_test": pred_stats,
            "predictions_bunched": bool(pred_spread < 0.5),
            "scale_pos_weight": round(scale_pos_weight, 3),
            "neg_to_pos_ratio_train": NEG_TO_POS_RATIO,
            "deployment_threshold": round(deploy_thr, 4),
            "split_date_ranges": split.date_ranges,
            "training_time_seconds": round(total_elapsed, 2),
            "tuning_time_seconds": round(t_tune_elapsed, 2),
            "ensemble_refit_time_seconds": round(t_refit_elapsed, 2),
        },
        "daily_stats": daily_stats,
        "summary": summary,
        "feature_importance_top": feature_importance[:20],
        "training_time_seconds": round(total_elapsed, 2),
    }

    # ==== STEP 9b: quality gates → promote or reject ═════════════════════
    # The trusted production model is replaced ONLY when the candidate
    # passes the gates. A worse (or invalid) candidate is preserved for
    # inspection under models/rejected/ and the incumbent keeps serving.
    log.info("==== STEP 9b: promotion gates ====")
    # With a champion split (CHAMPION_FRACTION > 0) the promotion decision runs
    # ENTIRELY on the champion block: candidate metrics are recomputed there and
    # the incumbent head-to-head scores the same champion rows. The FINAL test
    # numbers above are then reported AFTER — and independent of — this decision.
    if champion_df is not None and len(champion_df):
        X_champ = champion_df[feature_cols]
        y_champ = champion_df["_y_bin"].to_numpy()
        champ_proba = best_model.predict_proba(X_champ)
        gate_auc = float(roc_auc_score(y_champ, champ_proba)) if len(np.unique(y_champ)) > 1 else 0.0
        gate_ap = float(average_precision_score(y_champ, champ_proba)) if len(np.unique(y_champ)) > 1 else 0.0
        gate_brier = float(brier_score_loss(y_champ, champ_proba))
        gate_ece = expected_calibration_error(y_champ, champ_proba)
        gate_watch = thresholds_mod.threshold_metrics(y_champ, champ_proba, watch_thr)
        gate_X, gate_y, gate_prev = X_champ, y_champ, float(y_champ.mean())
        comparison_set = "champion (dedicated promotion split — final test untouched)"
        log.info("Promotion comparison on CHAMPION split: AUC=%.4f AP=%.4f", gate_auc, gate_ap)
    else:
        gate_auc, gate_ap, gate_brier, gate_ece = auc, ap, brier, test_ece
        gate_watch = watch_test_metrics
        gate_X, gate_y, gate_prev = X_test, y_test_bin, float(y_test_bin.mean())
        comparison_set = "held_out_test (champion-selection until CHAMPION_FRACTION > 0)"

    promoted, reason = _evaluate_promotion_gates(
        candidate_auc=gate_auc,
        candidate_ap=gate_ap,
        test_prevalence=gate_prev,
        X_test=gate_X,
        y_test_bin=gate_y,
        feature_cols=feature_cols,
        promoted_path=model_path,
        candidate_brier=gate_brier,
        candidate_ece=gate_ece,
        candidate_watch_recall=gate_watch.get("recall"),
        candidate_watch_precision=gate_watch.get("precision"),
        candidate_alert_rate=gate_watch.get("alert_rate"),
    )
    metadata["promotion"] = {
        "promoted": bool(promoted),
        "reason": reason,
        "model_version": metadata["trained_at"],
        "comparison_set": comparison_set,
        "champion_fraction": champion_fraction,
        "gates": {
            "min_auc": MODEL_GATE_MIN_AUC,
            "min_ap_uplift": MODEL_GATE_MIN_AP_UPLIFT,
            "max_auc_drop_vs_incumbent": MODEL_GATE_MAX_AUC_DROP,
            "max_ece_increase": MODEL_GATE_MAX_ECE_INCREASE,
            "min_watch_recall": MODEL_GATE_MIN_WATCH_RECALL,
            "min_watch_precision": MODEL_GATE_MIN_WATCH_PRECISION,
            "max_alert_rate": MODEL_GATE_MAX_ALERT_RATE,
            "allow_feature_change": MODEL_GATE_ALLOW_FEATURE_CHANGE,
        },
    }
    log.info("%s", reason)

    bundle = {"format": "bundle-v1", "model": best_model, "metadata": metadata}
    meta_path = os.path.join(p["meta_dir"], "dataset_info.json")

    if promoted:
        _promote_bundle(bundle, p["model_dir"], model_path)
        log.info("Promoted model bundle → %s (incumbent kept at previous.pkl)", model_path)
        # Display metadata AFTER the bundle swap — the bundle is the
        # authoritative pair; this JSON is for the frontend/docs.
        write_json(metadata, meta_path, indent=2, default=str)
        log.info("Saved metadata → %s", meta_path)
        history_path = os.path.join(p["model_dir"], "history",
                                    f"{timestamp}_lightgbm.pkl")
        write_pickle(bundle, history_path)
    else:
        rejected_dir = os.path.join(p["model_dir"], "rejected")
        os.makedirs(rejected_dir, exist_ok=True)
        rejected_path = os.path.join(rejected_dir, f"{timestamp}_lightgbm.pkl")
        write_pickle(bundle, rejected_path)
        log.warning("Candidate NOT promoted — kept for inspection at %s. "
                    "The trusted model and its metadata are unchanged.",
                    rejected_path)

    if not skip_risk_map and promoted:
        log.info("==== STEP 10: refresh risk map ====")
        # Free the largest training objects before risk_map loads its own
        # ~3.5 GB parquet. `del locals()[name]` is a Python no-op (locals()
        # returns a copy), so we delete each name explicitly then force a GC
        # cycle. This reliably reclaims 4–6 GB on a 22 GB laptop.
        del X_train, X_val, X_calib, X_test
        del y_train_bin, y_val_bin, y_calib_bin, y_test_bin
        del val_df, calib_df, test_df
        del train_df_us
        del sample_weight_train, search, estimator
        gc.collect()
        try:
            from risk_map import run as generate_risk_map
            generate_risk_map()
        except Exception as exc:
            log.warning("risk_map.run() failed: %s", exc)
    elif not promoted:
        log.info("Risk map NOT refreshed — last snapshot from the trusted "
                 "model keeps serving.")

    return metadata


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train fire-date prediction model (LightGBM)")
    p.add_argument("--n-iter", type=int, default=30, help="Bayes/Randomized search iterations (default 30)")
    p.add_argument("--n-splits", type=int, default=5, help="Purged date-grouped CV folds (7-day purge)")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--calib-fraction", type=float, default=0.10)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--grid-size", type=float, default=float(os.getenv("GRID_SIZE", "0.1")))
    p.add_argument("--min-confidence", type=int, default=0)
    p.add_argument("--n-ensemble", type=int, default=10)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--quick", action="store_true",
                   help="Fast iteration: n_iter=12, n_splits=3")
    p.add_argument("--fast", action="store_true",
                   help="Shorter train: n_iter=20, n_splits=3")
    p.add_argument("--skip-risk-map", action="store_true")
    p.add_argument("--predict-only", action="store_true")
    p.add_argument("--max-history-days", type=int, default=-1)
    # `--only` is kept for CLI backwards-compat with run.sh — it is ignored
    # because this build is LightGBM-only.
    p.add_argument("--only", type=str, default="lightgbm",
                   help="Ignored (LightGBM-only build).")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    if args.quick:
        args.n_iter = 12
        args.n_splits = 3
        log.info("Quick mode: n_iter=12, n_splits=3")
    elif args.fast:
        args.n_iter = 20
        args.n_splits = 3
        log.info("Fast mode: n_iter=20, n_splits=3")
    if args.only and args.only.lower() != "lightgbm":
        log.warning("--only=%s ignored (this build is LightGBM-only).", args.only)
    max_hist = args.max_history_days
    if max_hist < 0:
        max_hist = int(os.getenv("MAX_TRAIN_HISTORY_DAYS", "0") or 0)
    main(
        n_iter=args.n_iter,
        n_splits=args.n_splits,
        val_fraction=args.val_fraction,
        calib_fraction=args.calib_fraction,
        test_fraction=args.test_fraction,
        grid_size=args.grid_size,
        min_confidence=args.min_confidence,
        skip_risk_map=args.skip_risk_map,
        predict_only=args.predict_only,
        max_history_days=max_hist,
        n_ensemble=args.n_ensemble,
        early_stopping_rounds=args.early_stopping_rounds,
    )
