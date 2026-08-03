"""Operational alert-threshold selection — leakage-safe and drift-aware.

Two operating points, BOTH chosen on the calibration split only (never the
held-out test set):

  HIGH  — F1-optimal cutoff. Precision-oriented "urgent alert".
  WATCH — lowest cutoff still catching >= WATCH_TARGET_RECALL of calibration
          fires (recall-oriented "monitoring list"), with an optional minimum
          precision guard so a recall target can't drag precision to zero.

``MEDIUM`` is kept as an API/UI compatibility ALIAS of WATCH (same value) —
older frontends and GeoJSON snapshots key off ``MEDIUM``; new code should read
``WATCH``. LOW = 0.0 = no operational alert.

Why two thresholds: the model is well-ranked (ROC-AUC ~0.9) but the base rate
swings hard between fire season and off-season. A single F1-optimal threshold
picked on a high-prevalence calibration window catches almost nothing when the
test/deployment window is low-prevalence (recall collapses to ~0.13). A
recall-targeted WATCH cutoff is robust to that drift; HIGH stays for precision.

Everything here is a pure function over arrays / a calibration frame so it is
unit-testable without training a model. ``rolling_backtest.py`` and
``train.py`` both import these — no duplicated threshold logic.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

log = logging.getLogger("thresholds")

DEFAULT_WATCH_TARGET_RECALL = 0.90
# Menu of recall targets for the operator trade-off table. High base-rate drift
# means the single WATCH cutoff can miss most fires on a low-prevalence day, so
# we also publish lower-recall operating points the owner can choose by cost.
DEFAULT_WATCH_RECALL_TARGETS = (0.50, 0.70, 0.80, 0.90)
# Below this many positives a calibration group can't select a stable
# recall-target threshold, so we fall back to the global calibration set.
DEFAULT_MIN_POSITIVES = 20
# Recent-vs-calibration prevalence ratio beyond which we flag base-rate drift.
DEFAULT_DRIFT_RATIO = 3.0


# ─────────────────────────────────────────────────────────────────────────────
# Config (env-driven, validated, safe defaults)
# ─────────────────────────────────────────────────────────────────────────────

def _clamped_float(env: str, default: float, lo: float, hi: float) -> float:
    raw = os.getenv(env)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        v = float(raw)
    except (TypeError, ValueError):
        log.warning("%s=%r is not a number — using default %s", env, raw, default)
        return default
    if not (lo <= v <= hi):
        log.warning("%s=%s out of range [%s, %s] — clamping", env, v, lo, hi)
        return min(max(v, lo), hi)
    return v


def watch_target_recall() -> float:
    """Target recall for the WATCH list. Env WATCH_TARGET_RECALL, default 0.90,
    validated to (0, 1]."""
    v = _clamped_float("WATCH_TARGET_RECALL", DEFAULT_WATCH_TARGET_RECALL, 0.0, 1.0)
    return v if v > 0 else DEFAULT_WATCH_TARGET_RECALL


def watch_recall_targets() -> Tuple[float, ...]:
    """Recall targets for the trade-off table. Env WATCH_RECALL_TARGETS as a
    comma-separated list (e.g. "0.5,0.7,0.8,0.9"); invalid entries are dropped
    and the default menu is used when none are valid."""
    raw = os.getenv("WATCH_RECALL_TARGETS", "").strip()
    if not raw:
        return DEFAULT_WATCH_RECALL_TARGETS
    out = []
    for tok in raw.split(","):
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0.0 < v <= 1.0:
            out.append(round(v, 4))
    return tuple(sorted(set(out))) or DEFAULT_WATCH_RECALL_TARGETS


def watch_min_precision() -> Optional[float]:
    """Optional minimum precision guard for WATCH. Env WATCH_MIN_PRECISION in
    [0, 1]; unset ⇒ None (no guard)."""
    raw = os.getenv("WATCH_MIN_PRECISION")
    if raw is None or str(raw).strip() == "":
        return None
    return _clamped_float("WATCH_MIN_PRECISION", 0.0, 0.0, 1.0)


def drift_ratio_threshold() -> float:
    """Prevalence ratio above which base-rate drift is flagged. Env
    DRIFT_PREVALENCE_RATIO, default 3.0, validated to >= 1."""
    return _clamped_float("DRIFT_PREVALENCE_RATIO", DEFAULT_DRIFT_RATIO, 1.0, 1e6)


def validate_config() -> Dict[str, Any]:
    """Resolve + validate all threshold config into a plain dict (for logging
    / persisting). Never raises — invalid values fall back to safe defaults."""
    return {
        "watch_target_recall": watch_target_recall(),
        "watch_min_precision": watch_min_precision(),
        "drift_prevalence_ratio": drift_ratio_threshold(),
        "min_positives_per_group": DEFAULT_MIN_POSITIVES,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core threshold finders
# ─────────────────────────────────────────────────────────────────────────────

def _validate_inputs(y: np.ndarray, proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Coerce + validate scoring inputs. Raises ValueError on mismatched
    lengths; drops rows with non-finite probability rather than letting NaN/inf
    silently corrupt a threshold. Returns cleaned (y, proba)."""
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    if y.shape[0] != proba.shape[0]:
        raise ValueError(
            f"y and proba length mismatch: {y.shape[0]} vs {proba.shape[0]}")
    finite = np.isfinite(proba)
    if not finite.all():
        log.warning("Dropping %d non-finite probability value(s)",
                    int((~finite).sum()))
        y, proba = y[finite], proba[finite]
    return y.astype(int), proba


def f1_optimal_threshold(y: np.ndarray, proba: np.ndarray) -> float:
    """Threshold maximising F1 on (y, proba). 0.5 when a single class."""
    y, proba = _validate_inputs(y, proba)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, proba)
    if len(thr) == 0:
        return 0.5
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(prec), where=(prec + rec) > 0)
    best = int(np.argmax(f1[:-1]))     # PR curve has len(thr)+1 points
    return float(thr[best])


def recall_target_threshold(
    y: np.ndarray,
    proba: np.ndarray,
    target_recall: float,
    min_precision: Optional[float] = None,
) -> Tuple[float, Optional[str]]:
    """Lowest cutoff catching >= target_recall of positives. If ``min_precision``
    is set, raise the cutoff to the most-recall-preserving point that still
    meets it. Returns (threshold, fallback_reason|None)."""
    y, proba = _validate_inputs(y, proba)
    if len(y) == 0:
        return 0.5, "empty_input"
    if len(np.unique(y)) < 2:
        return 0.5, "single_class"
    order = np.argsort(-proba)
    y_sorted = y[order]
    p_sorted = proba[order]
    total_pos = float(y_sorted.sum())
    if total_pos == 0:
        return 0.5, "no_positives"
    cum_tp = np.cumsum(y_sorted)
    recall = cum_tp / total_pos
    hit = np.argmax(recall >= target_recall)  # first index reaching target
    thr = float(p_sorted[hit])
    reason: Optional[str] = None

    if min_precision is not None:
        # precision at each prefix cutoff = cum_tp / prefix_size
        prefix = np.arange(1, len(y_sorted) + 1)
        precision = cum_tp / prefix
        ok = precision >= min_precision
        ok_at_or_before = ok[: hit + 1]
        if ok_at_or_before.any():
            # keep the LAST index <= hit that still meets precision (highest
            # recall consistent with the precision floor)
            keep = np.where(ok_at_or_before)[0].max()
            thr = float(p_sorted[keep])
            if keep < hit:
                reason = "min_precision_guard"
        else:
            reason = "min_precision_unreachable"
    return thr, reason


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def threshold_metrics(y: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, Any]:
    """Precision / recall / F1 / alert-rate + confusion matrix at a cutoff."""
    y, proba = _validate_inputs(y, proba)
    if len(y) == 0:
        return {"threshold": round(float(thr), 4), "precision": 0.0,
                "recall": 0.0, "f1": 0.0, "alert_rate": 0.0,
                "confusion_matrix": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
                "n": 0, "positives": 0}
    pred = (proba >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n = len(y)
    return {
        "threshold": round(float(thr), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "alert_rate": round(pred.mean(), 4) if n else 0.0,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "n": n,
        "positives": int(y.sum()),
    }


def topk_metrics(
    y: np.ndarray,
    proba: np.ndarray,
    ks: Sequence[float] = (0.05, 0.10, 0.20),
) -> Dict[str, float]:
    """Precision@K and recall captured by the top-K% highest-probability cells.
    Top-K ranking is RELATIVE priority — distinct from the absolute tiers."""
    y, proba = _validate_inputs(y, proba)
    out: Dict[str, float] = {}
    if len(y) == 0:
        for k in ks:
            pct = int(round(k * 100))
            out[f"precision_at_top_{pct}pct"] = 0.0
            out[f"recall_captured_top_{pct}pct"] = 0.0
        return out
    order = np.argsort(-proba)
    total_pos = int(y.sum())
    for k in ks:
        pct = int(round(k * 100))
        n = max(1, int(len(proba) * k))
        top = order[:n]
        tp = int(y[top].sum())
        out[f"precision_at_top_{pct}pct"] = round(tp / n, 4)
        out[f"recall_captured_top_{pct}pct"] = round(
            tp / total_pos, 4) if total_pos else 0.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Drift monitoring
# ─────────────────────────────────────────────────────────────────────────────

def threshold_tradeoff_table(
    y: np.ndarray,
    proba: np.ndarray,
    recall_targets: Optional[Sequence[float]] = None,
    min_precision: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """A menu of operating points: for each target recall, the cutoff that hits
    it plus full metrics (precision / recall / F1 / alert-rate / confusion).

    Chosen ONLY on the arrays passed in (the calibration split) — never on test.
    Lets the system owner pick precision-vs-recall by real cost instead of a
    single fixed WATCH threshold that can miss most fires under base-rate drift.
    """
    y, proba = _validate_inputs(y, proba)
    targets = tuple(recall_targets) if recall_targets else watch_recall_targets()
    rows: List[Dict[str, Any]] = []
    for tr in sorted(set(round(float(t), 4) for t in targets)):
        thr, reason = recall_target_threshold(y, proba, tr, min_precision)
        m = threshold_metrics(y, proba, thr)
        rows.append({"target_recall": tr, **m, "fallback_reason": reason})
    return rows


def select_constrained_threshold(
    y: np.ndarray,
    proba: np.ndarray,
    *,
    target_recall: Optional[float] = None,
    min_precision: Optional[float] = None,
    max_alert_rate: Optional[float] = None,
    min_positives: int = DEFAULT_MIN_POSITIVES,
) -> Dict[str, Any]:
    """Pick a threshold on the CALIBRATION arrays that satisfies ALL of the
    given constraints (recall ≥ target, precision ≥ floor, alert-rate ≤ ceiling)
    simultaneously, preferring the highest precision then recall.

    Returns a STRUCTURED status — it never returns a threshold that pretends an
    unmet target passed:

      {feasible, threshold, metrics, unmet_constraints,
       best_available_operating_point, fallback_reason}

    When constraints conflict, ``feasible`` is False, ``threshold`` is None, and
    ``best_available_operating_point`` reports the closest achievable point with
    the constraints it violates. Selection consumes ONLY the arrays passed in —
    never test data.
    """
    y, proba = _validate_inputs(y, proba)
    pos = int(y.sum())
    if len(y) == 0 or len(np.unique(y)) < 2 or pos < min_positives:
        return {
            "feasible": False, "threshold": None, "metrics": None,
            "unmet_constraints": ["insufficient_data"],
            "best_available_operating_point": None,
            "fallback_reason": (
                f"insufficient calibration data (positives={pos} < "
                f"{min_positives} or single class)"),
        }

    _, _, thr = precision_recall_curve(y, proba)
    grid = np.unique(np.concatenate([np.asarray(thr, dtype=float), [0.0]]))

    def _violations(m: Dict[str, Any]) -> List[str]:
        v = []
        if target_recall is not None and m["recall"] < target_recall - 1e-12:
            v.append("recall")
        if min_precision is not None and m["precision"] < min_precision - 1e-12:
            v.append("precision")
        if max_alert_rate is not None and m["alert_rate"] > max_alert_rate + 1e-12:
            v.append("alert_rate")
        return v

    scored = [(float(t), threshold_metrics(y, proba, t)) for t in grid]
    feasible = [(t, m) for t, m in scored if not _violations(m)]
    if feasible:
        # highest precision, then highest recall, then higher threshold (fewer
        # alerts) — deterministic.
        t, m = max(feasible, key=lambda c: (c[1]["precision"], c[1]["recall"], c[0]))
        return {"feasible": True, "threshold": round(t, 4), "metrics": m,
                "unmet_constraints": [], "best_available_operating_point": m,
                "fallback_reason": None}

    # Infeasible: report the best-effort point (max F1) + which constraints it
    # still violates. NEVER claim the target was met.
    t, m = max(scored, key=lambda c: c[1]["f1"])
    return {
        "feasible": False, "threshold": None, "metrics": None,
        "unmet_constraints": _violations(m),
        "best_available_operating_point": {"threshold": round(t, 4), **m},
        "fallback_reason": "no threshold satisfies all constraints simultaneously",
    }


def prevalence_drift(
    calib_prevalence: float,
    recent_prevalence: float,
    ratio_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare recent observed prevalence with the calibration prevalence.
    Flags drift when they differ by more than ``ratio_threshold`` in either
    direction — the condition that wrecks a single fixed threshold."""
    rt = ratio_threshold if ratio_threshold is not None else drift_ratio_threshold()
    a = max(float(calib_prevalence), 1e-9)
    b = max(float(recent_prevalence), 1e-9)
    ratio = max(a / b, b / a)
    return {
        "calib_prevalence": round(float(calib_prevalence), 4),
        "recent_prevalence": round(float(recent_prevalence), 4),
        "ratio": round(ratio, 2),
        "ratio_threshold": rt,
        "drift_flag": bool(ratio >= rt),
        "direction": ("calibration_higher" if a > b else "recent_higher"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full operating-point selection (with seasonal grouping + fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _one_group_thresholds(
    y: np.ndarray,
    proba: np.ndarray,
    target_recall: float,
    min_precision: Optional[float],
) -> Dict[str, Any]:
    """HIGH + WATCH for a single (already-selected) calibration group, with
    full calibration metrics attached. Enforces HIGH > WATCH >= 0."""
    high = f1_optimal_threshold(y, proba)
    watch, watch_fallback = recall_target_threshold(
        y, proba, target_recall, min_precision)

    high_r = round(float(high), 4)
    watch_r = round(float(watch), 4)
    watch_available = True
    watch_disabled_reason = None

    # Enforce 1 >= HIGH > WATCH >= 0 AFTER rounding (ties/collisions included).
    if watch_r >= high_r:
        # Nudge WATCH one rounding step below HIGH so the invariant is strict.
        watch_r = round(high_r - 0.0001, 4)
        watch_fallback = watch_fallback or "clamped_below_high"
    if watch_r < 0.0:
        watch_r = 0.0
    # HIGH=0 leaves no strictly-lower non-negative WATCH → disable WATCH rather
    # than deploy a degenerate cutoff.
    if high_r <= 0.0:
        watch_available = False
        watch_disabled_reason = "high_threshold_zero"
        watch_r = 0.0
    # An unreachable minimum-precision guard must NOT silently deploy — mark
    # WATCH unavailable so serving can fall back to a safe state.
    if min_precision is not None and watch_fallback == "min_precision_unreachable":
        watch_available = False
        watch_disabled_reason = "min_precision_unreachable"

    wm = threshold_metrics(y, proba, watch_r)
    # If a precision guard is enabled and reachable, the resulting WATCH must
    # actually meet it — otherwise disable rather than mislead.
    if (watch_available and min_precision is not None
            and wm["precision"] + 1e-9 < min_precision):
        watch_available = False
        watch_disabled_reason = "min_precision_not_met"

    return {
        "HIGH": high_r,
        "WATCH": watch_r,
        "MEDIUM": watch_r,                  # compatibility alias of WATCH
        "LOW": 0.0,
        "watch_available": watch_available,
        "watch_disabled_reason": watch_disabled_reason,
        "high_metrics": threshold_metrics(y, proba, high_r),
        "watch_metrics": wm,
        "watch_target_recall": round(float(target_recall), 4),
        "watch_min_precision": min_precision,
        "watch_fallback_reason": watch_fallback,
    }


def select_operational_thresholds(
    y: np.ndarray,
    proba: np.ndarray,
    dates: Optional[Sequence] = None,
    *,
    target_recall: Optional[float] = None,
    min_precision: Optional[float] = None,
    group_by_month: bool = False,
    min_positives: int = DEFAULT_MIN_POSITIVES,
) -> Dict[str, Any]:
    """Select HIGH + WATCH on a calibration set (never test).

    When ``group_by_month`` and ``dates`` are provided, a per-month threshold is
    also produced for each month with >= ``min_positives`` positives — the
    season-matched path that resists base-rate drift. Any month below the floor
    falls back to the global thresholds with a recorded reason. The global
    (all-calibration) thresholds are always present as the safe default.

    Returns a metadata dict ready to persist in ``dataset_info.json``: absolute
    cutoffs, target recall, per-threshold calibration precision/recall/F1/
    prevalence/date-range/sample-size, selection method, and fallback reasons.
    """
    y, proba = _validate_inputs(y, proba)
    tr = target_recall if target_recall is not None else watch_target_recall()
    mp = min_precision if min_precision is not None else watch_min_precision()

    base = _one_group_thresholds(y, proba, tr, mp)
    n = len(y)
    pos = int(y.sum())
    date_range = None
    if dates is not None and len(dates):
        ds = pd.to_datetime(pd.Series(dates))
        date_range = [str(ds.min().date()), str(ds.max().date())]

    meta: Dict[str, Any] = {
        **base,
        "selected_on": "calib",
        "selection_method": "f1_optimal_high + recall_target_watch",
        "calib_prevalence": round(pos / n, 4) if n else 0.0,
        "calib_sample_size": n,
        "calib_positives": pos,
        "calib_date_range": date_range,
        # Operator trade-off menu (thresholds from calib; multiple recall
        # targets). These are ALTERNATIVE operating points, not applied to
        # serving unless the owner promotes one — HIGH/WATCH above stay default.
        "watch_tradeoff": threshold_tradeoff_table(y, proba, min_precision=mp),
    }

    if group_by_month and dates is not None and len(dates):
        months = pd.to_datetime(pd.Series(dates)).dt.month.to_numpy()
        seasonal: Dict[str, Any] = {}
        for m in sorted(set(months.tolist())):
            mask = months == m
            gy, gp = y[mask], proba[mask]
            gpos = int(gy.sum())
            if gpos < min_positives or len(np.unique(gy)) < 2:
                seasonal[str(int(m))] = {
                    "fallback": "insufficient_positives",
                    "positives": gpos,
                    "min_positives": min_positives,
                    "HIGH": base["HIGH"], "WATCH": base["WATCH"],
                }
                continue
            g = _one_group_thresholds(gy, gp, tr, mp)
            seasonal[str(int(m))] = {
                "fallback": None,
                "positives": gpos,
                "sample_size": int(mask.sum()),
                "HIGH": g["HIGH"], "WATCH": g["WATCH"],
                "watch_fallback_reason": g["watch_fallback_reason"],
            }
        # Count how many distinct calendar months carry their OWN (non-fallback)
        # threshold and how many independent dates back them. Seasonal
        # thresholds are only *applied* at serving time when this is genuinely
        # representative — a single-season calibration window (e.g. Apr–May)
        # cannot support month-specific claims. Until then they are DESCRIPTIVE
        # ONLY and serving uses the global WATCH/HIGH (Option B).
        own = [k for k, v in seasonal.items() if v.get("fallback") is None]
        n_indep_dates = int(pd.to_datetime(pd.Series(dates)).dt.normalize().nunique())
        distinct_months = int(len(set(months.tolist())))
        applied = len(own) >= 6 and n_indep_dates >= 60 and distinct_months >= 6
        meta["seasonal_thresholds"] = seasonal
        meta["seasonal_thresholds_applied"] = bool(applied)
        meta["seasonal_thresholds_status"] = (
            "applied" if applied else "descriptive_only")
        meta["seasonal_thresholds_reason"] = (
            None if applied else
            f"insufficient seasonal coverage: {len(own)} month(s) with own "
            f"threshold, {distinct_months} distinct month(s), "
            f"{n_indep_dates} independent calibration date(s) — need >=6 "
            f"months and >=60 dates. Serving uses the global WATCH/HIGH.")
        meta["independent_calib_dates"] = n_indep_dates
        if applied:
            meta["selection_method"] += " (+ applied per-month thresholds)"
        else:
            meta["selection_method"] += " (per-month DESCRIPTIVE ONLY, not applied)"

    # Invariant guard — HIGH must dominate WATCH which must be non-negative.
    assert meta["HIGH"] >= meta["WATCH"] >= 0.0, (
        f"threshold ordering violated: HIGH={meta['HIGH']} WATCH={meta['WATCH']}")
    return meta
