"""Single source of truth for turning the binary model into user-facing output.

Every prediction surface — ``risk_map.py`` (which writes the GeoJSON), and the
``/predictions/*``, ``/predict/location`` API endpoints — MUST score through the
functions here so they can never disagree. Before this module existed the API
ran a separate legacy "pseudo-days" path and reported an exact
``predicted_fire_date`` while the GeoJSON reported calibrated probabilities;
the two surfaces contradicted each other.

Honest semantics (do not weaken):
  • The model is a binary classifier. Its only real output is the calibrated
    probability of **at least one FIRMS hotspot in this 0.1° cell within the
    next ``horizon_days`` days** (default 3).
  • It does NOT predict an exact fire date, a day-of-week, or "fire in N days".
    Callers must never convert the probability, or any rank bucket derived
    from it, into a calendar date.
  • It only covers cells with ≥1 historical FIRMS detection (the densified
    universe). Cells outside that universe are unscored, not "no risk".

Risk-tier vocabulary (user-facing): ``High`` / ``Watch`` / ``Background``.
These map directly onto the absolute calibrated-probability cutoffs selected on
the calibration split at training time and persisted as
``probability_tier_thresholds`` — no new thresholds, no retraining.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# User-facing tier labels. High = deployment/alert threshold, Watch = the
# ≥85%-recall watch cutoff, Background = below both.
RISK_TIER_HIGH = "High"
RISK_TIER_WATCH = "Watch"
RISK_TIER_BACKGROUND = "Background"
RISK_TIERS: Tuple[str, str, str] = (RISK_TIER_HIGH, RISK_TIER_WATCH, RISK_TIER_BACKGROUND)

DEFAULT_HORIZON_DAYS = 3

# One canonical sentence describing what the probability means — reused by the
# API context block and the frontend so the wording never drifts.
RISK_DEFINITION = (
    "Calibrated probability of at least one FIRMS hotspot in this 0.1° cell "
    "within the next {horizon} days. Not a prediction of an exact fire date."
)


def tier_cutoffs_from_meta(
    tier_thresholds: Optional[Dict[str, Any]],
) -> Optional[Tuple[float, float]]:
    """Extract ``(high_cutoff, watch_cutoff)`` as calibrated probabilities from a
    persisted ``probability_tier_thresholds`` dict, or None when the artifact
    has no absolute probability cutoffs (legacy model → caller falls back to
    relative ranking and must label tiers as relative, never absolute risk).

    A dict whose HIGH/MEDIUM values exceed 1.0 is a legacy *days*-scale mapping,
    not probabilities, and is rejected here so it can't be misread as risk.
    """
    if not tier_thresholds:
        return None
    if "HIGH" not in tier_thresholds or "MEDIUM" not in tier_thresholds:
        return None
    try:
        high = float(tier_thresholds["HIGH"])
        watch = float(tier_thresholds["MEDIUM"])
    except (TypeError, ValueError):
        return None
    if not (0.0 <= watch <= high <= 1.0):
        return None
    return high, watch


def probability_to_tier(
    prob: float, cutoffs: Optional[Tuple[float, float]]
) -> Optional[str]:
    """Map one calibrated probability to a risk tier. Returns None when no
    absolute cutoffs are available (caller must not present an absolute tier)."""
    if cutoffs is None:
        return None
    high, watch = cutoffs
    if prob >= high:
        return RISK_TIER_HIGH
    if prob >= watch:
        return RISK_TIER_WATCH
    return RISK_TIER_BACKGROUND


def calibrated_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Calibrated P(≥1 hotspot within horizon) for each row of X.

    Uses the ensemble's Platt-calibrated ``predict_proba``. A very old artifact
    without ``predict_proba`` falls back to a rank-only proxy recovered from the
    monotone pseudo-days output — explicitly NOT a calibrated probability, only
    good for ordering, and flagged by the caller as such.
    """
    if hasattr(model, "predict_proba"):
        prob = np.asarray(model.predict_proba(X), dtype=float)
    else:  # legacy fallback — rank proxy only
        prob = 1.0 - (np.asarray(model.predict(X), dtype=float) - 1.0) / 6.0
    return np.clip(prob, 0.0, 1.0)


def score_cells(
    model: Any,
    cells: pd.DataFrame,
    feature_cols: List[str],
    cutoffs: Optional[Tuple[float, float]],
) -> pd.DataFrame:
    """Score a frame of candidate cells. Returns a NEW frame carrying only the
    identity columns plus ``probability`` and ``risk_tier`` — the canonical
    per-cell output every surface shares. Missing features are zero-filled
    exactly as at training input time.
    """
    missing = [c for c in feature_cols if c not in cells.columns]
    if missing:
        raise KeyError(
            f"score_cells: {len(missing)} feature column(s) absent from input, "
            f"e.g. {missing[:5]}. The feature contract and the frame disagree."
        )
    X = cells[feature_cols].fillna(0)
    prob = calibrated_probability(model, X)
    id_cols = [c for c in ("lat_grid", "lon_grid") if c in cells.columns]
    out = cells[id_cols].copy()
    out["probability"] = prob
    out["risk_tier"] = [probability_to_tier(float(p), cutoffs) for p in prob]
    return out


RANKING_POLICY_VERSION = "1.0"


def assign_operational_ranking(
    df: pd.DataFrame,
    high_threshold: Optional[float] = None,
    watch_top_pct: float = 10.0,
    extended_watch_top_pct: float = 20.0,
    probability_col: str = "probability",
) -> pd.DataFrame:
    """Relative resource-allocation ranking — SEPARATE from the absolute
    calibrated-probability tiers.

    Under severe base-rate drift the absolute WATCH cutoff can leave a whole day
    with zero alerts even though some cells are clearly higher-risk than others.
    Top-k ranking still points crews at the highest-priority cells WITHOUT
    changing any probability or claiming a fire is certain.

    Deterministic: cells are ordered by probability DESC, then lat_grid ASC,
    then lon_grid ASC (stable mergesort) so tied probabilities always break the
    same way and the WATCH/EXTENDED_WATCH counts are reproducible. Returns a NEW
    frame with ``rank`` (1-based), ``percentile`` (fraction from the top), and
    ``ranking_tier``:

      HIGH           — probability ≥ the absolute HIGH threshold (OVERRIDES the
                       relative tiers; a genuinely high-confidence alert)
      WATCH          — within the top ``watch_top_pct`` % by rank
      EXTENDED_WATCH — within the top ``extended_watch_top_pct`` %
      LOW            — everything else

    The ``probability`` column is never modified.
    """
    n = len(df)
    out = df.sort_values(
        by=[probability_col, "lat_grid", "lon_grid"],
        ascending=[False, True, True], kind="mergesort",
    ).reset_index(drop=True)
    if n == 0:
        out["rank"] = []
        out["percentile"] = []
        out["ranking_tier"] = []
        return out
    ranks = np.arange(1, n + 1)
    out["rank"] = ranks
    out["percentile"] = np.round(ranks / n, 6)
    watch_cut = max(1, int(round(n * watch_top_pct / 100.0)))
    ext_cut = max(watch_cut, int(round(n * extended_watch_top_pct / 100.0)))
    prob = out[probability_col].to_numpy(dtype=float)
    tiers = np.where(ranks <= watch_cut, "WATCH",
                     np.where(ranks <= ext_cut, "EXTENDED_WATCH", "LOW"))
    if high_threshold is not None:
        tiers = np.where(prob >= float(high_threshold), "HIGH", tiers)
    out["ranking_tier"] = tiers
    return out


def ranking_policy_metadata(
    high_threshold: Optional[float],
    watch_top_pct: float = 10.0,
    extended_watch_top_pct: float = 20.0,
    historical_coverage: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """The ranking-policy contract embedded in the GeoJSON/metadata so clients
    render Top-k identically and never confuse it with a calibrated alert."""
    return {
        "type": "relative_resource_allocation",
        "policy_version": RANKING_POLICY_VERSION,
        "watch_top_pct": watch_top_pct,
        "extended_watch_top_pct": extended_watch_top_pct,
        "absolute_high_threshold": high_threshold,
        "tie_breaking": ["probability_desc", "lat_grid_asc", "lon_grid_asc"],
        "note": "Relative priority ranking for resource allocation — NOT an "
                "absolute calibrated-probability alert and not a guarantee a "
                "fire will occur. HIGH (absolute threshold) overrides Top-k.",
        "historical_coverage_note": (
            "held-out HISTORICAL captured-recall estimates; not a guarantee for "
            "any single day"),
        "historical_coverage": historical_coverage or {},
    }


def snapshot_context(
    meta: Optional[Dict[str, Any]],
    base_date: Any,
    model_version: Optional[str],
    cutoffs: Optional[Tuple[float, float]],
) -> Dict[str, Any]:
    """The shared context block every prediction response embeds so clients can
    render tiers and wording identically regardless of which endpoint they hit.
    """
    horizon = int((meta or {}).get("imminent_days", DEFAULT_HORIZON_DAYS))
    tiers: Dict[str, Any] = {"absolute": cutoffs is not None}
    if cutoffs is not None:
        high, watch = cutoffs
        tiers["High"] = round(high, 4)
        tiers["Watch"] = round(watch, 4)
        tiers["Background"] = 0.0
    return {
        "model_version": model_version,
        "base_date": str(base_date) if base_date is not None else None,
        "horizon_days": horizon,
        "risk_definition": RISK_DEFINITION.format(horizon=horizon),
        "risk_tiers": tiers,
        "coverage_note": (
            "Only cells with ≥1 historical FIRMS detection are scored; "
            "unscored cells are outside the model's universe, not zero-risk."
        ),
        "output_semantics": "binary_fire_within_horizon",
    }
