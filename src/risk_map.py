# =========================================================
# FIRE DATE MAP GENERATION (OBSERVED + PREDICTED DATES)
# =========================================================
#
# All values written to fire_dates_all.geojson are derived from REAL data:
#   • observed:   NASA FIRMS VIIRS NRT detections (latest densified day)
#   • predicted:  model output on real features for the latest base date
#   • historical_fire_count_30d: literal sum of FIRMS detections in this
#                                grid cell over the 30 days ending at base_date
#   • urgency_level: derived via thresholds calibrated on real validation
#                    predictions (dataset_info.json["urgency_thresholds"])
# =========================================================

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

from features import (
    FEATURES_CORE,
    FEATURES_WEATHER,
    MAX_PREDICTION_DAYS,
    DEFAULT_URGENCY_THRESHOLDS,
    calibrate_urgency_thresholds,
    urgency_from_thresholds,
)
from storage import exists, read_json, read_pickle, read_table, write_json
from urban_areas import THAI_URBAN_AREAS, classify_urban
from thailand_boundary import is_in_thailand, find_province

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Honor OUTPUT_DIR like train.py / api.py — previously this module hardcoded
# BASE_DIR/outputs, so under a custom OUTPUT_DIR (Railway volume at
# /data/outputs) it silently read a stale model from the repo directory
# while train.py wrote the fresh one elsewhere.
_OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")
if not os.path.isabs(_OUTPUT_DIR):
    _OUTPUT_DIR = os.path.join(BASE_DIR, _OUTPUT_DIR)

MODEL_PATH = os.path.join(_OUTPUT_DIR, "models", "lgbm_fire_date_model.pkl")
DATA_PATH  = os.path.join(_OUTPUT_DIR, "features", "full_features.parquet")
META_PATH  = os.path.join(_OUTPUT_DIR, "metadata", "dataset_info.json")

RISKMAP_DIR  = os.path.join(_OUTPUT_DIR, "riskmap")
GEOJSON_PATH = os.path.join(RISKMAP_DIR, "fire_dates_all.geojson")
LATEST_PATH  = os.path.join(RISKMAP_DIR, "latest.json")

# NRT FIRMS cache — its max acq_datetime is the true "data as of" instant
# (finer than the daily base_date the model predicts from).
_FIRMS_PATH = os.getenv("FIRMS_PATH") or os.path.join(BASE_DIR, "data", "firms", "firms_all.parquet")
if not os.path.isabs(_FIRMS_PATH):
    _FIRMS_PATH = os.path.join(BASE_DIR, _FIRMS_PATH)


def _latest_data_time() -> Optional[str]:
    """Max FIRMS acquisition timestamp as a UTC ISO string — the real moment
    the freshest hotspot was observed, finer than the daily base_date. FIRMS
    acq times are UTC; a naive timestamp is localized to UTC so the frontend
    can convert to the viewer's zone. Returns None if unavailable."""
    try:
        if not os.path.exists(_FIRMS_PATH):
            return None
        col = pd.read_parquet(_FIRMS_PATH, columns=["acq_datetime"])["acq_datetime"]
        latest = pd.to_datetime(col, errors="coerce").max()
        if pd.isna(latest):
            return None
        ts = pd.Timestamp(latest)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.isoformat()
    except Exception:
        return None

os.makedirs(RISKMAP_DIR, exist_ok=True)

HISTORY_WINDOW_DAYS = 30

# Short-window filter: cells must have ≥ N fires in the last HISTORY_WINDOW_DAYS
# days. Catches "recently active" cells. Default 3 (was 1 — at 1 the dashboard
# showed ~975 Thailand cells, vs GISTDA's ~74 hotspots/day; bumping to 3
# brings the displayed scale closer to ground truth).
MIN_HISTORICAL_FIRES_FOR_DISPLAY = int(os.getenv("MIN_HISTORICAL_FIRES_FOR_DISPLAY", "3"))

# Long-window filter: cells must have ≥ N fires in the last
# LONG_HISTORY_WINDOW_DAYS days. Catches "structurally fire-prone" cells.
# This is the strongest signal you have for "this place actually burns" — a
# cell that's burned 0–2 times in 90 days is almost certainly not going to
# burn next week, regardless of what the model predicts. Default 3 fires in
# 90 days = "this place burns at least once per month on average".
# Set MIN_LONG_HISTORICAL_FIRES=0 to disable the long-window filter.
LONG_HISTORY_WINDOW_DAYS  = int(os.getenv("LONG_HISTORY_WINDOW_DAYS",  "90"))
MIN_LONG_HISTORICAL_FIRES = int(os.getenv("MIN_LONG_HISTORICAL_FIRES", "3"))

# Climatological filter: annualized fire-days per cell, computed across the
# ENTIRE available data record (not a fixed window). Captures the empirical
# base rate of fire activity for each location — the "is this place
# structurally fire-prone?" question that count-based windows can't answer
# cleanly. A cell with 0.5 fire-days/year is essentially never on fire; a
# cell with 5+ fire-days/year is a known hotspot. Default cutoff of 1.0
# fire-day/year is conservative — bump higher to be more selective.
# Set to 0 to disable.
MIN_FIRE_DAYS_PER_YEAR = float(os.getenv("MIN_FIRE_DAYS_PER_YEAR", "3.0"))

# Per-day cap: regression model outputs concentrate predictions in 2-3 day
# bucket (model bunching), so without a cap the dashboard would show ~500
# cells "all firing on day 2" — wildly above GISTDA's observed ~74/day.
# We rank cells within each predicted day by historical_fire_count_30d
# (cells already actively burning are most likely to keep burning) and keep
# only the top MAX_CELLS_PER_DAY. Default 100 ≈ GISTDA scale + headroom.
# Set to 0 to disable.
MAX_CELLS_PER_DAY = int(os.getenv("MAX_CELLS_PER_DAY", "100"))

# Urban-exclusion filter: drop predictions that fall inside (or near) major
# Thai cities. FIRMS detects garbage burning, industrial heat, and other
# non-wildfire heat sources in cities; predicting "wildfire in central
# Bangkok" is misleading. The curated list of urban areas + radii is in
# urban_areas.py — edit it to add/remove cities or adjust their footprints.
#
#   URBAN_FILTER_ENABLED  — set to "false" to disable the filter entirely
#   URBAN_BUFFER_KM       — extra km added beyond each city's radius. 0 =
#                           drop only cells inside the hand-tuned radius;
#                           5–10 = also drop suburban edge cells.
URBAN_FILTER_ENABLED = os.getenv("URBAN_FILTER_ENABLED", "true").lower() in ("1", "true", "yes")
URBAN_BUFFER_KM      = float(os.getenv("URBAN_BUFFER_KM", "0.0"))

# Country filter: drop cells that fall outside Thailand's land border.
# The FIRMS BBOX is rectangular and necessarily includes Myanmar, Laos,
# Cambodia, Vietnam, and northern Malaysia — useful for the model to learn
# regional fire patterns, but the dashboard is Thailand-focused. Default-on.
COUNTRY_FILTER_ENABLED = os.getenv("COUNTRY_FILTER_ENABLED", "true").lower() in ("1", "true", "yes")


# =========================================================
# METADATA HELPERS
# =========================================================

# Metadata embedded in the promoted model bundle (set by load_assets).
# Authoritative over dataset_info.json — the JSON can lag the bundle if a
# crash lands between the two writes; the bundle can't tear.
_BUNDLE_METADATA: Optional[dict] = None


def _load_metadata() -> dict:
    if _BUNDLE_METADATA is not None:
        return _BUNDLE_METADATA
    if not exists(META_PATH):
        return {}
    try:
        return read_json(META_PATH)
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_features(meta: dict, df: pd.DataFrame) -> list:
    """Prefer the persisted feature list (matches the deployed model exactly)."""
    feats = meta.get("features")
    if feats:
        return list(feats)
    # Fall back: any core/weather column that's actually present.
    return [c for c in (*FEATURES_CORE, *FEATURES_WEATHER) if c in df.columns]


def _resolve_thresholds(meta: dict) -> dict:
    t = meta.get("urgency_thresholds")
    if isinstance(t, dict) and {"CRITICAL", "HIGH", "MEDIUM", "LOW"} <= set(t):
        return {k: float(v) for k, v in t.items()}
    return dict(DEFAULT_URGENCY_THRESHOLDS)


# =========================================================
# LOAD MODEL & DATA
# =========================================================

def load_assets():
    """Load model + features parquet — memory-frugal.

    The full features parquet is ~200MB on disk but `pd.read_parquet()`
    expands to ~6GB in memory as float64, and even pyarrow's `to_pandas()`
    peaks much higher before downcasting. On a 22GB laptop with the browser
    + IDE open this OOM-kills the process during STEP 10 of train.py.

    Fix: iterate parquet row-groups, convert each to a pandas chunk with
    float32 already applied at the pyarrow→pandas boundary, then concat.
    Peak RAM stays under 2GB regardless of total size.
    """
    # Canonical loader (train.load_model_bundle): registers the pickled classes
    # in __main__ and unwraps bundle-v1. Stash embedded metadata so
    # _load_metadata() prefers it over dataset_info.json.
    import train as _train
    model, _meta = _train.load_model_bundle(MODEL_PATH)
    global _BUNDLE_METADATA
    _BUNDLE_METADATA = _meta if _meta is not None else None

    import pyarrow as pa
    import pyarrow.parquet as pq
    import gc

    pf = pq.ParquetFile(DATA_PATH)
    schema = pf.schema_arrow
    # Build a downcast schema: float64 → float32, int64 → int32, except for
    # lat/lon/date which we keep in their original precision.
    keep_original = {"lat_grid", "lon_grid", "date"}
    new_fields = []
    for f in schema:
        if f.name in keep_original:
            new_fields.append(f)
        elif pa.types.is_floating(f.type) and f.type != pa.float32():
            new_fields.append(pa.field(f.name, pa.float32(), nullable=f.nullable))
        elif pa.types.is_signed_integer(f.type) and f.type.bit_width >= 64:
            new_fields.append(pa.field(f.name, pa.int32(), nullable=f.nullable))
        else:
            new_fields.append(f)
    target_schema = pa.schema(new_fields)

    chunks = []
    for batch in pf.iter_batches(batch_size=200_000):
        # Cast each column in-place at Arrow level — much cheaper than
        # round-tripping through pandas float64.
        cast_cols = {}
        for i, field in enumerate(batch.schema):
            col = batch.column(i)
            target = target_schema.field(field.name).type
            if col.type != target:
                cast_cols[field.name] = col.cast(target)
            else:
                cast_cols[field.name] = col
        small_batch = pa.RecordBatch.from_arrays(
            list(cast_cols.values()), names=list(cast_cols.keys())
        )
        chunks.append(small_batch.to_pandas())

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    df["date"] = pd.to_datetime(df["date"]).dt.date
    return model, df


# =========================================================
# OBSERVED LAYER (latest real day from FIRMS)
# =========================================================

def build_observed(df: pd.DataFrame):
    observed_date = df["date"].max()
    obs = df[df["date"] == observed_date].copy()
    return obs, observed_date


# =========================================================
# HISTORICAL FIRE COUNT (real FIRMS detections in last N days)
# =========================================================

def historical_fire_counts(
    df: pd.DataFrame,
    base_date,
    window_days: int = HISTORY_WINDOW_DAYS,
    column_name: str = "historical_fire_count_30d",
) -> pd.DataFrame:
    """Real per-cell sum of fire_count over [base_date-window, base_date].

    No imputation, no extrapolation — just a groupby on the densified FIRMS
    frame. Cells with no detections in the window get exactly 0.

    Args:
        column_name: Name of the output count column. Default matches the
            legacy "historical_fire_count_30d" so existing callers keep working.
    """
    start = base_date - timedelta(days=window_days)
    window = df[(df["date"] >= start) & (df["date"] <= base_date)]
    counts = (
        window.groupby(["lat_grid", "lon_grid"], as_index=False)["fire_count"]
        .sum()
        .rename(columns={"fire_count": column_name})
    )
    counts[column_name] = counts[column_name].astype(int)
    return counts


def historical_fire_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell annualized fire frequency, computed across the full record.

    For each cell we count the number of distinct *fire-days* (days the cell
    had at least one FIRMS detection), then annualize:

        fire_days_per_year = fire_days_total × 365.25 / days_in_record

    Why fire-days instead of total detections? A single intense fire can
    produce dozens of detections in one day — those should count as one
    "fire event," not dozens. Fire-days is a more honest proxy for "how
    often does this cell actually burn?"

    Returns a frame with columns:
        lat_grid, lon_grid, fire_days_total, fire_days_per_year
    """
    if df.empty:
        return pd.DataFrame(columns=["lat_grid", "lon_grid",
                                     "fire_days_total", "fire_days_per_year"])
    days_in_record = (df["date"].max() - df["date"].min()).days + 1
    if days_in_record < 30:
        # Too short to estimate a yearly rate reliably; mark every cell as
        # "rate unknown" with NaN so downstream filtering can fall back to
        # the count-based filters.
        rate = (
            df.groupby(["lat_grid", "lon_grid"])["fire_count"]
            .apply(lambda x: int((x > 0).sum()))
            .reset_index(name="fire_days_total")
        )
        rate["fire_days_per_year"] = float("nan")
        return rate

    rate = (
        df.groupby(["lat_grid", "lon_grid"])["fire_count"]
        .apply(lambda x: int((x > 0).sum()))
        .reset_index(name="fire_days_total")
    )
    rate["fire_days_per_year"] = (
        rate["fire_days_total"] * 365.25 / days_in_record
    )
    return rate


# =========================================================
# PREDICTED LAYER
# =========================================================

def build_predicted(df: pd.DataFrame, model, base_date, meta: dict):
    base = df[df["date"] == base_date].copy()

    feature_cols = _resolve_features(meta, df)
    missing = [c for c in feature_cols if c not in base.columns]
    if missing:
        raise RuntimeError(
            f"Feature CSV is missing {len(missing)} columns expected by the model: "
            f"{missing[:5]}{'…' if len(missing) > 5 else ''}. Re-run train.py."
        )
    X = base[feature_cols].fillna(0)

    # ── PRIMARY OUTPUT: calibrated P(≥1 hotspot within IMMINENT_DAYS) ──────
    # The model is a binary "fire within 3 days" classifier. It does NOT
    # know whether a fire lands on day 1, 2 or 3 — so this module must never
    # manufacture an exact fire date from it. `probability` is the real,
    # calibrated model output; everything else below is display scaffolding.
    # Single source of truth: the API's /predictions/* and /predict/location
    # score through this exact function too, so the GeoJSON and the API can
    # never report a different probability for the same cell.
    import inference
    base["probability"] = inference.calibrated_probability(model, X)

    raw_pred = model.predict(X)   # monotone pseudo-days (compat / display bins)
    # Floor into integer bins. These bins are PROBABILITY-RANK buckets that
    # the dashboard's day selector groups by — bin 0 = highest probability
    # band, bin 7 = lowest. They are not date forecasts.
    days_floored = np.clip(np.floor(raw_pred), 0, MAX_PREDICTION_DAYS).astype(int)

    base["raw_prediction"] = raw_pred
    base["days_until_fire"] = days_floored
    # Honest date semantics: the ONLY date claim this model supports is
    # "within the next imminent_days days". Same window end for every cell.
    imminent_days = int(meta.get("imminent_days", 3))
    risk_window_end = (base_date + timedelta(days=imminent_days)).strftime("%Y-%m-%d")
    base["predicted_fire_date"] = risk_window_end  # kept for frontend/CSV compat
    base["prediction_confidence"] = base["probability"]  # legacy field alias

    # Attach real historical fire counts per cell at TWO time scales:
    #   • 30-day count → "is this cell active right now?"
    #   • long-window count (default 90d) → "is this cell structurally fire-prone?"
    # The long-window count is the strongest filter we have for "this place
    # actually burns" — it cuts cells that the model wants to predict on but
    # which haven't seen meaningful fire activity in months.
    counts_30d = historical_fire_counts(
        df, base_date,
        window_days=HISTORY_WINDOW_DAYS,
        column_name="historical_fire_count_30d",
    )
    long_col = f"historical_fire_count_{LONG_HISTORY_WINDOW_DAYS}d"
    counts_long = historical_fire_counts(
        df, base_date,
        window_days=LONG_HISTORY_WINDOW_DAYS,
        column_name=long_col,
    )
    base = base.merge(counts_30d,  on=["lat_grid", "lon_grid"], how="left")
    base = base.merge(counts_long, on=["lat_grid", "lon_grid"], how="left")
    base["historical_fire_count_30d"] = base["historical_fire_count_30d"].fillna(0).astype(int)
    base[long_col] = base[long_col].fillna(0).astype(int)

    # Climatological fire-rate (full-history annualized). This is the
    # "empirical base rate" filter — answers "does this cell actually
    # burn often enough to be worth predicting on?" Cells the model wants
    # to call CRITICAL but that have never historically burned will be
    # dropped here.
    rate = historical_fire_rate(df)
    base = base.merge(rate, on=["lat_grid", "lon_grid"], how="left")
    base["fire_days_total"] = base["fire_days_total"].fillna(0).astype(int)
    base["fire_days_per_year"] = base["fire_days_per_year"].fillna(0.0)

    # Apply ALL filters in sequence. Cells must pass each one to be displayed.
    n_total = len(base)
    if MIN_HISTORICAL_FIRES_FOR_DISPLAY > 0:
        before = len(base)
        base = base[base["historical_fire_count_30d"] >= MIN_HISTORICAL_FIRES_FOR_DISPLAY].copy()
        print(
            f"Short-window  filter: ≥{MIN_HISTORICAL_FIRES_FOR_DISPLAY} fire(s) in "
            f"last {HISTORY_WINDOW_DAYS}d → {len(base):,} / {before:,} kept "
            f"({len(base)*100/max(before,1):.1f}%)"
        )
    if MIN_LONG_HISTORICAL_FIRES > 0:
        before = len(base)
        base = base[base[long_col] >= MIN_LONG_HISTORICAL_FIRES].copy()
        print(
            f"Long-window   filter: ≥{MIN_LONG_HISTORICAL_FIRES} fire(s) in "
            f"last {LONG_HISTORY_WINDOW_DAYS}d → {len(base):,} / {before:,} kept "
            f"({len(base)*100/max(before,1):.1f}%)"
        )
    if MIN_FIRE_DAYS_PER_YEAR > 0:
        before = len(base)
        # Cells with NaN fire_days_per_year (very short data record) are
        # kept — we only filter when we have enough data to estimate a rate.
        mask = (
            base["fire_days_per_year"].isna()
            | (base["fire_days_per_year"] >= MIN_FIRE_DAYS_PER_YEAR)
        )
        base = base[mask].copy()
        print(
            f"Fire-rate     filter: ≥{MIN_FIRE_DAYS_PER_YEAR} fire-days/year "
            f"(full history) → {len(base):,} / {before:,} kept "
            f"({len(base)*100/max(before,1):.1f}%)"
        )

    # Urban-exclusion: drop cells that fall inside (or near) cities. Run AFTER
    # the cheaper count/rate filters so we do the haversine pass on the
    # smallest possible set. We also annotate every surviving cell with its
    # nearest urban area + distance, surfaced in the popup so users can see
    # context like "5 km from Chiang Mai" alongside the prediction.
    is_urban, urban_dist, urban_name = classify_urban(
        base["lat_grid"].to_numpy(),
        base["lon_grid"].to_numpy(),
        urban_areas=THAI_URBAN_AREAS,
        buffer_km=URBAN_BUFFER_KM,
    )
    base["nearest_urban_area"] = urban_name
    base["nearest_urban_distance_km"] = urban_dist
    if URBAN_FILTER_ENABLED:
        before = len(base)
        base = base[~is_urban].copy()
        n_dropped = before - len(base)
        print(
            f"Urban-area    filter: dropped {n_dropped:,} cells inside major "
            f"cities (+{URBAN_BUFFER_KM} km buffer); {len(base):,} / {before:,} kept "
            f"({len(base)*100/max(before,1):.1f}%)"
        )

    # Country filter — drop cells outside Thailand's land border. Done last
    # so the polygon-containment check (most expensive of the three filters)
    # only runs against the smallest possible set.
    if COUNTRY_FILTER_ENABLED and len(base) > 0:
        before = len(base)
        in_th = is_in_thailand(
            base["lat_grid"].to_numpy(),
            base["lon_grid"].to_numpy(),
        )
        base = base[in_th].copy()
        n_dropped = before - len(base)
        print(
            f"Country       filter: dropped {n_dropped:,} cells outside Thailand "
            f"(neighbour countries); {len(base):,} / {before:,} kept "
            f"({len(base)*100/max(before,1):.1f}%)"
        )

    # Per-cell province annotation — emitted into the GeoJSON properties so
    # the frontend can offer a provincial filter without doing point-in-
    # polygon on the client. Only runs against post-filter cells (small set).
    if len(base) > 0:
        provinces = find_province(
            base["lat_grid"].to_numpy(),
            base["lon_grid"].to_numpy(),
        )
        base["province"] = provinces

    # Display cap: within each probability bucket keep the top
    # MAX_CELLS_PER_DAY ranked PRIMARILY by calibrated model probability
    # (tie-break: recent activity). The previous ranking led with
    # historical_fire_count_30d × 1000, which made the "model's" map mostly
    # a recent-activity map. The number of suppressed cells is reported so
    # the operator knows this is a display cap, not the model's full output.
    n_suppressed_by_cap = 0
    if MAX_CELLS_PER_DAY > 0 and len(base) > 0:
        before = len(base)
        base = base.assign(
            _rank_score=base["probability"].fillna(0) * 1000
            + base["historical_fire_count_30d"].fillna(0).clip(upper=999) / 1000.0
        )
        base = (
            base.sort_values(["days_until_fire", "_rank_score"], ascending=[True, False])
                .groupby("days_until_fire", group_keys=False)
                .head(MAX_CELLS_PER_DAY)
                .drop(columns="_rank_score")
                .reset_index(drop=True)
        )
        n_suppressed_by_cap = before - len(base)
        per_day_counts = base.groupby("days_until_fire").size().to_dict()
        print(
            f"Display cap   : kept top {MAX_CELLS_PER_DAY} by model probability per bucket; "
            f"suppressed {n_suppressed_by_cap:,}; {len(base):,} / {before:,} kept "
            f"(per-bucket: {per_day_counts})"
        )
    base.attrs["n_suppressed_by_cap"] = n_suppressed_by_cap

    print(f"Total after history filters: {len(base):,} / {n_total:,} cells "
          f"({len(base)*100/max(n_total,1):.1f}%)")

    # ── Tier assignment: ABSOLUTE calibrated-probability thresholds ──
    # Cutoffs are selected on the training pipeline's calib split (never on
    # test) and persisted in dataset_info.json["probability_tier_thresholds"]:
    #   HIGH   ≥ deployment threshold (F1-optimal on calib)
    #   MEDIUM ≥ threshold catching ≥85% of calib fires
    #   LOW    = everything else
    # Because the thresholds are absolute, a genuinely low-risk day shows NO
    # HIGH cells — unlike the previous percentile pyramid, which forced the
    # top 10% of any snapshot to be "HIGH" even when nothing was likely to
    # burn ("high relative to today" masquerading as high absolute risk).
    probs = base["probability"].to_numpy()

    tier_meta = meta.get("probability_tier_thresholds") or {}
    if {"HIGH", "MEDIUM"} <= set(tier_meta):
        thr_high = float(tier_meta["HIGH"])
        thr_med = float(tier_meta["MEDIUM"])

        def _tier(p: float) -> str:
            if p >= thr_high:
                return "HIGH"
            if p >= thr_med:
                return "MEDIUM"
            return "LOW"

        base["urgency_level"] = [_tier(p) for p in probs]
        thresholds = {
            "CRITICAL": 1.01,  # unreachable sentinel — tier not used
            "HIGH": thr_high,
            "MEDIUM": thr_med,
            "LOW": 0.0,
        }
        thresholds_source = (
            "absolute calibrated probability, selected on calib split "
            f"(HIGH: F1-opt, precision={tier_meta.get('calib_precision_at_high')}, "
            f"recall={tier_meta.get('calib_recall_at_high')}; "
            f"MEDIUM: recall≥{tier_meta.get('calib_recall_target_medium')})"
        )
    elif len(base) >= 10:
        # Legacy artifact without persisted cutoffs — RELATIVE ranking only.
        # Explicitly labelled: "HIGH" here means "top of today's snapshot",
        # not high absolute risk. Retrain to get absolute thresholds.
        low_med_cut = float(np.percentile(probs, 60))
        med_hi_cut = float(np.percentile(probs, 90))
        base["urgency_level"] = [
            "HIGH" if p >= med_hi_cut else "MEDIUM" if p >= low_med_cut else "LOW"
            for p in probs
        ]
        thresholds = {"CRITICAL": 1.01, "HIGH": med_hi_cut, "MEDIUM": low_med_cut, "LOW": 0.0}
        thresholds_source = (
            "PRIORITY RANKING (relative percentiles — legacy model artifact "
            "has no calibrated tier cutoffs; retrain to enable absolute tiers)"
        )
    else:
        thresholds = _resolve_thresholds(meta)
        base["urgency_level"] = [
            urgency_from_thresholds(int(d), thresholds)
            for d in base["days_until_fire"]
        ]
        thresholds_source = "fixed-domain fallback (tiny snapshot)"

    tier_counts = base["urgency_level"].value_counts().to_dict() if len(base) else {}
    print(f"Urgency thresholds [{thresholds_source}]: "
          f"HIGH≥{thresholds['HIGH']:.3f} · MEDIUM≥{thresholds['MEDIUM']:.3f} "
          f"· tiers: {tier_counts}")
    # Relative operational ranking (Top-k) — SEPARATE from the absolute tiers
    # above. Deterministic; HIGH absolute threshold overrides. Adds rank /
    # percentile / ranking_tier WITHOUT touching probability or the row order
    # (merged back by cell so downstream display-cap logic is unaffected).
    import inference as _inf
    high_thr = float(thresholds.get("HIGH", 1.01)) if thresholds else None
    if len(base):
        ranked = _inf.assign_operational_ranking(
            base, high_threshold=high_thr, watch_top_pct=10,
            extended_watch_top_pct=20,
        )[["lat_grid", "lon_grid", "rank", "percentile", "ranking_tier"]]
        base = base.merge(ranked, on=["lat_grid", "lon_grid"], how="left")
    base.attrs["thresholds_source"] = thresholds_source

    return base, base_date, thresholds


# =========================================================
# WRITE GEOJSON (append-then-overwrite-current-base-date)
# =========================================================

def _cell_key(lat: float, lon: float, grid_size: float) -> Tuple[int, int]:
    """Canonical integer grid index for a coordinate — snap to the model's
    0.1° grid so 'same cell' means exactly the training label's cell, free of
    float-equality hazards. Shared definition for prediction and observation."""
    return (int(round(lat / grid_size)), int(round(lon / grid_size)))


def _revalidate_predictions(
    features: list,
    daily_df: pd.DataFrame,
    horizon_days: int,
    spatial_radius_km: float = 25.0,
    operational_tiers: Optional[set] = None,
    grid_size: float = 0.1,
) -> dict:
    """Two SEPARATE, explicitly-named evaluations of stored predictions.

    strict_model_target — the model's ACTUAL binary target: ≥1 FIRMS hotspot
      in the SAME 0.1° cell on any day in ``[base_date+1, base_date+horizon]``
      inclusive. This is the ONLY evaluation permitted to report model
      hit/miss (and downstream precision/recall/F1). Each feature's
      ``validation_status`` is this strict result.

    operational_spatial_tolerance — a proximity DIAGNOSTIC only: a real fire
      within ``spatial_radius_km`` of the cell within the same date window. It
      is NOT model precision/recall and is surfaced under its own key so it can
      never be mistaken for one. Stored per feature as ``spatial_diagnostic``.

    A prediction is ``future`` until observations cover the ENTIRE horizon
    (``latest_observed >= base_date + horizon``) — not merely +horizon−1. The
    previous code matched ±1 day around base_date+horizon, i.e. days +2..+4,
    with a 25 km radius; both are fixed here.
    """
    empty = {"hits": 0, "misses": 0, "future": 0, "hit_rate": None,
             "evaluation": "strict_model_target",
             "operational_spatial_tolerance": {"hits": 0, "misses": 0,
                                               "future": 0, "hit_rate": None}}
    if daily_df is None or daily_df.empty:
        return empty

    operational_tiers = operational_tiers or {"HIGH", "MEDIUM"}
    horizon_days = int(horizon_days)

    # latest_observed comes from the full densified frame (zero-fire days are
    # still observations) so maturity is judged correctly even before any fire.
    latest_observed = pd.to_datetime(daily_df["date"]).max().date()
    fired = daily_df[daily_df["fire_count"] > 0].copy()
    fired["date"] = pd.to_datetime(fired["date"]).dt.date

    # Strict lookup: set of (cell_index, date) that actually burned.
    fired_cells: set = set()
    # Spatial lookup: coords by date (for the radius diagnostic).
    obs_by_date: Dict[Any, List[Tuple[float, float]]] = {}
    for _, row in fired.iterrows():
        la, lo, d = float(row["lat_grid"]), float(row["lon_grid"]), row["date"]
        fired_cells.add((_cell_key(la, lo, grid_size), d))
        obs_by_date.setdefault(d, []).append((la, lo))

    def haversine_km(lat1, lon1, lat2, lon2):
        from math import radians, cos, sin, asin, sqrt
        r = 6371.0
        dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return r * 2 * asin(sqrt(a))

    def _acc():
        return {"hits": 0, "misses": 0, "future": 0,
                "hits_all": 0, "misses_all": 0, "future_all": 0}
    strict = _acc()
    spatial = _acc()
    per_snapshot: Dict[str, Dict[str, int]] = {}

    for f in features:
        props = f.get("properties", {})
        if props.get("source") != "predicted":
            continue
        bd_str = props.get("base_date")
        if not bd_str:
            continue
        try:
            base_date = pd.to_datetime(bd_str).date()
        except (ValueError, TypeError):
            continue

        # Window = [base_date+1, base_date+horizon] inclusive (the label).
        window = [base_date + timedelta(days=k) for k in range(1, horizon_days + 1)]
        window_end = window[-1]
        tier = props.get("urgency_level") or "NONE"
        is_op = tier in operational_tiers
        bucket = per_snapshot.setdefault(bd_str, {
            "hits": 0, "misses": 0, "future": 0,
            "hits_all": 0, "misses_all": 0, "future_all": 0,
        })

        lat = float(props.get("lat", props.get("lat_grid", 0)))
        lon = float(props.get("lon", props.get("lon_grid", 0)))

        # future until the ENTIRE horizon window is observed.
        if latest_observed < window_end:
            props["validation_status"] = "future"
            props["spatial_diagnostic"] = "future"
            for acc in (strict, spatial):
                acc["future_all"] += 1
                if is_op:
                    acc["future"] += 1
            bucket["future_all"] += 1
            if is_op:
                bucket["future"] += 1
            continue

        # STRICT: same cell burned on any day in the window.
        pkey = _cell_key(lat, lon, grid_size)
        strict_hit = any((pkey, d) in fired_cells for d in window)

        # SPATIAL DIAGNOSTIC: a fire within radius on any day in the window.
        lat_gate = max(0.3, spatial_radius_km / 80.0)
        spatial_hit = False
        for d in window:
            for olat, olon in obs_by_date.get(d, ()):
                if abs(olat - lat) > lat_gate:
                    continue
                if haversine_km(lat, lon, olat, olon) <= spatial_radius_km:
                    spatial_hit = True
                    break
            if spatial_hit:
                break

        props["validation_status"] = "hit" if strict_hit else "miss"
        props["spatial_diagnostic"] = "hit" if spatial_hit else "miss"

        for acc, is_hit in ((strict, strict_hit), (spatial, spatial_hit)):
            key_all = "hits_all" if is_hit else "misses_all"
            acc[key_all] += 1
            if is_op:
                acc["hits" if is_hit else "misses"] += 1
        bucket["hits_all" if strict_hit else "misses_all"] += 1
        if is_op:
            bucket["hits" if strict_hit else "misses"] += 1

    for _, b in per_snapshot.items():
        ov = b["hits"] + b["misses"]
        av = b["hits_all"] + b["misses_all"]
        b["hit_rate"] = round(b["hits"] / ov, 4) if ov else None
        b["hit_rate_all"] = round(b["hits_all"] / av, 4) if av else None

    def _summ(acc):
        ov = acc["hits"] + acc["misses"]
        av = acc["hits_all"] + acc["misses_all"]
        return {
            "hits": acc["hits"], "misses": acc["misses"], "future": acc["future"],
            "hit_rate": round(acc["hits"] / ov, 4) if ov else None,
            "hits_all": acc["hits_all"], "misses_all": acc["misses_all"],
            "future_all": acc["future_all"],
            "hit_rate_all": round(acc["hits_all"] / av, 4) if av else None,
        }

    result = _summ(strict)
    result.update({
        "evaluation": "strict_model_target",
        "match_rule": "same 0.1° cell, any day in [base+1, base+horizon]",
        "horizon_days": horizon_days,
        "operational_tiers": sorted(operational_tiers),
        "per_snapshot": per_snapshot,
        # Proximity diagnostic — NOT model precision/recall.
        "operational_spatial_tolerance": {
            **_summ(spatial),
            "match_radius_km": spatial_radius_km,
            "note": "operational proximity diagnostic only — not model "
                    "precision/recall",
        },
    })
    return result


def _ranking_policy_block(thresholds: dict, metrics: dict) -> dict:
    """Ranking-policy metadata + held-out historical Top-k coverage (from the
    model's test metrics — recall captured by the top 5/10/20%)."""
    import inference as _inf
    m = metrics or {}
    coverage = {}
    for pct in (5, 10, 20):
        v = m.get(f"recall_captured_top_{pct}pct")
        if v is not None:
            coverage[f"top_{pct}pct"] = v
    high_thr = float(thresholds.get("HIGH", 1.01)) if thresholds else None
    return _inf.ranking_policy_metadata(
        high_thr, watch_top_pct=10, extended_watch_top_pct=20,
        historical_coverage=coverage)


def append_geojson(observed: pd.DataFrame, predicted: pd.DataFrame, base_date, thresholds: dict, metrics: dict, daily_df: Optional[pd.DataFrame] = None):
    base_date_str = base_date.strftime("%Y-%m-%d")
    # The model's true horizon is imminent_days (default 3) — NOT the 7-day
    # pseudo-day display range. Every metadata surface reports this horizon and
    # the explicit risk window so API/GeoJSON/CSV/frontend agree.
    horizon_days = int((_load_metadata() or {}).get("imminent_days", 3))
    risk_window_start = (base_date + timedelta(days=1)).strftime("%Y-%m-%d")
    risk_window_end = (base_date + timedelta(days=horizon_days)).strftime("%Y-%m-%d")

    geojson = {"type": "FeatureCollection", "features": []}
    if exists(GEOJSON_PATH):
        try:
            geojson = read_json(GEOJSON_PATH)
        except json.JSONDecodeError:
            print("⚠️ Corrupted GeoJSON → recreate")

    # Retention. Without pruning the GeoJSON grows by ~one prediction snapshot
    # plus a fresh observed batch on every risk_map.run() — by month two the
    # file would be tens of MB and slow on first dashboard load. Two limits:
    #   • Keep the most recent MAX_BASE_DATE_SNAPSHOTS predicted snapshots
    #     (incl. the one we're about to write, replacing any same-base prior).
    #   • Drop observed features — we re-emit a fresh observed batch from the
    #     latest densified FIRMS day a few lines down. Stale observed rows
    #     just bloat the file; the date picker only needs predictions.
    MAX_BASE_DATE_SNAPSHOTS = int(os.getenv("MAX_BASE_DATE_SNAPSHOTS", "7"))
    existing = geojson.get("features", [])
    predicted_only = [f for f in existing if f["properties"].get("source") == "predicted"]
    # Snapshots to keep: most recent N-1, excluding the one we're rewriting.
    base_dates = sorted(
        {f["properties"].get("base_date") for f in predicted_only if f["properties"].get("base_date")},
        reverse=True,
    )
    # base_dates is sorted descending (newest first). Slice the list BEFORE
    # converting to set so we always keep the N-1 most-recent dates, not an
    # arbitrary subset. set(list(a_set)[:n]) is a known Python footgun because
    # iteration order of sets is non-deterministic — this was causing the
    # 2026-04-22 anomalous snapshot (2700 features) to survive instead of being
    # evicted when newer snapshots accumulated.
    keep_dates = set(
        [d for d in base_dates if d != base_date_str][: MAX_BASE_DATE_SNAPSHOTS - 1]
    )
    geojson["features"] = [
        f for f in predicted_only
        if f["properties"].get("base_date") in keep_dates
    ]

    # Top-level metadata so the frontend can read calibrated thresholds and
    # validation metrics without a separate fetch.
    geojson["metadata"] = {
        "base_date": base_date_str,
        # True observation instant of the freshest FIRMS hotspot (UTC ISO) and
        # when this snapshot was generated — lets the UI show "data as of …"
        # with a real time, not just the daily base_date.
        "latest_data_time": _latest_data_time(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizon_days": horizon_days,
        "risk_window_start": risk_window_start,
        "risk_window_end": risk_window_end,
        "output_semantics": (
            "probability = calibrated P(≥1 FIRMS hotspot in this 0.1° cell "
            f"during base_date+1..base_date+{horizon_days} inclusive). "
            "days_until_fire / predicted_fire_date are DEPRECATED display "
            "scaffolding, not a forecast of an exact fire date."
        ),
        "urgency_thresholds": thresholds,
        "thresholds_source": str(predicted.attrs.get("thresholds_source", "")) if hasattr(predicted, "attrs") else "",
        "suppressed_by_display_cap": int(predicted.attrs.get("n_suppressed_by_cap", 0)) if hasattr(predicted, "attrs") else 0,
        "metrics": metrics,
        # Relative Top-k operational ranking policy (separate from the absolute
        # calibrated tiers). Historical coverage = held-out captured-recall
        # estimates from the model's test metrics — NOT a same-day guarantee.
        "ranking_policy": _ranking_policy_block(thresholds, metrics),
        "history_window_days": HISTORY_WINDOW_DAYS,
        "long_history_window_days": LONG_HISTORY_WINDOW_DAYS,
    }

    # ---------- OBSERVED (latest densified FIRMS day) ----------
    observed_date_str = observed["date"].iloc[0].strftime("%Y-%m-%d") if len(observed) else base_date_str
    for _, r in observed.iterrows():
        if int(r.get("fire_count", 0)) <= 0:
            # Densified rows include cells with zero fires; only emit real detections.
            continue
        geojson["features"].append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(r["lon_grid"]), float(r["lat_grid"])],
            },
            "properties": {
                "date": observed_date_str,
                "source": "observed",
                "lat": float(r["lat_grid"]),
                "lon": float(r["lon_grid"]),
                "fire_count": int(r["fire_count"]),
                # urgency_level is not meaningful for observed detections (they
                # are real hotspots, not model predictions), but we set a safe
                # sentinel so frontend code that reads urgency_level without
                # null-checking doesn't crash on None.
                "urgency_level": "OBSERVED",
                # Province annotation — frontend Thailand-only filter on the
                # Live Fires page needs this. Predicted features already get
                # it via build_predicted; observed was missing it.
                "province": find_province(
                    [float(r["lat_grid"])], [float(r["lon_grid"])]
                )[0] or None,
            },
        })

    # ---------- PREDICTED ----------
    long_col = f"historical_fire_count_{LONG_HISTORY_WINDOW_DAYS}d"
    for _, r in predicted.iterrows():
        geojson["features"].append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(r["lon_grid"]), float(r["lat_grid"])],
            },
            "properties": {
                "base_date": base_date_str,
                "source": "predicted",
                # Probability-rank display bucket (0 = highest band). NOT an
                # exact-date forecast — the binary model can't resolve days.
                "days_until_fire": int(r["days_until_fire"]),
                # End of the risk window (base_date + imminent_days) — the
                # only date claim the model supports. Same for all cells.
                "predicted_fire_date": str(r["predicted_fire_date"]),
                "urgency_level": str(r["urgency_level"]),
                # Relative Top-k operational ranking (separate from urgency_level).
                "ranking_tier": str(r.get("ranking_tier", "LOW")),
                "rank": int(r["rank"]) if pd.notna(r.get("rank")) else None,
                "percentile": float(r["percentile"]) if pd.notna(r.get("percentile")) else None,
                "confidence": float(r["prediction_confidence"]),
                "raw_prediction": float(r["raw_prediction"]),
                # Calibrated P(≥1 hotspot within the risk window) — the
                # model's actual output; everything else derives from it.
                "probability": float(r.get("probability", 0.0)),
                "historical_fire_count_30d": int(r["historical_fire_count_30d"]),
                "historical_fire_count_long": int(r[long_col]),
                "fire_days_total": int(r["fire_days_total"]),
                "fire_days_per_year": float(r["fire_days_per_year"]),
                "nearest_urban_area": str(r["nearest_urban_area"]),
                "nearest_urban_distance_km": float(r["nearest_urban_distance_km"]),
                "province": str(r.get("province", "") or ""),
                # Hansen tree cover passed through so the dashboard can
                # categorise cells by land-cover bucket without re-doing
                # the per-cell lookup. Defaults to 0 if the cache wasn't
                # populated during training.
                "tree_cover_pct_2000": float(r.get("tree_cover_pct_2000", 0) or 0),
                "tree_loss_pct_recent": float(r.get("tree_loss_pct_recent", 0) or 0),
                "lat": float(r["lat_grid"]),
                "lon": float(r["lon_grid"]),
            },
        })

    # Retrospective validation: walk every predicted feature still in the
    # GeoJSON (current snapshot + retained history) and tag with
    # hit / miss / future against actual FIRMS observations from the
    # densified daily frame. Lets the dashboard show "model called this
    # cell to fire 3 days ago — did it actually burn?" at a glance.
    if daily_df is not None:
        validation_summary = _revalidate_predictions(
            geojson["features"], daily_df, horizon_days=horizon_days,
        )
        geojson["metadata"]["validation_summary"] = validation_summary
        if validation_summary.get("hit_rate") is not None:
            print(
                f"Strict-target validation (same cell, base+1..base+{horizon_days}): "
                f"{validation_summary['hits']} hits / "
                f"{validation_summary['hits'] + validation_summary['misses']} validatable "
                f"= {validation_summary['hit_rate']*100:.1f}% hit rate "
                f"({validation_summary['future']} still pending)"
            )

    write_json(geojson, GEOJSON_PATH, indent=2)

    write_json(
        {
            "base_date": base_date_str,
            "observed_date": observed_date_str,
            "horizon_days": horizon_days,
            "risk_window_start": risk_window_start,
            "risk_window_end": risk_window_end,
            "urgency_thresholds": thresholds,
            "metrics": metrics,
        },
        LATEST_PATH,
        indent=2,
    )


# =========================================================
# PIPELINE
# =========================================================

def run():
    print("🔄 Loading assets...")
    model, df = load_assets()
    meta = _load_metadata()

    print("📍 Building observed layer (real FIRMS detections)...")
    observed, obs_date = build_observed(df)

    print(f"🔮 Predicting fire dates for next {MAX_PREDICTION_DAYS} days...")
    predicted, base_date, thresholds = build_predicted(df, model, obs_date, meta)

    metrics = (meta.get("model") or {}).get("test_metrics", {}) or {}
    # Include feature_importance_top (lives at the root of dataset_info.json)
    # in the GeoJSON-side metrics so the Reports page can render it without
    # a second fetch.
    if "feature_importance_top" in meta:
        metrics["feature_importance_top"] = meta["feature_importance_top"][:20]
    # Bundle dataset + model hyperparameter info so the Reports page can render
    # a full "Model Training Summary" using real persisted values (no
    # fabrication). Everything below is read from dataset_info.json verbatim.
    _model_meta = meta.get("model") or {}
    metrics["training_summary"] = {
        "trained_at": meta.get("trained_at"),
        "data_source": meta.get("data_source"),
        "date_range": [meta.get("earliest_date"), meta.get("latest_date")],
        "total_days": meta.get("total_days"),
        "active_cells": meta.get("total_active_cells"),
        "grid_size_deg": meta.get("grid_size"),
        "training_rows": meta.get("training_rows"),
        "feature_count": meta.get("feature_count"),
        "weather_features_count": len(meta.get("weather_features_used") or []),
        "prediction_type": meta.get("prediction_type"),
        "imminent_days": meta.get("imminent_days"),
        "training_time_seconds": meta.get("training_time_seconds"),
        "model_type": _model_meta.get("type") or meta.get("best_model"),
        "search_method": _model_meta.get("search_method"),
        "search_iterations": _model_meta.get("n_iter"),
        "cv_n_splits": _model_meta.get("n_splits"),
        "cv_gap_days": _model_meta.get("cv_method") or _model_meta.get("ts_split_gap_days"),
        "ensemble_size": _model_meta.get("n_ensemble"),
        "early_stopping_rounds": _model_meta.get("early_stopping_rounds"),
        "best_params": _model_meta.get("best_params"),
    }
    # Pass the densified daily frame so append_geojson can revalidate every
    # predicted feature against actual FIRMS observations and tag it
    # hit / miss / future. Empowers the dashboard's "did it actually burn?"
    # comparison view.
    append_geojson(observed, predicted, base_date, thresholds, metrics, daily_df=df)

    # Mirror this snapshot into PostgreSQL when a DATABASE_URL is configured
    # (Railway). The GeoJSON stays the serving format; the DB is the durable
    # archive that survives redeploys and feeds long-term analytics. A DB
    # failure must never break the map refresh, hence the broad except.
    try:
        import db
        if db.enabled():
            n = db.insert_predictions(predicted, base_date)
            print(f"🗄️  Persisted {n} predictions to database (base_date={base_date})")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  DB persist skipped: {exc}")

    print("\n✅ FIRE DATE MAP UPDATED")
    print("Observed date :", obs_date)
    print("Base date     :", base_date)
    print("Thresholds    :", thresholds)
    print("GeoJSON       :", GEOJSON_PATH)

    urgency_counts = predicted["urgency_level"].value_counts()
    print("\n📊 URGENCY SUMMARY (calibrated thresholds):")
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]:
        print(f"  {level}: {int(urgency_counts.get(level, 0))} locations")


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    run()