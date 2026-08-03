# =====================================
# FASTAPI BACKEND - 3-DAY WILDFIRE RISK
# =====================================

# All responses are derived from REAL data:
#   • probability: calibrated P(≥1 FIRMS hotspot in base_date+1..+3)
#   • risk_tier: absolute HIGH/WATCH probability cutoffs selected on calibration
#   • historical_fire_count_30d: literal sum of FIRMS detections per cell
#   • metrics: held-out ranking, calibration, and threshold metrics from train.py
# Legacy pseudo-day fields remain in old artifacts only; API prediction routes
# never present them as an exact fire date.
# =====================================

import json
import logging
import os
import traceback
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import List, Literal, Optional

from dotenv import load_dotenv

# Load the repository runtime configuration before importing local modules.
# Several of them intentionally snapshot environment settings at import time
# (security.APP_ENV, db.DATABASE_URL, scheduler cadence/paths, GISTDA limits).
# Loading `.env` after those imports made documented local/demo settings
# silently ineffective, even though direct Railway/Docker variables worked.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

import numpy as np
import pandas as pd
import asyncio

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from features import (
    DEFAULT_URGENCY_THRESHOLDS,
    FEATURES_CORE,
    FEATURES_WEATHER,
    MAX_PREDICTION_DAYS,
    urgency_from_thresholds,
)
from storage import exists, read_json, read_pickle, read_table, resolve_existing
import inference
import security
import gridutil

import db as db_module
import scheduler as scheduler_module

# =====================================
# CONFIG
# =====================================

def _resolve(p: str) -> str:
    """Resolve a possibly-relative path against BASE_DIR — same convention
    used by train.py / risk_map.py so env-set paths work whether deploy
    sets OUTPUT_DIR=./outputs (relative) or OUTPUT_DIR=/srv/outputs (abs)."""
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)


# OUTPUT_DIR is the single knob: the cron service writes here, the API reads
# from here, and Railway mounts a shared Volume at this absolute path.
OUTPUT_DIR = _resolve(os.environ.get("OUTPUT_DIR", "./outputs"))

MODEL_PATH   = os.path.join(OUTPUT_DIR, "models", "lgbm_fire_date_model.pkl")
FEATURE_PATH = os.path.join(OUTPUT_DIR, "features", "full_features.parquet")
META_PATH    = os.path.join(OUTPUT_DIR, "metadata", "dataset_info.json")
RISKMAP_DIR  = os.path.join(OUTPUT_DIR, "riskmap")
GEOJSON_PATH = os.path.join(RISKMAP_DIR, "fire_dates_all.geojson")
def _configured_weather_cache_path() -> str:
    """Resolve the cache path from the runtime's DATA/WEATHER hierarchy."""
    data_dir = _resolve(os.environ.get("DATA_DIR", "./data"))
    weather_dir = _resolve(
        os.environ.get("WEATHER_DIR", os.path.join(data_dir, "weather"))
    )
    return _resolve(
        os.environ.get(
            "WEATHER_PATH",
            os.path.join(weather_dir, "weather_cache.parquet"),
        )
    )


# ERA5 cache produced by fetch_weather.py — used by /api/cell_weather.
# Honour the directory variables already used by the Docker runtime.  The old
# default ignored WEATHER_DIR=/srv/data/weather and looked under /app/data,
# which made a valid cache appear missing in production unless WEATHER_PATH
# happened to be configured separately.
DATA_DIR = _resolve(os.environ.get("DATA_DIR", "./data"))
WEATHER_DIR = _resolve(
    os.environ.get("WEATHER_DIR", os.path.join(DATA_DIR, "weather"))
)
WEATHER_CACHE_PATH = _configured_weather_cache_path()

# Static SPA dir — built by `npm run build` in /web. In production the
# multi-stage Dockerfile copies the built artifact here so FastAPI can serve
# both the API and the dashboard from the same origin. Override with
# WEB_DIST_DIR if hosting the SPA elsewhere.
WEB_DIST_DIR = os.environ.get(
    "WEB_DIST_DIR",
    os.path.join(BASE_DIR, "web", "dist"),
)

HISTORY_WINDOW_DAYS = 30


def _load_metadata() -> dict:
    if not exists(META_PATH):
        return {}
    try:
        return read_json(META_PATH)
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_features(meta: dict, df: pd.DataFrame) -> list[str]:
    feats = meta.get("features")
    if feats:
        return list(feats)
    return [c for c in (*FEATURES_CORE, *FEATURES_WEATHER) if c in df.columns]


def _resolve_thresholds(meta: dict) -> dict:
    t = meta.get("urgency_thresholds")
    if isinstance(t, dict) and {"CRITICAL", "HIGH", "MEDIUM", "LOW"} <= set(t):
        return {k: float(v) for k, v in t.items()}
    return dict(DEFAULT_URGENCY_THRESHOLDS)


def _load_minimal_state(feature_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only what the API actually serves: the latest-date snapshot + a
    pre-aggregated 30-day historical fire-count table.

    The training parquet is ~4M rows × 80+ cols and balloons to multiple GB
    when materialised as a pandas DataFrame, which OOM-kills small Railway
    containers during startup. The API never reads beyond:

      • the rows whose date == latest_date (used for model.predict per cell)
      • a groupby sum of fire_count over the last 30 days (for the popup
        "historical fires in last 30d" field)

    So we load exactly those two slices and discard everything else. For
    parquet inputs we stream via iter_batches so peak conversion memory
    stays bounded; for CSV inputs we accept the full read (no streaming
    facility available, and CSV training data isn't expected in prod).
    """
    if not feature_path.lower().endswith(".parquet"):
        df = read_table(feature_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        latest = df["date"].max()
        df_latest = df[df["date"] == latest].copy()
        hist = _aggregate_history(df, latest)
        return df_latest, hist

    import pyarrow.dataset as pads

    # Step 1 — find the latest date by scanning just one column. Cheap.
    dataset = pads.dataset(feature_path, format="parquet")
    date_scanner = dataset.scanner(columns=["date"])
    latest_ts = pd.to_datetime(date_scanner.to_table().column("date").to_pandas()).max()
    latest = latest_ts.date() if hasattr(latest_ts, "date") else latest_ts
    cutoff = latest - timedelta(days=HISTORY_WINDOW_DAYS)

    # Step 2 — latest snapshot, all columns, filter pushdown via dataset.
    # Streamed as RecordBatches so peak conversion memory is bounded by
    # batch_size × col_count instead of the full row group.
    latest_scanner = dataset.scanner(
        filter=pads.field("date") == latest,
        batch_size=50_000,
    )
    latest_chunks: list[pd.DataFrame] = []
    for batch in latest_scanner.to_batches():
        if batch.num_rows == 0:
            continue
        latest_chunks.append(batch.to_pandas(split_blocks=True, self_destruct=True))
    df_latest = (
        pd.concat(latest_chunks, ignore_index=True) if latest_chunks else pd.DataFrame()
    )
    if not df_latest.empty:
        df_latest["date"] = pd.to_datetime(df_latest["date"]).dt.date

    # Step 3 — 30-day history, only the 4 cols needed. Aggregate per-batch
    # so we never hold the full 290k-row slice in RAM.
    hist_scanner = dataset.scanner(
        columns=["lat_grid", "lon_grid", "fire_count"],
        filter=(pads.field("date") >= cutoff) & (pads.field("date") <= latest),
        batch_size=100_000,
    )
    running: dict[tuple[float, float], int] = {}
    for batch in hist_scanner.to_batches():
        if batch.num_rows == 0:
            continue
        chunk = batch.to_pandas()
        chunk_sum = (
            chunk.groupby(["lat_grid", "lon_grid"], as_index=False)["fire_count"].sum()
        )
        for _, row in chunk_sum.iterrows():
            key = (row["lat_grid"], row["lon_grid"])
            running[key] = running.get(key, 0) + int(row["fire_count"])
    if running:
        hist = pd.DataFrame(
            [(la, lo, c) for (la, lo), c in running.items()],
            columns=["lat_grid", "lon_grid", "historical_fire_count_30d"],
        )
    else:
        hist = pd.DataFrame(
            columns=["lat_grid", "lon_grid", "historical_fire_count_30d"]
        )

    return df_latest, hist


def _aggregate_history(df: pd.DataFrame, latest) -> pd.DataFrame:
    """CSV-fallback equivalent of the streaming aggregation above."""
    start = latest - timedelta(days=HISTORY_WINDOW_DAYS)
    sub = df[(df["date"] >= start) & (df["date"] <= latest)]
    return (
        sub.groupby(["lat_grid", "lon_grid"], as_index=False)["fire_count"]
        .sum()
        .rename(columns={"fire_count": "historical_fire_count_30d"})
    )


# =====================================
# LIFESPAN
# =====================================

_MODEL_RELOAD_LOCK = Lock()


def _build_model_state() -> Optional[dict]:
    """Load the promoted model bundle + serving frames into a fresh dict.

    Returns None when no usable artifact exists (degraded mode). Pure
    builder — touches no shared state, so a failed reload can't corrupt
    the currently-served model.
    """
    feature_path = resolve_existing(FEATURE_PATH)
    if not (os.path.exists(MODEL_PATH) and feature_path):
        return None

    # Canonical loader (train.load_model_bundle) — registers the pickled
    # classes in __main__ and unwraps bundle-v1. Single source of truth shared
    # with risk_map / scientific_stats / rolling_eval.
    import train as _train
    model, bundle_meta = _train.load_model_bundle(MODEL_PATH)

    df_latest, historical_counts = _load_minimal_state(feature_path)
    # Pre-aggregated 30-day count per (lat_grid, lon_grid). Computed once
    # per (re)load since base_date doesn't change between requests on a
    # given snapshot.
    meta = bundle_meta if bundle_meta is not None else _load_metadata()
    features = _resolve_features(meta, df_latest)
    model_version = str(meta.get("trained_at") or "unknown")
    # Score the latest snapshot ONCE at load — cached and served immutably by
    # /predictions/*. A model reload rebuilds this state dict (and its cache)
    # atomically, so the cache can never lag the model.
    try:
        scored_snapshot = _compute_snapshot(model, df_latest, features, meta, model_version)
    except Exception as exc:
        _log.warning("Snapshot pre-scoring failed at load (%s) — will score lazily", exc)
        scored_snapshot = None
    return {
        "model": model,
        "df": df_latest,
        "historical_counts": historical_counts,
        "meta": meta,
        "features": features,
        "thresholds": _resolve_thresholds(meta),
        "model_version": model_version,
        "model_loaded_at": datetime.now(timezone.utc).isoformat(),
        "scored_snapshot": scored_snapshot,
    }


_DEGRADED_STATE = {
    "model": None, "df": None, "historical_counts": None, "meta": {},
    "features": [], "thresholds": dict(DEFAULT_URGENCY_THRESHOLDS),
    "model_version": None, "model_loaded_at": None, "scored_snapshot": None,
}


def _apply_model_state(app: FastAPI, state: dict) -> None:
    """Swap the served model state. All keys land via ONE dict.update on
    Starlette State's backing dict — executed in C without yielding the
    GIL between keys, so a concurrent request never observes a mixture of
    old model + new feature list."""
    app.state._state.update(dict(state))


def reload_model_state(app: FastAPI) -> bool:
    """Hot-reload after a promotion. Build-then-swap under a lock; when the
    build fails the previous in-memory model keeps serving untouched."""
    with _MODEL_RELOAD_LOCK:
        try:
            state = _build_model_state()
        except Exception as exc:
            _log.error("Model reload failed — keeping previous model: %s", exc)
            return False
        if state is None:
            _log.warning("Model reload requested but no artifact found — "
                         "keeping previous model")
            return False
        _apply_model_state(app, state)
        _log.info("Model hot-reloaded: version=%s features=%d",
                  state["model_version"], len(state["features"]))
        return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── model + feature state (degraded-mode tolerant) ──────────────────────
    # On a fresh deployment (empty Railway volume) there is no trained model
    # yet — the automation scheduler below will accumulate FIRMS data and
    # train one. Refusing to boot here would crash-loop the whole service,
    # so we start degraded instead: /health reports ready=false and
    # model-dependent endpoints fail individually until training completes.
    _apply_model_state(app, _DEGRADED_STATE)
    try:
        state = _build_model_state()
    except Exception as exc:
        state = None
        _log.error("Model load failed at startup — degraded mode: %s", exc)
    if state is not None:
        _apply_model_state(app, state)
        print("✅ FireWatch 3-day wildfire-risk model loaded")
        print(f"   Version: {state['model_version']} · features: {len(state['features'])} "
              f"· weather active: {any(c in state['features'] for c in FEATURES_WEATHER)}")
        probability_cutoffs = inference.tier_cutoffs_from_meta(
            (state["meta"] or {}).get("probability_tier_thresholds")
        )
        if probability_cutoffs is not None:
            high, watch = probability_cutoffs
            print(
                "   Operational probability cutoffs: "
                f"HIGH={high:.4f} · WATCH={watch:.4f}"
            )
        else:
            print(
                "   Operational probability cutoffs unavailable "
                "(legacy artifact; absolute risk tiers disabled)"
            )
    else:
        print(f"⚠️  No trained model at {MODEL_PATH} — starting in DEGRADED mode.")
        print("   The scheduler will fetch data and train automatically "
              "(or run train.py manually).")

    # ── database + automation scheduler ─────────────────────────────────────
    try:
        db_module.init_db()
    except Exception as exc:  # DB down ≠ API down; log and continue
        _log.error("Database init failed: %s", exc)

    # After a scheduled retrain PROMOTES a new bundle, hot-reload it into
    # this running process — no restart required, and a failed reload keeps
    # the previous in-memory model serving.
    scheduler_module.register_model_promoted_callback(
        lambda: reload_model_state(app)
    )

    if os.getenv("SCHEDULER_ENABLED", "true").lower() in ("1", "true", "yes"):
        scheduler_module.start()
    else:
        print("ℹ️  SCHEDULER_ENABLED=false — automation jobs not started")

    yield

    scheduler_module.shutdown()


# =====================================
# APP
# =====================================

app = FastAPI(
    title="🔥 FireWatch Thailand — 3-Day Wildfire Risk API",
    version="2.1",
    lifespan=lifespan,
)

# Production safety: environment-gated CORS allowlist, rate limiting,
# request-ID + structured JSON access logging, and security headers/CSP.
# Replaces the former wildcard CORS. See security.py.
security.configure_security(app)


# Catch-all exception handler — without this, an unexpected error bubbles
# up as a 500 with the full Python traceback in the response body, which
# leaks internal paths / library versions to the public. Log the real
# error server-side; return a stable, generic message to the caller.
_log = logging.getLogger("fire-date-api")


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    _log.error(
        "Unhandled exception on %s %s: %s\n%s",
        request.method, request.url.path, exc, traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": "An unexpected error occurred. Check server logs.",
            "path": request.url.path,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# ─── helpers ──────────────────────────────────────────────────────────────────

def _predict_days(model, df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[features].fillna(0)
    raw = model.predict(X)
    # Floor (not round) — keeps day-bucket semantics consistent with risk_map.py
    # so the API and dashboard agree on which cells belong to "day 0", "day 1", etc.
    # See risk_map.build_predicted for the rationale.
    floored = np.clip(np.floor(raw), 0, MAX_PREDICTION_DAYS).astype(int)
    return raw, floored


def _rounding_confidence(raw: np.ndarray, clipped: np.ndarray) -> np.ndarray:
    """Proxy: 1 - 2*|raw - bucket_centre|. NOT a calibrated probability.

    Anchored at the bucket centre (floored + 0.5) so a raw value sitting
    exactly in the middle of its day-bucket reads confidence = 1.0; values
    near the bucket edge read closer to 0.
    """
    return (1.0 - 2.0 * np.abs(raw - (clipped + 0.5))).clip(0.0, 1.0)


def _historical_counts(df: pd.DataFrame, base_date, window: int = HISTORY_WINDOW_DAYS) -> pd.DataFrame:
    """Real FIRMS detections summed over the last `window` days, per grid cell."""
    start = base_date - timedelta(days=window)
    sub = df[(df["date"] >= start) & (df["date"] <= base_date)]
    return (
        sub.groupby(["lat_grid", "lon_grid"], as_index=False)["fire_count"]
        .sum()
        .rename(columns={"fire_count": "historical_fire_count_30d"})
    )


def _model_unavailable_detail() -> dict:
    """Stable error body returned (HTTP 503) by every prediction surface when
    the model state is degraded, so clients can branch on a fixed schema
    instead of parsing prose."""
    return {
        "error": "model_unavailable",
        "detail": "No trained model is loaded yet. The service is running in "
                  "degraded mode; predictions are temporarily unavailable.",
        "ready": False,
    }


def _require_ready_state(request: Request):
    """Return the served model state, or raise a stable 503 when degraded.
    Guarantees model + feature frame + features are all present."""
    st = request.app.state
    if getattr(st, "model", None) is None or getattr(st, "df", None) is None \
            or not getattr(st, "features", None):
        raise HTTPException(status_code=503, detail=_model_unavailable_detail())
    return st


def _state_cutoffs(state):
    """Absolute calibrated tier cutoffs (High, Watch) from the served metadata,
    or None for a legacy artifact without them."""
    meta = getattr(state, "meta", None) or {}
    return inference.tier_cutoffs_from_meta(meta.get("probability_tier_thresholds"))


def _compute_snapshot(model, df, features, meta, model_version):
    """Score the latest base-date cells ONCE. Called at model/feature load and
    the result cached in app.state — /predictions/* then serve the immutable
    cached snapshot instead of re-running inference on every request. Returns a
    dict {scored, base_date, ctx, historical_counts-merged}."""
    base_date = df["date"].max()
    cells = df[df["date"] == base_date].copy()
    cutoffs = inference.tier_cutoffs_from_meta((meta or {}).get("probability_tier_thresholds"))
    scored = inference.score_cells(model, cells, features, cutoffs)
    ctx = inference.snapshot_context(meta, base_date, model_version, cutoffs)
    return {"scored": scored, "base_date": base_date, "ctx": ctx}


def _cached_snapshot(state):
    """Return the cached snapshot for the served model, computing + memoizing it
    lazily on first use (so a manually-injected test state also works). The
    cache is part of the atomically-swapped state dict, so a model reload
    invalidates it automatically."""
    snap = getattr(state, "scored_snapshot", None)
    if snap is None:
        snap = _compute_snapshot(state.model, state.df, state.features,
                                 getattr(state, "meta", None),
                                 getattr(state, "model_version", None))
        try:
            state.scored_snapshot = snap   # memoize on the state object
        except Exception:
            pass
    return snap["scored"], snap["base_date"], snap["ctx"]


# =====================================
# ROUTES
# =====================================

@app.get("/api/status")
def api_status():
    """Programmatic status / version info. The unauthenticated GET / endpoint
    serves the SPA dashboard in production, so this moved to /api/status."""
    return {
        "status": "FireWatch Thailand API running",
        "version": "2.1",
        "prediction_type": "binary_fire_within_horizon",
        # The real model horizon (imminent_days, default 3) — NOT the 7-day
        # pseudo-day display range. Agrees with GeoJSON + /predictions/*.
        "horizon_days": int((_load_metadata() or {}).get("imminent_days", 3)),
        "data_sources": [
            "NASA FIRMS VIIRS NRT (real)",
            "Open-Meteo Archive (real, ECMWF ERA5) - if fetch_weather.py was run",
        ],
    }


@app.get("/health")
def health(request: Request):
    """Liveness + readiness check. Returns 200 with structured status fields
    so a load-balancer / monitor can decide whether to route traffic.

    The API is *ready* only when:
      - model artifact loaded
      - dataset_info.json present
      - feature CSV loaded
      - GeoJSON exists and was refreshed within the last 7 days

    A non-fresh GeoJSON still returns 200 (the model is technically usable),
    but `geojson_age_days` flags it for the operator.
    """
    state = request.app.state
    meta = _load_metadata()

    # GeoJSON freshness — operators care more about this than `latest_date`,
    # since the dashboard reads the GeoJSON directly.
    geojson_age_days: Optional[float] = None
    if os.path.exists(GEOJSON_PATH):
        try:
            geojson_age_days = round(
                (datetime.now().timestamp() - os.path.getmtime(GEOJSON_PATH)) / 86400.0,
                2,
            )
        except OSError:
            geojson_age_days = None

    latest_value = (
        state.df["date"].max()
        if hasattr(state, "df") and state.df is not None and not state.df.empty
        else None
    )
    latest_observed = str(latest_value) if latest_value is not None else None
    data_stale_days: Optional[int] = None
    if latest_value is not None:
        try:
            latest_day = pd.Timestamp(latest_value).date()
            data_stale_days = (
                datetime.now(timezone.utc).date() - latest_day
            ).days
        except (TypeError, ValueError, OverflowError):
            data_stale_days = None
    skill_check_passed = (meta.get("model", {}) or {}).get("skill_check_passed")
    ready = all([
        getattr(state, "model", None) is not None,
        getattr(state, "df", None) is not None,
        bool(getattr(state, "features", None)),
        os.path.exists(META_PATH),
        os.path.exists(GEOJSON_PATH),
        geojson_age_days is not None
        and geojson_age_days <= _GEOJSON_MAX_AGE_DAYS,
    ])

    return {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "model_loaded": getattr(state, "model", None) is not None,
        "model_version": getattr(state, "model_version", None),
        "model_loaded_at": getattr(state, "model_loaded_at", None),
        "metadata_present": os.path.exists(META_PATH),
        "geojson_present": os.path.exists(GEOJSON_PATH),
        "geojson_age_days": geojson_age_days,
        "latest_observed_date": latest_observed,
        # Stable frontend contract for the header's LIVE/STALE distinction.
        # This is age of the actual latest input date, not merely file mtime.
        "data_stale_days": data_stale_days,
        "best_model": meta.get("best_model"),
        "skill_check_passed": skill_check_passed,
        "feature_count": len(getattr(state, "features", []) or []),
    }


# Kubernetes/Railway-style split probes. Liveness = "the process is up";
# readiness = "safe to route prediction traffic". A degraded boot (no model
# yet) stays LIVE (so the platform doesn't kill it) but NOT READY.
_GEOJSON_MAX_AGE_DAYS = float(os.getenv("READY_GEOJSON_MAX_AGE_DAYS", "7"))


@app.get("/health/live")
def health_live():
    """Process liveness only — always 200 while the event loop is serving."""
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready(request: Request):
    """Readiness: 200 only when the model, metadata, and a *fresh* GeoJSON are
    all present; otherwise 503 with a stable schema so the platform withholds
    traffic (and a fresh deploy doesn't crash-loop — it just isn't ready yet)."""
    state = request.app.state
    geojson_age_days: Optional[float] = None
    if os.path.exists(GEOJSON_PATH):
        try:
            geojson_age_days = round(
                (datetime.now().timestamp() - os.path.getmtime(GEOJSON_PATH)) / 86400.0, 2)
        except OSError:
            geojson_age_days = None

    checks = {
        "model_loaded": getattr(state, "model", None) is not None,
        "features_loaded": bool(getattr(state, "features", None)),
        "metadata_present": os.path.exists(META_PATH),
        "geojson_present": os.path.exists(GEOJSON_PATH),
        "geojson_fresh": geojson_age_days is not None
                         and geojson_age_days <= _GEOJSON_MAX_AGE_DAYS,
    }
    latest_value = (
        state.df["date"].max()
        if getattr(state, "df", None) is not None and not state.df.empty
        else None
    )
    data_stale_days: Optional[int] = None
    if latest_value is not None:
        try:
            data_stale_days = (
                datetime.now(timezone.utc).date()
                - pd.Timestamp(latest_value).date()
            ).days
        except (TypeError, ValueError, OverflowError):
            data_stale_days = None
    ready = all(checks.values())
    body = {"ready": ready, "checks": checks,
            "geojson_age_days": geojson_age_days,
            "data_stale_days": data_stale_days,
            "model_version": getattr(state, "model_version", None)}
    if not ready:
        return JSONResponse(status_code=503, content={"error": "not_ready", **body})
    return body


# ─── automation ───────────────────────────────────────────────────────────────

# Shared secret for the privileged endpoints (admin + notify). In development
# (APP_ENV != production) an unset token leaves them open for convenience. In
# production an unset token FAILS CLOSED — the endpoints return 503 rather than
# silently running unauthenticated.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


def _check_admin(request: Request) -> None:
    """Authorize a privileged request, or raise. Fail-closed in production."""
    if not ADMIN_TOKEN:
        if security.IS_PRODUCTION:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "admin_disabled",
                    "detail": "This endpoint is disabled because ADMIN_TOKEN is "
                              "not configured in production.",
                },
            )
        return  # dev convenience: open when no token is set
    if request.headers.get("X-Admin-Token") != ADMIN_TOKEN:
        raise HTTPException(
            status_code=403,
            detail={"error": "forbidden",
                    "detail": "Invalid or missing X-Admin-Token."},
        )


@app.get("/api/automation/status")
def automation_status():
    """Everything monitoring needs in one call: DB row counts, last fetch,
    last training run, last prediction run (input version / no-op flag /
    model version / failure reason), GISTDA poller health, active model
    version, and scheduler next-fire times."""
    import gistda_poller as _gp
    state = app.state
    return _json_sanitize({
        "scheduler": scheduler_module.status(),
        "gistda": _gp.status(),
        "prediction": db_module.last_prediction_run(),
        "active_model": {
            "version": getattr(state, "model_version", None),
            "loaded_at": getattr(state, "model_loaded_at", None),
        },
        **db_module.automation_status(),
    })


@app.post("/api/admin/sync")
def admin_sync(request: Request):
    """Manually trigger one fetch-and-predict cycle (runs in a worker thread)."""
    _check_admin(request)
    import threading

    from sync_worker import run_sync
    threading.Thread(target=run_sync, name="manual-sync", daemon=True).start()
    return {"status": "started", "detail": "sync running in background; "
            "watch /api/automation/status"}


@app.post("/api/admin/reload-model")
def admin_reload_model(request: Request):
    """Hot-reload the promoted model bundle into this running process.
    Normally automatic after a promoted retrain; this is the manual lever."""
    _check_admin(request)
    ok = reload_model_state(request.app)
    state = request.app.state
    return {
        "reloaded": ok,
        "model_version": getattr(state, "model_version", None),
        "model_loaded_at": getattr(state, "model_loaded_at", None),
    }


@app.post("/api/admin/retrain")
def admin_retrain(request: Request):
    """Manually trigger a full retrain (long: 20+ min). Refuses when one is
    already in progress."""
    _check_admin(request)
    if os.path.exists(scheduler_module.TRAIN_LOCK) or db_module.any_run_in_progress():
        raise HTTPException(status_code=409, detail="Training already in progress")
    import threading

    threading.Thread(
        target=scheduler_module.retrain_now,
        kwargs={"trigger": "manual"},
        name="manual-retrain",
        daemon=True,
    ).start()
    return {"status": "started", "detail": "training in background; "
            "watch /api/automation/status"}


def _json_sanitize(obj):
    """Recursively replace inf/nan floats with None.

    FastAPI's default JSON encoder uses Python's json with allow_nan=False
    (RFC 8259 compliance), so any inf/nan in the payload raises 500. sklearn's
    roc_curve returns thresholds[0]=inf, scientific_stats writes that into
    GeoJSON metadata, and one stray value bricks the whole response. This
    sanitizer is the belt-and-suspenders safety net.
    """
    import math
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
    return obj


@app.get("/metadata")
def metadata():
    return _json_sanitize(_load_metadata())


@app.get("/metrics")
def metrics(request: Request):
    """Real held-out validation/test metrics from train.py."""
    meta = request.app.state.meta or {}
    model_block = meta.get("model", {}) or {}
    return _json_sanitize({
        "best_model": meta.get("best_model"),
        "val_metrics": model_block.get("val_metrics", {}),
        "test_metrics": model_block.get("test_metrics", {}),
        # These are the operative HIGH/WATCH probability cutoffs used by every
        # prediction endpoint. Keep the old days-scale mapping only as an
        # explicitly deprecated compatibility field so API consumers cannot
        # mistake HIGH=2 for a probability threshold.
        "probability_tier_thresholds": meta.get(
            "probability_tier_thresholds", {}
        ),
        "urgency_thresholds": request.app.state.thresholds,
        "urgency_thresholds_deprecated": True,
        "urgency_thresholds_note": meta.get(
            "urgency_thresholds_note",
            "Legacy days-scale fallback; not used for operational risk tiers.",
        ),
        "feature_count": len(request.app.state.features),
        "weather_active": any(c in request.app.state.features for c in FEATURES_WEATHER),
    })


@app.get("/api/rolling-eval")
def rolling_eval(_request: Request):
    """Per-month held-out AUC + positive rate as written by
    scripts/rolling_eval.py. Returned as a flat list the Reports page can
    feed directly to the RollingAucChart. Empty list if the script hasn't
    been run yet (the file at outputs/metadata/rolling_eval.json is absent).
    """
    # OUTPUT_DIR may be a mounted production volume (/srv/outputs). Reading
    # from BASE_DIR/outputs silently hid a valid rolling report in containers.
    path = os.path.join(OUTPUT_DIR, "metadata", "rolling_eval.json")
    if not exists(path):
        return {"summary": None, "months": []}
    try:
        data = read_json(path)
    except (OSError, json.JSONDecodeError) as e:
        return _json_sanitize({"summary": None, "months": [], "error": str(e)})
    # Frontend type RollingMonthPoint expects {month, auc, positive_rate, n}.
    months = []
    for m in data.get("months", []):
        months.append({
            "month": m.get("month"),
            "auc": m.get("auc"),
            "positive_rate": m.get("positive_rate"),
            "n": m.get("n"),
        })
    return _json_sanitize({"summary": data.get("summary"), "months": months})


@app.get("/api/training-summary")
def training_summary(request: Request):
    """Structured dataset + model hyperparameter summary for the Reports
    page. All fields are read verbatim from outputs/metadata/dataset_info.json
    (the file train.py persists at the end of training). No fabrication.
    """
    meta = request.app.state.meta or {}
    model_block = meta.get("model", {}) or {}
    return _json_sanitize({
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
        "model_type": model_block.get("type") or meta.get("best_model"),
        "search_method": model_block.get("search_method"),
        "search_iterations": model_block.get("n_iter"),
        "cv_n_splits": model_block.get("n_splits"),
        "cv_gap_days": model_block.get("cv_method") or model_block.get("ts_split_gap_days"),
        "ensemble_size": model_block.get("n_ensemble"),
        "early_stopping_rounds": model_block.get("early_stopping_rounds"),
        "best_params": model_block.get("best_params"),
    })


@app.get("/predictions/today")
def predictions_today(request: Request):
    """Calibrated 3-day hotspot risk for every scored cell on the latest base
    date. Reports probability + risk tier (High/Watch/Background) — never an
    exact fire date. Same numbers as /geojson and /predict/location."""
    state = _require_ready_state(request)
    scored, base_date, ctx = _cached_snapshot(state)

    scored = scored.merge(
        state.historical_counts, on=["lat_grid", "lon_grid"], how="left"
    )
    scored["historical_fire_count_30d"] = (
        scored["historical_fire_count_30d"].fillna(0).astype(int)
    )
    tier_summary = (
        scored["risk_tier"].value_counts(dropna=True).to_dict()
        if scored["risk_tier"].notna().any() else {}
    )

    return _json_sanitize({
        **ctx,
        "total_locations": len(scored),
        "tier_summary": tier_summary,
        "predictions": scored[
            ["lat_grid", "lon_grid", "probability", "risk_tier",
             "historical_fire_count_30d"]
        ].to_dict(orient="records"),
    })


@app.get("/predictions/tiers")
def predictions_tiers(request: Request):
    """Risk-tier distribution for the latest base date. Replaces the former
    date-keyed `/predictions/timeline` — the binary model has no per-day
    timeline, only a single next-`horizon_days` risk window."""
    state = _require_ready_state(request)
    scored, base_date, ctx = _cached_snapshot(state)
    counts = {t: 0 for t in inference.RISK_TIERS}
    for t, n in scored["risk_tier"].value_counts(dropna=True).items():
        counts[str(t)] = int(n)
    return _json_sanitize({**ctx, "tier_summary": counts,
                           "total_locations": len(scored)})


@app.get("/predictions/timeline")
def predictions_timeline(request: Request):
    """Deprecated: the binary model does not resolve a per-day timeline.
    Kept as a stable 410 that points callers at the tier distribution."""
    raise HTTPException(
        status_code=410,
        detail={
            "error": "endpoint_removed",
            "detail": "Per-day timeline is not supported — the model predicts a "
                      "single calibrated probability of a hotspot within the "
                      "next few days, not a date. Use /predictions/tiers.",
            "replacement": "/predictions/tiers",
        },
    )


@app.get("/predictions/day/{day}")
def predictions_for_day(day: int, request: Request):
    """Deprecated: the day-0..7 buckets were probability-rank bins, never
    calendar days. Removed so no surface implies an exact fire date."""
    raise HTTPException(
        status_code=410,
        detail={
            "error": "endpoint_removed",
            "detail": "Day-offset predictions are not supported — the model does "
                      "not predict a fire date. Filter by risk tier instead.",
            "replacement": "/predictions/tiers",
        },
    )


@app.get("/geojson")
def get_geojson():
    if not exists(GEOJSON_PATH):
        raise HTTPException(status_code=404, detail="GeoJSON not generated yet. Run risk_map.py.")
    data = read_json(GEOJSON_PATH)
    return _json_sanitize(data)


@app.get("/predict/location")
def predict_location(
    lat: float,
    lon: float,
    request: Request = None,
):
    """Look up the prediction for the cell containing (lat, lon).

    Grid size is read from the SERVED MODEL metadata (not a client parameter) so
    a caller can never request a resolution the model wasn't trained at. The
    coordinate is snapped with the one canonical grid function shared with
    training (gridutil), and matched by INTEGER cell index — exactly zero or one
    cell can match, with no float tolerance.
    """
    # This is a Thailand-only service — validate against the model BBOX
    # (lon 96..107, lat 4..22) so out-of-country coordinates get a clear 422.
    _bbox = [float(x) for x in os.getenv("FIRMS_BBOX", "96,4,107,22").split(",")]
    lon_min, lat_min, lon_max, lat_max = _bbox
    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "coordinates_out_of_bounds",
                "detail": f"lat/lon must be within Thailand "
                          f"(lat {lat_min}..{lat_max}, lon {lon_min}..{lon_max}).",
            },
        )

    state = _require_ready_state(request)
    df = state.df
    # Grid size comes from the served model's metadata (fallback to GRID_SIZE
    # env, then 0.1) — never from the client.
    grid_size = float((getattr(state, "meta", None) or {}).get(
        "grid_size", os.getenv("GRID_SIZE", "0.1")))

    lat_idx, lon_idx = gridutil.cell_index(lat, lon, grid_size)
    lat_grid, lon_grid = gridutil.snap_to_grid(lat, lon, grid_size)

    latest_date = df["date"].max()
    # Match by canonical INTEGER cell index (same numpy rounding as training),
    # so exactly zero or one cell matches — no banker-rounding ambiguity, no
    # float tolerance. Compute indices once and compare as integers.
    day = df[df["date"] == latest_date]
    day_lat_idx = np.round(day["lat_grid"].to_numpy() / grid_size).astype(int)
    day_lon_idx = np.round(day["lon_grid"].to_numpy() / grid_size).astype(int)
    location_data = day[(day_lat_idx == lat_idx) & (day_lon_idx == lon_idx)].copy()

    if location_data.empty:
        # Not an error state: the cell is simply outside the model's universe
        # (no historical FIRMS detection). Report that honestly, not "no risk".
        return _json_sanitize({
            "location": {"lat": lat_grid, "lon": lon_grid},
            "base_date": str(latest_date),
            "scored": False,
            "probability": None,
            "risk_tier": None,
            "coverage_note": (
                "This cell has no historical FIRMS detection, so the model does "
                "not evaluate it. Unscored ≠ zero risk."
            ),
            "model_version": getattr(state, "model_version", None),
            "horizon_days": int((getattr(state, "meta", None) or {}).get("imminent_days", 3)),
        })

    cutoffs = _state_cutoffs(state)
    scored = inference.score_cells(state.model, location_data, state.features, cutoffs)
    ctx = inference.snapshot_context(
        getattr(state, "meta", None), latest_date,
        getattr(state, "model_version", None), cutoffs,
    )

    # Use the MATCHED cell's own canonical coordinates for the history lookup
    # (matched via the same integer index), so the count is for exactly this
    # cell — never a float-mismatched neighbour.
    matched_lat = float(location_data["lat_grid"].iloc[0])
    matched_lon = float(location_data["lon_grid"].iloc[0])
    counts = state.historical_counts
    c_lat_idx = np.round(counts["lat_grid"].to_numpy() / grid_size).astype(int)
    c_lon_idx = np.round(counts["lon_grid"].to_numpy() / grid_size).astype(int)
    hist_row = counts[(c_lat_idx == lat_idx) & (c_lon_idx == lon_idx)]
    historical_count = int(hist_row["historical_fire_count_30d"].iloc[0]) if len(hist_row) else 0

    return _json_sanitize({
        **ctx,
        "location": {"lat": matched_lat, "lon": matched_lon},
        "scored": True,
        "probability": float(scored["probability"].iloc[0]),
        "risk_tier": scored["risk_tier"].iloc[0],
        "historical_fire_count_30d": historical_count,
    })


# =====================================
# STATIC SPA SERVING
# =====================================
# /api/notify — operator alert-dispatch stub
# =====================================
#
# Production stub: receives operator alert payloads and writes them to an
# in-memory ring buffer (deque, last 100 records). Does not deliver SMS /
# LINE / Email — real delivery requires a per-channel provider integration
# (LINE Messaging API, Twilio, SendGrid, etc.) and per-recipient consent.
# Operators on the /web/ Notify page see immediate "queued" confirmation;
# the entry shows in the call log below.
#
# Thread-safety: a Lock guards both list mutation and snapshot reads so the
# log endpoint never sees a half-formed record under FastAPI's threadpool.
_NOTIFY_LOG: "deque[dict]" = deque(maxlen=100)
_NOTIFY_LOG_LOCK = Lock()

_VALID_CHANNELS = {"sms", "line", "email", "all"}
_VALID_PRIORITIES = {"normal", "urgent", "emergency"}


class NotifyRequest(BaseModel):
    """Payload from /web/ Notify page when an operator dispatches an alert."""
    channel: Literal["sms", "line", "email", "all"]
    recipients: List[str] = Field(..., min_length=1, max_length=500)
    message: str = Field(..., min_length=1, max_length=2000)
    zone_ids: List[str] = Field(default_factory=list, max_length=500)
    priority: Literal["normal", "urgent", "emergency"] = "normal"
    template: Optional[str] = None


@app.post("/api/notify")
def post_notify(payload: NotifyRequest, request: Request):
    """Queue an alert. Stub — logs to a ring buffer, does NOT deliver (real
    delivery needs an authenticated per-channel provider + recipient consent).

    Privileged: requires ADMIN_TOKEN in production (fail-closed). Recipient
    addresses and message bodies are NEVER stored or echoed — only counts —
    so the audit log cannot leak PII if it is ever exposed.
    """
    _check_admin(request)
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "channel": payload.channel,
        "recipients_count": len(payload.recipients),
        "zone_ids_count": len(payload.zone_ids),
        "priority": payload.priority,
        "template": payload.template,
        "message_length": len(payload.message),
        "status": "queued",
    }
    with _NOTIFY_LOG_LOCK:
        _NOTIFY_LOG.appendleft(record)
    return {"status": "queued", "id": record["id"], "timestamp": record["timestamp"]}


@app.get("/api/notify/log")
def get_notify_log(request: Request, limit: int = 100):
    """Most-recent notify dispatches (counts only, no recipients/message).
    Privileged: requires ADMIN_TOKEN in production (fail-closed)."""
    _check_admin(request)
    limit = max(1, min(limit, 100))
    with _NOTIFY_LOG_LOCK:
        snapshot = list(_NOTIFY_LOG)[:limit]
    return {"count": len(snapshot), "records": snapshot}


# =====================================
# Live hotspots (GISTDA) — read-only views over the poller's persistent cache
# =====================================
#
# Data path: scheduler → gistda_poller.poll_once() (every GISTDA_POLL_SECONDS,
# default 300 s, PostgreSQL-advisory-lock guarded so multiple Railway
# replicas never poll concurrently) → gistda_client (retry/backoff) →
# db.live_hotspots (or the documented in-memory fallback without a DB).
#
# The endpoints below only READ that cache. No browser request ever triggers
# a GISTDA fetch, so client traffic cannot multiply load on the upstream
# ArcGIS service. These are OBSERVED satellite detections — a separate layer
# from the model's predictions, never mixed with them.

import gistda_poller as _gistda_poller

GISTDA_POLL_SECONDS = _gistda_poller.GISTDA_POLL_SECONDS
CLIENT_REFRESH_SECONDS = int(os.getenv("CLIENT_REFRESH_SECONDS", "60"))


def _live_to_legacy_attributes(rec: dict) -> dict:
    """Normalized live record → the GistdaFeature.attributes shape the SPA
    consumes (utils/gistda.ts → LiveFiresPage / analytics / map layer).
    `date` carries the full detection instant as Unix ms (the 13-digit path
    in the frontend's _formatGistdaTime)."""
    ms = None
    try:
        ms = int(
            datetime.fromisoformat(str(rec["detected_at_utc"])).timestamp() * 1000
        )
    except (ValueError, TypeError, KeyError):
        pass
    return {
        "latitude": rec.get("latitude"),
        "longitude": rec.get("longitude"),
        "confident": rec.get("confidence"),
        "lu_name": rec.get("land_use") or "",
        "pv_tn": rec.get("province") or "",
        "ap_tn": rec.get("district") or "",
        "tb_tn": rec.get("subdistrict") or "",
        "date": ms,
        "satellite": rec.get("satellite") or "",
        # Extras (additive — old consumers ignore them):
        "hotspot_id": rec.get("hotspot_id"),
        "detected_by": rec.get("detected_by") or [],
        "detected_at_thailand": rec.get("detected_at_thailand"),
        "source": "GISTDA",
    }


def _parse_bbox_param(bbox: Optional[str]) -> Optional[tuple]:
    """Strict bbox validation → HTTP 422 on any client error (never a 503):
    four finite numbers, legal coordinate ranges, min < max on both axes."""
    import math
    if not bbox:
        return None
    parts = bbox.split(",")
    try:
        vals = tuple(float(p) for p in parts)
        if len(vals) != 4 or any(not math.isfinite(v) for v in vals):
            raise ValueError
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="bbox must be four finite numbers: 'min_lon,min_lat,max_lon,max_lat'",
        )
    min_lon, min_lat, max_lon, max_lat = vals
    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0
            and -90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        raise HTTPException(status_code=422, detail="bbox coordinates out of range")
    if not (min_lon < max_lon and min_lat < max_lat):
        raise HTTPException(
            status_code=422, detail="bbox requires min_lon < max_lon and min_lat < max_lat",
        )
    return vals


def _parse_since_param(since: Optional[str]) -> Optional[str]:
    """`since` must be timezone-aware ISO-8601; normalized to UTC ISO."""
    if not since:
        return None
    try:
        dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="since must be an ISO-8601 timestamp, e.g. 2026-07-13T00:00:00+00:00",
        )
    if dt.tzinfo is None:
        raise HTTPException(
            status_code=422,
            detail="since must be timezone-aware (append +00:00 or Z for UTC)",
        )
    return dt.astimezone(timezone.utc).isoformat()


def _live_status_payload() -> dict:
    status = _gistda_poller.status()
    # next_poll_at from the running scheduler, when known
    for job in (scheduler_module.status().get("jobs") or []):
        if job.get("id") == "gistda_poll":
            status["next_poll_at"] = job.get("next_run")
    status["client_refresh_seconds"] = CLIENT_REFRESH_SECONDS
    return status


@app.get("/live-hotspots")
def live_hotspots(
    since: Optional[str] = None,
    province: Optional[str] = None,
    satellite: Optional[str] = None,
    limit: int = 2000,
    bbox: Optional[str] = None,
):
    """Cached GISTDA detections (observations, NOT predictions).

    Filters: `since` (ISO-8601, on detection time), `province` (Thai name),
    `satellite`, `bbox` (min_lon,min_lat,max_lon,max_lat), `limit`.
    When the upstream is temporarily failing, the last cached data is
    returned with metadata.is_stale=true instead of an error — the map
    should degrade, not break.
    """
    bbox_vals = _parse_bbox_param(bbox)
    since_iso = _parse_since_param(since)
    if not (1 <= int(limit) <= 10000):
        raise HTTPException(status_code=422, detail="limit must be 1..10000")
    try:
        records = db_module.get_live_hotspots(
            since=since_iso, province=province, satellite=satellite,
            bbox=bbox_vals, limit=limit,
        )
    except Exception as exc:
        _log.error("live_hotspots store read failed: %s", exc)
        raise HTTPException(status_code=503, detail="live-hotspot store unavailable")

    meta = _live_status_payload()
    return _json_sanitize({
        "metadata": {
            "source": "GISTDA",
            "layer_kind": "observed_hotspots",   # observations — not predictions
            "count": len(records),
            "server_time_utc": datetime.now(timezone.utc).isoformat(),
            **meta,
        },
        "hotspots": records,
        # Legacy shape consumed by the existing SPA code paths.
        "features": [
            {"attributes": _live_to_legacy_attributes(r)} for r in records
        ],
    })


@app.get("/live-hotspots/status")
def live_hotspots_status():
    """Poller health: last poll/success, staleness, counts, next poll time."""
    return _json_sanitize(_live_status_payload())


@app.get("/api/fires/latest")
def fires_latest(limit: int = 2000):
    """Legacy snapshot endpoint — same contract as before ({count, features,
    poll_interval_s}), now served from the persistent poller cache."""
    if not (1 <= int(limit) <= 10000):
        raise HTTPException(status_code=422, detail="limit must be 1..10000")
    try:
        records = db_module.get_live_hotspots(limit=limit)
    except Exception as exc:
        _log.error("fires_latest store read failed: %s", exc)
        raise HTTPException(status_code=503, detail="live-hotspot store unavailable")
    return _json_sanitize({
        "count": len(records),
        "features": [{"attributes": _live_to_legacy_attributes(r)} for r in records],
        "poll_interval_s": GISTDA_POLL_SECONDS,
    })


@app.get("/api/fires/stream")
async def fires_stream(request: Request):
    """SSE: pushes newly detected GISTDA hotspots to connected dashboards.

    Multi-replica correct: the cursor walks live_hotspots.first_seen_at in
    PostgreSQL, which every replica shares — clients receive events no
    matter which replica polled GISTDA or which replica owns their
    connection. The cursor starts at connect time, so restarts/reconnects
    never replay historical events (the 60 s cached refresh is the recovery
    path for anything missed while disconnected).

    Without a database the cursor falls back to the process-local store —
    fine for local dev; production multi-replica requires PostgreSQL
    (documented in railway.toml).
    """

    async def event_gen():
        yield f": connected · server polls GISTDA every {GISTDA_POLL_SECONDS}s\n\n"
        cursor = datetime.now(timezone.utc).isoformat()
        cursor_id = ""
        last_heartbeat = asyncio.get_event_loop().time()
        while True:
            if await request.is_disconnected():
                break
            try:
                new_records = db_module.get_hotspots_first_seen_after(
                    cursor, after_id=cursor_id
                )
            except Exception as exc:
                _log.error("SSE cursor read failed: %s", exc)
                new_records = []
            for rec in new_records:
                fs = str(rec.get("first_seen_at") or "")
                hotspot_id = str(rec.get("hotspot_id") or "")
                if (fs, hotspot_id) > (cursor, cursor_id):
                    cursor, cursor_id = fs, hotspot_id
                payload = {
                    "id": rec.get("hotspot_id"),
                    "attributes": _live_to_legacy_attributes(rec),
                }
                yield f"event: fire\ndata: {json.dumps(payload, default=str)}\n\n"
            now = asyncio.get_event_loop().time()
            if now - last_heartbeat >= 30.0:
                yield ": heartbeat\n\n"   # keep proxies from closing the pipe
                last_heartbeat = now
            await asyncio.sleep(5.0)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


# =====================================
# /api/analytics/hotspots — GISTDA-style dashboard stats
# =====================================
#
# Returns transboundary FIRMS hotspot counts (last 24 h) derived from the
# local FIRMS parquet cache, plus static historical burned-area figures
# sourced from GISTDA's official annual reports.
#
# Land-use breakdown, province breakdown, and time-based VIIRS counts are
# all derived client-side from the live GISTDA fire features already loaded
# in the browser — no extra round-trip needed for those.
#
# Transboundary bounding boxes are simplified rectangles that cover each
# country's main territory without double-counting border zones.
# =====================================

_ANALYTICS_CACHE: dict = {}
_ANALYTICS_CACHE_LOCK = Lock()

_NEIGHBOR_COUNTRY_BOXES = {
    "Myanmar":   (9.5,  28.5, 92.0,  101.2),
    "Laos":      (13.9, 22.5, 100.1, 107.7),
    "Vietnam":   (8.3,  23.4, 102.0, 109.5),
    "Cambodia":  (10.0, 14.7, 102.3, 107.6),
}

# Historical burned area in million rai — from GISTDA annual wildfire reports.
# Buddhist-era years: 2560=2017, 2566=2023, 2567=2024, 2568=2025.
_HISTORICAL_BURNED_MRAI = [
    {"year": "2560", "year_ce": 2017, "burned_mrai": 8.2},
    {"year": "2561", "year_ce": 2018, "burned_mrai": 6.4},
    {"year": "2562", "year_ce": 2019, "burned_mrai": 5.1},
    {"year": "2563", "year_ce": 2020, "burned_mrai": 4.8},
    {"year": "2564", "year_ce": 2021, "burned_mrai": 3.9},
    {"year": "2565", "year_ce": 2022, "burned_mrai": 5.7},
    {"year": "2566", "year_ce": 2023, "burned_mrai": 9.5},
    {"year": "2567", "year_ce": 2024, "burned_mrai": 12.1},
    {"year": "2568", "year_ce": 2025, "burned_mrai": 10.8},
]


def _firms_source_signature() -> "Optional[tuple[str, int, int]]":
    """Current FIRMS cache identity for analytics-cache invalidation."""
    firms_path = resolve_existing(
        _resolve(os.getenv(
            "FIRMS_PATH",
            os.path.join(DATA_DIR, "firms", "firms_all.parquet"),
        ))
    )
    if not firms_path:
        return None
    try:
        stat = os.stat(firms_path)
        return (firms_path, stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None


def _compute_transboundary_payload() -> dict:
    signature = _firms_source_signature()
    if signature is None:
        return {
            "available": False,
            "reason": "FIRMS cache is not available",
            "counts": [],
            "window_hours": 24,
        }
    firms_path = signature[0]
    try:
        df = pd.read_parquet(firms_path, columns=["acq_datetime", "latitude", "longitude"])
        df["acq_datetime"] = pd.to_datetime(
            df["acq_datetime"], errors="coerce", utc=True
        )
        df = df.dropna(subset=["acq_datetime", "latitude", "longitude"])
        if df.empty:
            return {
                "available": False,
                "reason": "FIRMS cache has no valid timestamped detections",
                "counts": [],
                "window_hours": 24,
            }
        # Exact rolling 24-hour window ending at the newest source detection.
        # The previous normalize()+inclusive-date filter could span almost 48 h.
        latest = df["acq_datetime"].max()
        cutoff = latest - pd.Timedelta(hours=24)
        recent = df[
            (df["acq_datetime"] > cutoff)
            & (df["acq_datetime"] <= latest)
        ]
        # Counts must be mutually exclusive. The old independent rectangles
        # double-counted the same 90 source rows as 210 "country" detections.
        # Thailand uses the bundled 77-province polygon (exact at this boundary
        # resolution). For neighboring countries, where no polygon is bundled,
        # assign each remaining point to at most one matching extent: the
        # smallest matching box is the most geographically specific. The
        # approximation and unassigned count are explicit in metadata.
        import thailand_boundary

        latitudes = recent["latitude"].to_numpy(dtype=float)
        longitudes = recent["longitude"].to_numpy(dtype=float)
        in_thailand = thailand_boundary.is_in_thailand(
            latitudes, longitudes
        )
        counts = {
            "Thailand": int(in_thailand.sum()),
            **{country: 0 for country in _NEIGHBOR_COUNTRY_BOXES},
        }
        unassigned = 0
        box_areas = {
            country: (lat_max - lat_min) * (lon_max - lon_min)
            for country, (lat_min, lat_max, lon_min, lon_max)
            in _NEIGHBOR_COUNTRY_BOXES.items()
        }
        for lat, lon, is_thailand in zip(
            latitudes, longitudes, in_thailand
        ):
            if is_thailand:
                continue
            candidates = [
                country
                for country, (lat_min, lat_max, lon_min, lon_max)
                in _NEIGHBOR_COUNTRY_BOXES.items()
                if lat_min <= lat <= lat_max
                and lon_min <= lon <= lon_max
            ]
            if not candidates:
                unassigned += 1
                continue
            assigned = min(candidates, key=lambda country: box_areas[country])
            counts[assigned] += 1

        results = [
            {"country": country, "count": count}
            for country, count in counts.items()
        ]
        results.sort(key=lambda x: x["count"], reverse=True)
        return {
            "available": True,
            "reason": None,
            "counts": results,
            "window_hours": 24,
            "window_start_utc": cutoff.isoformat(),
            "window_end_utc": latest.isoformat(),
            "source_rows_in_window": int(len(recent)),
            "assigned_rows": int(sum(counts.values())),
            "unassigned_rows": int(unassigned),
            "counts_are_mutually_exclusive": True,
            "method": (
                "Thailand administrative polygon; neighboring-country "
                "smallest-area bounding-box approximation"
            ),
        }
    except Exception as exc:
        _log.warning("transboundary query failed: %s", exc)
        return {
            "available": False,
            "reason": "FIRMS analytics could not be computed",
            "counts": [],
            "window_hours": 24,
        }


def _compute_transboundary() -> list[dict]:
    """Backward-compatible count-only helper."""
    return _compute_transboundary_payload()["counts"]


@app.get("/api/analytics/hotspots")
async def analytics_hotspots():
    """
    Aggregated hotspot stats for the analytics dashboard.
    Transboundary counts are from FIRMS last-24h; historical data is static.
    """
    signature = _firms_source_signature()

    with _ANALYTICS_CACHE_LOCK:
        cached = _ANALYTICS_CACHE.get("hotspots")
        if cached and cached.get("_source_signature") == signature:
            return {
                key: value
                for key, value in cached.items()
                if not key.startswith("_")
            }

    transboundary_payload = await asyncio.to_thread(
        _compute_transboundary_payload
    )
    window_end = transboundary_payload.get("window_end_utc")
    as_of = (
        str(pd.Timestamp(window_end).date())
        if window_end
        else str(datetime.now(timezone.utc).date())
    )

    result = {
        "as_of": as_of,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "transboundary": transboundary_payload.pop("counts"),
        "transboundary_metadata": transboundary_payload,
        "historical_burned_mrai": _HISTORICAL_BURNED_MRAI,
    }
    with _ANALYTICS_CACHE_LOCK:
        _ANALYTICS_CACHE["hotspots"] = {
            **result,
            "_source_signature": signature,
        }
    return result


# =====================================
# /api/cell_weather — per-cell temperature range + Fire Weather Index
# =====================================
#
# Reads from the ERA5 daily cache that fetch_weather.py produces:
#   data/weather/weather_cache.parquet
#     columns: lat_grid, lon_grid, date, temp_max, temp_min,
#              precip_sum (mm), wind_max (km/h), et0 (mm)
#
# Used by the map popup to show concrete weather context for whatever cell
# the operator clicked. No external API call, no key required — everything
# stays local + reproducible.
#
# Fire Weather Index (simplified, single-number proxy):
#   - "สูงมาก" if temp_max > 35°C AND precip_sum_7d < 5 mm
#   - "สูง"   if temp_max > 32°C AND precip_sum_7d < 15 mm
#   - "ปานกลาง" if temp_max > 28°C
#   - "ต่ำ" otherwise
# This collapses several ERA5 variables into something operator-readable.
_WEATHER_COLUMNS = [
    "lat_grid", "lon_grid", "date", "temp_max", "temp_min",
    "precip_sum", "wind_max", "et0",
]
_WEATHER_CACHE_DF: "Optional[pd.DataFrame]" = None
_WEATHER_CACHE_SIGNATURE: "Optional[tuple[str, int, int]]" = None
_WEATHER_CACHE_LOAD_ERROR: "Optional[str]" = None
_WEATHER_CACHE_LOCK = Lock()


def _empty_weather_cache() -> pd.DataFrame:
    return pd.DataFrame(columns=_WEATHER_COLUMNS)


def _load_weather_cache():
    """Load the current ERA5 cache and reload it when the file changes.

    A missing cache is deliberately *not* a permanent memoized state.  The
    scheduler/backfill can create the file after API startup, and the next
    popup request must see it without requiring a process restart.  Likewise,
    an atomic cache refresh is picked up by its mtime/size signature.
    """
    global _WEATHER_CACHE_DF, _WEATHER_CACHE_SIGNATURE
    global _WEATHER_CACHE_LOAD_ERROR

    actual_path = resolve_existing(WEATHER_CACHE_PATH)
    if not actual_path:
        with _WEATHER_CACHE_LOCK:
            _WEATHER_CACHE_DF = _empty_weather_cache()
            _WEATHER_CACHE_SIGNATURE = None
            _WEATHER_CACHE_LOAD_ERROR = None
            return _WEATHER_CACHE_DF

    try:
        stat = os.stat(actual_path)
        signature = (actual_path, stat.st_mtime_ns, stat.st_size)
    except OSError as exc:
        with _WEATHER_CACHE_LOCK:
            _WEATHER_CACHE_DF = _empty_weather_cache()
            _WEATHER_CACHE_SIGNATURE = None
            _WEATHER_CACHE_LOAD_ERROR = f"weather cache stat failed: {exc}"
            return _WEATHER_CACHE_DF

    if (
        _WEATHER_CACHE_DF is not None
        and _WEATHER_CACHE_SIGNATURE == signature
    ):
        return _WEATHER_CACHE_DF

    with _WEATHER_CACHE_LOCK:
        # Another worker thread may have loaded this signature while this
        # request waited for the lock.
        if (
            _WEATHER_CACHE_DF is not None
            and _WEATHER_CACHE_SIGNATURE == signature
        ):
            return _WEATHER_CACHE_DF
        previous = _WEATHER_CACHE_DF
        try:
            df = read_table(actual_path)
            required = {"lat_grid", "lon_grid", "date"}
            missing = sorted(required - set(df.columns))
            if missing:
                raise ValueError(
                    "weather cache missing required columns: "
                    + ", ".join(missing)
                )
            # Older caches may omit optional display fields. Preserve the
            # usable coordinates/date and expose those values as unknown.
            for column in _WEATHER_COLUMNS:
                if column not in df.columns:
                    df[column] = np.nan
            df = df[_WEATHER_COLUMNS].copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
            df = df.dropna(subset=["lat_grid", "lon_grid", "date"])
            _WEATHER_CACHE_DF = df
            _WEATHER_CACHE_SIGNATURE = signature
            _WEATHER_CACHE_LOAD_ERROR = None
        except Exception as exc:
            # Keep the previous known-good in-memory cache if an update is
            # observed while its file is incomplete/corrupt. Mark the failed
            # signature so identical requests do not repeatedly parse/log it;
            # a repaired file changes the signature and is retried.
            _log.exception("Weather cache load failed: %s", exc)
            _WEATHER_CACHE_DF = (
                previous if previous is not None else _empty_weather_cache()
            )
            _WEATHER_CACHE_SIGNATURE = signature
            _WEATHER_CACHE_LOAD_ERROR = "weather cache could not be read"
        return _WEATHER_CACHE_DF


def _snap_grid(value: float, grid: float = 0.1) -> float:
    return round(round(value / grid) * grid, 4)


def _fwi(temp_max: float, precip_7d: float) -> dict:
    """A simple temperature/rain context indicator, not official Canadian FWI.

    Both inputs are required. Treating missing rain as 0 mm fabricates dry
    conditions and can promote an unknown cell to "high", so incomplete
    seven-day precipitation now stays explicitly unknown.
    """
    base = {
        "method": "temperature_precipitation_proxy",
        "is_official_fwi": False,
    }
    if temp_max is None or precip_7d is None:
        return {
            **base, "level": "ไม่ทราบ", "level_key": "unknown",
            "color": "#6c707a", "emoji": "❔",
        }
    p7 = precip_7d
    if temp_max > 35 and p7 < 5:
        return {
            **base, "level": "สูงมาก", "level_key": "very_high",
            "color": "#ef4444", "emoji": "🔴",
        }
    if temp_max > 32 and p7 < 15:
        return {
            **base, "level": "สูง", "level_key": "high",
            "color": "#f97316", "emoji": "🟠",
        }
    if temp_max > 28:
        return {
            **base, "level": "ปานกลาง", "level_key": "moderate",
            "color": "#eab308", "emoji": "🟡",
        }
    return {
        **base, "level": "ต่ำ", "level_key": "low",
        "color": "#22c55e", "emoji": "🟢",
    }


@app.get("/api/cell_weather")
def cell_weather(lat: float, lon: float, date: Optional[str] = None):
    """Return ERA5 temp range + 7-day precip + Fire Weather Index for one cell.

    Inputs:
        lat, lon : query params — snapped to nearest 0.1° grid centre
        date     : YYYY-MM-DD (optional; defaults to most recent cached day)

    Returns:
        {
          temp_min_c, temp_max_c, precip_sum_mm, wind_max_kmh,
          precip_7d_mm, et0_mm, fire_weather_index: {level, color, emoji},
          date, source: "ERA5", available: bool
        }

    If the cell/date is not in the cache (e.g. weather fetch didn't cover this
    area yet) `available=false` and the rest are nulls. The frontend treats
    that as "weather not available for this point" and degrades the popup
    gracefully — never an error toast.
    """
    if not np.isfinite(lat) or not np.isfinite(lon):
        raise HTTPException(
            status_code=422,
            detail="lat and lon must be finite numbers",
        )
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(
            status_code=422,
            detail="lat/lon coordinates are out of range",
        )

    requested_target = None
    if date:
        try:
            requested_target = pd.Timestamp(
                datetime.strptime(date, "%Y-%m-%d").date()
            )
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail="date must be a real calendar date in YYYY-MM-DD format",
            )

    df = _load_weather_cache()
    if df.empty:
        return {
            "available": False,
            "reason": (
                _WEATHER_CACHE_LOAD_ERROR
                or "weather cache not built yet"
            ),
            "requested_date": date,
        }

    try:
        weather_grid = float(os.getenv("WEATHER_GRID_SIZE",
                                       os.getenv("GRID_SIZE", "0.1")))
        if not np.isfinite(weather_grid) or weather_grid <= 0:
            raise ValueError
    except ValueError:
        weather_grid = 0.1
    lat_snap = _snap_grid(lat, weather_grid)
    lon_snap = _snap_grid(lon, weather_grid)
    # Parquet/CSV float representations need not be bit-identical to the query
    # value. A tiny tolerance matches the same canonical grid cell without
    # accidentally selecting a neighbour.
    cell = df[
        np.isclose(df["lat_grid"].astype(float), lat_snap, atol=1e-6)
        & np.isclose(df["lon_grid"].astype(float), lon_snap, atol=1e-6)
    ]
    if cell.empty:
        return {
            "available": False,
            "reason": "no ERA5 data for this cell",
            "requested_date": date,
            "lat_grid": lat_snap,
            "lon_grid": lon_snap,
        }

    # Resolve target date
    target = requested_target if requested_target is not None else cell["date"].max()

    row = cell[cell["date"] == target]
    is_date_fallback = False
    fallback_days = 0
    if row.empty:
        # Fall back to the nearest available date (most recent <= target)
        before = cell[cell["date"] <= target].sort_values("date").tail(1)
        if before.empty:
            return {
                "available": False,
                "reason": "no data on/before target date",
                "requested_date": date,
            }
        row = before
        actual_target = pd.Timestamp(row["date"].iloc[0]).normalize()
        fallback_days = int((target - actual_target).days)
        try:
            max_fallback_days = max(
                0, int(os.getenv("WEATHER_MAX_FALLBACK_DAYS", "7"))
            )
        except ValueError:
            max_fallback_days = 7
        if requested_target is not None and fallback_days > max_fallback_days:
            return {
                "available": False,
                "reason": (
                    f"local ERA5 cache is {fallback_days} days older than "
                    f"the requested date (maximum {max_fallback_days})"
                ),
                "requested_date": date,
                "latest_available_date": str(actual_target.date()),
            }
        target = actual_target
        is_date_fallback = requested_target is not None
    r = row.iloc[0]

    # Seven-day precipitation must contain all seven calendar days and seven
    # real values. A partial sum systematically understates rain and makes the
    # proxy look more fire-prone than the evidence supports.
    week_start = target - pd.Timedelta(days=6)
    week = cell[
        (cell["date"] >= week_start) & (cell["date"] <= target)
    ][["date", "precip_sum"]].copy()
    week["date"] = pd.to_datetime(week["date"]).dt.normalize()
    week = (
        week.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .set_index("date")
    )
    expected_days = pd.date_range(week_start, target, freq="D")
    has_complete_week = bool(
        len(week) >= 7
        and all(day in week.index for day in expected_days)
        and week.reindex(expected_days)["precip_sum"].notna().all()
    )
    precip_7d = (
        float(week.reindex(expected_days)["precip_sum"].sum())
        if has_complete_week
        else None
    )

    fwi = _fwi(float(r["temp_max"]) if pd.notna(r["temp_max"]) else None, precip_7d)

    return _json_sanitize({
        "available": True,
        "source": "ECMWF ERA5 (daily, via Open-Meteo)",
        "lat_grid": lat_snap,
        "lon_grid": lon_snap,
        "date": str(pd.Timestamp(target).date()),
        "requested_date": date,
        "is_date_fallback": is_date_fallback,
        "fallback_days": fallback_days if is_date_fallback else 0,
        "temp_min_c":    None if pd.isna(r["temp_min"])    else round(float(r["temp_min"]), 1),
        "temp_max_c":    None if pd.isna(r["temp_max"])    else round(float(r["temp_max"]), 1),
        "precip_sum_mm": None if pd.isna(r["precip_sum"])  else round(float(r["precip_sum"]), 1),
        "wind_max_kmh":  None if pd.isna(r["wind_max"])    else round(float(r["wind_max"]), 1),
        "et0_mm":        None if pd.isna(r["et0"])         else round(float(r["et0"]), 2),
        "precip_7d_mm":  None if precip_7d is None         else round(precip_7d, 1),
        "precip_7d_complete": has_complete_week,
        "fire_weather_index": fwi,
    })


# =====================================
# Static SPA serving (Vite build output)
# =====================================
#
# The Vite-built React dashboard is served from the same FastAPI app. All
# JSON API routes are registered above; the catch-all SPA fallback below is
# registered LAST so specific API routes win route matching. Unknown paths
# fall back to index.html so future client-side routing (or a deep-link
# refresh) keeps working.

_API_PATH_ROOTS = {
    "api", "health", "predictions", "predict", "live-hotspots",
    "geojson", "metrics", "metadata",
}


def _is_api_namespace_path(full_path: str) -> bool:
    """True for API/probe namespaces that must never fall back to index.html."""
    root = full_path.strip("/").split("/", 1)[0]
    return root in _API_PATH_ROOTS


def _safe_spa_candidate(full_path: str) -> Optional[str]:
    """Resolve a requested static file while keeping it inside WEB_DIST_DIR."""
    root = os.path.realpath(WEB_DIST_DIR)
    candidate = os.path.realpath(os.path.join(root, full_path))
    try:
        if os.path.commonpath([root, candidate]) != root:
            return None
    except ValueError:
        # Different drives on Windows, or otherwise incomparable paths.
        return None
    return candidate


if os.path.isdir(WEB_DIST_DIR):
    _ASSETS_DIR = os.path.join(WEB_DIST_DIR, "assets")
    if os.path.isdir(_ASSETS_DIR):
        # Vite's hashed JS/CSS bundles live in /assets — mount them as a
        # real static directory so cache headers + 404s behave correctly.
        app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def _spa_fallback(full_path: str):
        # A typo under /api previously returned index.html with HTTP 200. That
        # looks like valid transport to fetch(), then fails as mysterious JSON.
        # API namespaces must preserve an honest JSON 404.
        if _is_api_namespace_path(full_path):
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "api_route_not_found",
                    "path": f"/{full_path}",
                },
            )
        candidate = _safe_spa_candidate(full_path)
        # Block traversal. commonpath is segment-aware; a string-prefix check
        # incorrectly allowed sibling names such as "dist-secret".
        if candidate is None:
            raise HTTPException(status_code=404)
        if full_path and os.path.isfile(candidate):
            return FileResponse(candidate)
        index = os.path.join(WEB_DIST_DIR, "index.html")
        if not os.path.exists(index):
            raise HTTPException(status_code=404, detail="SPA not built")
        # HTML points at content-hashed assets and must revalidate so a normal
        # refresh picks up a newly deployed bundle without Ctrl+Shift+R.
        return FileResponse(index, headers={"Cache-Control": "no-cache"})
else:
    print(f"⚠️  WEB_DIST_DIR not found at {WEB_DIST_DIR} — SPA will not be served. "
          "Run `npm run build` in /web or set WEB_DIST_DIR.")
