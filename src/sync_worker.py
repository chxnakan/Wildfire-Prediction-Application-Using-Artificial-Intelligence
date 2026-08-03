"""Sync worker: the every-10-minutes job body.

Flow:
  1. fetch_firms.update_firms() — pulls VIIRS NRT, merges into the parquet
     cache (kept for backward compat: train.py reads it).
  2. Detect new detections two ways:
       - parquet row-count delta (works with no database)
       - INSERT .. ON CONFLICT DO NOTHING rowcount (exact, when DB enabled)
  3. If anything new arrived → subprocess `train.py --predict-only` which
     rebuilds full_features.parquet and refreshes the risk map (and, via the
     risk_map DB hook, the risk_predictions table).
  4. Always append a fetch_log row so monitoring can tell "no new data"
     apart from "sync is dead".

Never raises: scheduler jobs must not die on a transient FIRMS 400/timeout.
"""

import logging
import os
import subprocess
import sys
from datetime import timedelta

import pandas as pd

import db
from fetch_firms import OUT_FILE, update_firms
from storage import read_table, resolve_existing

log = logging.getLogger("sync_worker")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")

# Lock file created by scheduler.retrain_now while train.py runs — sync must
# not rebuild features mid-training (both write full_features.parquet).
# Must resolve OUTPUT_DIR identically to scheduler.py or the two modules
# would guard different paths on Railway (OUTPUT_DIR=/data/outputs).
_OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")
if not os.path.isabs(_OUTPUT_DIR):
    _OUTPUT_DIR = os.path.join(BASE_DIR, _OUTPUT_DIR)
TRAIN_LOCK = os.path.join(_OUTPUT_DIR, ".train.lock")

FETCH_DAYS = int(os.getenv("SYNC_FETCH_DAYS", "1"))
PREDICT_MAX_HISTORY_DAYS = int(os.getenv("PREDICT_MAX_HISTORY_DAYS", "180"))
PREDICT_TIMEOUT_S = int(os.getenv("PREDICT_TIMEOUT_S", "1800"))


def _cache_row_count() -> int:
    path = resolve_existing(OUT_FILE)
    if not path or os.path.getsize(path) == 0:
        return 0
    try:
        return len(read_table(path))
    except Exception:
        return 0


def _mirror_new_rows_to_db() -> int:
    """Push cache rows newer than the DB's high-water mark into Postgres."""
    if not db.enabled():
        return 0
    path = resolve_existing(OUT_FILE)
    if not path:
        return 0
    df = read_table(path)
    df["acq_datetime"] = pd.to_datetime(df["acq_datetime"], errors="coerce")
    df = df.dropna(subset=["acq_datetime"])

    high_water = db.latest_hotspot_time()
    if high_water is not None:
        # 1-day margin: late-arriving detections for an already-seen day.
        cutoff = pd.Timestamp(high_water).tz_localize(None) - timedelta(days=1)
        df = df[df["acq_datetime"].dt.tz_localize(None) > cutoff]
    return db.insert_hotspots(df)


def _refresh_predictions() -> bool:
    """Rebuild features + risk map in a subprocess (memory isolation — the
    feature build can spike RAM and we don't want that inside the API proc)."""
    if os.path.exists(TRAIN_LOCK):
        log.info("Training in progress — skipping prediction refresh this cycle")
        return False
    cmd = [
        sys.executable, "train.py",
        "--predict-only",
        "--max-history-days", str(PREDICT_MAX_HISTORY_DAYS),
    ]
    log.info("Refreshing predictions: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, cwd=SRC_DIR, timeout=PREDICT_TIMEOUT_S,
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        log.error("Prediction refresh timed out after %ss", PREDICT_TIMEOUT_S)
        return False
    if proc.returncode != 0:
        log.error("Prediction refresh failed (rc=%s): %s",
                  proc.returncode, (proc.stderr or "")[-2000:])
        return False
    log.info("Prediction refresh complete")
    return True


# ── input-data versioning (prediction idempotency) ───────────────────────────
#
# The model's inputs are the FIRMS cache (and, when present, the weather
# cache). A deterministic version string over those inputs decides whether a
# prediction-only run is needed at all:
#
#   version = sha1(latest acq_datetime | row_count | weather mtime | schema)
#
# Properties: pure function of committed data (restart-safe — recomputed
# identically after a redeploy, unlike the old before/after row-count delta),
# cheap (no full-file hashing), and monotonic in practice (FIRMS appends).
# GISTDA data is deliberately NOT part of the version: the model has no
# GISTDA features, so a GISTDA-only change must not rerun an identical
# prediction. If GISTDA ever becomes a (past-only) model feature, add its
# high-water mark here.
INPUT_SCHEMA_VERSION = "firms-v1"
_DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
if not os.path.isabs(_DATA_DIR):
    _DATA_DIR = os.path.join(BASE_DIR, _DATA_DIR)
_WEATHER_DIR = os.getenv("WEATHER_DIR", os.path.join(_DATA_DIR, "weather"))
if not os.path.isabs(_WEATHER_DIR):
    _WEATHER_DIR = os.path.join(BASE_DIR, _WEATHER_DIR)
WEATHER_PATH = os.getenv(
    "WEATHER_PATH", os.path.join(_WEATHER_DIR, "weather_cache.parquet")
)
if not os.path.isabs(WEATHER_PATH):
    WEATHER_PATH = os.path.join(BASE_DIR, WEATHER_PATH)

# File fallback so idempotency survives restarts even without a database.
_VERSION_STATE_FILE = os.path.join(_OUTPUT_DIR, ".last_prediction_version")


def compute_input_version() -> tuple:
    """(version_string, row_count) for the current model-input data."""
    import hashlib

    path = resolve_existing(OUT_FILE)
    if not path:
        return ("empty", 0)
    df = read_table(path)
    n = len(df)
    latest = str(pd.to_datetime(df["acq_datetime"], errors="coerce").max())
    weather_mtime = ""
    if os.path.exists(WEATHER_PATH):
        weather_mtime = str(int(os.path.getmtime(WEATHER_PATH)))
    key = f"{latest}|{n}|{weather_mtime}|{INPUT_SCHEMA_VERSION}"
    return ("v1-" + hashlib.sha1(key.encode()).hexdigest()[:16], n)


def _last_predicted_version() -> str:
    v = db.last_predicted_input_version()
    if v:
        return v
    try:
        with open(_VERSION_STATE_FILE) as f:
            return f.read().strip()
    except OSError:
        return ""


def _remember_predicted_version(version: str) -> None:
    try:
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        with open(_VERSION_STATE_FILE, "w") as f:
            f.write(version)
    except OSError:
        log.warning("Could not persist prediction version state file")


def _current_model_version() -> str:
    try:
        import json
        with open(os.path.join(_OUTPUT_DIR, "metadata", "dataset_info.json")) as f:
            return str(json.load(f).get("trained_at") or "unknown")
    except Exception:
        return "unknown"


def run_sync() -> dict:
    """One sync cycle:

        fetch FIRMS → compute input version → unchanged? no-op
                                            → changed?  ONE prediction-only
                                              run (advisory-lock guarded,
                                              idempotent by version)

    GISTDA is polled by its own scheduler job and never triggers prediction
    (the model has no GISTDA features — rerunning on identical inputs would
    produce an identical snapshot). Never raises.
    """
    summary = {"new_rows": 0, "db_rows": 0, "noop": True,
               "input_version": None, "predictions_refreshed": False}
    try:
        before = _cache_row_count()
        update_firms(days=FETCH_DAYS)
        after = _cache_row_count()
        new_rows = max(after - before, 0)
        summary["new_rows"] = new_rows

        db_rows = _mirror_new_rows_to_db()
        summary["db_rows"] = db_rows

        version, _n = compute_input_version()
        summary["input_version"] = version
        last_version = _last_predicted_version()

        if version == last_version or version == "empty":
            log.info("Model input unchanged (version %s) — no-op", version)
            db.record_prediction_run("done", input_version=version, noop=True,
                                     model_version=_current_model_version())
        else:
            # Advisory lock: replicas must not run duplicate prediction jobs.
            with db.advisory_lock(db.PREDICTION_LOCK_KEY) as acquired:
                if not acquired:
                    log.info("Prediction skipped — another instance is predicting")
                    summary["noop"] = True
                else:
                    summary["noop"] = False
                    log.info(
                        "New model input (version %s ≠ %s, +%d rows) — "
                        "running prediction-only ONCE",
                        version, last_version or "∅", new_rows,
                    )
                    ok = _refresh_predictions()
                    summary["predictions_refreshed"] = ok
                    if ok:
                        _remember_predicted_version(version)
                        db.record_prediction_run(
                            "done", input_version=version, new_rows=new_rows,
                            model_version=_current_model_version(),
                        )
                    else:
                        # Old snapshot keeps serving; version NOT remembered,
                        # so the next cycle retries.
                        db.record_prediction_run(
                            "failed", input_version=version, new_rows=new_rows,
                            model_version=_current_model_version(),
                            error="prediction subprocess failed — serving last snapshot",
                        )

        db.log_fetch(
            new_rows=max(new_rows, db_rows),
            status="ok",
            detail=f"cache+{new_rows} db+{db_rows} version={version} "
                   f"noop={summary['noop']} refreshed={summary['predictions_refreshed']}",
        )
    except Exception as exc:  # noqa: BLE001 — job must survive anything
        log.exception("Sync cycle failed")
        summary["error"] = str(exc)
        try:
            db.log_fetch(new_rows=0, status="error", detail=str(exc))
        except Exception:
            pass
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    print(run_sync())
