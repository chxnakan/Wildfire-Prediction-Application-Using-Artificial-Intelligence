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


def run_sync() -> dict:
    """One sync cycle. Returns a summary dict (also logged to fetch_log)."""
    summary = {"new_rows": 0, "db_rows": 0, "predictions_refreshed": False}
    try:
        before = _cache_row_count()
        update_firms(days=FETCH_DAYS)
        after = _cache_row_count()
        new_rows = max(after - before, 0)
        summary["new_rows"] = new_rows

        db_rows = _mirror_new_rows_to_db()
        summary["db_rows"] = db_rows

        if new_rows > 0 or db_rows > 0:
            log.info("New data: %d cache rows / %d db rows — refreshing predictions",
                     new_rows, db_rows)
            summary["predictions_refreshed"] = _refresh_predictions()
        else:
            log.info("No new FIRMS data this cycle")

        db.log_fetch(
            new_rows=max(new_rows, db_rows),
            status="ok",
            detail=f"cache+{new_rows} db+{db_rows} "
                   f"refreshed={summary['predictions_refreshed']}",
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
