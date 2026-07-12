"""APScheduler wiring — runs inside the FastAPI process (Railway).

Jobs:
  sync_job          every SYNC_INTERVAL_MINUTES (default 10)
                    → sync_worker.run_sync()
  retrain_check_job every 6 hours
                    → retrains when the last successful run is older than
                      RETRAIN_EVERY_DAYS (default 3)

Why "check every 6 h" instead of IntervalTrigger(days=3): Railway restarts
the process on every deploy, which would reset a 3-day interval clock and
push retraining forever into the future. The check job is stateless — it
compares against a persisted timestamp (model_runs table, or the trained_at
field in dataset_info.json when no DB is configured) so restarts are safe.

Overlap safety:
  - max_instances=1 + coalesce=True on every job (no pile-ups after sleep)
  - outputs/.train.lock guards train.py vs. the sync job's predict subprocess
    (both write full_features.parquet)
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import db

log = logging.getLogger("scheduler")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")


def _resolve(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)


OUTPUT_DIR = _resolve(os.environ.get("OUTPUT_DIR", "./outputs"))
META_PATH = os.path.join(OUTPUT_DIR, "metadata", "dataset_info.json")
TRAIN_LOCK = os.path.join(OUTPUT_DIR, ".train.lock")

SYNC_INTERVAL_MINUTES = int(os.getenv("SYNC_INTERVAL_MINUTES", "10"))
RETRAIN_EVERY_DAYS = float(os.getenv("RETRAIN_EVERY_DAYS", "3"))
RETRAIN_CHECK_HOURS = int(os.getenv("RETRAIN_CHECK_HOURS", "6"))
TRAIN_ARGS = os.getenv("TRAIN_ARGS", "--quick").split()
TRAIN_TIMEOUT_S = int(os.getenv("TRAIN_TIMEOUT_S", str(3 * 3600)))

_scheduler = None


# ─── jobs ─────────────────────────────────────────────────────────────────────

def sync_job():
    from sync_worker import run_sync
    run_sync()


def _last_train_time():
    """Last successful training, from DB first, dataset_info.json fallback."""
    ts = db.last_successful_run_at()
    if ts is not None:
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        with open(META_PATH) as f:
            trained_at = json.load(f).get("trained_at")
        if trained_at:
            parsed = datetime.fromisoformat(str(trained_at).replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return None


def _read_metrics_from_meta() -> dict:
    try:
        with open(META_PATH) as f:
            meta = json.load(f)
        tm = (meta.get("model") or {}).get("test_metrics", {}) or {}
        return {
            "roc_auc": tm.get("roc_auc"),
            "f1": tm.get("f1"),
            "n_train_rows": meta.get("training_rows"),
            "n_positive": tm.get("n_positive"),
        }
    except (OSError, json.JSONDecodeError):
        return {}


def retrain_check_job():
    last = _last_train_time()
    if last is not None:
        age = datetime.now(timezone.utc) - last
        if age < timedelta(days=RETRAIN_EVERY_DAYS):
            log.info("Model is %.1f days old (< %s) — no retrain needed",
                     age.total_seconds() / 86400, RETRAIN_EVERY_DAYS)
            return
    if os.path.exists(TRAIN_LOCK) or db.any_run_in_progress():
        log.info("Training already in progress — skipping")
        return
    retrain_now(trigger="scheduled")


def retrain_now(trigger: str = "manual") -> dict:
    """Blocking retrain (runs on a scheduler worker thread, not the event
    loop). Wraps train.py in a lock file + model_runs bookkeeping."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_id = db.start_model_run(trigger=trigger)
    with open(TRAIN_LOCK, "w") as f:
        f.write(str(os.getpid()))
    log.info("Retraining started (trigger=%s, args=%s)", trigger, TRAIN_ARGS)
    try:
        proc = subprocess.run(
            [sys.executable, "train.py", *TRAIN_ARGS],
            cwd=SRC_DIR, timeout=TRAIN_TIMEOUT_S,
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            metrics = _read_metrics_from_meta()
            db.finish_model_run(run_id, "done", metrics=metrics)
            log.info("Retraining finished OK — metrics: %s", metrics)
            return {"status": "done", "metrics": metrics}
        err = (proc.stderr or "")[-4000:]
        db.finish_model_run(run_id, "failed", error=err)
        log.error("Retraining failed (rc=%s): %s", proc.returncode, err)
        return {"status": "failed", "error": err}
    except subprocess.TimeoutExpired:
        db.finish_model_run(run_id, "failed", error="timeout")
        log.error("Retraining timed out after %ss", TRAIN_TIMEOUT_S)
        return {"status": "failed", "error": "timeout"}
    except Exception as exc:  # noqa: BLE001
        db.finish_model_run(run_id, "failed", error=str(exc))
        log.exception("Retraining crashed")
        return {"status": "failed", "error": str(exc)}
    finally:
        try:
            os.remove(TRAIN_LOCK)
        except OSError:
            pass


# ─── lifecycle ────────────────────────────────────────────────────────────────

def start() -> bool:
    """Start the background scheduler. Called from api.py's lifespan."""
    global _scheduler
    if _scheduler is not None:
        return True
    from apscheduler.schedulers.background import BackgroundScheduler

    # A crashed process can leave a stale lock behind — clear it on boot.
    if os.path.exists(TRAIN_LOCK):
        log.warning("Removing stale train lock from previous run")
        try:
            os.remove(TRAIN_LOCK)
        except OSError:
            pass

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        sync_job, "interval",
        minutes=SYNC_INTERVAL_MINUTES,
        id="sync", max_instances=1, coalesce=True,
        misfire_grace_time=120,
        next_run_time=datetime.now(timezone.utc) + timedelta(seconds=30),
    )
    _scheduler.add_job(
        retrain_check_job, "interval",
        hours=RETRAIN_CHECK_HOURS,
        id="retrain_check", max_instances=1, coalesce=True,
        misfire_grace_time=3600,
        next_run_time=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    _scheduler.start()
    log.info("Scheduler started: sync every %d min, retrain check every %d h "
             "(retrain when model older than %s days)",
             SYNC_INTERVAL_MINUTES, RETRAIN_CHECK_HOURS, RETRAIN_EVERY_DAYS)
    return True


def shutdown():
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        log.info("Scheduler stopped")


def status() -> dict:
    if _scheduler is None:
        return {"running": False}
    return {
        "running": True,
        "jobs": [
            {
                "id": j.id,
                "next_run": j.next_run_time.isoformat() if j.next_run_time else None,
            }
            for j in _scheduler.get_jobs()
        ],
    }
