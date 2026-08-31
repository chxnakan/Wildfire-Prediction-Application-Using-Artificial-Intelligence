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

Each job family has its own opt-out switch.  This matters for presentation
mode: the live GISTDA observation layer may keep polling while FIRMS sync and
model retraining remain disabled.
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from threading import Lock

import db
from process_lock import process_lock_active

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
# The regional archive causally densifies to tens of millions of rows when the
# history is unbounded. Keep scheduled retrains on the same reviewed 400-day
# window used by the promoted model; operators can still override TRAIN_ARGS.
TRAIN_ARGS = os.getenv(
    "TRAIN_ARGS", "--quick --max-history-days 400"
).split()
TRAIN_TIMEOUT_S = int(os.getenv("TRAIN_TIMEOUT_S", str(3 * 3600)))

_scheduler = None
_prediction_reload_pending = False
_prediction_reload_lock = Lock()


def _env_enabled(name: str, default: bool = True) -> bool:
    """Read an opt-in/opt-out environment flag without changing defaults."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ─── jobs ─────────────────────────────────────────────────────────────────────

def sync_job():
    global _prediction_reload_pending
    from sync_worker import run_sync
    summary = run_sync()
    # predict-only keeps the promoted model but replaces the serving feature
    # frame and GeoJSON. Refresh the API's in-memory snapshot only after that
    # subprocess completed successfully; failed/no-op cycles keep the current
    # atomic state untouched.
    refreshed = summary.get("predictions_refreshed") is True
    with _prediction_reload_lock:
        if refreshed:
            _prediction_reload_pending = True
        attempted = _prediction_reload_pending
        reload_succeeded = None
        if attempted:
            reload_succeeded = _fire_prediction_refreshed()
            _prediction_reload_pending = not reload_succeeded
        summary["serving_state_reload_attempted"] = attempted
        summary["serving_state_reloaded"] = reload_succeeded
        summary["serving_state_reload_pending"] = _prediction_reload_pending
    return summary


def gistda_poll_job():
    # Single-poller safety lives inside poll_once (PostgreSQL advisory
    # lock) — this job body stays a thin shim so replicas that lose the
    # lock exit in microseconds.
    from gistda_poller import poll_once
    poll_once()


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


def _read_promotion_from_meta() -> dict:
    """Promotion block written by train.py's quality gates. Note: on a
    REJECTED candidate, train.py does not touch dataset_info.json, so a
    stale promotion block here means 'incumbent unchanged' — the model
    pkl's mtime disambiguates."""
    try:
        with open(META_PATH) as f:
            return json.load(f).get("promotion") or {}
    except (OSError, json.JSONDecodeError):
        return {}


# Callbacks fired (best-effort) after a training run PROMOTES a new model —
# api.py registers its hot-reload here at startup, avoiding a circular
# import (api imports scheduler; scheduler must not import api).
_on_model_promoted: list = []
_on_prediction_refreshed: list = []


def register_model_promoted_callback(fn) -> None:
    _on_model_promoted.append(fn)


def register_prediction_refreshed_callback(fn) -> None:
    """Register a local-process reload after predict-only artifact refresh."""
    _on_prediction_refreshed.append(fn)


def _fire_model_promoted() -> None:
    for fn in list(_on_model_promoted):
        try:
            fn()
        except Exception:
            log.exception("model-promoted callback failed")


def _fire_prediction_refreshed() -> bool:
    succeeded = True
    for fn in list(_on_prediction_refreshed):
        try:
            result = fn()
            if result is False:
                succeeded = False
                log.error(
                    "prediction-refreshed callback could not reload serving state"
                )
        except Exception:
            succeeded = False
            # The refreshed files stay on disk, while reload_model_state's
            # build-then-swap contract preserves the last good in-memory state.
            log.exception("prediction-refreshed callback failed")
    return succeeded


def retrain_check_job():
    last = _last_train_time()
    if last is not None:
        age = datetime.now(timezone.utc) - last
        if age < timedelta(days=RETRAIN_EVERY_DAYS):
            log.info("Model is %.1f days old (< %s) — no retrain needed",
                     age.total_seconds() / 86400, RETRAIN_EVERY_DAYS)
            return
    if process_lock_active(
        TRAIN_LOCK, max_age_seconds=TRAIN_TIMEOUT_S
    ) or db.any_run_in_progress():
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
        model_path = os.path.join(OUTPUT_DIR, "models", "lgbm_fire_date_model.pkl")
        mtime_before = os.path.getmtime(model_path) if os.path.isfile(model_path) else 0
        proc = subprocess.run(
            [sys.executable, "train.py", *TRAIN_ARGS],
            cwd=SRC_DIR, timeout=TRAIN_TIMEOUT_S,
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            # Promotion happened iff train.py's gates replaced the bundle —
            # the pkl mtime moving is the ground truth (a rejected candidate
            # leaves both the pkl and dataset_info.json untouched).
            mtime_after = os.path.getmtime(model_path) if os.path.isfile(model_path) else 0
            promoted = mtime_after > mtime_before
            promo = _read_promotion_from_meta() if promoted else {}
            reason = promo.get("reason") if promoted else \
                "candidate rejected by quality gates (incumbent retained)"
            metrics = _read_metrics_from_meta()
            db.finish_model_run(run_id, "done", metrics=metrics,
                                promoted=promoted, promotion_reason=reason)
            log.info("Retraining finished — promoted=%s (%s)", promoted, reason)
            if promoted:
                _fire_model_promoted()
            return {"status": "done", "promoted": promoted,
                    "reason": reason, "metrics": metrics}
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

    sync_enabled = _env_enabled("SCHEDULER_SYNC_ENABLED")
    retrain_enabled = _env_enabled("SCHEDULER_RETRAIN_ENABLED")
    gistda_enabled = _env_enabled("GISTDA_ENABLED")

    # A crashed process can leave a stale lock behind — clear it on boot.
    # A GISTDA-only presentation process never needs this lock and therefore
    # must not mutate it.
    if (sync_enabled or retrain_enabled) and os.path.exists(TRAIN_LOCK):
        log.warning("Removing stale train lock from previous run")
        try:
            os.remove(TRAIN_LOCK)
        except OSError:
            pass

    _scheduler = BackgroundScheduler(timezone="UTC")
    enabled_jobs: list[str] = []
    if sync_enabled:
        _scheduler.add_job(
            sync_job, "interval",
            minutes=SYNC_INTERVAL_MINUTES,
            id="sync", max_instances=1, coalesce=True,
            misfire_grace_time=120,
            next_run_time=datetime.now(timezone.utc) + timedelta(seconds=30),
        )
        enabled_jobs.append("sync")
    if retrain_enabled:
        _scheduler.add_job(
            retrain_check_job, "interval",
            hours=RETRAIN_CHECK_HOURS,
            id="retrain_check", max_instances=1, coalesce=True,
            misfire_grace_time=3600,
            next_run_time=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        enabled_jobs.append("retrain_check")
    # GISTDA live-hotspot poller — every GISTDA_POLL_SECONDS (default 300).
    # This is the ONLY place GISTDA polling starts; the browser and the
    # API endpoints never trigger fetches themselves.
    if gistda_enabled:
        _scheduler.add_job(
            gistda_poll_job, "interval",
            seconds=int(os.getenv("GISTDA_POLL_SECONDS", "300")),
            id="gistda_poll", max_instances=1, coalesce=True,
            misfire_grace_time=60,
            next_run_time=datetime.now(timezone.utc) + timedelta(seconds=10),
        )
        enabled_jobs.append("gistda_poll")
    _scheduler.start()
    log.info(
        "Scheduler started with jobs: %s",
        ", ".join(enabled_jobs) if enabled_jobs else "none",
    )
    return True


def shutdown():
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        log.info("Scheduler stopped")


def status() -> dict:
    if _scheduler is None:
        return {
            "running": False,
            "prediction_reload_pending": _prediction_reload_pending,
        }
    return {
        "running": True,
        "prediction_reload_pending": _prediction_reload_pending,
        "jobs": [
            {
                "id": j.id,
                "next_run": j.next_run_time.isoformat() if j.next_run_time else None,
            }
            for j in _scheduler.get_jobs()
        ],
    }
