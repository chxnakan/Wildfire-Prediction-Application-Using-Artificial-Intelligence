"""Server-side GISTDA live-hotspot poller — ONE poll cycle per interval.

Flow (scheduler.py fires this every GISTDA_POLL_SECONDS, default 300):

    acquire PostgreSQL advisory lock ──(held elsewhere? skip quietly)
        → gistda_client.fetch_live_hotspots()   (retry/backoff inside)
        → db.upsert_live_hotspots()             (insert new / touch seen)
        → db.cleanup_live_hotspots()            (retention, default 7 days)
        → db.log_live_poll()                    (audit row, ok or error)

The advisory lock makes multiple Railway replicas safe: only the instance
that wins the lock polls; the rest skip the cycle without error. Without a
database (local dev) the lock degrades to process-local — documented in
db.advisory_lock.

Never raises — a failed poll leaves the previous cached state untouched and
records a sanitized error for /live-hotspots/status.

New-record notification: after each successful poll, `poll_generation` is
bumped and the newly inserted records are stashed in `consume_new_records()`
so api.py's SSE stream can push alerts without a second GISTDA query.
Thread-safety: the poller runs on an APScheduler worker thread while SSE
readers run on the asyncio loop — everything shared goes through _NEW_LOCK.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, List, Optional

import db
import gistda_client

log = logging.getLogger("gistda_poller")

GISTDA_ENABLED = os.getenv("GISTDA_ENABLED", "true").lower() in ("1", "true", "yes")
GISTDA_POLL_SECONDS = int(os.getenv("GISTDA_POLL_SECONDS", "300"))
GISTDA_RETENTION_DAYS = int(os.getenv("GISTDA_RETENTION_DAYS", "7"))
GISTDA_STALE_AFTER_SECONDS = int(os.getenv("GISTDA_STALE_AFTER_SECONDS", "900"))
# Strict country validation: detections must fall inside Thailand's actual
# land-border polygon, with a small tolerance ring for border-straddling
# satellite pixels (a VIIRS pixel is 375 m; default 2 km covers geolocation
# jitter without admitting genuinely-foreign fires).
GISTDA_REQUIRE_IN_THAILAND = os.getenv(
    "GISTDA_REQUIRE_IN_THAILAND", "true").lower() in ("1", "true", "yes")
GISTDA_BORDER_TOLERANCE_KM = float(os.getenv("GISTDA_BORDER_TOLERANCE_KM", "2.0"))

# Cross-thread hand-off of newly seen records (for the SSE alert stream).
# Buffer entries are (generation, record); each SSE connection keeps its own
# cursor so EVERY connected dashboard sees every new fire (a drain-style
# queue would hand each event to exactly one client).
_NEW_LOCK = threading.Lock()
_NEW_BUFFER: List[tuple] = []          # bounded [(generation, record), ...]
_NEW_BUFFER_MAX = 500
_GENERATION = 0
_LAST_ERROR: Optional[str] = None
_FIRST_POLL_DONE = False


def poll_generation() -> int:
    with _NEW_LOCK:
        return _GENERATION


def records_since(generation: int) -> tuple:
    """(latest_generation, records newer than `generation`) — non-destructive."""
    with _NEW_LOCK:
        recs = [r for g, r in _NEW_BUFFER if g > generation]
        return _GENERATION, recs


def _filter_thailand(records: list) -> tuple:
    """Strict country validation against the 77-province polygon.

    Border tolerance: a point is accepted when the point itself OR any of
    four probe points offset by GISTDA_BORDER_TOLERANCE_KM (N/S/E/W) falls
    inside the polygon — a cheap, deterministic approximation of "within
    tolerance of the border" that avoids per-point polygon-distance math.
    Satellite pixels are 375 m and geolocation jitters, so detections just
    outside the border line may still be Thai fires.
    """
    if not GISTDA_REQUIRE_IN_THAILAND or not records:
        return records, 0
    from thailand_boundary import is_in_thailand

    lats = [r["latitude"] for r in records]
    lons = [r["longitude"] for r in records]
    inside = is_in_thailand(lats, lons)

    tol_lat = GISTDA_BORDER_TOLERANCE_KM / 111.32
    tol_lon = GISTDA_BORDER_TOLERANCE_KM / (111.32 * 0.974)
    for dlat, dlon in ((tol_lat, 0), (-tol_lat, 0), (0, tol_lon), (0, -tol_lon)):
        if inside.all():
            break
        probe = is_in_thailand(
            [la + dlat for la in lats], [lo + dlon for lo in lons]
        )
        inside = inside | probe

    kept = [r for r, ok in zip(records, inside) if ok]
    return kept, len(records) - len(kept)


def poll_once() -> Dict[str, object]:
    """One guarded poll cycle. Returns a summary dict; never raises.

    Status values (consistent across log rows, /live-hotspots/status and
    the return value): ok · partial (some layers failed, data still landed)
    · error · skipped_lock · disabled.
    """
    global _GENERATION, _LAST_ERROR, _FIRST_POLL_DONE
    if not GISTDA_ENABLED:
        return {"status": "disabled"}

    t0 = time.monotonic()
    with db.advisory_lock(db.GISTDA_POLLER_LOCK_KEY) as acquired:
        if not acquired:
            log.info("GISTDA poll skipped — another instance holds the lock")
            return {"status": "skipped_lock"}

        try:
            records, stats = gistda_client.fetch_live_hotspots()
        except gistda_client.GistdaRequestError as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            err = str(exc)[:200]
            with _NEW_LOCK:
                _LAST_ERROR = err
            try:
                db.log_live_poll("error", duration_ms=duration_ms, error=err)
            except Exception:
                log.exception("live_poll_log write failed")
            log.error("GISTDA poll failed after %dms: %s", duration_ms, err)
            return {"status": "error", "error": err}

        # Drop detections already older than the retention window BEFORE
        # upserting — GISTDA's "daily" layers can retain months-old records
        # in the off-season, and inserting rows that the very next statement
        # deletes would churn the table (and inflate inserted counts) every
        # cycle for nothing.
        from datetime import datetime, timedelta, timezone
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=GISTDA_RETENTION_DAYS)
        ).isoformat()
        n_outside_window = len(records)
        records = [r for r in records if r["detected_at_utc"] >= cutoff]
        n_outside_window -= len(records)

        # Strict Thailand land-border validation (polygon + border tolerance).
        records, n_outside_th = _filter_thailand(records)
        stats["outside_thailand_count"] = n_outside_th

        try:
            inserted, updated, inserted_ids = db.upsert_live_hotspots(records)
            removed = db.cleanup_live_hotspots(GISTDA_RETENTION_DAYS)
        except Exception as exc:  # DB down ≠ crash the scheduler
            duration_ms = int((time.monotonic() - t0) * 1000)
            err = f"store: {type(exc).__name__}"
            with _NEW_LOCK:
                _LAST_ERROR = err
            log.exception("GISTDA store write failed")
            try:
                db.log_live_poll("error", stats=stats, duration_ms=duration_ms, error=err)
            except Exception:
                pass
            return {"status": "error", "error": err}

        duration_ms = int((time.monotonic() - t0) * 1000)
        stats.update({"inserted": inserted, "updated": updated})

        # partial = data landed but at least one layer failed. The failed
        # layers stay visible in the log row and in status(); a later
        # all-layers success clears them naturally by writing an 'ok' row.
        cycle_status = "partial" if stats.get("layers_failed") else "ok"
        layer_err = (
            "; ".join(f"{k}: {v}" for k, v in (stats.get("errors") or {}).items())
            or None
        )
        try:
            db.log_live_poll(cycle_status, stats=stats,
                             duration_ms=duration_ms, error=layer_err or "")
        except Exception:
            log.exception("live_poll_log write failed")

        # Stash genuinely-new detections for the SSE alert stream — exactly
        # the IDs the store reported as inserted. The very first poll after
        # boot seeds the store; suppress its flood so a restart doesn't
        # re-alert operators about hours-old fires.
        new_ids = set(inserted_ids) if _FIRST_POLL_DONE else set()
        with _NEW_LOCK:
            _GENERATION += 1
            if new_ids:
                _NEW_BUFFER.extend(
                    (_GENERATION, r) for r in records if r["hotspot_id"] in new_ids
                )
                del _NEW_BUFFER[:-_NEW_BUFFER_MAX]
            # Keep the layer error visible on partial success — one healthy
            # layer must not mask the other one being down.
            _LAST_ERROR = layer_err
        _FIRST_POLL_DONE = True

        log.info(
            "GISTDA poll %s: raw=%d normalized=%d outside_retention=%d "
            "outside_thailand=%d inserted=%d updated=%d invalid=%d merged=%d "
            "near_splits=%d removed_by_retention=%d layers_failed=%s duration=%dms",
            cycle_status,
            stats.get("raw_count", 0), stats.get("normalized_count", 0),
            n_outside_window, n_outside_th, inserted, updated,
            stats.get("invalid_count", 0), stats.get("merged_count", 0),
            stats.get("near_split_count", 0), removed,
            stats.get("layers_failed") or "-", duration_ms,
        )
        return {"status": cycle_status, "inserted": inserted, "updated": updated,
                "layers_failed": stats.get("layers_failed") or [],
                "duration_ms": duration_ms}


def status() -> Dict[str, object]:
    """Metadata block for /live-hotspots and /live-hotspots/status."""
    from datetime import datetime, timezone

    base = db.live_poll_status()
    last_success = base.get("last_success_at")
    is_stale = True
    if last_success:
        try:
            age = (
                datetime.now(timezone.utc)
                - datetime.fromisoformat(str(last_success))
            ).total_seconds()
            is_stale = age > GISTDA_STALE_AFTER_SECONDS
        except ValueError:
            pass
    with _NEW_LOCK:
        last_error = _LAST_ERROR
    return {
        "source": "GISTDA",
        "enabled": GISTDA_ENABLED,
        "polling_interval_seconds": GISTDA_POLL_SECONDS,
        "retention_days": GISTDA_RETENTION_DAYS,
        "stale_after_seconds": GISTDA_STALE_AFTER_SECONDS,
        "is_stale": is_stale,
        "last_error": last_error,
        **base,
    }
