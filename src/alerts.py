"""Cached FIRMS detections for in-app alerts, independent of model training."""

import csv
import hashlib
import math
import os
import threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from io import StringIO

import requests

DATASETS = ("VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")
POLL_SECONDS = 15 * 60
DEFAULT_LIMIT = 100
MAX_LIMIT = 5000


def utc_now():
    return datetime.now(timezone.utc)


def parse_detections(body, dataset):
    reader = csv.DictReader(StringIO(body))
    required = {"latitude", "longitude", "acq_date", "acq_time"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Unexpected FIRMS response")
    detections = {}
    for row in reader:
        try:
            lat, lon = float(row["latitude"]), float(row["longitude"])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            observed = datetime.strptime(
                f'{row["acq_date"]} {row["acq_time"].zfill(4)}', "%Y-%m-%d %H%M"
            ).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError, AttributeError):
            continue
        identity = f"{dataset}|{lat}|{lon}|{observed.isoformat()}"
        alert_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
        try:
            frp = float(row.get("frp", ""))
            frp = frp if math.isfinite(frp) else None
        except (ValueError, TypeError):
            frp = None
        detections[alert_id] = {
            "id": alert_id,
            "latitude": lat,
            "longitude": lon,
            "observed_at": observed.isoformat(),
            "satellite": dataset.removeprefix("VIIRS_").removesuffix("_NRT"),
            "confidence": {"l": "Low", "n": "Nominal", "h": "High"}.get(
                (row.get("confidence") or "").lower(), "Unknown"
            ),
            "frp_mw": frp,
        }
    return list(detections.values())


class FireAlerts:
    def __init__(self):
        self._lock = threading.Lock()
        self._by_source = {}
        self._source_updated = {}
        self._state = {
            "status": "starting",
            "last_checked": None,
            "last_success": None,
            "failed_sources": [],
            "bbox": os.getenv("FIRMS_BBOX", "96,4,107,22"),
        }

    def refresh(self):
        # Read at each refresh so a server restart is sufficient after editing .env.
        key = os.getenv("FIRMS_API_KEY", "").strip()
        if not key or key == "YOUR_FIRMS_API_KEY_HERE":
            with self._lock:
                self._state.update(status="missing_key", last_checked=utc_now().isoformat())
            return
        updates, failed = {}, []
        for dataset in DATASETS:
            try:
                # Two calendar days cover a rolling 24-hour window across midnight UTC.
                response = requests.get(
                    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                    f"{key}/{dataset}/{self._state['bbox']}/2",
                    timeout=(10, 30),
                )
                response.raise_for_status()
                updates[dataset] = parse_detections(response.text, dataset)
            except (requests.RequestException, ValueError, csv.Error):
                # Never surface exception URLs: they contain the private MAP_KEY.
                failed.append(dataset)
        checked = utc_now().isoformat()
        with self._lock:
            self._by_source.update(updates)
            self._source_updated.update({source: checked for source in updates})
            self._state.update(
                status="error" if not updates else "partial" if failed else "ok",
                failed_sources=failed,
                last_checked=checked,
            )
            if not failed:
                self._state["last_success"] = checked

    def snapshot(self, now=None, limit=DEFAULT_LIMIT, location_filter=None):
        now = now or utc_now()
        limit = max(1, min(int(limit), MAX_LIMIT))
        cutoff = now - timedelta(hours=24)
        with self._lock:
            state = dict(self._state)
            detections = [
                dict(item, fetched_at=self._source_updated[source])
                for source, items in self._by_source.items()
                for item in items
                if cutoff <= datetime.fromisoformat(item["observed_at"]) <= now
            ]
        if location_filter is not None:
            detections = [item for item in detections if location_filter(item["latitude"], item["longitude"])]
        detections.sort(key=lambda item: (item["observed_at"], item["id"]), reverse=True)
        state.update(
            alerts=detections[:limit],
            total=len(detections),
            by_satellite=dict(Counter(item["satellite"] for item in detections)),
            window_hours=24,
            poll_seconds=POLL_SECONDS,
            stale=state["status"] != "ok" or not state["last_success"] or (
                now - datetime.fromisoformat(state["last_success"])
            ).total_seconds() > POLL_SECONDS * 2,
        )
        return state


def monitor(service, stop):
    while not stop.is_set():
        service.refresh()
        stop.wait(POLL_SECONDS)
