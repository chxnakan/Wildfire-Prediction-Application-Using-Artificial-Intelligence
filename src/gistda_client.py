"""GISTDA live-hotspot client: fetch + normalize + stable IDs + event dedupe.

This is the single reusable client for GISTDA's public ArcGIS REST hotspot
layers (the same endpoints fetch_gistda_hotspots.py uses for historical
pulls). It powers the server-side live poller (gistda_poller.py) — the
browser never talks to GISTDA directly.

Data rules (do not relax):
  - GISTDA hotspots are OBSERVATIONS (satellite detections), not model
    predictions. They must never enter model training features, and must
    never be concatenated with FIRMS rows without space-time deduplication
    (both reprocess the same VIIRS/MODIS overpasses).
  - Acquisition time (detected_at_*) is preserved separately from fetch
    time (fetched_at_utc). Polling every N seconds does NOT mean the
    satellite source updates that often — GISTDA refreshes per overpass.
  - Missing scientific values stay missing (None); nothing is invented.

Identity model (two levels — do not conflate):
  source_detection_id   deterministic ID of ONE raw source detection:
                        hash(layer | satellite | lat 4dp | lon 4dp | acq
                        instant). Changes when any of those change. Used to
                        track which raw detections contributed to an event.
  hotspot_id            canonical PHYSICAL-EVENT identity: hash of the
                        detection's spatial grid cell (side =
                        GISTDA_EVENT_DISTANCE_METERS) + acquisition time
                        bucket (GISTDA_EVENT_TIME_TOLERANCE_SECONDS). It is
                        a pure function of each detection's own coordinates
                        and time — NEVER of which layers/satellites happened
                        to respond in a given poll — so it is stable when:
                          - only VIIRS responds first, MODIS arrives later
                          - layer response order changes
                          - the same data is polled repeatedly
                        Two detections in the same cell+bucket = one event
                        (their satellites union via DB enrichment).
                        Documented trade-off: two detections < D metres
                        apart but straddling a cell edge (or a time-bucket
                        edge) stay TWO events. This is conservative
                        (never over-merges) and, critically, deterministic —
                        the alternative (cluster-derived IDs) changes
                        identity when late layers alter cluster membership,
                        which is exactly the duplicate-row bug this design
                        removes. In-poll, cross-cell near-duplicates are
                        still counted via haversine and reported as
                        `near_split_count` for observability.

Normalized record schema (stable — API + DB + tests depend on it):
  hotspot_id            canonical event ID (see above)
  source_detection_id   raw-detection ID (see above)
  source                "GISTDA"
  latitude / longitude  float, validated to Thailand-region bounds
  detected_at_utc       ISO-8601 UTC
  detected_at_thailand  ISO-8601 UTC+07:00 (same instant, Thai wall clock)
  time_precision        "second" when the layer provided a time-of-day;
                        "date_only" when only a date was available (the
                        timestamp is then date midnight UTC and MUST be
                        read together with this flag — we never invent a
                        time of day)
  satellite             str  (e.g. "N" = Suomi-NPP, "T"/"A" = Terra/Aqua)
  confidence            float 0-100 or None
  province / district / subdistrict   Thai admin names (pv_tn/ap_tn/tb_tn)
  land_use              GISTDA lu_name
  source_layer          "hotspot_npp_daily" | "hotspot_daily"
  source_layers         list of layers that observed this event
  source_detection_ids  list of contributing raw-detection IDs
  source_object_id      ArcGIS FID (informational only — the daily layers
                        truncate-and-reload, so FIDs are NOT stable across
                        refreshes and are never used for identity)
  source_url            layer URL the record came from
  detected_by           list of satellites that observed this event
  fetched_at_utc        ISO-8601 UTC of our poll

Timezone note (verified against live data 2026-07-13): the NPP layer's
`time` field is UTC. Evidence: a sampled detection carried time 06:27,
which is 13:27 Asia/Bangkok — exactly the Suomi-NPP afternoon overpass
(~13:30 local). Were the field Bangkok-local, 06:27 local = 23:27 UTC
would match no VIIRS overpass. The MODIS layer's `datetime` is epoch ms
(inherently UTC).
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger("gistda_client")

GISTDA_BASE = "https://gistdaportal.gistda.or.th/data/rest/services/FR_Fire"
# layer key → (url, outFields). outFields are PER LAYER because the two
# schemas differ (verified against the live services): the VIIRS-NPP layer
# has FID/date/time/tb_tn; the MODIS layer has FID/datetime and spells the
# subdistrict field `tb_tb` (typo in GISTDA's own schema). Requesting a
# field a layer doesn't have makes ArcGIS return error 400.
LAYERS = {
    "hotspot_npp_daily": (
        f"{GISTDA_BASE}/hotspot_npp_daily/MapServer/0",
        "FID,latitude,longitude,confident,lu_name,pv_tn,ap_tn,tb_tn,date,time,satellite",
    ),
    "hotspot_daily": (
        f"{GISTDA_BASE}/hotspot_daily/MapServer/0",
        "FID,latitude,longitude,confident,lu_name,pv_tn,ap_tn,tb_tb,datetime,satellite",
    ),
}
TH_GEOMETRY = "96,4,107,22"  # ArcGIS envelope bbox — matches FIRMS_BBOX

# Validation bounds: generous margin around the Thailand request bbox. A
# coordinate outside this is a data error, not a fire.
LAT_BOUNDS = (0.0, 30.0)
LON_BOUNDS = (90.0, 115.0)

# ── Event-identity tolerances ────────────────────────────────────────────────
# Two detections belong to the SAME physical event when they fall in the
# same spatial grid cell (side ≈ GISTDA_EVENT_DISTANCE_METERS, default one
# VIIRS 375 m pixel) AND the same acquisition-time bucket
# (GISTDA_EVENT_TIME_TOLERANCE_SECONDS, default 600 s — covers scan-overlap
# double reports). Distinct overpasses (> tolerance apart) are SEPARATE
# events on purpose: re-detection hours later is real signal, not a dupe.
# See the module docstring for why identity is cell-based (deterministic
# under partial layer availability) rather than cluster-based.
EVENT_DISTANCE_M = float(os.getenv("GISTDA_EVENT_DISTANCE_METERS", "375"))
EVENT_TIME_TOLERANCE_S = int(
    os.getenv("GISTDA_EVENT_TIME_TOLERANCE_SECONDS",
              os.getenv("GISTDA_EVENT_TIME_TOLERANCE_S", "600"))  # legacy name
)

# Metres → degrees for the identity grid. Latitude: 1° ≈ 111 320 m
# everywhere. Longitude shrinks with latitude; using the value at Thailand's
# mid-latitude (~13°N, cos ≈ 0.974) keeps cells near-square over the whole
# domain (4–22°N spans cos 0.93–1.00, ±4% — irrelevant at 375 m scale).
_LAT_DEG_PER_M = 1.0 / 111_320.0
_LON_DEG_PER_M = 1.0 / (111_320.0 * 0.974)
CELL_LAT_DEG = EVENT_DISTANCE_M * _LAT_DEG_PER_M
CELL_LON_DEG = EVENT_DISTANCE_M * _LON_DEG_PER_M

# ── HTTP behaviour ────────────────────────────────────────────────────────────
REQUEST_TIMEOUT_S = float(os.getenv("GISTDA_REQUEST_TIMEOUT_SECONDS", "15"))
CONNECT_TIMEOUT_S = 10.0
MAX_RETRIES = int(os.getenv("GISTDA_MAX_RETRIES", "4"))
BACKOFF_BASE_S = 1.5
BACKOFF_CAP_S = 30.0

TH_TZ = timezone(timedelta(hours=7))

CONF_STR_MAP = {"high": 80.0, "nominal": 50.0, "low": 20.0}


class GistdaRequestError(Exception):
    """Raised when a layer query exhausts retries or hits a permanent error."""


def _sleep_backoff(attempt: int, retry_after: Optional[str] = None) -> None:
    """Exponential backoff with full jitter; honors Retry-After when given."""
    if retry_after:
        try:
            time.sleep(min(float(retry_after), BACKOFF_CAP_S))
            return
        except ValueError:
            pass  # non-numeric Retry-After → fall through to backoff
    delay = min(BACKOFF_BASE_S * (2 ** attempt), BACKOFF_CAP_S)
    time.sleep(random.uniform(0, delay))


def query_layer(
    layer_key: str,
    session: Optional[requests.Session] = None,
    where: str = "1=1",
    _sleep=_sleep_backoff,
) -> List[dict]:
    """One ArcGIS /query with retry.

    Retries (with exponential backoff + jitter): connection errors, timeouts,
    HTTP 5xx, 408, and 429 (honoring Retry-After). Permanent 4xx fail
    immediately. Raises GistdaRequestError when retries are exhausted.
    """
    url, out_fields = LAYERS[layer_key]
    sess = session or requests.Session()
    params = {
        "where": where,
        "geometry": TH_GEOMETRY,
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": out_fields,
        "f": "json",
    }

    last_err: Optional[str] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            res = sess.get(
                f"{url}/query", params=params,
                timeout=(CONNECT_TIMEOUT_S, REQUEST_TIMEOUT_S),
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_err = f"network: {type(exc).__name__}"
            if attempt < MAX_RETRIES:
                _sleep(attempt)
                continue
            raise GistdaRequestError(f"{layer_key}: {last_err}") from exc

        if res.status_code in (408, 429) or res.status_code >= 500:
            last_err = f"HTTP {res.status_code}"
            if attempt < MAX_RETRIES:
                _sleep(attempt, res.headers.get("Retry-After"))
                continue
            raise GistdaRequestError(f"{layer_key}: {last_err} (retries exhausted)")
        if res.status_code >= 400:
            # Permanent client error — retrying identical input won't help.
            raise GistdaRequestError(f"{layer_key}: HTTP {res.status_code} (permanent)")

        try:
            payload = res.json()
        except ValueError as exc:
            last_err = "invalid JSON"
            if attempt < MAX_RETRIES:
                _sleep(attempt)
                continue
            raise GistdaRequestError(f"{layer_key}: {last_err}") from exc

        # ArcGIS reports errors in-band with HTTP 200.
        if isinstance(payload, dict) and "error" in payload:
            code = (payload["error"] or {}).get("code")
            raise GistdaRequestError(f"{layer_key}: ArcGIS error code={code}")

        features = payload.get("features", []) or []
        if payload.get("exceededTransferLimit"):
            log.warning("%s: transfer limit exceeded — result truncated", layer_key)
        return features

    raise GistdaRequestError(f"{layer_key}: {last_err or 'unknown failure'}")


# ─── normalization ────────────────────────────────────────────────────────────

def _parse_detected_at(attrs: dict, layer_key: str) -> Optional[Tuple[datetime, str]]:
    """Layer-specific acquisition-time parsing → (UTC datetime, precision).

    hotspot_daily (MODIS): `datetime` is epoch ms — a full UTC instant.
    hotspot_npp_daily (VIIRS): `date` is epoch ms at date midnight UTC and
    `time` is a separate "HH:MM[:SS]" (or "HHMM") string **in UTC** — see
    the module docstring for the overpass-based evidence. When `time` is
    absent or unparseable we keep midnight UTC but return precision
    "date_only" so downstream never mistakes it for a real time of day —
    we do not invent acquisition times.
    """
    raw = attrs.get("datetime") if layer_key == "hotspot_daily" else attrs.get("date")
    if raw is None:
        return None
    try:
        dt = datetime.fromtimestamp(float(raw) / 1000.0, tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return None

    if layer_key == "hotspot_daily":
        return dt, "second"

    time_str = str(attrs.get("time") or "").strip()
    if time_str:
        parts = time_str.split(":")
        try:
            if len(parts) == 3:
                return dt.replace(hour=int(parts[0]), minute=int(parts[1]),
                                  second=int(parts[2])), "second"
            if len(parts) == 2:
                return dt.replace(hour=int(parts[0]), minute=int(parts[1])), "second"
            if len(parts) == 1 and len(time_str) in (3, 4):  # "HHMM"
                padded = time_str.zfill(4)
                return dt.replace(hour=int(padded[:2]), minute=int(padded[2:])), "second"
        except ValueError:
            pass
    return dt, "date_only"


def _parse_confidence(attrs: dict) -> Optional[float]:
    raw = attrs.get("confident")
    if raw is None:
        return None
    mapped = CONF_STR_MAP.get(str(raw).strip().lower())
    if mapped is not None:
        return mapped
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if 0.0 <= val <= 100.0 else None


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    import math
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(math.sqrt(a))


def source_detection_id(
    layer_key: str,
    satellite: str,
    latitude: float,
    longitude: float,
    detected_at_utc: datetime,
) -> str:
    """Deterministic identity of ONE raw source detection (layer-scoped).

    The ArcGIS daily layers truncate-and-reload, so their FID changes
    between refreshes — hashing the detection content (layer, satellite,
    coordinates at 4 dp ≈ 11 m, acquisition instant) is what stays stable
    across repeated polls. NOT the canonical event ID — see event_id().
    """
    key = "|".join([
        layer_key,
        satellite or "",
        f"{latitude:.4f}",
        f"{longitude:.4f}",
        detected_at_utc.strftime("%Y-%m-%dT%H:%M:%S"),
    ])
    return "gistda-det-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def event_id(latitude: float, longitude: float, detected_at_utc: datetime) -> str:
    """Canonical physical-event identity — a pure function of WHERE and WHEN.

    Quantizes the coordinates onto a fixed grid (cell side =
    GISTDA_EVENT_DISTANCE_METERS) and the acquisition time onto fixed
    buckets (GISTDA_EVENT_TIME_TOLERANCE_SECONDS). Because the key depends
    only on the detection's own position/time — never on layer, satellite,
    FID, or which sources responded this poll — the same physical event
    maps to the same ID no matter which subset of layers reported it or in
    what order. That is the property that prevents duplicate DB rows when
    a layer fails one poll and recovers the next.
    """
    cell_lat = int(latitude // CELL_LAT_DEG)
    cell_lon = int(longitude // CELL_LON_DEG)
    bucket = int(detected_at_utc.timestamp() // EVENT_TIME_TOLERANCE_S)
    key = f"{cell_lat}|{cell_lon}|{bucket}"
    return "gistda-ev-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def normalize_feature(
    feature: dict,
    layer_key: str,
    fetched_at_utc: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """One ArcGIS feature → normalized record. Returns None when invalid
    (missing/unparseable coordinates or timestamp)."""
    attrs = feature.get("attributes") or feature.get("properties") or {}

    try:
        lat = float(attrs.get("latitude"))
        lon = float(attrs.get("longitude"))
    except (TypeError, ValueError):
        return None
    if not (LAT_BOUNDS[0] <= lat <= LAT_BOUNDS[1] and LON_BOUNDS[0] <= lon <= LON_BOUNDS[1]):
        return None

    parsed = _parse_detected_at(attrs, layer_key)
    if parsed is None:
        return None
    detected_at, time_precision = parsed

    satellite = str(attrs.get("satellite") or "").strip() or (
        "N" if layer_key == "hotspot_npp_daily" else ""
    )
    fetched = fetched_at_utc or datetime.now(timezone.utc)
    url, _ = LAYERS[layer_key]
    det_id = source_detection_id(layer_key, satellite, lat, lon, detected_at)

    return {
        "hotspot_id": event_id(lat, lon, detected_at),
        "source_detection_id": det_id,
        "source": "GISTDA",
        "latitude": lat,
        "longitude": lon,
        "detected_at_utc": detected_at.isoformat(),
        "detected_at_thailand": detected_at.astimezone(TH_TZ).isoformat(),
        "time_precision": time_precision,
        "satellite": satellite,
        "confidence": _parse_confidence(attrs),
        "province": str(attrs.get("pv_tn") or "") or None,
        "district": str(attrs.get("ap_tn") or "") or None,
        # MODIS layer spells the subdistrict field "tb_tb" (upstream typo).
        "subdistrict": str(attrs.get("tb_tn") or attrs.get("tb_tb") or "") or None,
        "land_use": str(attrs.get("lu_name") or "") or None,
        "source_layer": layer_key,
        "source_layers": [layer_key],
        "source_detection_ids": [det_id],
        "source_object_id": attrs.get("FID", attrs.get("objectid")),
        "source_url": url,
        "detected_by": [satellite] if satellite else [],
        "fetched_at_utc": fetched.isoformat(),
    }


def _merge_into(kept: Dict[str, Any], rec: Dict[str, Any]) -> None:
    """Fold a same-event detection into the kept record: union the
    contribution lists, fill missing annotation fields, keep max confidence.
    Mirrors the DB-side ON CONFLICT enrichment so in-poll and cross-poll
    merging follow one documented rule set."""
    for field in ("detected_by", "source_layers", "source_detection_ids"):
        for v in rec.get(field) or []:
            if v not in kept[field]:
                kept[field].append(v)
    for field in ("province", "district", "subdistrict", "land_use"):
        if kept.get(field) is None and rec.get(field) is not None:
            kept[field] = rec[field]
    if rec.get("confidence") is not None:
        kept["confidence"] = (
            rec["confidence"] if kept.get("confidence") is None
            else max(kept["confidence"], rec["confidence"])
        )
    # Earliest acquisition wins as the event timestamp; "second" precision
    # beats "date_only" when instants are equal.
    if (rec["detected_at_utc"], rec["time_precision"] == "date_only") < (
        kept["detected_at_utc"], kept["time_precision"] == "date_only"
    ):
        kept["detected_at_utc"] = rec["detected_at_utc"]
        kept["detected_at_thailand"] = rec["detected_at_thailand"]
        kept["time_precision"] = rec["time_precision"]


def dedupe_events(
    records: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Collapse detections that share a canonical event ID; measure near-splits.

    Returns (deduped_records, n_merged, n_near_splits).

    Merging is strictly by `hotspot_id` (same identity cell + time bucket),
    so the output is deterministic under partial layer availability. The
    third value counts pairs in ADJACENT cells within the same time bucket
    whose true haversine distance is under GISTDA_EVENT_DISTANCE_METERS —
    physically-close detections the conservative identity scheme keeps as
    separate events. It is an observability metric (logged + persisted),
    not a merge: collapsing across cells would tie an event's identity to
    which sources responded, re-creating the duplicate-row bug.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    merged = 0
    for rec in records:
        eid = rec["hotspot_id"]
        if eid in groups:
            _merge_into(groups[eid], rec)
            merged += 1
        else:
            groups[eid] = rec

    out = list(groups.values())

    # Near-split audit: bucket events by identity cell, haversine-check only
    # the 8 neighbouring cells within the same time bucket (no O(n²) sweep).
    cells: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
    for rec in out:
        dt = datetime.fromisoformat(rec["detected_at_utc"])
        key = (
            int(rec["latitude"] // CELL_LAT_DEG),
            int(rec["longitude"] // CELL_LON_DEG),
            int(dt.timestamp() // EVENT_TIME_TOLERANCE_S),
        )
        cells.setdefault(key, []).append(rec)

    near_splits = 0
    for (cx, cy, cb), members in cells.items():
        for dx, dy in ((1, -1), (1, 0), (1, 1), (0, 1)):  # forward half — no double count
            for other in cells.get((cx + dx, cy + dy, cb), []):
                for rec in members:
                    if haversine_m(rec["latitude"], rec["longitude"],
                                   other["latitude"], other["longitude"]) <= EVENT_DISTANCE_M:
                        near_splits += 1
    return out, merged, near_splits


def fetch_live_hotspots(
    session: Optional[requests.Session] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch + normalize + event-dedupe both daily layers.

    Returns (records, stats). One layer failing does not black out the
    other; a layer's sanitized error string lands in stats["errors"].
    Raises GistdaRequestError only when EVERY layer fails.
    """
    fetched_at = datetime.now(timezone.utc)
    sess = session or requests.Session()

    records: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "raw_count": 0, "normalized_count": 0, "invalid_count": 0,
        "merged_count": 0, "near_split_count": 0,
        "errors": {}, "layers_ok": [], "layers_failed": [],
    }

    for layer_key in LAYERS:
        try:
            features = query_layer(layer_key, session=sess)
            stats["layers_ok"].append(layer_key)
        except GistdaRequestError as exc:
            stats["errors"][layer_key] = str(exc)[:200]
            stats["layers_failed"].append(layer_key)
            continue
        stats["raw_count"] += len(features)
        for feat in features:
            rec = normalize_feature(feat, layer_key, fetched_at_utc=fetched_at)
            if rec is None:
                stats["invalid_count"] += 1
            else:
                records.append(rec)

    if not stats["layers_ok"]:
        raise GistdaRequestError(
            "all GISTDA layers failed: " + "; ".join(stats["errors"].values())
        )

    records, merged, near_splits = dedupe_events(records)
    stats["merged_count"] = merged
    stats["near_split_count"] = near_splits
    stats["normalized_count"] = len(records)
    return records, stats
