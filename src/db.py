"""PostgreSQL persistence layer for the automation pipeline.

Design contract:
  - Reads DATABASE_URL from env (Railway injects it automatically when a
    PostgreSQL service is attached). When it is NOT set, every function
    here becomes a silent no-op returning a benign default — so local dev
    without Postgres keeps working exactly as before (parquet + GeoJSON).
  - All tables are created idempotently by init_db() at API startup.
  - Hotspot inserts use ON CONFLICT DO NOTHING against a natural key
    (latitude, longitude, acq_datetime) so re-inserting the whole FIRMS
    cache is safe and the returned rowcount == genuinely-new detections.

Tables:
  firms_hotspots    — cleaned VIIRS detections (mirror of firms_all.parquet)
  risk_predictions  — one row per predicted cell per base_date snapshot
  model_runs        — training history + headline metrics
  fetch_log         — one row per sync attempt (audit + "is it alive?" checks)
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import pandas as pd

log = logging.getLogger("db")

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

_engine = None


def enabled() -> bool:
    """True when a database is configured for this deployment."""
    return bool(DATABASE_URL)


def _normalize_url(url: str) -> str:
    # Railway/Heroku hand out postgres:// or postgresql:// — SQLAlchemy would
    # default those to psycopg2, which we don't ship. Pin the psycopg3 dialect.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def get_engine():
    """Lazy singleton engine. pool_pre_ping recovers from idle disconnects
    (Railway kills idle connections after a few minutes)."""
    global _engine
    if _engine is None:
        if not enabled():
            raise RuntimeError("DATABASE_URL not set — db module is disabled")
        from sqlalchemy import create_engine

        _engine = create_engine(
            _normalize_url(DATABASE_URL),
            pool_size=3,
            max_overflow=2,
            pool_pre_ping=True,
            pool_recycle=300,
        )
    return _engine


@contextmanager
def _conn():
    eng = get_engine()
    with eng.begin() as conn:
        yield conn


# ─── schema ───────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS firms_hotspots (
    id           BIGSERIAL PRIMARY KEY,
    latitude     DOUBLE PRECISION NOT NULL,
    longitude    DOUBLE PRECISION NOT NULL,
    acq_datetime TIMESTAMPTZ      NOT NULL,
    frp          DOUBLE PRECISION,
    bright_ti4   DOUBLE PRECISION,
    bright_ti5   DOUBLE PRECISION,
    confidence   DOUBLE PRECISION,
    dataset      VARCHAR(32),
    inserted_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (latitude, longitude, acq_datetime)
);
CREATE INDEX IF NOT EXISTS idx_hotspots_acq_date ON firms_hotspots (acq_datetime);

CREATE TABLE IF NOT EXISTS risk_predictions (
    id                  BIGSERIAL PRIMARY KEY,
    base_date           DATE  NOT NULL,
    lat_grid            DOUBLE PRECISION NOT NULL,
    lon_grid            DOUBLE PRECISION NOT NULL,
    days_until_fire     INTEGER,
    predicted_fire_date DATE,
    probability         DOUBLE PRECISION,
    raw_prediction      DOUBLE PRECISION,
    urgency_level       VARCHAR(16),
    province            VARCHAR(100),
    nearest_urban_area  VARCHAR(100),
    fire_count_30d      INTEGER,
    generated_at        TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (base_date, lat_grid, lon_grid)
);
CREATE INDEX IF NOT EXISTS idx_predictions_base_date ON risk_predictions (base_date);

CREATE TABLE IF NOT EXISTS model_runs (
    id           SERIAL PRIMARY KEY,
    started_at   TIMESTAMPTZ DEFAULT NOW(),
    finished_at  TIMESTAMPTZ,
    status       VARCHAR(16) DEFAULT 'running',
    trigger      VARCHAR(16) DEFAULT 'scheduled',
    roc_auc      DOUBLE PRECISION,
    f1           DOUBLE PRECISION,
    n_train_rows INTEGER,
    n_positive   INTEGER,
    error        TEXT
);

CREATE TABLE IF NOT EXISTS fetch_log (
    id         SERIAL PRIMARY KEY,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    new_rows   INTEGER DEFAULT 0,
    status     VARCHAR(16) DEFAULT 'ok',
    detail     TEXT
);

CREATE TABLE IF NOT EXISTS live_hotspots (
    hotspot_id           VARCHAR(64) PRIMARY KEY,
    source               VARCHAR(16) NOT NULL DEFAULT 'GISTDA',
    latitude             DOUBLE PRECISION NOT NULL,
    longitude            DOUBLE PRECISION NOT NULL,
    detected_at_utc      TIMESTAMPTZ NOT NULL,
    time_precision       VARCHAR(12) DEFAULT 'second',
    satellite            VARCHAR(16),
    confidence           DOUBLE PRECISION,
    province             VARCHAR(100),
    district             VARCHAR(100),
    subdistrict          VARCHAR(100),
    land_use             VARCHAR(120),
    source_layer         VARCHAR(40),
    detected_by          JSONB DEFAULT '[]'::jsonb,
    source_layers        JSONB DEFAULT '[]'::jsonb,
    source_detection_ids JSONB DEFAULT '[]'::jsonb,
    first_seen_at        TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at         TIMESTAMPTZ DEFAULT NOW(),
    fetched_at           TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_live_hotspots_detected ON live_hotspots (detected_at_utc);
CREATE INDEX IF NOT EXISTS idx_live_hotspots_province ON live_hotspots (province);
CREATE INDEX IF NOT EXISTS idx_live_hotspots_first_seen ON live_hotspots (first_seen_at);

CREATE TABLE IF NOT EXISTS live_poll_log (
    id            SERIAL PRIMARY KEY,
    polled_at     TIMESTAMPTZ DEFAULT NOW(),
    status        VARCHAR(16) DEFAULT 'ok',
    raw_count     INTEGER DEFAULT 0,
    inserted      INTEGER DEFAULT 0,
    updated       INTEGER DEFAULT 0,
    invalid       INTEGER DEFAULT 0,
    merged        INTEGER DEFAULT 0,
    duration_ms   INTEGER,
    error         TEXT
);

CREATE TABLE IF NOT EXISTS prediction_runs (
    id             SERIAL PRIMARY KEY,
    started_at     TIMESTAMPTZ DEFAULT NOW(),
    finished_at    TIMESTAMPTZ,
    status         VARCHAR(16) DEFAULT 'running',
    input_version  VARCHAR(64),
    new_rows       INTEGER DEFAULT 0,
    noop           BOOLEAN DEFAULT FALSE,
    model_version  VARCHAR(64),
    error          TEXT
);
"""

# Idempotent migrations for databases created by earlier revisions of this
# module (Railway PostgreSQL that already has data). Every statement is safe
# to re-run. Kept as a LIST (not a big string) because the DO $$ block
# contains semicolons and must execute as a single statement.
_MIGRATIONS = [
    "ALTER TABLE live_hotspots ADD COLUMN IF NOT EXISTS time_precision VARCHAR(12) DEFAULT 'second'",
    "ALTER TABLE live_hotspots ADD COLUMN IF NOT EXISTS source_layers JSONB DEFAULT '[]'::jsonb",
    "ALTER TABLE live_hotspots ADD COLUMN IF NOT EXISTS source_detection_ids JSONB DEFAULT '[]'::jsonb",
    """
    DO $$
    BEGIN
        IF (SELECT data_type FROM information_schema.columns
            WHERE table_name = 'live_hotspots' AND column_name = 'detected_by') = 'text' THEN
            ALTER TABLE live_hotspots
                ALTER COLUMN detected_by TYPE JSONB
                USING COALESCE(detected_by, '[]')::jsonb;
            ALTER TABLE live_hotspots
                ALTER COLUMN detected_by SET DEFAULT '[]'::jsonb;
        END IF;
    END $$
    """,
    "ALTER TABLE live_poll_log ADD COLUMN IF NOT EXISTS layers_failed TEXT",
    "ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS promoted BOOLEAN",
    "ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS promotion_reason TEXT",
]


def init_db() -> bool:
    """Create all tables if missing + apply idempotent migrations.
    Safe against an existing Railway database. Returns True when ready."""
    if not enabled():
        log.info("DATABASE_URL not set — running without a database")
        return False
    from sqlalchemy import text

    with _conn() as conn:
        for stmt in _SCHEMA.split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
        for stmt in _MIGRATIONS:
            conn.execute(text(stmt))
    log.info("Database schema ready")
    return True


# ─── hotspots ─────────────────────────────────────────────────────────────────

_HOTSPOT_COLS = [
    "latitude", "longitude", "acq_datetime",
    "frp", "bright_ti4", "bright_ti5", "confidence", "dataset",
]


def insert_hotspots(df: pd.DataFrame) -> int:
    """Insert cleaned FIRMS rows; duplicates (natural key) are skipped.
    Returns the number of genuinely-new rows."""
    if not enabled() or df.empty:
        return 0
    from sqlalchemy import text

    cols = [c for c in _HOTSPOT_COLS if c in df.columns]
    records = df[cols].where(pd.notna(df[cols]), None).to_dict("records")
    stmt = text(
        f"INSERT INTO firms_hotspots ({', '.join(cols)}) "
        f"VALUES ({', '.join(':' + c for c in cols)}) "
        "ON CONFLICT (latitude, longitude, acq_datetime) DO NOTHING"
    )
    inserted = 0
    with _conn() as conn:
        # Chunked so one bad batch doesn't roll back hours of data and the
        # driver doesn't build a single giant statement.
        for i in range(0, len(records), 5000):
            res = conn.execute(stmt, records[i : i + 5000])
            inserted += res.rowcount or 0
    return inserted


def latest_hotspot_time() -> Optional[datetime]:
    if not enabled():
        return None
    from sqlalchemy import text

    with _conn() as conn:
        row = conn.execute(text("SELECT MAX(acq_datetime) FROM firms_hotspots")).scalar()
    return row


# ─── predictions ──────────────────────────────────────────────────────────────

def insert_predictions(df: pd.DataFrame, base_date) -> int:
    """Persist one snapshot of risk_map predictions. Re-running the same
    base_date overwrites that snapshot (delete + insert) so the table always
    holds exactly one row per (base_date, cell).

    An empty frame is a valid snapshot: it means the model produced no cells
    for that date.  It must therefore delete any previously persisted rows for
    the same date instead of leaving a stale snapshot behind.
    """
    if not enabled():
        return 0
    from sqlalchemy import text

    snapshot_date = pd.to_datetime(base_date).date()
    delete_stmt = text("DELETE FROM risk_predictions WHERE base_date = :bd")

    if df.empty:
        with _conn() as conn:
            conn.execute(delete_stmt, {"bd": snapshot_date})
        return 0

    col_map = {
        "lat_grid": "lat_grid",
        "lon_grid": "lon_grid",
        "days_until_fire": "days_until_fire",
        "predicted_fire_date": "predicted_fire_date",
        "probability": "probability",
        "raw_prediction": "raw_prediction",
        "urgency_level": "urgency_level",
        "province": "province",
        "nearest_urban_area": "nearest_urban_area",
        "historical_fire_count_30d": "fire_count_30d",
    }
    present = {src: dst for src, dst in col_map.items() if src in df.columns}
    sub = df[list(present)].rename(columns=present).copy()
    sub["base_date"] = snapshot_date
    if "predicted_fire_date" in sub.columns:
        sub["predicted_fire_date"] = pd.to_datetime(
            sub["predicted_fire_date"], errors="coerce"
        ).dt.date
    records = sub.where(pd.notna(sub), None).to_dict("records")

    cols = list(sub.columns)
    stmt = text(
        f"INSERT INTO risk_predictions ({', '.join(cols)}) "
        f"VALUES ({', '.join(':' + c for c in cols)}) "
        "ON CONFLICT (base_date, lat_grid, lon_grid) DO NOTHING"
    )
    with _conn() as conn:
        conn.execute(delete_stmt, {"bd": snapshot_date})
        conn.execute(stmt, records)
    return len(records)


# ─── model runs ───────────────────────────────────────────────────────────────

def start_model_run(trigger: str = "scheduled") -> Optional[int]:
    if not enabled():
        return None
    from sqlalchemy import text

    with _conn() as conn:
        run_id = conn.execute(
            text("INSERT INTO model_runs (trigger) VALUES (:t) RETURNING id"),
            {"t": trigger},
        ).scalar()
    return run_id


def finish_model_run(
    run_id: Optional[int],
    status: str,
    metrics: Optional[dict] = None,
    error: Optional[str] = None,
    promoted: Optional[bool] = None,
    promotion_reason: Optional[str] = None,
) -> None:
    if not enabled() or run_id is None:
        return
    from sqlalchemy import text

    m = metrics or {}
    with _conn() as conn:
        conn.execute(
            text(
                "UPDATE model_runs SET finished_at = NOW(), status = :s, "
                "roc_auc = :auc, f1 = :f1, n_train_rows = :rows, "
                "n_positive = :pos, error = :err, promoted = :promoted, "
                "promotion_reason = :reason WHERE id = :id"
            ),
            {
                "s": status,
                "auc": m.get("roc_auc"),
                "f1": m.get("f1"),
                "rows": m.get("n_train_rows"),
                "pos": m.get("n_positive"),
                "err": (error or "")[:4000] or None,
                "promoted": promoted,
                "reason": (promotion_reason or "")[:1000] or None,
                "id": run_id,
            },
        )


def last_successful_run_at() -> Optional[datetime]:
    if not enabled():
        return None
    from sqlalchemy import text

    with _conn() as conn:
        return conn.execute(
            text("SELECT MAX(finished_at) FROM model_runs WHERE status = 'done'")
        ).scalar()


def any_run_in_progress() -> bool:
    """A 'running' row younger than 6 h means a training subprocess is live
    (older ones are treated as crashed and ignored)."""
    if not enabled():
        return False
    from sqlalchemy import text

    with _conn() as conn:
        n = conn.execute(
            text(
                "SELECT COUNT(*) FROM model_runs WHERE status = 'running' "
                "AND started_at > NOW() - INTERVAL '6 hours'"
            )
        ).scalar()
    return bool(n)


# ─── fetch log ────────────────────────────────────────────────────────────────

def log_fetch(new_rows: int, status: str = "ok", detail: str = "") -> None:
    if not enabled():
        return
    from sqlalchemy import text

    with _conn() as conn:
        conn.execute(
            text(
                "INSERT INTO fetch_log (new_rows, status, detail) "
                "VALUES (:n, :s, :d)"
            ),
            {"n": int(new_rows), "s": status, "d": detail[:2000]},
        )


def automation_status() -> dict:
    """Snapshot for the /api/automation/status endpoint and monitoring."""
    if not enabled():
        return {"database": "disabled"}
    from sqlalchemy import text

    with _conn() as conn:
        last_fetch = conn.execute(
            text(
                "SELECT fetched_at, new_rows, status FROM fetch_log "
                "ORDER BY id DESC LIMIT 1"
            )
        ).mappings().first()
        last_run = conn.execute(
            text(
                "SELECT started_at, finished_at, status, roc_auc, trigger "
                "FROM model_runs ORDER BY id DESC LIMIT 1"
            )
        ).mappings().first()
        counts = conn.execute(
            text(
                "SELECT (SELECT COUNT(*) FROM firms_hotspots) AS hotspots, "
                "(SELECT COUNT(*) FROM risk_predictions) AS predictions"
            )
        ).mappings().first()

    def _iso(v):
        return v.isoformat() if hasattr(v, "isoformat") else v

    return {
        "database": "connected",
        "last_fetch": {k: _iso(v) for k, v in dict(last_fetch).items()} if last_fetch else None,
        "last_model_run": {k: _iso(v) for k, v in dict(last_run).items()} if last_run else None,
        "row_counts": dict(counts) if counts else {},
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── live hotspots (GISTDA poller store) ─────────────────────────────────────
#
# Persistent store for the live-detection layer. When DATABASE_URL is set
# (Railway PostgreSQL) everything below hits the live_hotspots table, so
# state survives redeploys and is shared across replicas. When it is NOT set
# (local development), a module-level in-memory dict provides the same
# interface — clearly a fallback: state is lost on restart and is
# per-process. Production must run with a database.

_MEM_LIVE: Dict[str, dict] = {}
_MEM_POLL_LOG: list = []


def _live_json(rec: dict) -> dict:
    """Normalize a client record into DB row shape."""
    import json as _json
    return {
        "hotspot_id": rec["hotspot_id"],
        "source": rec.get("source", "GISTDA"),
        "latitude": float(rec["latitude"]),
        "longitude": float(rec["longitude"]),
        "detected_at_utc": rec["detected_at_utc"],
        "time_precision": rec.get("time_precision", "second"),
        "satellite": rec.get("satellite"),
        "confidence": rec.get("confidence"),
        "province": rec.get("province"),
        "district": rec.get("district"),
        "subdistrict": rec.get("subdistrict"),
        "land_use": rec.get("land_use"),
        "source_layer": rec.get("source_layer"),
        "detected_by": _json.dumps(rec.get("detected_by") or []),
        "source_layers": _json.dumps(rec.get("source_layers") or []),
        "source_detection_ids": _json.dumps(rec.get("source_detection_ids") or []),
        "fetched_at": rec.get("fetched_at_utc"),
    }


def _merge_mem_row(existing: dict, row: dict) -> None:
    """In-memory mirror of the SQL enrichment rules below."""
    import json as _json
    for field in ("detected_by", "source_layers", "source_detection_ids"):
        old = _json.loads(existing.get(field) or "[]")
        new = _json.loads(row.get(field) or "[]")
        existing[field] = _json.dumps(old + [v for v in new if v not in old])
    for field in ("province", "district", "subdistrict", "land_use", "satellite"):
        if existing.get(field) is None and row.get(field) is not None:
            existing[field] = row[field]
    if row.get("confidence") is not None:
        existing["confidence"] = (
            row["confidence"] if existing.get("confidence") is None
            else max(existing["confidence"], row["confidence"])
        )
    existing["last_seen_at"] = datetime.now(timezone.utc).isoformat()
    existing["fetched_at"] = row["fetched_at"]


def upsert_live_hotspots(records: list) -> tuple:
    """Upsert by canonical hotspot_id with ENRICHMENT on conflict.

    Later polls can carry better information (a second layer recovered, a
    satellite joined the event, admin names filled in). Documented rules:
      - detected_by / source_layers / source_detection_ids → set union
      - province / district / subdistrict / land_use → fill only when the
        stored value is NULL (never overwrite non-null with null or churn
        existing annotations)
      - confidence → maximum of stored and incoming (non-null wins)
      - first_seen_at preserved; last_seen_at / fetched_at refreshed
    Returns (inserted, updated, inserted_ids); inserted_ids feed SSE alerts.
    """
    if not records:
        return (0, 0, [])
    inserted_ids: list = []
    if not enabled():
        updated = 0
        now = datetime.now(timezone.utc).isoformat()
        for rec in records:
            row = _live_json(rec)
            hid = row["hotspot_id"]
            if hid in _MEM_LIVE:
                _merge_mem_row(_MEM_LIVE[hid], row)
                updated += 1
            else:
                row["first_seen_at"] = now
                row["last_seen_at"] = now
                _MEM_LIVE[hid] = row
                inserted_ids.append(hid)
        return (len(inserted_ids), updated, inserted_ids)

    from sqlalchemy import text

    # jsonb set-union helper pattern:  (SELECT COALESCE(jsonb_agg(DISTINCT v),
    # '[]') FROM jsonb_array_elements(a || b) t(v)) — dedupes while unioning.
    def _union(col: str) -> str:
        return (
            f"(SELECT COALESCE(jsonb_agg(DISTINCT v), '[]'::jsonb) "
            f"FROM jsonb_array_elements("
            f"COALESCE(live_hotspots.{col}, '[]'::jsonb) || "
            f"COALESCE(EXCLUDED.{col}, '[]'::jsonb)) AS t(v))"
        )

    stmt = text(
        "INSERT INTO live_hotspots (hotspot_id, source, latitude, longitude, "
        "detected_at_utc, time_precision, satellite, confidence, province, "
        "district, subdistrict, land_use, source_layer, detected_by, "
        "source_layers, source_detection_ids, fetched_at) "
        "VALUES (:hotspot_id, :source, :latitude, :longitude, "
        ":detected_at_utc, :time_precision, :satellite, :confidence, "
        ":province, :district, :subdistrict, :land_use, :source_layer, "
        "CAST(:detected_by AS jsonb), CAST(:source_layers AS jsonb), "
        "CAST(:source_detection_ids AS jsonb), :fetched_at) "
        "ON CONFLICT (hotspot_id) DO UPDATE SET "
        f"detected_by = {_union('detected_by')}, "
        f"source_layers = {_union('source_layers')}, "
        f"source_detection_ids = {_union('source_detection_ids')}, "
        "province = COALESCE(live_hotspots.province, EXCLUDED.province), "
        "district = COALESCE(live_hotspots.district, EXCLUDED.district), "
        "subdistrict = COALESCE(live_hotspots.subdistrict, EXCLUDED.subdistrict), "
        "land_use = COALESCE(live_hotspots.land_use, EXCLUDED.land_use), "
        "satellite = COALESCE(live_hotspots.satellite, EXCLUDED.satellite), "
        "confidence = GREATEST(live_hotspots.confidence, EXCLUDED.confidence), "
        "last_seen_at = NOW(), fetched_at = EXCLUDED.fetched_at "
        "RETURNING (xmax = 0) AS inserted"
    )
    updated = 0
    with _conn() as conn:
        for rec in records:
            was_inserted = conn.execute(stmt, _live_json(rec)).scalar()
            if was_inserted:
                inserted_ids.append(rec["hotspot_id"])
            else:
                updated += 1
    return (len(inserted_ids), updated, inserted_ids)


def get_hotspots_first_seen_after(
    cursor_iso: str,
    after_id: Optional[str] = None,
    limit: int = 500,
) -> list:
    """Events first seen after `cursor_iso` — the DB-backed SSE cursor.

    Works across replicas: the poller (whichever replica holds the advisory
    lock) inserts rows; every replica's SSE stream reads new rows from here.
    Restart-safe: a fresh connection starts its cursor at connect time, so
    clients never get a replay flood of historical events.

    The cursor is the compound ``(first_seen_at, hotspot_id)`` key. Poll
    inserts commonly share one transaction timestamp; a timestamp-only cursor
    skipped every row after the first 500 in such a batch.
    """
    import json as _json
    if not enabled():
        if after_id is None:
            rows = [
                r for r in _MEM_LIVE.values()
                if str(r.get("first_seen_at") or "") > cursor_iso
            ]
        else:
            rows = [
                r for r in _MEM_LIVE.values()
                if (
                    str(r.get("first_seen_at") or ""),
                    str(r.get("hotspot_id") or ""),
                ) > (cursor_iso, after_id)
            ]
        rows.sort(key=lambda r: (
            str(r.get("first_seen_at") or ""),
            str(r.get("hotspot_id") or ""),
        ))
        out = []
        for r in rows[:limit]:
            rec = dict(r)
            for f in ("detected_by", "source_layers", "source_detection_ids"):
                if isinstance(rec.get(f), str):
                    rec[f] = _json.loads(rec[f])
            out.append(rec)
        return out

    from sqlalchemy import text

    if after_id is None:
        query = (
            "SELECT * FROM live_hotspots "
            "WHERE first_seen_at > CAST(:cur AS timestamptz) "
            "ORDER BY first_seen_at ASC, hotspot_id ASC LIMIT :lim"
        )
        params = {"cur": cursor_iso, "lim": int(limit)}
    else:
        query = (
            "SELECT * FROM live_hotspots "
            "WHERE first_seen_at > CAST(:cur AS timestamptz) "
            "OR (first_seen_at = CAST(:cur AS timestamptz) "
            "AND hotspot_id > :after_id) "
            "ORDER BY first_seen_at ASC, hotspot_id ASC LIMIT :lim"
        )
        params = {
            "cur": cursor_iso, "after_id": after_id, "lim": int(limit),
        }
    with _conn() as conn:
        rows = conn.execute(text(query), params).mappings().all()
    out = []
    for r in rows:
        rec = {k: (v.isoformat() if hasattr(v, "isoformat") else v)
               for k, v in dict(r).items()}
        for f in ("detected_by", "source_layers", "source_detection_ids"):
            v = rec.get(f)
            if isinstance(v, str):
                try:
                    rec[f] = _json.loads(v)
                except (TypeError, ValueError):
                    rec[f] = []
        out.append(rec)
    return out


def get_live_hotspots(
    since: Optional[str] = None,
    province: Optional[str] = None,
    satellite: Optional[str] = None,
    bbox: Optional[tuple] = None,   # (min_lon, min_lat, max_lon, max_lat)
    limit: int = 2000,
) -> list:
    """Query the live store with optional filters, newest detections first."""
    import json as _json
    limit = max(1, min(int(limit), 10000))

    if not enabled():
        rows = list(_MEM_LIVE.values())
        if since:
            rows = [r for r in rows if str(r["detected_at_utc"]) >= since]
        if province:
            rows = [r for r in rows if (r.get("province") or "") == province]
        if satellite:
            rows = [r for r in rows if (r.get("satellite") or "") == satellite]
        if bbox:
            rows = [
                r for r in rows
                if bbox[0] <= r["longitude"] <= bbox[2]
                and bbox[1] <= r["latitude"] <= bbox[3]
            ]
        rows.sort(key=lambda r: str(r["detected_at_utc"]), reverse=True)
        out = []
        for r in rows[:limit]:
            rec = dict(r)
            for f in ("detected_by", "source_layers", "source_detection_ids"):
                if isinstance(rec.get(f), str):
                    rec[f] = _json.loads(rec[f])
            out.append(rec)
        return out

    from sqlalchemy import text

    clauses, params = [], {"limit": limit}
    if since:
        clauses.append("detected_at_utc >= :since")
        params["since"] = since
    if province:
        clauses.append("province = :province")
        params["province"] = province
    if satellite:
        clauses.append("satellite = :satellite")
        params["satellite"] = satellite
    if bbox:
        clauses.append(
            "longitude BETWEEN :min_lon AND :max_lon "
            "AND latitude BETWEEN :min_lat AND :max_lat"
        )
        params.update({"min_lon": bbox[0], "min_lat": bbox[1],
                       "max_lon": bbox[2], "max_lat": bbox[3]})
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with _conn() as conn:
        rows = conn.execute(
            text(
                f"SELECT * FROM live_hotspots {where} "
                "ORDER BY detected_at_utc DESC LIMIT :limit"
            ),
            params,
        ).mappings().all()

    out = []
    for r in rows:
        rec = {k: (v.isoformat() if hasattr(v, "isoformat") else v)
               for k, v in dict(r).items()}
        # JSONB columns arrive as lists from psycopg; legacy TEXT as strings.
        for f in ("detected_by", "source_layers", "source_detection_ids"):
            v = rec.get(f)
            if isinstance(v, str):
                try:
                    rec[f] = _json.loads(v)
                except (TypeError, ValueError):
                    rec[f] = []
            elif v is None:
                rec[f] = []
        out.append(rec)
    return out


def cleanup_live_hotspots(retention_days: int = 7) -> int:
    """Delete live hotspots older than retention_days. Returns rows removed."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    if not enabled():
        stale = [
            hid for hid, r in _MEM_LIVE.items()
            if str(r["detected_at_utc"]) < cutoff.isoformat()
        ]
        for hid in stale:
            del _MEM_LIVE[hid]
        return len(stale)

    from sqlalchemy import text

    with _conn() as conn:
        res = conn.execute(
            text("DELETE FROM live_hotspots WHERE detected_at_utc < :cutoff"),
            {"cutoff": cutoff},
        )
    return res.rowcount or 0


def log_live_poll(status: str, stats: Optional[dict] = None,
                  duration_ms: int = 0, error: str = "") -> None:
    """Append one poll-cycle outcome (audit trail + status endpoint source).

    `status` ∈ {ok, partial, error} here; the poller additionally reports
    skipped_lock / disabled without logging a row (nothing was attempted).
    """
    s = stats or {}
    layers_failed = ",".join(s.get("layers_failed") or []) or None
    row = {
        "polled_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "raw_count": int(s.get("raw_count", 0)),
        "inserted": int(s.get("inserted", 0)),
        "updated": int(s.get("updated", 0)),
        "invalid": int(s.get("invalid_count", 0)),
        "merged": int(s.get("merged_count", 0)),
        "duration_ms": int(duration_ms),
        "error": (error or "")[:500] or None,
        "layers_failed": layers_failed,
    }
    if not enabled():
        _MEM_POLL_LOG.append(row)
        del _MEM_POLL_LOG[:-50]   # keep the tail bounded
        return
    from sqlalchemy import text

    with _conn() as conn:
        conn.execute(
            text(
                "INSERT INTO live_poll_log (status, raw_count, inserted, "
                "updated, invalid, merged, duration_ms, error, layers_failed) "
                "VALUES (:status, :raw_count, :inserted, :updated, :invalid, "
                ":merged, :duration_ms, :error, :layers_failed)"
            ),
            row,
        )


def live_poll_status() -> dict:
    """Latest poll outcome + last success — feeds /live-hotspots/status.
    `partial` counts as a success for freshness (some data landed), but the
    failed layers stay visible in last_poll.layers_failed."""
    if not enabled():
        last = _MEM_POLL_LOG[-1] if _MEM_POLL_LOG else None
        last_ok = next(
            (r for r in reversed(_MEM_POLL_LOG)
             if r["status"] in ("ok", "partial")), None
        )
        return {
            "storage": "memory",
            "hotspot_count": len(_MEM_LIVE),
            "last_poll": last,
            "last_success_at": last_ok["polled_at"] if last_ok else None,
        }
    from sqlalchemy import text

    with _conn() as conn:
        last = conn.execute(
            text("SELECT * FROM live_poll_log ORDER BY id DESC LIMIT 1")
        ).mappings().first()
        last_ok_at = conn.execute(
            text("SELECT MAX(polled_at) FROM live_poll_log "
                 "WHERE status IN ('ok', 'partial')")
        ).scalar()
        count = conn.execute(text("SELECT COUNT(*) FROM live_hotspots")).scalar()

    def _iso(v):
        return v.isoformat() if hasattr(v, "isoformat") else v

    return {
        "storage": "postgres",
        "hotspot_count": int(count or 0),
        "last_poll": {k: _iso(v) for k, v in dict(last).items()} if last else None,
        "last_success_at": _iso(last_ok_at),
    }


# ─── prediction runs (input-version idempotency + audit) ─────────────────────

# Advisory-lock key for prediction-only jobs — distinct from the GISTDA
# poller's key so the two subsystems never contend.
PREDICTION_LOCK_KEY = 0x70726564  # "pred" in hex

_MEM_PRED_RUNS: list = []


def last_predicted_input_version() -> Optional[str]:
    """Input-data version of the most recent SUCCESSFUL prediction run."""
    if not enabled():
        for r in reversed(_MEM_PRED_RUNS):
            if r["status"] == "done" and not r.get("noop"):
                return r.get("input_version")
        return None
    from sqlalchemy import text

    with _conn() as conn:
        return conn.execute(
            text(
                "SELECT input_version FROM prediction_runs "
                "WHERE status = 'done' AND NOT noop "
                "ORDER BY id DESC LIMIT 1"
            )
        ).scalar()


def record_prediction_run(
    status: str,
    input_version: Optional[str] = None,
    new_rows: int = 0,
    noop: bool = False,
    model_version: Optional[str] = None,
    error: str = "",
) -> None:
    row = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "input_version": input_version,
        "new_rows": int(new_rows),
        "noop": bool(noop),
        "model_version": model_version,
        "error": (error or "")[:500] or None,
    }
    if not enabled():
        _MEM_PRED_RUNS.append(row)
        del _MEM_PRED_RUNS[:-50]
        return
    from sqlalchemy import text

    with _conn() as conn:
        conn.execute(
            text(
                "INSERT INTO prediction_runs (started_at, finished_at, status, "
                "input_version, new_rows, noop, model_version, error) "
                "VALUES (:started_at, :finished_at, :status, :input_version, "
                ":new_rows, :noop, :model_version, :error)"
            ),
            row,
        )


def last_prediction_run() -> Optional[dict]:
    if not enabled():
        return dict(_MEM_PRED_RUNS[-1]) if _MEM_PRED_RUNS else None
    from sqlalchemy import text

    with _conn() as conn:
        row = conn.execute(
            text("SELECT * FROM prediction_runs ORDER BY id DESC LIMIT 1")
        ).mappings().first()
    if not row:
        return None
    return {k: (v.isoformat() if hasattr(v, "isoformat") else v)
            for k, v in dict(row).items()}


# ─── advisory lock (single-poller protection across Railway replicas) ─────────

# Arbitrary but fixed application-wide key for the GISTDA poller lock.
GISTDA_POLLER_LOCK_KEY = 0x67697374  # "gist" in hex

_MEM_LOCKS: set = set()


class advisory_lock:
    """Context manager around pg_try_advisory_lock.

    Usage:
        with db.advisory_lock(db.GISTDA_POLLER_LOCK_KEY) as acquired:
            if not acquired:
                return   # another instance is polling — skip quietly

    Without a database this degrades to a process-local set — adequate for
    single-process local dev, NOT cross-instance safe (documented: multiple
    replicas without PostgreSQL each poll independently).
    """

    def __init__(self, key: int) -> None:
        self.key = int(key)
        self.acquired = False
        self._conn_ctx = None
        self._conn = None

    def __enter__(self) -> bool:
        if not enabled():
            if self.key in _MEM_LOCKS:
                return False
            _MEM_LOCKS.add(self.key)
            self.acquired = True
            return True
        from sqlalchemy import text

        # Hold ONE dedicated connection for the lock's lifetime — advisory
        # locks are connection-scoped, so it must not go back to the pool
        # until released.
        self._conn = get_engine().connect()
        got = self._conn.execute(
            text("SELECT pg_try_advisory_lock(:k)"), {"k": self.key}
        ).scalar()
        self.acquired = bool(got)
        if not self.acquired:
            self._conn.close()
            self._conn = None
        return self.acquired

    def __exit__(self, *exc) -> None:
        if not enabled():
            if self.acquired:
                _MEM_LOCKS.discard(self.key)
            return
        if self._conn is not None:
            from sqlalchemy import text
            try:
                self._conn.execute(
                    text("SELECT pg_advisory_unlock(:k)"), {"k": self.key}
                )
            finally:
                self._conn.close()
                self._conn = None
