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
from datetime import datetime, timezone
from typing import Optional

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
"""


def init_db() -> bool:
    """Create all tables if missing. Returns True when DB is ready."""
    if not enabled():
        log.info("DATABASE_URL not set — running without a database")
        return False
    from sqlalchemy import text

    with _conn() as conn:
        for stmt in _SCHEMA.split(";"):
            if stmt.strip():
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
    holds exactly one row per (base_date, cell)."""
    if not enabled() or df.empty:
        return 0
    from sqlalchemy import text

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
    sub["base_date"] = pd.to_datetime(base_date).date()
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
        conn.execute(
            text("DELETE FROM risk_predictions WHERE base_date = :bd"),
            {"bd": sub["base_date"].iloc[0]},
        )
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
                "n_positive = :pos, error = :err WHERE id = :id"
            ),
            {
                "s": status,
                "auc": m.get("roc_auc"),
                "f1": m.get("f1"),
                "rows": m.get("n_train_rows"),
                "pos": m.get("n_positive"),
                "err": (error or "")[:4000] or None,
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
