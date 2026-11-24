"""Serve saved hotspot forecasts, evaluation results and independent live alerts."""

import json
import os
import re
import threading
from collections import Counter
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from src.alerts import FireAlerts, monitor
from src.evaluation import evaluate
from src.risk_map import build_geojson
from src.detection_store import DetectionStore
from src.pipeline import HORIZON, observation_paths
from src.forecasting import forecast_horizon
from src.geography import in_thailand, thailand_detections
from src.security import headers_for

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
OUTPUT = Path(os.getenv("OUTPUT_DIR", str(ROOT / "outputs")))
alerts = FireAlerts()
SNAPSHOT_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}_[0-9a-f]{12}\.json$")
_cache = {}
detection_store = DetectionStore()


def read_json(relative):
    path = OUTPUT / relative
    if not path.exists():
        raise HTTPException(503, "Run python src/training.py to generate forecast and evaluation assets")
    return json.loads(path.read_text())


def cached(path, loader):
    """Reload a derived asset only when the file on disk changes."""
    if not path.exists():
        raise HTTPException(503, "Run python src/training.py to generate forecast and evaluation assets")
    stamp = path.stat().st_mtime_ns
    hit = _cache.get(path)
    if hit and hit[0] == stamp:
        return hit[1]
    value = loader(path)
    _cache[path] = (stamp, value)
    return value


def detections(grid_size=None, target_policy=None):
    if grid_size is None:
        grid_size = read_json("metadata/dataset_info.json")["grid_size"]
    sources = observation_paths(ROOT, os.getenv("RAW_DIR"), os.getenv("FIRMS_PATH"))
    try:
        return detection_store.daily(OUTPUT, grid_size, sources, target_policy)
    except FileNotFoundError as exc:
        raise HTTPException(503, "Original detection coordinates unavailable; run python src/training.py") from exc


def utc_today():
    return datetime.now(timezone.utc).date()


def snapshot_path(name):
    if not SNAPSHOT_NAME.fullmatch(name):
        raise HTTPException(404, "Unknown forecast")
    path = OUTPUT / "forecasts" / name
    if not path.is_file():
        raise HTTPException(404, "Unknown forecast")
    return path


def load_snapshot(name):
    return cached(snapshot_path(name), lambda path: json.loads(path.read_text()))


@asynccontextmanager
async def lifespan(app):
    stop = threading.Event()
    worker = threading.Thread(target=monitor, args=(alerts, stop), daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()


app = FastAPI(title="Hotspot Forecast and Alerts API", version="3.0", lifespan=lifespan)
app.mount("/dashboard", StaticFiles(directory=ROOT / "frontend", html=True), name="dashboard")


@app.middleware("http")
async def revalidate_dashboard(request, call_next):
    """Make browsers revalidate dashboard files (ETag) so an edited script is never served stale."""
    response = await call_next(request)
    if request.url.path.startswith("/dashboard"):
        response.headers["Cache-Control"] = "no-cache"
    response.headers.update(headers_for(request.url.path))
    return response


@app.get("/")
def root():
    latest = OUTPUT / "riskmap/latest.json"
    horizon = forecast_horizon(json.loads(latest.read_text())) if latest.exists() else HORIZON
    return {"status": "running", "version": "3.0", "prediction_type": "satellite_hotspot", "horizon_days": horizon}


@app.get("/alerts")
def fire_alerts(limit: int = Query(default=100, ge=1, le=5000), scope: Literal["region", "thailand"] = "region"):
    """Live detections from the past 24 hours, newest first; ``total`` counts all of them."""
    return {**alerts.snapshot(limit=limit, location_filter=in_thailand if scope == "thailand" else None), "scope": scope}


@app.get("/metadata")
def metadata():
    return read_json("metadata/dataset_info.json")


@app.get("/performance")
def performance():
    prospective = OUTPUT / "evaluation/prospective.json"
    return {"holdout": read_json("evaluation/metrics.json"),
            "prospective": json.loads(prospective.read_text()) if prospective.exists() else None}


@app.get("/geojson")
def get_geojson():
    return read_json("riskmap/fire_dates_all.geojson")


@app.get("/predictions/today")
def predictions_today():
    """Compatibility route: returns the latest issued forecast, with its real base date."""
    geojson = get_geojson()
    records = [feature["properties"] for feature in geojson["features"]
               if feature["properties"]["source"] == "predicted"]
    return {**geojson["metadata"], "prediction_horizon_days": forecast_horizon(geojson["metadata"]), "total_locations": len(records),
            "urgency_summary": dict(Counter(row["urgency_level"] for row in records)),
            "predictions": records}


@app.get("/predictions/timeline")
def predictions_timeline():
    forecast = predictions_today()
    base = date.fromisoformat(forecast["base_date"])
    timeline = {}
    for day in range(1, forecast_horizon(forecast) + 1):
        rows = [row for row in forecast["predictions"] if row["days_until_fire"] == day]
        timeline[(base + timedelta(days=day)).isoformat()] = {
            "days_from_base": day, "forecast_count": len(rows),
            "locations": [{"lat_grid": row["lat_grid"], "lon_grid": row["lon_grid"]} for row in rows],
        }
    return {"base_date": forecast["base_date"], "kind": forecast["kind"],
            "horizon_days": forecast_horizon(forecast), "timeline": timeline}


@app.get("/predict/location")
def predict_location(lat: float = Query(ge=-90, le=90), lon: float = Query(ge=-180, le=180),
                     grid_size: float | None = Query(default=None, gt=0)):
    forecast = predictions_today()
    grid = forecast["grid_size"]
    if grid_size is not None and abs(grid_size - grid) > 1e-9:
        raise HTTPException(400, "grid_size must match the trained model")
    lat_cell, lon_cell = round(lat / grid), round(lon / grid)
    row = next((row for row in forecast["predictions"]
                if row["lat_cell"] == lat_cell and row["lon_cell"] == lon_cell), None)
    if row is None:
        raise HTTPException(404, "Location is outside the previously observed candidate grids")
    return {**row, "model_version": forecast["model_version"], "kind": forecast["kind"],
            "location": {"lat": row["lat_grid"], "lon": row["lon_grid"]}}


@app.get("/forecasts")
def list_forecasts():
    directory = OUTPUT / "forecasts"
    latest = OUTPUT / "riskmap/latest.json"
    latest_name = json.loads(latest.read_text()).get("snapshot") if latest.exists() else None
    rows = []
    for path in directory.glob("*.json") if directory.exists() else []:
        if not SNAPSHOT_NAME.fullmatch(path.name):
            continue
        snapshot = load_snapshot(path.name)
        rows.append({"name": path.name, "base_date": snapshot["base_date"], "issued_at": snapshot["issued_at"],
                     "horizon_days": forecast_horizon(snapshot),
                     "model_version": snapshot["model_version"], "kind": snapshot["kind"],
                     "schema_version": snapshot["schema_version"], "calibrated": bool(snapshot.get("calibrated")),
                     "flagged": sum(1 for row in snapshot["predictions"] if row["forecast_detection"]),
                     "candidates": len(snapshot["predictions"]), "latest": path.name == latest_name})
    rows.sort(key=lambda row: (row["base_date"], row["issued_at"]), reverse=True)
    return {"latest": latest_name, "forecasts": rows}


@app.get("/forecasts/{name}/geojson")
def forecast_geojson(name: str):
    snapshot = load_snapshot(name)
    geojson = build_geojson(snapshot, detections(snapshot["grid_size"]))
    geojson["metadata"]["snapshot"] = name
    return geojson


@app.get("/forecasts/{name}/outcomes")
def forecast_outcomes(name: str, scope: Literal["region", "thailand"] = "region"):
    """Retrospective comparison with later recorded hotspots; not prospective verification."""
    snapshot = load_snapshot(name)
    base = pd.Timestamp(snapshot["base_date"])
    horizon = forecast_horizon(snapshot)
    window = pd.date_range(base + pd.Timedelta(days=1), periods=horizon)
    # Partial observations on the current UTC day cannot complete a label.
    completed_day = pd.Timestamp(utc_today()) - pd.Timedelta(days=1)
    observed = detections(snapshot["grid_size"])
    observed = observed[observed.date <= completed_day]
    result = {"name": name, "base_date": snapshot["base_date"], "kind": snapshot["kind"], "horizon_days": horizon,
              "window_start": str(window[0].date()), "window_end": str(window[-1].date()),
              "retrospective": not snapshot.get("prospective"), "threshold": snapshot["threshold"], "scope": scope,
              "target_policy": snapshot.get("target_policy", "all_hotspots")}
    if observed.empty or window[-1] > observed.date.max():
        return {**result, "status": "pending", "cells": [], "metrics": None}
    missing = sorted(set(window) - set(observed.date.unique()))
    if missing:
        return {**result, "status": "unknown_coverage", "missing_dates": [str(d.date()) for d in missing],
                "cells": [], "metrics": None}
    # Coverage comes from all raw detections; outcome labels follow the exact
    # policy saved with each issuance, including legacy all-hotspot snapshots.
    if snapshot.get("target_policy"):
        observed = detections(snapshot["grid_size"], snapshot["target_policy"])
    future = observed[observed.date.isin(window)]
    first = future.groupby(["lat_cell", "lon_cell"]).date.min()
    rows = pd.DataFrame(snapshot["predictions"])
    if scope == "thailand":
        rows = rows[[in_thailand(lat, lon) for lat, lon in zip(rows.lat_grid, rows.lon_grid)]]
        if rows.empty:
            return {**result, "status": "no_candidates", "cells": [], "metrics": None}
    actual = np.array([(first.loc[(lat, lon)] - base).days if (lat, lon) in first.index else -1
                       for lat, lon in zip(rows.lat_cell, rows.lon_cell)])
    scores = rows.probability if "probability" in rows else rows.detection_score
    metrics = evaluate(actual > 0, actual, scores, rows.estimated_days_if_detection, snapshot["threshold"])
    cells = [{"lat_grid": float(lat), "lon_grid": float(lon), "flagged": bool(flag), "actual_day": int(day),
              "predicted_day": int(pred) if flag else None}
             for lat, lon, flag, day, pred in zip(rows.lat_grid, rows.lon_grid, rows.forecast_detection,
                                                  actual, rows.days_until_fire)
             if flag or day > 0]
    return {**result, "status": "ok", "metrics": metrics, "cells": cells}


@app.get("/cell")
def cell_history(lat_cell: int, lon_cell: int, until: date, snapshot_name: str | None = None):
    grid_size = load_snapshot(snapshot_name)["grid_size"] if snapshot_name else None
    observed = detections(grid_size)
    end = pd.Timestamp(until)
    rows = observed[(observed.lat_cell == lat_cell) & (observed.lon_cell == lon_cell) & (observed.date <= end)]

    def since(days):
        return rows[rows.date > end - pd.Timedelta(days=days)]

    return {"lat_cell": lat_cell, "lon_cell": lon_cell, "until": str(end.date()),
            "detections_7d": int(since(7).fire_count.sum()), "detections_30d": int(since(30).fire_count.sum()),
            "fire_days_30d": int(since(30).date.nunique()), "fire_days_365d": int(since(365).date.nunique()),
            "history_days": int((end - observed.date.min()).days + 1) if not observed.empty else 0,
            "last_detection": str(rows.date.max().date()) if not rows.empty else None}


@app.get("/history/daily")
def daily_history(scope: Literal["region", "thailand"] = "region"):
    observed = detections()
    if scope == "thailand":
        observed = thailand_detections(observed, read_json("metadata/dataset_info.json")["grid_size"])
    daily = observed.groupby("date").agg(detections=("fire_count", "sum"), cells=("lat_cell", "size"))
    return {"days": [{"date": str(day.date()), "detections": int(row.detections), "cells": int(row.cells)}
                     for day, row in daily.iterrows()]}
