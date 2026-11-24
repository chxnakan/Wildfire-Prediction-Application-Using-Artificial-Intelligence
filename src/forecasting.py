"""Shared inference and immutable forecast records for later verification."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import evaluate
from src.pipeline import FEATURES, target_observations

SCHEMA_VERSION = 4


def forecast_horizon(record):
    """Preserve seven-day semantics of older bundles lacking an explicit horizon."""
    horizon = record.get("horizon_days", 7)
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in (3, 7):
        raise ValueError("Forecast horizon must be 3 or 7 days")
    return horizon


def predict(bundle, frame):
    if bundle.get("schema_version") != SCHEMA_VERSION or "calibrator" not in bundle:
        raise ValueError("Retrain with the calibrated daily hotspot pipeline before forecasting")
    result = frame[["date", "lat_cell", "lon_cell"]].copy()
    result["lat_grid"] = result.lat_cell * bundle["grid_size"]
    result["lon_grid"] = result.lon_cell * bundle["grid_size"]
    features = bundle.get("features", FEATURES)
    missing = [name for name in features if name not in frame]
    if missing:
        raise ValueError(f"Forecast inputs missing required features: {', '.join(missing)}")
    if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
        raise ValueError("Forecast inputs contain missing values; refresh complete inputs before forecasting")
    scores = bundle["classifier"].predict_proba(frame[features])[:, 1]
    probability = np.asarray(bundle["calibrator"].transform(scores), dtype=float)
    days = np.clip(bundle["regressor"].predict(frame[features]), 1, forecast_horizon(bundle))
    selected = probability >= bundle["threshold"]
    rounded = np.rint(days).astype(int)
    result["detection_score"] = scores
    result["probability"] = probability
    result["forecast_detection"] = selected
    result["estimated_days_if_detection"] = days
    result["days_until_fire"] = np.where(selected, rounded, -1)
    result["predicted_fire_date"] = pd.Series([
        (pd.Timestamp(date) + pd.Timedelta(days=int(day))).date().isoformat() if active else None
        for date, day, active in zip(frame.date, rounded, selected)
    ], index=result.index, dtype=object)
    result["urgency_level"] = np.where(~selected, "NONE", np.where(rounded <= 2, "HIGH", np.where(rounded <= 4, "MEDIUM", "LOW")))
    return result


def save_snapshot(predictions, bundle, directory, issued_at=None):
    """One immutable issuance per model/base date; historical output is explicit."""
    issued = issued_at or datetime.now(timezone.utc)
    base = pd.Timestamp(predictions.date.max()).date()
    prospective = base == issued.date() and pd.Timestamp(bundle["training_labels_through"]).date() < issued.date()
    snapshot = {
        "schema_version": SCHEMA_VERSION, "issued_at": issued.isoformat(), "base_date": base.isoformat(),
        "feature_cutoff": (pd.Timestamp(base) - pd.Timedelta(days=1)).date().isoformat(),
        "horizon_days": forecast_horizon(bundle), "model_version": bundle["model_version"],
        "training_labels_through": bundle["training_labels_through"],
        "grid_size": bundle["grid_size"], "threshold": bundle["threshold"],
        "prospective": prospective, "kind": "prospective" if prospective else "historical_demo",
        "calibrated": "probability" in predictions,
        "predictions": predictions.drop(columns="date").to_dict(orient="records"),
    }
    if bundle.get("weather"):
        snapshot["weather"] = bundle["weather"]
        snapshot["features"] = bundle["features"]
    for name in ["target_policy", "context", "candidate_population", "selection"]:
        if name in bundle:
            snapshot[name] = bundle[name]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{base}_{bundle['model_version']}.json"
    serialized = json.dumps(snapshot, allow_nan=False)
    # Link a complete temporary file into place atomically without overwriting
    # an existing issuance, even when two processes publish concurrently.
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
    try:
        os.link(temporary, path)
    except FileExistsError:
        pass
    finally:
        temporary.unlink()
    return path


def score_snapshots(directory, observations, observation_end):
    """Only score forecasts genuinely issued before their future outcome window."""
    end = pd.Timestamp(observation_end).normalize()
    observed = observations[observations.date <= end].copy()
    regional_dates = set(observed.date)
    report = {"evaluated_runs": [], "pending_runs": 0, "historical_runs_excluded": 0,
              "unknown_coverage_runs": 0, "invalid_runs_excluded": 0}
    for path in sorted(Path(directory).glob("*.json")):
        snapshot = json.loads(path.read_text())
        if not snapshot.get("prospective"):
            report["historical_runs_excluded"] += 1
            continue
        base = pd.Timestamp(snapshot["base_date"])
        issued = pd.Timestamp(snapshot["issued_at"])
        if (issued.tzinfo is None or issued.tz_convert("UTC").date() != base.date()
                or pd.Timestamp(snapshot["training_labels_through"]) >= base
                or pd.Timestamp(snapshot["feature_cutoff"]) >= base):
            report["invalid_runs_excluded"] += 1
            continue
        try:
            horizon = forecast_horizon(snapshot)
        except ValueError:
            report["invalid_runs_excluded"] += 1
            continue
        dates = pd.date_range(base + pd.Timedelta(days=1), periods=horizon)
        if dates[-1] > end:
            report["pending_runs"] += 1
            continue
        if not set(dates).issubset(regional_dates):
            report["unknown_coverage_runs"] += 1
            continue
        future = target_observations(observed[observed.date.isin(dates)], snapshot.get("target_policy")).copy()
        future["lat_cell"] = np.rint(future.latitude / snapshot["grid_size"]).astype(int)
        future["lon_cell"] = np.rint(future.longitude / snapshot["grid_size"]).astype(int)
        first = future.groupby(["lat_cell", "lon_cell"]).date.min()
        rows = pd.DataFrame(snapshot["predictions"])
        actual_days = np.array([
            (first.loc[(lat, lon)] - base).days if (lat, lon) in first.index else -1
            for lat, lon in zip(rows.lat_cell, rows.lon_cell)
        ])
        scores = rows.probability if "probability" in rows else rows.detection_score
        metrics = evaluate(actual_days > 0, actual_days, scores,
                           rows.estimated_days_if_detection, snapshot["threshold"])
        report["evaluated_runs"].append({"file": path.name, "base_date": snapshot["base_date"],
                                         "horizon_days": horizon,
                                         "target_policy": snapshot.get("target_policy", "all_hotspots"),
                                         "model_version": snapshot["model_version"], "metrics": metrics})
    return report
