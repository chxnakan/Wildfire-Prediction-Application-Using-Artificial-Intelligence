"""Publish a current forecast map and an immutable issuance record."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from dotenv import load_dotenv
from src.forecasting import predict, save_snapshot
from src.detection_store import DetectionStore
from src.pipeline import observation_paths


def build_geojson(snapshot, detections):
    """Map features for one issued snapshot plus detections on its feature-cutoff day."""
    grid = snapshot["grid_size"]
    if "grid_size" in detections.attrs and not abs(detections.attrs["grid_size"] - grid) < 1e-9:
        raise ValueError("Observation grid does not match the forecast grid")
    cutoff = pd.Timestamp(snapshot["feature_cutoff"])
    features = []
    for row in detections[detections.date == cutoff].itertuples():
        lat, lon = row.lat_cell * grid, row.lon_cell * grid
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                         "properties": {"source": "observed", "date": str(cutoff.date()),
                                        "fire_count": int(row.fire_count), "lat": lat, "lon": lon}})
    for row in snapshot["predictions"]:
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [row["lon_grid"], row["lat_grid"]]},
                         "properties": {**row, "source": "predicted", "base_date": snapshot["base_date"]}})
    return {"type": "FeatureCollection", "features": features,
            "metadata": {key: value for key, value in snapshot.items() if key != "predictions"}}


def run(output=None):
    load_dotenv(ROOT / ".env")
    output = Path(output or os.getenv("OUTPUT_DIR", str(ROOT / "outputs")))
    bundle = joblib.load(output / "models/lgbm_fire_date_model.pkl")
    frame = pd.read_csv(output / "features/latest_features.csv", parse_dates=["date"])
    predictions = predict(bundle, frame)
    snapshot_path = save_snapshot(predictions, bundle, output / "forecasts")
    snapshot = json.loads(snapshot_path.read_text())
    detections = DetectionStore().daily(output, snapshot["grid_size"],
        observation_paths(ROOT, os.getenv("RAW_DIR"), os.getenv("FIRMS_PATH")))
    geojson = build_geojson(snapshot, detections)
    geojson["metadata"]["snapshot"] = snapshot_path.name
    directory = output / "riskmap"
    directory.mkdir(exist_ok=True)
    (directory / "fire_dates_all.geojson").write_text(json.dumps(geojson, allow_nan=False))
    (directory / "latest.json").write_text(json.dumps(geojson["metadata"], indent=2))
    print(f"Map: {snapshot['base_date']} ({snapshot['kind']}), {len(predictions):,} candidate grids", flush=True)
    return geojson


if __name__ == "__main__":
    run()
