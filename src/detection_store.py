"""Retain original detection coordinates so archived grids remain reproducible."""

import math
import os
import tempfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline import load_observations, target_observations


def save_observation_coordinates(observations, output):
    """Keep one row per deduplicated satellite detection, including coincident pixels."""
    directory = Path(output) / "features"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        columns = ["date", "latitude", "longitude"] + [c for c in ["hotspot_type", "daynight", "confidence", "satellite", "observed_at"] if c in observations]
        observations[columns].to_csv(
            temporary, index=False, compression="gzip")
        os.replace(temporary, directory / "observation_coordinates.csv.gz")
    finally:
        temporary.unlink(missing_ok=True)


class DetectionStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._revision = None
        self._observations = None
        self._grids = {}

    def daily(self, output, grid_size, source_paths=(), target_policy=None):
        grid_size = float(grid_size)
        if not math.isfinite(grid_size) or grid_size <= 0:
            raise ValueError("Grid size must be finite and positive")
        coordinates = Path(output) / "features/observation_coordinates.csv.gz"
        paths = [coordinates] if coordinates.exists() else list(source_paths)
        if not paths:
            raise FileNotFoundError("Original detection coordinates unavailable; run the training pipeline")
        revision = tuple((str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
        with self._lock:
            if revision != self._revision:
                if paths == [coordinates]:
                    observations = pd.read_csv(coordinates, parse_dates=["date"])
                else:
                    # Compatibility for pre-existing outputs: recover coordinates
                    # from input CSVs, never guess them from an aggregated grid.
                    observations, _ = load_observations(paths)
                self._observations = observations
                self._revision = revision
                self._grids.clear()
            key = (grid_size, target_policy)
            if key not in self._grids:
                selected = target_observations(self._observations, target_policy)
                frame = selected.assign(
                    lat_cell=np.rint(selected.latitude / grid_size).astype("int32"),
                    lon_cell=np.rint(selected.longitude / grid_size).astype("int32"),
                )
                daily = frame.groupby(["lat_cell", "lon_cell", "date"]).size().rename("fire_count").reset_index()
                daily.attrs["grid_size"] = grid_size
                daily.attrs["target_policy"] = target_policy
                self._grids[key] = daily
            return self._grids[key]
