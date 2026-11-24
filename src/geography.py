"""Thailand membership using the same bundled province polygons as the dashboard."""

import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd


@lru_cache(maxsize=1)
def polygons():
    path = Path(__file__).resolve().parents[1] / "frontend/data/thailand-provinces.geojson"
    features = json.loads(path.read_text())["features"]
    result = []
    for feature in features:
        geometry = feature["geometry"]
        parts = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        for poly in parts:
            xs, ys = zip(*poly[0])
            result.append(((min(xs), min(ys), max(xs), max(ys)), poly))
    return result


def in_ring(x, y, ring):
    inside = False
    xj, yj = ring[-1]
    for xi, yi in ring:
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        xj, yj = xi, yi
    return inside


@lru_cache(maxsize=65536)
def in_thailand(lat, lon):
    for (west, south, east, north), poly in polygons():
        if west <= lon <= east and south <= lat <= north:
            if in_ring(lon, lat, poly[0]) and not any(in_ring(lon, lat, ring) for ring in poly[1:]):
                return True
    return False


def thailand_detections(daily, grid_size):
    """Filter by cell centre, testing each distinct cell only once."""
    if daily.empty:
        return daily
    cells = daily[["lat_cell", "lon_cell"]].drop_duplicates()
    keep = [in_thailand(lat * grid_size, lon * grid_size) for lat, lon in cells.itertuples(index=False, name=None)]
    index = pd.MultiIndex.from_frame(cells[keep])
    return daily[pd.MultiIndex.from_frame(daily[["lat_cell", "lon_cell"]]).isin(index)]


def thailand_cells(grid_size):
    """Candidate geometry depends on country boundaries, never future detections."""
    if not math.isfinite(grid_size) or not .02 <= grid_size <= 1:
        raise ValueError("Country candidate grid must be between 0.02 and 1 degree")
    boxes = [box for box, _ in polygons()]
    west, south = min(b[0] for b in boxes), min(b[1] for b in boxes)
    east, north = max(b[2] for b in boxes), max(b[3] for b in boxes)
    cells = [(lat, lon) for lat in range(math.ceil(south / grid_size), math.floor(north / grid_size) + 1)
             for lon in range(math.ceil(west / grid_size), math.floor(east / grid_size) + 1)
             if in_thailand(lat * grid_size, lon * grid_size)]
    return pd.DataFrame(cells, columns=["lat_cell", "lon_cell"])
