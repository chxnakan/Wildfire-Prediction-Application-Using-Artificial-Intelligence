"""Public, pre-forecast land cover and terrain context, cached by cell geometry.

WorldCover 2021 is a static baseline, not current vegetation/fuel moisture.
Fractions approximate the distribution of 300 m overview classes within the
entire hotspot cell. Copernicus GLO-90 is a surface model, not bare-earth terrain.
"""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
           60: "bare", 70: "snow", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}
LAND_FEATURES = [f"land_{name}_fraction" for name in CLASSES.values()]
TERRAIN_FEATURES = ["terrain_elevation_mean_m", "terrain_elevation_sd_m", "terrain_slope_mean_deg"]
POLICY = {
    "land_cover": "ESA WorldCover 2021 v200, released 2022; static historical context",
    "land_documentation": "https://esa-worldcover.org/en/data-access",
    "land_attribution": "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium; CC BY 4.0",
    "land_sampling": "Class fractions from 32x COG overview (0.002667 degrees, about 300 m); approximate, not native 10 m area fractions",
    "terrain": "Copernicus GLO-90 public DSM; mean elevation, elevation variation and slope at ~90 m",
    "terrain_documentation": "https://copernicus-dem-90m.s3.amazonaws.com/readme.html",
    "terrain_attribution": "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018; provided under COPERNICUS by the European Union and ESA",
    "availability_cutoff": "2023-01-01",
    "limitations": "Static land cover cannot establish ignition cause, distinguish controlled burns, identify crop species, or measure current fuel moisture. DSM slope includes vegetation/infrastructure and tile-edge approximation.",
    "schema": 1,
}


def _atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_tile(url, directory, factor=1):
    """Cache the actual analysis raster plus georeferencing and source URL."""
    key = hashlib.sha256(f"{url}|{factor}|v1".encode()).hexdigest()
    path = Path(directory) / f"{key}.npz"
    if not path.exists():
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                          GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_MAX_RETRY="2"):
            with rasterio.open(url) as src:
                shape = (math.ceil(src.height / factor), math.ceil(src.width / factor))
                array = src.read(1, out_shape=shape, masked=True).astype("float32").filled(np.nan)
                bounds = np.array(src.bounds)
        _atomic_npz(path, data=array, bounds=bounds)
    with np.load(path) as cached:
        array, bounds = cached["data"], cached["bounds"]
    source = {"url": url, "factor": factor, "cached_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return array, bounds, source


def summarize_tile(array, bounds, grid, terrain=False):
    """Aggregate pixel centres into the same nearest-centre cells as FIRMS."""
    west, south, east, north = bounds
    height, width = array.shape
    dy, dx = (north - south) / height, (east - west) / width
    lat = north - (np.arange(height) + .5) * dy
    lon = west + (np.arange(width) + .5) * dx
    # Integer composite key avoids a multi-million-row pandas intermediate.
    lat_ids = np.rint(lat / grid).astype("int32")
    lon_ids = np.rint(lon / grid).astype("int32")
    keys = (lat_ids[:, None].astype("int64") * 100000 + lon_ids[None, :]).ravel()
    unique, inverse = np.unique(keys, return_inverse=True)
    valid = np.isfinite(array).ravel()
    if not terrain:
        valid &= np.isin(array.ravel(), list(CLASSES))
    count = np.bincount(inverse[valid], minlength=len(unique))
    result = pd.DataFrame({"lat_cell": unique // 100000, "lon_cell": unique % 100000, "count": count})
    if terrain:
        # Finite differences in physical metres; preserve missing values.
        gy, gx = np.gradient(array.astype(float))
        slope = np.degrees(np.arctan(np.hypot(gy / (dy * 111320), gx / (dx * 111320 * np.cos(np.radians(lat))[:, None]))))
        valid &= np.isfinite(slope).ravel()
        result["count"] = np.bincount(inverse[valid], minlength=len(unique))
        for name, values in [("elevation_sum", array), ("elevation_squared_sum", array.astype(float) ** 2), ("slope_sum", slope)]:
            result[name] = np.bincount(inverse[valid], weights=values.ravel()[valid], minlength=len(unique))
    else:
        for code, name in CLASSES.items():
            result[f"land_{name}_fraction"] = np.bincount(inverse[valid & (array.ravel() == code)], minlength=len(unique))
    return result[result["count"] > 0]


def tiles_for_cells(cells, grid, size):
    tiles = set()
    for lat, lon in cells.itertuples(index=False, name=None):
        south, north = lat * grid - grid / 2, lat * grid + grid / 2
        west, east = lon * grid - grid / 2, lon * grid + grid / 2
        for y in range(math.floor(south / size) * size, math.floor((north - 1e-9) / size) * size + 1, size):
            for x in range(math.floor(west / size) * size, math.floor((east - 1e-9) / size) * size + 1, size):
                tiles.add((y, x))
    return sorted(tiles)


def cell_context(cells, grid, directory):
    cells = cells[["lat_cell", "lon_cell"]].drop_duplicates().sort_values(["lat_cell", "lon_cell"])
    if not 0 < grid <= 1:
        raise ValueError("Context extraction supports positive grids up to one degree")
    if not (cells.lat_cell.gt(0).all() and cells.lon_cell.between(1, 99999).all()):
        raise ValueError("This context extractor is scoped to Thailand's northern/eastern coordinates")
    signature = hashlib.sha256((cells.to_csv(index=False) + json.dumps(POLICY, sort_keys=True) + str(grid)).encode()).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    table_path, meta_path = directory / f"{signature}.csv", directory / f"{signature}.json"
    if table_path.exists() and meta_path.exists():
        return pd.read_csv(table_path), json.loads(meta_path.read_text())
    tables, sources = {}, []
    for kind, size, factor in [("land", 3, 32), ("terrain", 1, 1)]:
        tiles = tiles_for_cells(cells, grid, size)
        parts = []
        for i, (lat, lon) in enumerate(tiles, 1):
            if kind == "land":
                name = f"ESA_WorldCover_10m_2021_v200_N{lat:02d}E{lon:03d}_Map.tif"
                url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{name}"
            else:
                name = f"Copernicus_DSM_COG_30_N{lat:02d}_00_E{lon:03d}_00_DEM"
                url = f"https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif"
            array, bounds, source = read_tile(url, directory / "tiles", factor)
            parts.append(summarize_tile(array, bounds, grid, terrain=kind == "terrain").merge(cells, on=["lat_cell", "lon_cell"]))
            sources.append(source)
            if i == 1 or i % 10 == 0 or i == len(tiles):
                print(f"Context {kind}: {i}/{len(tiles)} tiles", flush=True)
        combined = pd.concat(parts).groupby(["lat_cell", "lon_cell"]).sum()
        if kind == "land":
            combined[LAND_FEATURES] = combined[LAND_FEATURES].div(combined["count"], axis=0)
            tables[kind] = combined[LAND_FEATURES].reset_index()
        else:
            mean = combined.elevation_sum / combined["count"]
            combined[TERRAIN_FEATURES[0]] = mean
            combined[TERRAIN_FEATURES[1]] = np.sqrt(np.maximum(0, combined.elevation_squared_sum / combined["count"] - mean ** 2))
            combined[TERRAIN_FEATURES[2]] = combined.slope_sum / combined["count"]
            tables[kind] = combined[TERRAIN_FEATURES].reset_index()
    table = cells.merge(tables["land"], on=["lat_cell", "lon_cell"], how="left").merge(tables["terrain"], on=["lat_cell", "lon_cell"], how="left")
    if not np.isfinite(table[LAND_FEATURES + TERRAIN_FEATURES].to_numpy()).all():
        raise ValueError("Land/terrain coverage incomplete; do not substitute zero for unavailable context")
    metadata = {**POLICY, "cells": len(cells), "sources": sources, "grid_size": grid}
    table.to_csv(table_path, index=False)
    metadata["table_sha256"] = hashlib.sha256(table_path.read_bytes()).hexdigest()
    meta_path.write_text(json.dumps(metadata, indent=2))
    return table, metadata


def attach_context(panel, grid, directory):
    if panel.date.min() < pd.Timestamp(POLICY["availability_cutoff"]):
        raise ValueError("Static context was not available before the requested historical period")
    table, metadata = cell_context(panel, grid, directory)
    return panel.merge(table, on=["lat_cell", "lon_cell"], how="left", validate="many_to_one"), metadata
