"""Canonical 0.1° grid snapping — ONE definition shared by training
(data_loader), risk_map validation, and the /predict/location API so a
coordinate always resolves to exactly the same cell everywhere.

data_loader snaps with pandas ``(lat/g).round() * g`` (numpy round-half-to-even)
then ``round(6)``. Reproducing that here with numpy — not Python's builtin
``round`` — avoids banker-rounding drift between surfaces. The integer
``cell_index`` is the unambiguous identity: two coordinates share a cell iff
their integer indices are equal (no float tolerance needed)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

DEFAULT_GRID_SIZE = 0.1


def cell_index(lat: float, lon: float, grid_size: float) -> Tuple[int, int]:
    """Integer (lat_idx, lon_idx) grid identity. Uses numpy round to match the
    training-time snapping exactly. Equality of indices == same cell."""
    return (int(np.round(lat / grid_size)), int(np.round(lon / grid_size)))


def snap_to_grid(lat: float, lon: float, grid_size: float) -> Tuple[float, float]:
    """Canonical (lat_grid, lon_grid) for a coordinate — identical to how
    data_loader.grid_and_aggregate snaps detections (numpy round then round 6)."""
    la, lo = cell_index(lat, lon, grid_size)
    return (round(la * grid_size, 6), round(lo * grid_size, 6))
