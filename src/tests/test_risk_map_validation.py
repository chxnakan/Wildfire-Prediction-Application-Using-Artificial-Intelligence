"""Strict model-target validation vs operational spatial diagnostic.

The model target is: >=1 FIRMS hotspot in the SAME 0.1° cell during
base_date+1 .. base_date+horizon inclusive. Strict evaluation must count fires
on +1/+2/+3 and must NOT count +4, and must not use a spatial radius. A
neighbouring cell may only register in the separately-named spatial diagnostic.

    cd src && ../.venv/bin/python -m pytest tests -q
"""

import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

import risk_map


BASE = date(2026, 3, 1)
HORIZON = 3
GRID = 0.1
CELL_LAT, CELL_LON = 18.5, 98.5


def _prediction(lat=CELL_LAT, lon=CELL_LON, tier="HIGH"):
    return {
        "properties": {
            "source": "predicted", "base_date": BASE.isoformat(),
            "lat": lat, "lon": lon, "urgency_level": tier,
        }
    }


def _daily(fire_day_offsets, cell=(CELL_LAT, CELL_LON), observed_through=None):
    """Densified daily frame: a fire in `cell` on each given +offset day, plus
    an explicit last-observed row so the horizon-maturity check is exercised."""
    rows = []
    for off in fire_day_offsets:
        rows.append({"date": BASE + timedelta(days=off), "lat_grid": cell[0],
                     "lon_grid": cell[1], "fire_count": 1})
    last = observed_through if observed_through else BASE + timedelta(days=HORIZON)
    rows.append({"date": last, "lat_grid": 0.0, "lon_grid": 0.0, "fire_count": 0})
    return pd.DataFrame(rows)


def _run(features, daily):
    return risk_map._revalidate_predictions(
        features, daily, horizon_days=HORIZON, grid_size=GRID)


@pytest.mark.parametrize("offset", [1, 2, 3])
def test_strict_hit_on_plus_1_2_3(offset):
    feats = [_prediction()]
    res = _run(feats, _daily([offset]))
    assert res["evaluation"] == "strict_model_target"
    assert res["hits"] == 1 and res["misses"] == 0
    assert feats[0]["properties"]["validation_status"] == "hit"


def test_strict_miss_on_plus_4():
    # Fire on +4 is OUTSIDE the [+1,+3] window → strict miss. Need obs through
    # +4 so the prediction is mature, not 'future'.
    feats = [_prediction()]
    res = _run(feats, _daily([4], observed_through=BASE + timedelta(days=4)))
    assert res["hits"] == 0 and res["misses"] == 1
    assert feats[0]["properties"]["validation_status"] == "miss"


def test_future_until_full_horizon_observed():
    # Observed only through +2 (< +3) → not yet mature.
    feats = [_prediction()]
    res = _run(feats, _daily([], observed_through=BASE + timedelta(days=2)))
    assert res["future"] == 1 and res["hits"] == 0 and res["misses"] == 0
    assert feats[0]["properties"]["validation_status"] == "future"


def test_neighbor_cell_is_strict_miss_but_spatial_hit():
    # Fire one cell north (~11 km) — NOT the same cell → strict miss, but the
    # spatial proximity diagnostic (25 km) registers it.
    feats = [_prediction()]
    neighbor = (round(CELL_LAT + GRID, 4), CELL_LON)
    res = _run(feats, _daily([2], cell=neighbor))
    assert res["hits"] == 0 and res["misses"] == 1
    assert feats[0]["properties"]["validation_status"] == "miss"
    spatial = res["operational_spatial_tolerance"]
    assert spatial["hits"] == 1
    assert feats[0]["properties"]["spatial_diagnostic"] == "hit"
    assert "not model precision/recall" in spatial["note"]


def test_spatial_diagnostic_is_separate_key_not_headline():
    res = _run([_prediction()], _daily([2]))
    # strict numbers are top-level; spatial lives only under its own key
    assert "operational_spatial_tolerance" in res
    assert res["operational_spatial_tolerance"]["match_radius_km"] == 25.0
    assert res["match_rule"].startswith("same 0.1° cell")


def test_same_cell_float_snap_still_hits():
    # Prediction coord with float noise still snaps to the same grid cell.
    feats = [_prediction(lat=CELL_LAT + 0.0003, lon=CELL_LON - 0.0004)]
    res = _run(feats, _daily([1]))
    assert res["hits"] == 1
