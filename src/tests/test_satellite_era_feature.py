"""Satellite-era feature is archive coverage, not target confirmation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data_loader import densify_active_cells, grid_and_aggregate


def _hotspot(lat, lon, day, dataset):
    return {
        "lat": lat, "lon": lon, "date": pd.Timestamp(day).date(),
        "frp": 10.0, "bright_main": 320.0, "confidence": 50.0,
        "thai_hour": 14, "dataset": dataset,
    }


def test_active_satellites_is_date_level_and_survives_zero_fire_rows():
    raw = pd.DataFrame([
        _hotspot(18.0, 98.0, "2024-01-01", "VIIRS_SNPP_NRT"),
        _hotspot(18.1, 98.1, "2024-01-01", "VIIRS_SNPP_NRT"),
        _hotspot(18.0, 98.0, "2024-01-03", "VIIRS_SNPP_NRT"),
        _hotspot(18.1, 98.1, "2024-01-03", "VIIRS_NOAA20_NRT"),
    ])
    daily = grid_and_aggregate(raw)
    assert set(daily.loc[daily["date"] == pd.Timestamp("2024-01-01").date(),
                         "n_active_satellites"]) == {1}
    assert set(daily.loc[daily["date"] == pd.Timestamp("2024-01-03").date(),
                         "n_active_satellites"]) == {2}

    dense = densify_active_cells(daily)
    day2 = dense[dense["date"] == pd.Timestamp("2024-01-02").date()]
    assert (day2["fire_count"] == 0).all()
    assert (day2["n_active_satellites"] == 1).all()
    day3 = dense[dense["date"] == pd.Timestamp("2024-01-03").date()]
    assert (day3["n_active_satellites"] == 2).all()
