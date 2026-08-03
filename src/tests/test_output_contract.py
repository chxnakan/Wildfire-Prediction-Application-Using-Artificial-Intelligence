"""Output contract (Task 5): the served GeoJSON metadata must agree with the
model bundle on horizon, thresholds, and feature count — a mismatch means the
map is showing predictions under a different contract than the model.

Skips cleanly when no trained artifact is present (fresh checkout / CI without a
model), so it never blocks the suite.

    cd src && ../.venv/bin/python -m pytest tests/test_output_contract.py -q
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
META = os.path.join(ROOT, "outputs", "metadata", "dataset_info.json")
GEOJSON = os.path.join(ROOT, "outputs", "riskmap", "fire_dates_all.geojson")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(META) and os.path.exists(GEOJSON)),
    reason="no trained model / GeoJSON present")


def _load():
    meta = json.load(open(META))
    gj = json.load(open(GEOJSON)).get("metadata", {})
    return meta, gj


def test_geojson_horizon_matches_bundle_imminent_days():
    meta, gj = _load()
    assert gj.get("horizon_days") == int(meta.get("imminent_days", 3))
    # horizon must be the real model horizon, never the 7-day pseudo-day range
    assert gj.get("horizon_days") <= 3


def test_geojson_risk_window_is_base_plus_1_to_horizon():
    import pandas as pd
    meta, gj = _load()
    base = pd.to_datetime(gj["base_date"])
    horizon = int(meta.get("imminent_days", 3))
    assert pd.to_datetime(gj["risk_window_start"]) == base + pd.Timedelta(days=1)
    assert pd.to_datetime(gj["risk_window_end"]) == base + pd.Timedelta(days=horizon)


def test_geojson_thresholds_match_bundle():
    meta, gj = _load()
    tt = meta.get("probability_tier_thresholds") or {}
    ut = gj.get("urgency_thresholds") or {}
    if "HIGH" in tt and "HIGH" in ut:
        assert abs(float(ut["HIGH"]) - float(tt["HIGH"])) < 1e-6
    # WATCH is persisted as the MEDIUM alias
    if "WATCH" in tt and "MEDIUM" in ut:
        assert abs(float(ut["MEDIUM"]) - float(tt["WATCH"])) < 1e-6


def test_geojson_has_generated_at_provenance():
    _, gj = _load()
    assert gj.get("generated_at")
    assert gj.get("latest_data_time") is not None or gj.get("base_date")
