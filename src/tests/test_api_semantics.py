"""Prediction-surface semantics: every API surface must report the SAME
calibrated probability + risk tier, none may present an exact fire date, and
the degraded state must return a stable 503.

Hermetic: a tiny stub model is injected into app.state, so these tests never
load the real (multi-GB) trained artifact. Run with the suite:
    cd src && ../.venv/bin/python -m pytest tests -q
"""

import os
import sys
from datetime import date
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


FEATURES = ["f_a", "f_b"]
# Absolute calibrated tier cutoffs (probability scale) — High ≥ .60, Watch ≥ .40.
TIER_THRESHOLDS = {"HIGH": 0.60, "MEDIUM": 0.40, "LOW": 0.0, "selected_on": "calib"}


class _StubModel:
    """Deterministic predict_proba keyed off feature f_a so tiers are stable."""
    def predict_proba(self, X):
        return np.clip(np.asarray(X["f_a"], dtype=float), 0.0, 1.0)

    def predict(self, X):  # pseudo-days compat path (should never drive output)
        return np.full(len(X), 3.0)


def _stub_state():
    latest = date(2026, 7, 14)
    # three cells with probabilities 0.9 (High), 0.5 (Watch), 0.1 (Background)
    df = pd.DataFrame({
        "date": [latest] * 3,
        "lat_grid": [18.5, 18.6, 18.7],
        "lon_grid": [98.5, 98.6, 98.7],
        "fire_count": [2, 0, 1],
        "f_a": [0.9, 0.5, 0.1],
        "f_b": [0.0, 0.0, 0.0],
    })
    hist = pd.DataFrame({
        "lat_grid": [18.5, 18.6, 18.7],
        "lon_grid": [98.5, 98.6, 98.7],
        "historical_fire_count_30d": [5, 0, 2],
    })
    return {
        "model": _StubModel(),
        "df": df,
        "historical_counts": hist,
        "meta": {"imminent_days": 3, "probability_tier_thresholds": TIER_THRESHOLDS},
        "features": FEATURES,
        "thresholds": {"CRITICAL": 1.01, "HIGH": 0.60, "MEDIUM": 0.40, "LOW": 0.0},
        "model_version": "2026-07-14T00:00:00+00:00",
        "model_loaded_at": "2026-07-14T00:00:00+00:00",
    }


@pytest.fixture
def client():
    # Prevent the lifespan from loading the real artifact — start degraded, fast.
    import api
    with mock.patch.object(api, "_build_model_state", return_value=None):
        with TestClient(api.app) as c:
            api._apply_model_state(c.app, _stub_state())
            yield c


@pytest.fixture
def degraded_client():
    import api
    with mock.patch.object(api, "_build_model_state", return_value=None):
        with TestClient(api.app) as c:
            api._apply_model_state(c.app, dict(api._DEGRADED_STATE))
            yield c


# ── no exact-date claims anywhere ────────────────────────────────────────────

_FORBIDDEN_DATE_KEYS = {"predicted_fire_date", "days_until_fire", "day_offset",
                        "fire_date"}


def _assert_no_exact_date(obj, where=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert k not in _FORBIDDEN_DATE_KEYS, f"exact-date key '{k}' at {where}"
            _assert_no_exact_date(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5]):
            _assert_no_exact_date(v, f"{where}[{i}]")


def test_today_reports_probability_and_tier_no_date(client):
    r = client.get("/predictions/today")
    assert r.status_code == 200
    body = r.json()
    _assert_no_exact_date(body, "today")
    assert body["horizon_days"] == 3
    assert "risk_definition" in body and "fire date" in body["risk_definition"]
    p0 = body["predictions"][0]
    assert set(["lat_grid", "lon_grid", "probability", "risk_tier"]) <= set(p0)
    assert body["tier_summary"] == {"High": 1, "Watch": 1, "Background": 1}


def test_tiers_endpoint_is_tier_distribution(client):
    r = client.get("/predictions/tiers")
    assert r.status_code == 200
    body = r.json()
    _assert_no_exact_date(body, "tiers")
    assert body["tier_summary"] == {"High": 1, "Watch": 1, "Background": 1}


def test_metrics_exposes_operational_probability_cutoffs(client):
    body = client.get("/metrics").json()
    assert body["probability_tier_thresholds"] == TIER_THRESHOLDS
    assert body["urgency_thresholds_deprecated"] is True
    assert "not used for operational risk tiers" in body["urgency_thresholds_note"]


def test_timeline_and_day_are_gone(client):
    for path in ("/predictions/timeline", "/predictions/day/2"):
        r = client.get(path)
        assert r.status_code == 410, path
        assert r.json()["detail"]["replacement"] == "/predictions/tiers"


def test_all_surfaces_agree_on_probability_and_tier(client):
    today = client.get("/predictions/today").json()
    cell = today["predictions"][0]
    loc = client.get(
        "/predict/location",
        params={"lat": cell["lat_grid"], "lon": cell["lon_grid"]},
    ).json()
    assert loc["scored"] is True
    assert abs(loc["probability"] - cell["probability"]) < 1e-9
    assert loc["risk_tier"] == cell["risk_tier"] == "High"
    _assert_no_exact_date(loc, "location")


def test_location_outside_universe_is_unscored_not_zero(client):
    r = client.get("/predict/location", params={"lat": 4.2, "lon": 96.3})
    assert r.status_code == 200
    body = r.json()
    assert body["scored"] is False
    assert body["probability"] is None and body["risk_tier"] is None
    assert "coverage_note" in body


def test_degraded_returns_stable_503(degraded_client):
    for path in ("/predictions/today", "/predictions/tiers"):
        r = degraded_client.get(path)
        assert r.status_code == 503, path
        assert r.json()["detail"]["error"] == "model_unavailable"
    r = degraded_client.get("/predict/location", params={"lat": 18.5, "lon": 98.5})
    assert r.status_code == 503


# ── snapshot caching (#12): repeated reads must NOT re-run inference ──────────

class _CountingModel(_StubModel):
    def __init__(self):
        self.calls = 0

    def predict_proba(self, X):
        self.calls += 1
        return super().predict_proba(X)


def test_predictions_snapshot_is_cached(client):
    import api
    counter = _CountingModel()
    st = _stub_state()
    st["model"] = counter
    st["scored_snapshot"] = None            # force lazy compute on first read
    api._apply_model_state(client.app, st)
    for _ in range(5):
        assert client.get("/predictions/today").status_code == 200
        assert client.get("/predictions/tiers").status_code == 200
    # 10 reads, but inference ran at most once (cached snapshot).
    assert counter.calls <= 1


# ── /predict/location canonical grid (#13) ───────────────────────────────────

def test_location_snaps_offset_coord_to_same_cell(client):
    # A coordinate a little off the cell centre snaps to the SAME 0.1° cell.
    exact = client.get("/predict/location", params={"lat": 18.5, "lon": 98.5}).json()
    off = client.get("/predict/location", params={"lat": 18.517, "lon": 98.483}).json()
    assert exact["scored"] and off["scored"]
    assert off["location"] == exact["location"]
    assert abs(off["probability"] - exact["probability"]) < 1e-9


def test_location_unpopulated_cell_is_unscored(client):
    # In-bounds but no scored cell there → unscored (not an error, not zero-risk).
    r = client.get("/predict/location", params={"lat": 19.9, "lon": 99.9})
    assert r.status_code == 200 and r.json()["scored"] is False


def test_location_rejects_client_grid_size_param(client):
    # grid_size is no longer a client parameter — supplying it is ignored, and
    # the endpoint still resolves from model metadata (query still succeeds).
    r = client.get("/predict/location",
                   params={"lat": 18.5, "lon": 98.5, "grid_size": 999})
    assert r.status_code == 200 and r.json()["scored"] is True
