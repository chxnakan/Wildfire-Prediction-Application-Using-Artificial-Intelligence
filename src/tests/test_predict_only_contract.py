"""Fail-closed guarantees for prediction-only snapshot refreshes."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

import train


def test_predict_only_rejects_frame_missing_deployed_features(monkeypatch):
    monkeypatch.setattr(
        train,
        "load_model_bundle",
        lambda _path: (object(), {"features": ["fire_lag_1", "temp_max_today"]}),
    )
    frame = pd.DataFrame({"fire_lag_1": [0.0]})

    with pytest.raises(RuntimeError, match=r"1 deployed feature\(s\).+temp_max_today"):
        train._validate_predict_only_feature_contract(frame, "model.pkl")


def test_predict_only_accepts_exact_deployed_contract(monkeypatch):
    monkeypatch.setattr(
        train,
        "load_model_bundle",
        lambda _path: (object(), {"features": ["fire_lag_1", "temp_max_today"]}),
    )
    frame = pd.DataFrame({"fire_lag_1": [0.0], "temp_max_today": [31.0]})

    train._validate_predict_only_feature_contract(frame, "model.pkl")


def test_risk_map_failure_propagates_to_prediction_subprocess(monkeypatch):
    def fail():
        raise RuntimeError("geojson publication failed")

    monkeypatch.setitem(sys.modules, "risk_map", SimpleNamespace(run=fail))

    with pytest.raises(RuntimeError, match="geojson publication failed"):
        train._generate_risk_map()


def test_risk_map_failure_restores_previous_feature_artifact(
    tmp_path, monkeypatch
):
    destination = tmp_path / "full_features.parquet"
    trusted = pd.DataFrame({"date": ["2026-07-16"], "signal": [0.25]})
    candidate = pd.DataFrame({"date": ["2026-08-24"], "signal": [0.99]})
    trusted.to_parquet(destination, index=False)

    def fail():
        raise RuntimeError("geojson publication failed")

    monkeypatch.setitem(sys.modules, "risk_map", SimpleNamespace(run=fail))

    with pytest.raises(RuntimeError, match="geojson publication failed"):
        with train._preserve_artifact_on_failure(str(destination)):
            # This mirrors the predict-only order: candidate feature frame is
            # visible to risk_map, then restored if map publication fails.
            train.write_table(candidate, str(destination))
            train._generate_risk_map()

    pd.testing.assert_frame_equal(pd.read_parquet(destination), trusted)
    assert list(tmp_path.glob(".full_features.parquet.*.rollback")) == []
