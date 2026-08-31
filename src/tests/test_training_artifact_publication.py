"""Serving-generation publication invariants for full training."""

from __future__ import annotations

import inspect
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train
from storage import read_pickle, write_json, write_pickle


def _bundle(version: str) -> dict:
    return {
        "format": "bundle-v1",
        "model": f"model-{version}",
        "metadata": {"trained_at": version, "features": [f"feature_{version}"]},
    }


def test_rejected_candidate_feature_stage_never_touches_incumbent(tmp_path):
    feature_path = tmp_path / "full_features.parquet"
    trusted = pd.DataFrame({"feature_v1": [1.0]})
    candidate = pd.DataFrame({"feature_v2": [2.0]})
    trusted.to_parquet(feature_path, index=False)

    with train._candidate_feature_artifact(str(feature_path)) as staged:
        train.write_table(candidate, staged)
        # A rejected gate simply exits without calling publication.
        pd.testing.assert_frame_equal(pd.read_parquet(feature_path), trusted)
        assert os.path.isfile(staged)

    pd.testing.assert_frame_equal(pd.read_parquet(feature_path), trusted)
    assert not os.path.exists(staged)


def test_successful_publication_installs_one_complete_generation(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    feature_path = tmp_path / "full_features.parquet"
    model_path = model_dir / "model.pkl"
    metadata_path = tmp_path / "dataset_info.json"
    trusted = pd.DataFrame({"feature_v1": [1.0]})
    candidate = pd.DataFrame({"feature_v2": [2.0]})
    trusted.to_parquet(feature_path, index=False)
    write_pickle(_bundle("v1"), str(model_path))
    write_json({"trained_at": "v1"}, str(metadata_path))

    observed = {}

    def verify_risk_map_reads_candidate_generation():
        observed["features"] = pd.read_parquet(feature_path)
        observed["model"] = read_pickle(str(model_path))
        observed["metadata"] = json.loads(metadata_path.read_text())

    monkeypatch.setattr(train, "_generate_risk_map", verify_risk_map_reads_candidate_generation)

    with train._candidate_feature_artifact(str(feature_path)) as staged:
        train.write_table(candidate, staged)
        train._publish_promoted_candidate(
            _bundle("v2"),
            {"trained_at": "v2"},
            candidate_feature_path=staged,
            feature_path=str(feature_path),
            model_dir=str(model_dir),
            promoted_path=str(model_path),
            metadata_path=str(metadata_path),
            refresh_risk_map=True,
        )

    pd.testing.assert_frame_equal(observed["features"], candidate)
    assert observed["model"]["metadata"]["trained_at"] == "v2"
    assert observed["metadata"]["trained_at"] == "v2"
    pd.testing.assert_frame_equal(pd.read_parquet(feature_path), candidate)
    assert read_pickle(str(model_path))["metadata"]["trained_at"] == "v2"
    assert read_pickle(str(model_dir / "previous.pkl"))["metadata"]["trained_at"] == "v1"


def test_risk_map_failure_restores_entire_incumbent_generation(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    feature_path = tmp_path / "full_features.parquet"
    model_path = model_dir / "model.pkl"
    previous_path = model_dir / "previous.pkl"
    metadata_path = tmp_path / "dataset_info.json"
    trusted = pd.DataFrame({"feature_v1": [1.0]})
    candidate = pd.DataFrame({"feature_v2": [2.0]})
    trusted.to_parquet(feature_path, index=False)
    write_pickle(_bundle("v1"), str(model_path))
    write_pickle(_bundle("v0"), str(previous_path))
    write_json({"trained_at": "v1"}, str(metadata_path))

    def fail_after_observing_candidate():
        assert read_pickle(str(model_path))["metadata"]["trained_at"] == "v2"
        pd.testing.assert_frame_equal(pd.read_parquet(feature_path), candidate)
        raise RuntimeError("geojson publication failed")

    monkeypatch.setattr(train, "_generate_risk_map", fail_after_observing_candidate)

    with pytest.raises(RuntimeError, match="geojson publication failed"):
        with train._candidate_feature_artifact(str(feature_path)) as staged:
            train.write_table(candidate, staged)
            train._publish_promoted_candidate(
                _bundle("v2"),
                {"trained_at": "v2"},
                candidate_feature_path=staged,
                feature_path=str(feature_path),
                model_dir=str(model_dir),
                promoted_path=str(model_path),
                metadata_path=str(metadata_path),
                refresh_risk_map=True,
            )

    pd.testing.assert_frame_equal(pd.read_parquet(feature_path), trusted)
    assert read_pickle(str(model_path))["metadata"]["trained_at"] == "v1"
    assert read_pickle(str(previous_path))["metadata"]["trained_at"] == "v0"
    assert json.loads(metadata_path.read_text())["trained_at"] == "v1"
    assert not list(tmp_path.rglob("*.rollback"))
    assert not (model_dir / ".candidate.pkl.tmp").exists()


def test_model_candidate_temp_is_removed_when_serialization_fails(
    tmp_path, monkeypatch
):
    model_path = tmp_path / "model.pkl"
    write_pickle(_bundle("v1"), str(model_path))

    def fail_write(_bundle_obj, path, **_kwargs):
        with open(path, "wb") as stream:
            stream.write(b"partial")
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(train, "write_pickle", fail_write)
    with pytest.raises(RuntimeError, match="serialization failed"):
        train._promote_bundle(_bundle("v2"), str(tmp_path), str(model_path))

    assert read_pickle(str(model_path))["metadata"]["trained_at"] == "v1"
    assert not (tmp_path / ".candidate.pkl.tmp").exists()


def test_memory_safe_ensemble_default_is_five(monkeypatch):
    assert inspect.signature(train.main).parameters["n_ensemble"].default == 5
    monkeypatch.setattr(sys, "argv", ["train.py"])
    assert train._cli().n_ensemble == 5
