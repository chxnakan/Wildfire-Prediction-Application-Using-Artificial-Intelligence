"""Canonical model-bundle loader (train.load_model_bundle).

Covers: valid bundle-v1, legacy bare model, corrupt/missing model field,
embedded metadata, __main__ pickle registration, and clear errors. No real
model artifact is touched — tiny picklable stubs only.

    cd src && ../.venv/bin/python -m pytest tests -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import train
from storage import write_pickle


class _StubModel:
    """Tiny picklable stand-in with the model contract."""
    def __init__(self, scale=1.0):
        self.scale = scale

    def predict_proba(self, X):
        return np.full(len(X), 0.5 * self.scale)

    def predict(self, X):
        return np.full(len(X), 3.0)


def _bundle(model, meta):
    return {"format": "bundle-v1", "model": model, "metadata": meta}


def test_load_valid_bundle(tmp_path):
    p = str(tmp_path / "m.pkl")
    write_pickle(_bundle(_StubModel(2.0), {"features": ["a"], "feature_count": 1}), p)
    model, meta = train.load_model_bundle(p)
    assert hasattr(model, "predict_proba")
    assert meta == {"features": ["a"], "feature_count": 1}


def test_embedded_metadata_returned(tmp_path):
    p = str(tmp_path / "m.pkl")
    write_pickle(_bundle(_StubModel(), {"trained_at": "2026-01-01", "features": ["x", "y"]}), p)
    _, meta = train.load_model_bundle(p)
    assert meta["trained_at"] == "2026-01-01" and meta["features"] == ["x", "y"]


def test_load_legacy_bare_model(tmp_path):
    p = str(tmp_path / "legacy.pkl")
    write_pickle(_StubModel(), p)          # bare model, no bundle wrapper
    model, meta = train.load_model_bundle(p)
    assert hasattr(model, "predict_proba") and meta is None


def test_corrupt_bundle_missing_model_field(tmp_path):
    p = str(tmp_path / "bad.pkl")
    write_pickle({"format": "bundle-v1", "model": None, "metadata": {}}, p)
    with pytest.raises(ValueError, match="missing 'model'"):
        train.load_model_bundle(p)


def test_unrecognized_dict_artifact_rejected(tmp_path):
    p = str(tmp_path / "weird.pkl")
    write_pickle({"format": "something-else", "model": _StubModel()}, p)
    with pytest.raises(ValueError, match="Unrecognized dict artifact"):
        train.load_model_bundle(p)


def test_non_model_object_rejected(tmp_path):
    p = str(tmp_path / "notamodel.pkl")
    write_pickle(12345, p)                  # int, no predict/predict_proba
    with pytest.raises(ValueError, match="not a usable model"):
        train.load_model_bundle(p)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        train.load_model_bundle("/nonexistent/model.pkl")


def test_register_puts_classes_in_main():
    main_mod = sys.modules["__main__"]
    for n in ("_EnsembleRegressor", "_prob_to_days_for_compat"):
        if hasattr(main_mod, n):
            delattr(main_mod, n)
    train.register_model_classes_in_main()
    assert hasattr(main_mod, "_EnsembleRegressor")
    assert hasattr(main_mod, "_prob_to_days_for_compat")
    # idempotent — second call must not raise or duplicate
    train.register_model_classes_in_main()


def test_main_pickle_compatibility(tmp_path):
    """A real _EnsembleRegressor pickled as if under __main__ still loads —
    the exact scenario that broke joblib.load in the scripts."""
    import pickle
    ens = train._EnsembleRegressor([_StubModel(), _StubModel(3.0)])
    # Force the pickle to reference the class as __main__._EnsembleRegressor,
    # reproducing how `python train.py` pickles it.
    train.register_model_classes_in_main()
    ens.__class__.__module__ = "__main__"
    p = str(tmp_path / "mainpickled.pkl")
    with open(p, "wb") as fh:
        pickle.dump(_bundle(ens, {"features": ["a"]}), fh)
    ens.__class__.__module__ = "train"      # restore so other tests are clean
    model, meta = train.load_model_bundle(p)
    assert hasattr(model, "predict_proba") and meta["features"] == ["a"]
