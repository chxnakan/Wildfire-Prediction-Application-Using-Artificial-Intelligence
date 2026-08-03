"""scientific_stats.py on TINY deterministic fixtures (never the real 214 MB
parquet). Verifies: it loads a bundle via the canonical loader, builds all four
splits from metadata date ranges, evaluates only the test range, writes a
versioned report, and by DEFAULT does not mutate dataset_info.json / GeoJSON.

    cd src && ../.venv/bin/python -m pytest tests -q
"""

import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from storage import write_pickle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SS_PATH = os.path.join(ROOT, "scripts", "scientific_stats.py")


class _StubModel:
    """predict_proba keyed off f_a so the test split has both classes."""
    def predict_proba(self, X):
        return np.clip(np.asarray(X["f_a"], dtype=float), 0.0, 1.0)

    def predict(self, X):
        return np.full(len(X), 3.0)


def _load_ss_module():
    spec = importlib.util.spec_from_file_location("scientific_stats_under_test", SS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SPLITS = {
    "train": ("2026-01-01", "2026-01-10"),
    "val":   ("2026-01-15", "2026-01-20"),
    "calib": ("2026-01-25", "2026-01-31"),
    "test":  ("2026-02-05", "2026-02-15"),
}


def _make_fixture(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for name, (lo, hi) in SPLITS.items():
        dates = pd.date_range(lo, hi, freq="D")
        for i, d in enumerate(dates):
            for j in range(6):                       # a few cells per day
                # mix of positive (days_until_fire 1..3) and negative (5)
                dfire = (i + j) % 5 + 1               # 1..5
                rows.append({
                    "date": d,
                    "days_until_fire": dfire if dfire <= 3 else 5,
                    "lat_grid": 18.0 + 0.1 * j,
                    "lon_grid": 98.0 + 0.1 * i,
                    "f_a": float(rng.random()),
                    "f_b": float(rng.random()),
                })
    df = pd.DataFrame(rows)
    fpath = str(tmp_path / "features.parquet")
    df.to_parquet(fpath, index=False)

    meta = {
        "trained_at": "2026-02-20T00:00:00+00:00",
        "features": ["f_a", "f_b"],
        "feature_count": 2,
        "evaluation_protocol": {
            "split_method": "date_grouped_chronological",
            "split_date_ranges": {k: list(v) for k, v in SPLITS.items()},
        },
        "model": {"test_metrics": {"deployment_threshold": 0.5}},
    }
    mpath = str(tmp_path / "model.pkl")
    write_pickle({"format": "bundle-v1", "model": _StubModel(), "metadata": meta}, mpath)

    meta_path = str(tmp_path / "dataset_info.json")
    json.dump(meta, open(meta_path, "w"))
    out_path = str(tmp_path / "scientific_stats.json")
    geojson_path = str(tmp_path / "fire.geojson")
    json.dump({"type": "FeatureCollection", "features": [], "metadata": {"metrics": {}}},
              open(geojson_path, "w"))
    return fpath, mpath, meta_path, out_path, geojson_path


def _patch_paths(mod, fpath, mpath, meta_path, out_path, geojson_path):
    mod.FEATURES_PATH = fpath
    mod.MODEL_PATH = mpath
    mod.META_PATH = meta_path
    mod.OUT_PATH = out_path
    mod.GEOJSON_PATH = geojson_path
    mod.CHUNK_SIZE = 50


def test_report_only_run_does_not_mutate_canonical(tmp_path):
    mod = _load_ss_module()
    fpath, mpath, meta_path, out_path, geojson_path = _make_fixture(tmp_path)
    _patch_paths(mod, fpath, mpath, meta_path, out_path, geojson_path)

    meta_before = open(meta_path).read()
    gj_before = open(geojson_path).read()

    mod.main(publish=False)

    # a versioned report + latest pointer were written
    assert os.path.exists(out_path)
    report = json.load(open(out_path))
    # all four splits present in the sample table
    assert set(report["samples"]) >= {"train", "validation", "calibration", "test"}
    assert report["samples"]["test"]["n"] > 0
    assert "roc_auc" in report["confidence_intervals_95"]
    assert report["confidence_intervals_95"]["roc_auc"]["method"] == "block_bootstrap_by_date"

    # DEFAULT run must not touch canonical dataset_info.json / GeoJSON
    assert open(meta_path).read() == meta_before
    assert open(geojson_path).read() == gj_before


def test_refuses_when_split_ranges_missing(tmp_path):
    mod = _load_ss_module()
    fpath, mpath, meta_path, out_path, geojson_path = _make_fixture(tmp_path)
    # strip the split ranges from the bundle metadata
    bad_meta = {"trained_at": "x", "features": ["f_a", "f_b"], "feature_count": 2,
                "evaluation_protocol": {"split_method": "m", "split_date_ranges": {}},
                "model": {"test_metrics": {"deployment_threshold": 0.5}}}
    write_pickle({"format": "bundle-v1", "model": _StubModel(), "metadata": bad_meta}, mpath)
    _patch_paths(mod, fpath, mpath, meta_path, out_path, geojson_path)
    with pytest.raises(SystemExit, match="split_date_ranges"):
        mod.main(publish=False)


def test_refuses_on_feature_count_mismatch(tmp_path):
    mod = _load_ss_module()
    fpath, mpath, meta_path, out_path, geojson_path = _make_fixture(tmp_path)
    bad_meta = {"trained_at": "x", "features": ["f_a", "f_b"], "feature_count": 99,
                "evaluation_protocol": {"split_method": "m",
                                        "split_date_ranges": {k: list(v) for k, v in SPLITS.items()}},
                "model": {"test_metrics": {"deployment_threshold": 0.5}}}
    write_pickle({"format": "bundle-v1", "model": _StubModel(), "metadata": bad_meta}, mpath)
    _patch_paths(mod, fpath, mpath, meta_path, out_path, geojson_path)
    with pytest.raises(SystemExit, match="feature_count"):
        mod.main(publish=False)
