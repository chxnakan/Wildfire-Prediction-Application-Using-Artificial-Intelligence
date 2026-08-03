"""Post-hoc recalibration (Phase 3): chronological calib split, calibrator
comparison on identical rows, reliability-bin integrity, provenance, and no
overwrite of the promoted model. Tiny deterministic fixtures only.

    cd src && ../.venv/bin/python -m pytest tests/test_recalibrate.py -q
"""

import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from storage import read_pickle, write_pickle
import train

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RC_PATH = os.path.join(ROOT, "scripts", "recalibrate.py")


def _rc():
    spec = importlib.util.spec_from_file_location("recalibrate_under_test", RC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _StubEnsemble:
    """Frozen base with a raw score + a (replaceable) Platt calibrator slot."""
    def __init__(self):
        self.calibrator = None

    def _raw_proba(self, X):
        return np.clip(np.asarray(X["f_a"], dtype=float), 0.0, 1.0)

    def predict_proba(self, X):
        return self._raw_proba(X)


# ── pure-helper integrity ────────────────────────────────────────────────────

def test_reliability_bin_counts_sum_to_n():
    rc = _rc()
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(int)
    p = rng.random(500)
    bins = rc._reliability(y, p, n_bins=10)
    assert sum(b["count"] for b in bins) == len(y)


def test_platt_and_isotonic_fit_and_apply():
    rc = _rc()
    rng = np.random.default_rng(1)
    raw = rng.random(400)
    y = (raw + rng.normal(0, 0.1, 400) > 0.5).astype(int)
    platt = rc._fit_platt(raw, y)
    iso = rc._fit_isotonic(raw, y)
    for fn in (platt, iso):
        out = fn(raw)
        assert out.min() >= 0.0 and out.max() <= 1.0
    # isotonic is monotone non-decreasing in raw
    order = np.argsort(raw)
    iso_sorted = iso(raw)[order]
    assert np.all(np.diff(iso_sorted) >= -1e-9)


def test_metrics_have_required_calibration_fields():
    rc = _rc()
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
    p = np.linspace(0.1, 0.9, 10)
    d = pd.to_datetime(["2026-04-01"] * 5 + ["2026-04-02"] * 5).to_numpy()
    m = rc._metrics(y, p, d)
    for k in ("brier", "ece", "log_loss", "calibration_slope",
              "calibration_intercept", "n", "positives", "independent_dates",
              "reliability_bins"):
        assert k in m


# ── integration: report-only run never overwrites the promoted model ─────────

def _make_fixture(tmp_path):
    rows = []
    # calib window 2026-03-29..2026-05-07 spanning many dates + both classes
    for i, d in enumerate(pd.date_range("2026-03-29", "2026-05-07", freq="D")):
        for j in range(8):
            rows.append({"date": d, "days_until_fire": ((i + j) % 5) + 1
                         if (i + j) % 2 == 0 else 5,
                         "f_a": (i * 8 + j) / 320.0, "f_b": 0.0,
                         "lat_grid": 18.0, "lon_grid": 98.0})
    df = pd.DataFrame(rows)
    fpath = str(tmp_path / "features.parquet"); df.to_parquet(fpath, index=False)
    meta = {"trained_at": "2026-05-10T00:00:00+00:00", "features": ["f_a", "f_b"],
            "feature_count": 2,
            "evaluation_protocol": {"split_method": "date_grouped_chronological",
                                    "split_date_ranges": {
                                        "train": ["2026-01-01", "2026-02-01"],
                                        "val": ["2026-02-10", "2026-03-01"],
                                        "calib": ["2026-03-29", "2026-05-07"],
                                        "test": ["2026-05-15", "2026-06-01"]}}}
    mpath = str(tmp_path / "model.pkl")
    write_pickle({"format": "bundle-v1", "model": _StubEnsemble(), "metadata": meta}, mpath)
    return fpath, mpath


def test_report_only_run_does_not_write_candidate_or_touch_promoted(tmp_path):
    rc = _rc()
    fpath, mpath = _make_fixture(tmp_path)
    rc.FEATURES_PATH = fpath
    rc.MODEL_PATH = mpath
    rc.OUT_DIR = str(tmp_path)
    rc.CANDIDATE_PATH = str(tmp_path / "candidate.pkl")
    rc.CHUNK = 50
    before = open(mpath, "rb").read()

    rc.main(write_candidate=False)

    report = json.load(open(os.path.join(str(tmp_path), "recalibration_report.json")))
    assert report["task"] == "post_hoc_recalibration"
    assert report["provenance"]["selected_on"].startswith("chronological")
    assert set(report["calibrator_comparison"]) >= {"platt", "isotonic"}
    assert report["selected_calibrator"] in ("platt", "isotonic", "beta")
    # milestones reported as real booleans (no fabrication)
    assert set(report["milestones"]) == {
        "milestone_1_brier<=0.035_ece<=0.035", "milestone_2_brier<=0.030_ece<=0.020"}
    # promoted model untouched + no candidate written on a report-only run
    assert open(mpath, "rb").read() == before
    assert not os.path.exists(str(tmp_path / "candidate.pkl"))


def test_refuses_when_calib_range_not_covered(tmp_path):
    rc = _rc()
    fpath, mpath = _make_fixture(tmp_path)
    # move calib range to dates the parquet doesn't contain
    from storage import read_pickle
    b = read_pickle(mpath)
    b["metadata"]["evaluation_protocol"]["split_date_ranges"]["calib"] = \
        ["2030-01-01", "2030-02-01"]
    write_pickle(b, mpath)
    rc.FEATURES_PATH = fpath; rc.MODEL_PATH = mpath
    rc.OUT_DIR = str(tmp_path); rc.CANDIDATE_PATH = str(tmp_path / "c.pkl"); rc.CHUNK = 50
    with pytest.raises(SystemExit, match="does not cover the calib range|no calibration rows"):
        rc.main(write_candidate=False)


class _ProbModel:
    def __init__(self, p):
        self.p = float(p)

    def predict_proba(self, X):
        p = np.full(len(X), self.p)
        return np.column_stack([1.0 - p, p])


def test_ensemble_isotonic_calibrator_pickle_round_trip(tmp_path):
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip").fit(
        np.array([0.0, 0.25, 0.75, 1.0]), np.array([0, 0, 1, 1]))
    ens = train._EnsembleRegressor([_ProbModel(0.6)], calibrator=iso)
    before = ens.predict_proba(pd.DataFrame({"x": [1, 2, 3]}))
    path = str(tmp_path / "isotonic_bundle.pkl")
    write_pickle({"format": "bundle-v1", "model": ens, "metadata": {"features": ["x"]}}, path)
    loaded, meta = train.load_model_bundle(path)
    after = loaded.predict_proba(pd.DataFrame({"x": [1, 2, 3]}))
    assert np.allclose(before, after)
    assert np.all((after >= 0) & (after <= 1))
    assert meta["features"] == ["x"]


def test_write_candidate_supports_selected_isotonic(tmp_path, monkeypatch):
    rc = _rc()
    fpath, mpath = _make_fixture(tmp_path)
    rc.FEATURES_PATH = fpath; rc.MODEL_PATH = mpath
    rc.OUT_DIR = str(tmp_path); rc.CANDIDATE_PATH = str(tmp_path / "candidate.pkl")
    rc.CHUNK = 50

    # Make selection deterministic while still exercising a real fitted
    # IsotonicRegression in the candidate bundle.
    monkeypatch.setattr(rc, "_fit_platt", lambda raw, y: lambda r: np.full(len(r), 0.99))
    rc.main(write_candidate=True)
    report = json.load(open(os.path.join(str(tmp_path), "recalibration_report.json")))
    assert report["selected_calibrator"] == "isotonic"
    bundle = read_pickle(rc.CANDIDATE_PATH)
    assert bundle["metadata"]["recalibration"]["method"] == "isotonic_refit_full_calib"
    assert bundle["model"].calibrator.__class__.__name__ == "IsotonicRegression"
