"""Rolling-origin backtest invariants (Task 4): temporal ordering, purge gap,
no-leakage window construction, honest skip handling, and provenance.

Pure functions only — no model training. The full backtest is expensive and is
run separately (documented command), never in unit tests.

    cd src && ../.venv/bin/python -m pytest tests/test_rolling_backtest.py -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

import rolling_backtest as rb


# ── fold window temporal ordering + purge gap (no leakage) ───────────────────

def test_fold_windows_are_chronologically_ordered_with_purge():
    w = rb.fold_windows(2026, (1, 4), calib_days=30, max_train_years=3.0)
    assert w["fit_start"] < w["fit_end"] < w["calib_start"] < w["calib_end"] < w["test_start"] < w["test_end"]
    # purge gap between fit→calib and calib→test is strictly > PURGE_DAYS
    fit_calib_gap = (w["calib_start"] - w["fit_end"]).days
    calib_test_gap = (w["test_start"] - w["calib_end"]).days
    assert fit_calib_gap > rb.PURGE_DAYS
    assert calib_test_gap > rb.PURGE_DAYS


def test_test_window_matches_requested_season_months():
    w = rb.fold_windows(2025, (1, 4), calib_days=30, max_train_years=2.0)
    assert w["test_start"] == pd.Timestamp("2025-01-01")
    assert w["test_end"] == pd.Timestamp("2025-04-30")


def test_train_window_respects_max_train_years():
    w2 = rb.fold_windows(2026, (1, 4), 30, max_train_years=2.0)
    w4 = rb.fold_windows(2026, (1, 4), 30, max_train_years=4.0)
    span2 = (w2["fit_end"] - w2["fit_start"]).days
    span4 = (w4["fit_end"] - w4["fit_start"]).days
    assert span4 > span2
    assert abs(span2 - 2 * 365.25) < 2 and abs(span4 - 4 * 365.25) < 2


def test_weather_coverage_gate_is_scoped_to_the_fold_window():
    recent = pd.DataFrame({c: [1.0, 2.0, np.nan, 4.0]
                           for c in rb.WEATHER_COLUMNS})
    kept = rb._drop_sparse_weather(recent, min_coverage=0.20)
    assert set(rb.WEATHER_COLUMNS) <= set(kept.columns)

    sparse = pd.DataFrame({c: [np.nan, np.nan, np.nan, 4.0]
                           for c in rb.WEATHER_COLUMNS})
    dropped = rb._drop_sparse_weather(sparse, min_coverage=0.30)
    assert not set(rb.WEATHER_COLUMNS) & set(dropped.columns)


# ── honest skip handling (never zero-filled) ─────────────────────────────────

def test_aggregate_separates_valid_and_skipped_folds():
    folds = [
        {"year": 2020, "status": "ok", "roc_auc": 0.80,
         "average_precision": 0.20,
         "deployment": {"recall": 0.7, "precision": 0.4},
         "watch_list": {"recall": 0.9, "precision": 0.3}},
        {"year": 2021, "status": "ok", "roc_auc": 0.84,
         "average_precision": 0.40,
         "deployment": {"recall": 0.6, "precision": 0.5},
         "watch_list": {"recall": 0.95, "precision": 0.35}},
        {"year": 2022, "status": "skipped", "reason": "single_class_split"},
    ]
    agg = rb._aggregate(folds)
    assert agg["n_folds_total"] == 3
    assert agg["valid_folds"] == 2
    assert agg["skipped_folds"] == 1
    assert agg["skipped_reasons"] == {"single_class_split": 1}
    # mean AUC is over the 2 valid folds only — the skip is NOT counted as 0
    assert abs(agg["auc_mean"] - 0.82) < 1e-9
    assert agg["auc_min"] == 0.80
    assert agg["ap_mean"] == 0.30
    assert agg["ap_median"] == 0.30
    assert agg["ap_min"] == 0.20


def test_aggregate_all_skipped_has_no_metric_keys():
    agg = rb._aggregate([{"year": 2022, "status": "skipped", "reason": "no_rows_in_season"}])
    assert agg["valid_folds"] == 0 and "auc_mean" not in agg


# ── provenance binds output to code + data + config ──────────────────────────

def test_provenance_has_required_fields(tmp_path):
    prov = rb._provenance(
        {"raw_dir": str(tmp_path)}, ["b", "a", "c"], n_estimators=600,
        max_train_years=3.0, calib_days=30, season_months=(1, 4))
    for k in ("generated_at", "code_version", "evaluation", "feature_count",
              "feature_contract_sha256", "raw_data_fingerprint",
              "model_configuration"):
        assert k in prov
    assert prov["feature_count"] == 3
    assert "out_of_sample" in prov["evaluation"]
    assert prov["model_configuration"]["purge_days"] == rb.PURGE_DAYS
    # feature contract hash is order-independent
    prov2 = rb._provenance({"raw_dir": str(tmp_path)}, ["c", "a", "b"], 600, 3.0, 30, (1, 4))
    assert prov["feature_contract_sha256"] == prov2["feature_contract_sha256"]
