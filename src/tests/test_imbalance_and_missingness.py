"""Audited-bottleneck fixes: compounded class weighting (item 2), tuning
objective (item 3), weather missingness (item 6), multi-sat estimand toggle
(item 9). Pure/unit level — no training runs.

    cd src && ../.venv/bin/python -m pytest tests/test_imbalance_and_missingness.py -q
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

import train


# ── item 2: class weighting applied exactly once ─────────────────────────────

def _y_dates(n=1000, prevalence=0.2, seed=0):
    rng = np.random.default_rng(seed)
    y = pd.Series((rng.random(n) < prevalence).astype(int))
    # daily dates with duplicates: rows sharing a date share the recency term
    dates = pd.Series(pd.date_range("2026-01-01", periods=max(1, n // 10),
                                    freq="D").repeat(10)[:n])
    return y, dates


def test_class_weighting_off_removes_class_dependence():
    y, dates = _y_dates()
    sw = train._compute_sample_weights(y, dates, class_weighting=False)
    # same-date rows must get the same weight regardless of class
    df = pd.DataFrame({"y": y, "d": dates.dt.date, "w": sw})
    for _, g in df.groupby("d"):
        if g["y"].nunique() == 2:
            w_pos = g[g["y"] == 1]["w"].mean()
            w_neg = g[g["y"] == 0]["w"].mean()
            assert abs(w_pos - w_neg) < 1e-9
            break


def test_class_weighting_on_upweights_minority():
    y, dates = _y_dates(prevalence=0.2)
    sw = train._compute_sample_weights(y, dates, class_weighting=True)
    w_pos = sw[y.to_numpy() == 1].mean()
    w_neg = sw[y.to_numpy() == 0].mean()
    assert w_pos > w_neg * 2      # minority (20%) clearly up-weighted


def test_legacy_stack_quantified_effective_weight():
    """Document the legacy (mode D) compounding: at 3.24% natural prevalence,
    undersample(4:1) × class-weights(≈4) × scale_pos_weight(≈4) up-weights
    positives ≈ 120× vs natural — the audited pathology."""
    natural_neg_over_pos = (1 - 0.0324) / 0.0324          # ≈ 29.9
    undersample_boost = natural_neg_over_pos / 4.0        # ≈ 7.5
    effective = undersample_boost * 4.0 * 4.0             # ≈ 119.5
    assert 100 < effective < 140


def test_prior_correction_formula_recovers_natural_probability():
    # If negatives are kept at rate beta, corrected p = beta*p/(beta*p + 1 - p).
    # Sanity: beta=1 → identity; smaller beta pulls p down toward natural scale.
    p = np.array([0.5, 0.8, 0.2])
    beta = 1.0
    corrected = beta * p / (beta * p + (1 - p))
    assert np.allclose(corrected, p)
    beta = 0.1
    corrected = beta * p / (beta * p + (1 - p))
    assert (corrected < p).all()
    assert abs(corrected[0] - (0.1 * 0.5) / (0.1 * 0.5 + 0.5)) < 1e-12


def test_multisat_none_removes_boost():
    y, dates = _y_dates()
    sat = np.zeros(len(y), dtype=int); sat[:100] = 3
    sw_with = train._compute_sample_weights(y, dates, multi_sat=sat, class_weighting=False)
    sw_without = train._compute_sample_weights(y, dates, multi_sat=None, class_weighting=False)
    # weights are mean-normalized, so compare RATIOS: boosted rows carry a
    # strictly larger with/without ratio than un-boosted rows, and un-boosted
    # ratios are all identical (only the global normalizer moved them).
    ratio = sw_with / sw_without
    assert (ratio[:100] > ratio[100:].max() + 1e-12).all()
    assert np.allclose(ratio[100:], ratio[100:][0])


# ── item 6: weather missingness indicators (opt-in, causal) ──────────────────

def _weather_frame():
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "lat_grid": 18.0, "lon_grid": 98.0, "date": dates,
        "fire_count": 0, "frp_sum": 0.0, "frp_max": 0.0,
        "bright_mean": 0.0, "bright_max": 0.0, "confidence_mean": 0.0,
        "temp_max": [25.0] * 15 + [np.nan] * 15,   # second half missing
    })
    return df


def test_missing_indicator_emitted_only_when_enabled(monkeypatch):
    import features
    # default off → no indicator column in contract
    assert not any(c.endswith("_missing") for c in features.FEATURES_WEATHER)

    monkeypatch.setenv("WEATHER_MISSING_INDICATORS", "true")
    importlib.reload(features)
    try:
        assert any(c.endswith("_missing") for c in features.FEATURES_WEATHER)
        assert "temp_max_missing" in features.FEATURES_WEATHER
    finally:
        monkeypatch.delenv("WEATHER_MISSING_INDICATORS", raising=False)
        importlib.reload(features)      # restore default contract for other tests
    assert len(features.FEATURES_WEATHER) == 37


def test_missing_indicator_flags_nan_before_zero_fill(monkeypatch):
    import features
    monkeypatch.setenv("WEATHER_MISSING_INDICATORS", "true")
    importlib.reload(features)
    try:
        df = features.add_temporal_features(_weather_frame())
        assert "temp_max_missing" in df.columns
        assert df["temp_max_missing"].iloc[:15].sum() == 0      # observed
        assert df["temp_max_missing"].iloc[15:].sum() == 15     # missing flagged
        # and the filled value is 0 (unchanged legacy behaviour)
        assert (df["temp_max_today"].iloc[15:] == 0).all()
    finally:
        monkeypatch.delenv("WEATHER_MISSING_INDICATORS", raising=False)
        importlib.reload(features)


def test_drought_features_are_past_only_and_missing_is_not_dry(monkeypatch):
    import features
    monkeypatch.delenv("WEATHER_MISSING_INDICATORS", raising=False)
    importlib.reload(features)
    df = _weather_frame()
    # A measurable sequence: wet day, two dry days, missing day, dry day.
    precip = np.full(len(df), 2.0)
    precip[:5] = [4.0, 0.0, 0.5, np.nan, 0.0]
    df["precip_sum"] = precip
    df["et0"] = 3.0
    out = features.add_temporal_features(df)

    # Today's rain cannot affect today's streak: day 2 sees day 1 (wet), day
    # 3 sees one prior dry day, and the missing day breaks the next-day streak.
    assert out["consecutive_dry_days"].iloc[:5].tolist() == [0, 0, 1, 2, 0]
    assert out["precip_sum_30d"].iloc[1] == 4.0
    assert out["precip_sum_30d"].iloc[2] == 4.0
    assert out["precip_deficit_30d"].iloc[2] == 2.0  # (3+3) - (4+0)
    for c in ("consecutive_dry_days", "precip_sum_30d", "precip_deficit_90d"):
        assert c in features.resolve_features(out)


# ── item 3: tuning objective configurable, AP by default ─────────────────────

def test_tune_scoring_default_and_override(monkeypatch):
    import numpy as np
    from eval_protocol import purged_date_folds
    dates = pd.Series(pd.date_range("2026-01-01", periods=120, freq="D").repeat(4))
    folds = purged_date_folds(dates, n_splits=2, purge_days=7)
    est = train._build_lgbm()
    monkeypatch.delenv("TUNE_SCORING", raising=False)
    s, _ = train._build_search(est, n_iter=1, cv=folds, random_state=0)
    assert s.scoring == "average_precision"
    monkeypatch.setenv("TUNE_SCORING", "roc_auc")
    s, _ = train._build_search(est, n_iter=1, cv=folds, random_state=0)
    assert s.scoring == "roc_auc"
    monkeypatch.setenv("TUNE_SCORING", "junk")
    s, _ = train._build_search(est, n_iter=1, cv=folds, random_state=0)
    assert s.scoring == "average_precision"
