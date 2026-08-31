"""Top-k operational ranking (Phase 1): deterministic, correct sizes, HIGH
override, metadata contract, and NO mutation of the calibrated probability.

    cd src && ../.venv/bin/python -m pytest tests/test_ranking.py -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import inference as I


def _grid(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "lat_grid": rng.integers(0, 50, n) / 10,
        "lon_grid": rng.integers(0, 50, n) / 10,
        "probability": rng.random(n),
    })


def test_topk_sizes_are_correct():
    df = _grid(100)
    r = I.assign_operational_ranking(df, high_threshold=None,
                                     watch_top_pct=10, extended_watch_top_pct=20)
    assert (r["ranking_tier"] == "RELATIVE_TOP_10").sum() == 10
    assert (r["ranking_tier"] == "RELATIVE_TOP_20").sum() == 10
    assert set(r["ranking_tier"]) <= {
        "RELATIVE_TOP_10", "RELATIVE_TOP_20", "UNRANKED"
    }


def test_ranking_is_deterministic_under_ties_and_shuffling():
    df = pd.DataFrame({
        "lat_grid": [5, 4, 3, 2, 1], "lon_grid": [1, 1, 1, 1, 1],
        "probability": [0.5, 0.5, 0.5, 0.5, 0.5],   # all tied
    })
    r1 = I.assign_operational_ranking(df, watch_top_pct=20, extended_watch_top_pct=40)
    r2 = I.assign_operational_ranking(df.sample(frac=1, random_state=7).reset_index(drop=True),
                                      watch_top_pct=20, extended_watch_top_pct=40)
    # Coordinates make serialization stable but do NOT manufacture different
    # risk ranks or split the tie into relative priority bands.
    assert r1["lat_grid"].tolist() == r2["lat_grid"].tolist() == [1, 2, 3, 4, 5]
    assert r1["rank"].tolist() == [1, 1, 1, 1, 1]
    assert set(r1["ranking_tier"]) == {"UNRANKED"}
    assert set(r1["percentile"]) == {1.0}


def test_high_absolute_threshold_overrides_relative_tier():
    df = _grid(100)
    r = I.assign_operational_ranking(df, high_threshold=0.5,
                                     watch_top_pct=10, extended_watch_top_pct=20)
    # every cell >= 0.5 is explicitly marked as an absolute override without
    # reusing the user-facing HIGH/WATCH vocabulary for relative ranking.
    assert (
        (r["probability"] >= 0.5)
        == (r["ranking_tier"] == "ABSOLUTE_HIGH")
    ).all()


def test_probability_is_never_mutated():
    df = _grid(50)
    before = sorted(df["probability"].tolist())
    r = I.assign_operational_ranking(df, high_threshold=0.7)
    assert sorted(r["probability"].tolist()) == before


def test_percentile_is_monotone_with_rank():
    df = _grid(100)
    r = I.assign_operational_ranking(df)
    assert r["percentile"].is_monotonic_increasing
    assert abs(r["percentile"].iloc[-1] - 1.0) < 1e-9


def test_empty_input_is_safe():
    r = I.assign_operational_ranking(
        pd.DataFrame({"lat_grid": [], "lon_grid": [], "probability": []}))
    assert len(r) == 0 and "ranking_tier" in r.columns


def test_ranking_policy_metadata_contract():
    m = I.ranking_policy_metadata(0.5236, 10, 20,
                                  historical_coverage={"top_10pct": 0.5247})
    assert m["type"] == "relative_resource_allocation"
    assert m["absolute_high_threshold"] == 0.5236
    assert m["tie_breaking"][0] == "calibrated_probability_desc"
    assert "precalibration_model_score" in m["tie_breaking"][1]
    assert m["ties_split_across_priority_bands"] is False
    assert "not an absolute" in m["note"].lower() or "NOT an absolute" in m["note"]
    assert m["historical_coverage"]["top_10pct"] == 0.5247
    assert "historical" in m["historical_coverage_note"].lower()


def test_raw_signal_breaks_calibration_plateau_before_coordinates():
    df = pd.DataFrame({
        "lat_grid": [1.0, 2.0, 3.0],
        "lon_grid": [100.0, 100.0, 100.0],
        "probability": [0.1, 0.1, 0.1],
        "_ranking": [0.2, 0.9, 0.8],
    })
    ranked = I.assign_operational_ranking(
        df,
        watch_top_pct=34,
        extended_watch_top_pct=67,
        ranking_signal_col="_ranking",
    )
    assert ranked["lat_grid"].tolist() == [2.0, 3.0, 1.0]
    assert ranked["probability"].tolist() == [0.1, 0.1, 0.1]


def test_calibrated_probability_has_lexicographic_precedence_over_history():
    """A large history count must never overturn a genuinely higher risk."""
    df = pd.DataFrame({
        "lat_grid": [10.0, 11.0],
        "lon_grid": [100.0, 100.0],
        "probability": [0.500001, 0.5],
        "historical_fire_count_30d": [0, 999],
    })

    ranked = I.assign_operational_ranking(
        df,
        evidence_cols=["historical_fire_count_30d"],
    )

    assert ranked["lat_grid"].tolist() == [10.0, 11.0]
    assert ranked["probability"].tolist() == [0.500001, 0.5]


class _PlateauModel:
    def _raw_proba(self, frame):
        return frame["signal"].to_numpy(dtype=float)

    def predict_proba(self, frame):
        return np.full(len(frame), 0.01)


def test_model_ranking_signal_is_ordering_only_and_has_safe_fallback():
    frame = pd.DataFrame({"signal": [0.2, 0.8]})
    calibrated = I.calibrated_probability(_PlateauModel(), frame)
    signal, source = I.model_ranking_signal(
        _PlateauModel(), frame, calibrated=calibrated
    )
    assert signal.tolist() == [0.2, 0.8]
    assert calibrated.tolist() == [0.01, 0.01]
    assert source == "precalibration_model_score"

    legacy = type("Legacy", (), {"predict_proba": lambda self, x: calibrated})()
    fallback, fallback_source = I.model_ranking_signal(
        legacy, frame, calibrated=calibrated
    )
    assert fallback.tolist() == calibrated.tolist()
    assert fallback_source == "calibrated_probability_fallback"


def test_watch_is_canonical_cutoff_with_medium_legacy_fallback():
    assert I.tier_cutoffs_from_meta({
        "HIGH": 0.7, "WATCH": 0.3, "MEDIUM": 0.4,
    }) == (0.7, 0.3)
    assert I.tier_cutoffs_from_meta({"HIGH": 0.7, "MEDIUM": 0.4}) == (0.7, 0.4)
    assert I.probability_to_tier(0.7, (0.7, 0.3)) == "High"
    assert I.probability_to_tier(0.3, (0.7, 0.3)) == "Watch"
