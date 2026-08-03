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
    assert (r["ranking_tier"] == "WATCH").sum() == 10
    # EXTENDED_WATCH is ranks 11..20 (top-20 minus the top-10 WATCH)
    assert (r["ranking_tier"] == "EXTENDED_WATCH").sum() == 10
    assert set(r["ranking_tier"]) <= {"WATCH", "EXTENDED_WATCH", "LOW"}


def test_ranking_is_deterministic_under_ties_and_shuffling():
    df = pd.DataFrame({
        "lat_grid": [5, 4, 3, 2, 1], "lon_grid": [1, 1, 1, 1, 1],
        "probability": [0.5, 0.5, 0.5, 0.5, 0.5],   # all tied
    })
    r1 = I.assign_operational_ranking(df, watch_top_pct=20, extended_watch_top_pct=40)
    r2 = I.assign_operational_ranking(df.sample(frac=1, random_state=7).reset_index(drop=True),
                                      watch_top_pct=20, extended_watch_top_pct=40)
    # tie-break = lat_grid asc → order is [1,2,3,4,5] both times
    assert r1["lat_grid"].tolist() == r2["lat_grid"].tolist() == [1, 2, 3, 4, 5]
    assert r1["rank"].tolist() == [1, 2, 3, 4, 5]


def test_high_absolute_threshold_overrides_relative_tier():
    df = _grid(100)
    r = I.assign_operational_ranking(df, high_threshold=0.5,
                                     watch_top_pct=10, extended_watch_top_pct=20)
    # every cell >= 0.5 is HIGH regardless of its rank
    assert ((r["probability"] >= 0.5) == (r["ranking_tier"] == "HIGH")).all()


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
    assert m["tie_breaking"] == ["probability_desc", "lat_grid_asc", "lon_grid_asc"]
    assert "not an absolute" in m["note"].lower() or "NOT an absolute" in m["note"]
    assert m["historical_coverage"]["top_10pct"] == 0.5247
    assert "historical" in m["historical_coverage_note"].lower()
