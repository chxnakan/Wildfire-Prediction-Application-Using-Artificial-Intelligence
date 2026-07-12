"""Protocol-invariant tests for eval_protocol.py + the train.py conventions.

Run from src/:  python tests/test_eval_protocol.py   (plain asserts, no pytest
needed — prints one line per passing test). Also pytest-compatible.

These tests use SYNTHETIC dataframes — that does not violate the project's
"real data only" rule, which governs model FEATURES; test fixtures exercising
split logic are exactly what synthetic data is for.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from eval_protocol import (
    SplitResult,
    assert_split_valid,
    date_grouped_split,
    observed_label_mask,
    purged_date_folds,
)


def _synthetic_frame(n_days=200, n_cells=50, pos_rate=0.03, seed=7):
    """Dense (cell × day) frame mimicking the densified FIRMS layout."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for c in range(n_cells):
        for d in dates:
            rows.append((c, d))
    df = pd.DataFrame(rows, columns=["cell", "date"])
    df["y"] = (rng.random(len(df)) < pos_rate).astype(int)
    return df


def test_split_dates_are_disjoint():
    df = _synthetic_frame()
    res = date_grouped_split(df["date"], purge_days=7)
    sets = {
        name: set(pd.to_datetime(df.loc[mask, "date"]).dt.date)
        for name, mask in res.masks.items()
    }
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (sets[a] & sets[b]), f"{a} ∩ {b} not empty"
    print("ok  test_split_dates_are_disjoint")


def test_purge_gap_is_wide_enough():
    df = _synthetic_frame()
    purge = 7
    res = date_grouped_split(df["date"], purge_days=purge)
    d = pd.to_datetime(df["date"])
    order = ["train", "val", "calib", "test"]
    for i, earlier in enumerate(order):
        for later in order[i + 1:]:
            gap = (d[res.masks[later]].min() - d[res.masks[earlier]].max()).days
            assert gap > purge, f"{earlier}->{later} gap {gap} ≤ purge {purge}"
    print("ok  test_purge_gap_is_wide_enough")


def test_every_row_in_at_most_one_split():
    df = _synthetic_frame()
    res = date_grouped_split(df["date"], purge_days=7)
    total = np.zeros(len(df), dtype=int)
    for mask in res.masks.values():
        total += mask.astype(int)
    assert total.max() <= 1, "a row belongs to two splits"
    # purged rows belong to none — verify some were actually purged
    assert (total == 0).sum() > 0, "expected purged rows between blocks"
    print("ok  test_every_row_in_at_most_one_split")


def test_assert_split_valid_catches_leakage():
    df = _synthetic_frame(n_days=100)
    res = date_grouped_split(df["date"], purge_days=7)
    # Corrupt: copy the val mask into train (shared dates)
    bad_masks = {k: v.copy() for k, v in res.masks.items()}
    bad_masks["train"] = bad_masks["train"] | bad_masks["val"]
    bad = SplitResult(
        masks=bad_masks, date_ranges=res.date_ranges,
        purge_days=res.purge_days, n_dates_purged=res.n_dates_purged,
    )
    try:
        assert_split_valid(bad, df["date"])
    except AssertionError as e:
        assert "LEAKAGE" in str(e)
        print("ok  test_assert_split_valid_catches_leakage")
        return
    raise AssertionError("leaked split passed validation")


def test_purged_cv_folds_have_date_gap():
    df = _synthetic_frame(n_days=180)
    folds = purged_date_folds(df["date"], n_splits=3, purge_days=7)
    d = pd.to_datetime(df["date"]).to_numpy()
    assert len(folds) == 3
    for k, (tr, te) in enumerate(folds):
        assert len(np.intersect1d(tr, te)) == 0
        gap = (d[te].min() - d[tr].max()) / np.timedelta64(1, "D")
        assert gap > 7, f"fold {k}: CV gap {gap} days ≤ 7"
    print("ok  test_purged_cv_folds_have_date_gap")


def test_tail_labels_are_censored():
    df = _synthetic_frame(n_days=50)
    horizon = 3
    observed = observed_label_mask(df["date"], horizon_days=horizon)
    d = pd.to_datetime(df["date"])
    max_date = d.max()
    # every unobserved row is within `horizon` days of the end
    assert ((max_date - d[~observed]).dt.days < horizon).all()
    # every observed row has at least `horizon` future days available
    assert ((max_date - d[observed]).dt.days >= horizon).all()
    # exactly the last `horizon` dates are censored
    n_censored_dates = pd.Series(d[~observed]).dt.date.nunique()
    assert n_censored_dates == horizon, f"censored {n_censored_dates} ≠ {horizon}"
    print("ok  test_tail_labels_are_censored")


def test_natural_prevalence_preserved_outside_train():
    """Undersampling train must not change val/calib/test prevalence."""
    df = _synthetic_frame(n_days=300, n_cells=40, pos_rate=0.05)
    res = date_grouped_split(df["date"], purge_days=7)

    prevalence_before = {
        name: df.loc[mask, "y"].mean() for name, mask in res.masks.items()
    }
    # Simulate the pipeline: undersample negatives WITHIN TRAIN ONLY
    train = df[res.masks["train"]]
    pos, neg = train[train.y == 1], train[train.y == 0]
    train_us = pd.concat([pos, neg.sample(len(pos) * 4, random_state=0)])
    assert abs(train_us.y.mean() - 0.2) < 0.02  # train is now ~1:4 by design

    for name in ("val", "calib", "test"):
        after = df.loc[res.masks[name], "y"].mean()
        assert after == prevalence_before[name], (
            f"{name} prevalence changed by train undersampling"
        )
    print("ok  test_natural_prevalence_preserved_outside_train")


def test_calibration_period_independent_of_fit_data():
    """calib dates must not intersect train∪val (the model-fitting data)."""
    df = _synthetic_frame()
    res = date_grouped_split(df["date"], purge_days=7)
    d = pd.to_datetime(df["date"]).dt.date
    fit_dates = set(d[res.masks["train"]]) | set(d[res.masks["val"]])
    calib_dates = set(d[res.masks["calib"]])
    assert not (fit_dates & calib_dates)
    # and calib strictly precedes test (threshold chosen before test is seen)
    test_min = min(d[res.masks["test"]])
    assert max(calib_dates) < test_min
    print("ok  test_calibration_period_independent_of_fit_data")


def test_split_too_short_raises():
    df = _synthetic_frame(n_days=20)
    try:
        date_grouped_split(df["date"], purge_days=7)
    except ValueError:
        print("ok  test_split_too_short_raises")
        return
    raise AssertionError("short dataset should refuse a purged 4-way split")


if __name__ == "__main__":
    test_split_dates_are_disjoint()
    test_purge_gap_is_wide_enough()
    test_every_row_in_at_most_one_split()
    test_assert_split_valid_catches_leakage()
    test_purged_cv_folds_have_date_gap()
    test_tail_labels_are_censored()
    test_natural_prevalence_preserved_outside_train()
    test_calibration_period_independent_of_fit_data()
    test_split_too_short_raises()
    print("\nALL PROTOCOL TESTS PASSED")
