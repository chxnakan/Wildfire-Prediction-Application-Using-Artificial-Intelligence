"""Evaluation-protocol utilities: leakage-safe splitting, purging, censoring.

Why this module exists
----------------------
The original pipeline had four protocol bugs that inflated reported skill:

  1. Row-based chronological split → one calendar date could land in both
     train and validation (confirmed: 2025-11-10 appeared in both).
  2. sklearn TimeSeriesSplit(gap=7) purges 7 ROWS, not 7 days — with
     thousands of cells per date the effective gap was ~zero.
  3. Negatives were undersampled BEFORE splitting, so val/test prevalence
     was an artificial 4:1 instead of the natural base rate.
  4. Tail rows whose future window extends past the last observed date were
     labelled "no fire" when the truth is "not observed yet".

Everything here is a pure function over a date Series so it can be unit
tested without touching the training pipeline (tests/test_eval_protocol.py).

Terminology: a "split" is one of train/val/calib/test. All splits are
whole-date: a calendar date belongs to exactly one split, and `purge_days`
calendar days are dropped entirely before each split boundary so that a
label whose future window starts near a boundary can never overlap the next
split's period.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SplitResult:
    """Boolean row masks + audit info for a 4-way chronological split."""
    masks: Dict[str, np.ndarray]              # split name → bool mask (row-aligned)
    date_ranges: Dict[str, Tuple[str, str]]   # split name → (min_date, max_date)
    purge_days: int
    n_dates_purged: int
    info: dict = field(default_factory=dict)


def _to_date_series(dates) -> pd.Series:
    s = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    return s.dt.normalize()


def date_grouped_split(
    dates,
    val_fraction: float = 0.15,
    calib_fraction: float = 0.10,
    test_fraction: float = 0.15,
    purge_days: int = 7,
    champion_fraction: float = 0.0,
) -> SplitResult:
    """Chronological train/val/calib/[champion]/test split on WHOLE calendar dates.

    Fractions are of the unique-date count (not row count) so a burst of
    hotspots on one day cannot straddle a boundary. Between consecutive
    splits, `purge_days` calendar days are removed from the dataset
    entirely (assigned to no split) so that:

      - a training row's forward-looking label window (≤ purge_days) can
        never contain a validation/calib/test day, and
      - a validation row's rolling-history features (which look backward)
        share no day with training labels' forward windows.

    ``champion_fraction > 0`` inserts a dedicated PROMOTION/champion block
    between calib and test:

        [ train ][purge][ val ][purge][ calib ][purge][ champion ][purge][ test ]

    Candidate-vs-incumbent promotion decisions run on the champion block so the
    FINAL test stays untouched by model selection (repeated promotions on the
    test set leak selection decisions into the reported number). Default 0.0
    keeps the legacy 4-way layout for backward compatibility.

    Raises ValueError when the data span is too short to give every split
    at least one date after purging.
    """
    if val_fraction + calib_fraction + test_fraction + champion_fraction >= 1.0:
        raise ValueError("val + calib + champion + test fractions must sum to < 1.0")

    s = _to_date_series(dates)
    unique_dates = np.array(sorted(s.unique()))
    n = len(unique_dates)

    n_champion = max(int(round(n * champion_fraction)), 1) if champion_fraction > 0 else 0
    n_purges = 4 if n_champion else 3
    n_test = max(int(round(n * test_fraction)), 1)
    n_calib = max(int(round(n * calib_fraction)), 1)
    n_val = max(int(round(n * val_fraction)), 1)
    n_train = n - n_val - n_calib - n_champion - n_test - n_purges * purge_days
    if n_train < 1:
        raise ValueError(
            f"Not enough distinct dates ({n}) for a "
            f"{'5' if n_champion else '4'}-way split with "
            f"{purge_days}-day purges — need at least "
            f"{n_val + n_calib + n_champion + n_test + n_purges * purge_days + 1}."
        )

    # Chronological block layout:
    # [ train ][purge][ val ][purge][ calib ][purge][ (champion)[purge] ][ test ]
    bounds = {}
    i = n_train
    bounds["train"] = unique_dates[:i]
    i += purge_days
    bounds["val"] = unique_dates[i : i + n_val]
    i += n_val + purge_days
    bounds["calib"] = unique_dates[i : i + n_calib]
    i += n_calib + purge_days
    if n_champion:
        bounds["champion"] = unique_dates[i : i + n_champion]
        i += n_champion + purge_days
    bounds["test"] = unique_dates[i:]

    masks: Dict[str, np.ndarray] = {}
    date_ranges: Dict[str, Tuple[str, str]] = {}
    for name, block in bounds.items():
        block_set = set(block)
        masks[name] = s.isin(block_set).to_numpy()
        date_ranges[name] = (
            str(pd.Timestamp(block[0]).date()),
            str(pd.Timestamp(block[-1]).date()),
        )

    result = SplitResult(
        masks=masks,
        date_ranges=date_ranges,
        purge_days=purge_days,
        # A champion block adds a fourth split boundary; report all actual
        # purged dates rather than the legacy four-way constant.
        n_dates_purged=n_purges * purge_days,
        info={
            "n_unique_dates": n,
            "n_dates_per_split": {k: len(v) for k, v in bounds.items()},
        },
    )
    assert_split_valid(result, s)
    return result


def assert_split_valid(result: SplitResult, dates) -> None:
    """Hard assertions: whole-date membership, disjointness, purge width.

    Raises AssertionError — these are protocol invariants, not conditions
    to handle gracefully. If one fires, the reported metrics are invalid.
    """
    s = _to_date_series(dates)
    date_sets: Dict[str, set] = {}
    for name, mask in result.masks.items():
        date_sets[name] = set(s[mask].unique())

    names = list(date_sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = date_sets[a] & date_sets[b]
            assert not overlap, (
                f"LEAKAGE: splits '{a}' and '{b}' share {len(overlap)} "
                f"calendar dates, e.g. {sorted(overlap)[:3]}"
            )

    # Purge width: min date of each later split must be > purge_days after
    # max date of every earlier split.
    order = [n for n in ("train", "val", "calib", "champion", "test") if date_sets.get(n)]
    for i, earlier in enumerate(order):
        for later in order[i + 1:]:
            gap = (min(date_sets[later]) - max(date_sets[earlier])).days
            assert gap > result.purge_days, (
                f"PURGE VIOLATION: gap between '{earlier}' and '{later}' is "
                f"{gap} days, need > {result.purge_days}"
            )


def purged_date_folds(
    dates,
    n_splits: int = 3,
    purge_days: int = 7,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV folds grouped by calendar date with a purge gap.

    Drop-in replacement for sklearn TimeSeriesSplit in *SearchCV (their `cv`
    parameter accepts an explicit list of (train_idx, test_idx) tuples).
    Guarantees: every fold's train dates strictly precede its test dates
    with > purge_days calendar days between them; no date is in both.
    """
    s = _to_date_series(dates)
    unique_dates = np.array(sorted(s.unique()))
    n = len(unique_dates)
    # n_splits+1 equal date blocks: fold k trains on blocks [0..k], tests on k+1
    block = n // (n_splits + 1)
    if block <= purge_days:
        raise ValueError(
            f"Too few distinct dates ({n}) for {n_splits} purged folds with "
            f"purge={purge_days}d — each block spans only {block} dates."
        )

    positions = s.to_numpy()
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(1, n_splits + 1):
        train_end_date = unique_dates[k * block - 1]
        test_start_date = train_end_date + np.timedelta64(purge_days + 1, "D")
        test_end_date = unique_dates[min((k + 1) * block - 1, n - 1)]
        train_idx = np.where(positions <= train_end_date)[0]
        test_idx = np.where(
            (positions >= test_start_date) & (positions <= test_end_date)
        )[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(
                f"Purged fold {k} is degenerate (train={len(train_idx)}, "
                f"test={len(test_idx)} rows) — not enough dates."
            )
        folds.append((train_idx, test_idx))
    return folds


def observed_label_mask(dates, horizon_days: int) -> np.ndarray:
    """True where a forward-looking label of `horizon_days` is fully observed.

    A row dated d claims to know whether a fire happens in (d, d+horizon].
    That claim is only verifiable when the dataset extends at least
    horizon_days past d. Rows failing this are CENSORED — usable for live
    inference, but never for supervised training or evaluation (their
    'negative' labels would really mean 'future not observed yet').
    """
    s = _to_date_series(dates)
    max_date = s.max()
    return (s <= max_date - pd.Timedelta(days=horizon_days)).to_numpy()
