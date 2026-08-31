"""Fail-closed source completeness checks used before feature generation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import train


def _daily(dates):
    return pd.DataFrame({"date": pd.to_datetime(dates), "fire_count": 1})


def test_continuity_trims_isolated_fetch_dates_after_last_safe_run():
    dates = list(pd.date_range("2026-07-01", periods=10, freq="D"))
    dates += [pd.Timestamp("2026-07-17"), pd.Timestamp("2026-07-25")]

    report = train._firms_continuity_report(_daily(dates), 7)

    assert report["safe_through"] == "2026-07-10"
    assert report["discarded_tail_dates"] == 2
    assert report["status"] == "trimmed"


def test_continuity_accepts_latest_consecutive_tail():
    dates = pd.date_range("2026-08-10", periods=10, freq="D")

    report = train._firms_continuity_report(_daily(dates), 7)

    assert report["safe_through"] == "2026-08-19"
    assert report["discarded_tail_dates"] == 0
    assert report["status"] == "complete"


def test_short_recent_backfill_island_cannot_bridge_a_missing_feature_window():
    historic = list(pd.date_range("2026-01-01", periods=100, freq="D"))
    # A successful 10-day NRT retry after a long outage is useful source data,
    # but it cannot repair the missing 90-day causal feature history.
    recent_island = list(pd.date_range("2026-08-15", periods=10, freq="D"))

    report = train._firms_continuity_report(
        _daily(historic + recent_island),
        required_consecutive_days=90,
    )

    assert report["safe_through"] == "2026-04-10"
    assert report["discarded_tail_dates"] == 10
    assert report["status"] == "trimmed"


def test_history_window_is_anchored_after_untrusted_tail_is_removed():
    trusted_dates = pd.date_range("2026-01-01", "2026-04-10", freq="D")
    untrusted_dates = pd.to_datetime(["2026-08-15", "2026-08-24"])
    frame = _daily(list(trusted_dates) + list(untrusted_dates))

    trimmed, continuity_removed, history_removed = (
        train._trim_to_trusted_history(
            frame,
            safe_through="2026-04-10",
            max_history_days=30,
        )
    )

    kept_dates = pd.to_datetime(trimmed["date"])
    assert continuity_removed == 2
    assert history_removed > 0
    assert kept_dates.min() == pd.Timestamp("2026-03-11")
    assert kept_dates.max() == pd.Timestamp("2026-04-10")
