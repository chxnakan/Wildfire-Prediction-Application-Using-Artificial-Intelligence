"""NASA FIRMS Area API request-window and fail-closed invariants."""

from __future__ import annotations

from datetime import date
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fetch_firms as ff


def test_ten_day_sync_is_split_at_current_five_day_api_limit():
    assert ff._request_windows(10, today=date(2026, 8, 26)) == [
        (date(2026, 8, 17), 5),
        (date(2026, 8, 22), 5),
    ]


def test_historical_interval_is_inclusive_and_never_exceeds_limit():
    windows = ff._request_windows(
        1, start_date="2026-07-17", end_date="2026-08-25"
    )
    assert windows[0] == (date(2026, 7, 17), 5)
    assert windows[-1] == (date(2026, 8, 21), 5)
    assert sum(days for _start, days in windows) == 40
    assert max(days for _start, days in windows) == 5


@pytest.mark.parametrize(
    ("start", "end"),
    [("2026-08-01", None), (None, "2026-08-01"), ("2026-08-02", "2026-08-01")],
)
def test_invalid_date_intervals_fail_before_network(start, end):
    with pytest.raises(ValueError):
        ff._request_windows(1, start_date=start, end_date=end)
