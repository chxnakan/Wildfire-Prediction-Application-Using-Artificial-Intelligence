"""FIRMS downtime backfill window selection."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import sync_worker as sw


def _cache(path, latest):
    pd.DataFrame({"acq_datetime": [latest]}).to_parquet(path, index=False)


def test_healthy_sync_keeps_small_fetch_window(tmp_path, monkeypatch):
    cache = tmp_path / "firms.parquet"
    _cache(cache, pd.Timestamp.now("UTC"))
    monkeypatch.setattr(sw, "OUT_FILE", str(cache))
    monkeypatch.setattr(sw, "FETCH_DAYS", 1)
    monkeypatch.setattr(sw, "MAX_BACKFILL_DAYS", 10)

    assert sw._fetch_days_for_cycle() == 1


def test_sync_backfills_gap_from_pre_update_high_water(tmp_path, monkeypatch):
    cache = tmp_path / "firms.parquet"
    latest = pd.Timestamp.now("UTC").normalize() - pd.Timedelta(days=6)
    _cache(cache, latest)
    monkeypatch.setattr(sw, "OUT_FILE", str(cache))
    monkeypatch.setattr(sw, "FETCH_DAYS", 1)
    monkeypatch.setattr(sw, "MAX_BACKFILL_DAYS", 10)

    assert sw._fetch_days_for_cycle() == 7


def test_missing_cache_requests_maximum_recoverable_window(tmp_path, monkeypatch):
    monkeypatch.setattr(sw, "OUT_FILE", str(tmp_path / "missing.parquet"))
    monkeypatch.setattr(sw, "MAX_BACKFILL_DAYS", 10)

    assert sw._fetch_days_for_cycle() == 10
