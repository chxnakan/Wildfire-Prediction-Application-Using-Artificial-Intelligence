"""Regression tests for atomic replacement of persisted risk snapshots."""

from contextlib import contextmanager
from datetime import date

import pandas as pd

import db


class _RecordingConnection:
    def __init__(self):
        self.calls = []

    def execute(self, statement, parameters=None):
        self.calls.append((str(statement), parameters))
        return object()


def _enable_recording_db(monkeypatch):
    connection = _RecordingConnection()

    @contextmanager
    def recording_connection():
        yield connection

    monkeypatch.setattr(db, "enabled", lambda: True)
    monkeypatch.setattr(db, "_conn", recording_connection)
    return connection


def test_empty_prediction_snapshot_deletes_stale_rows(monkeypatch):
    connection = _enable_recording_db(monkeypatch)

    inserted = db.insert_predictions(pd.DataFrame(), "2026-08-25T09:15:00Z")

    assert inserted == 0
    assert len(connection.calls) == 1
    sql, parameters = connection.calls[0]
    assert "DELETE FROM risk_predictions WHERE base_date = :bd" in sql
    assert parameters == {"bd": date(2026, 8, 25)}


def test_nonempty_prediction_snapshot_deletes_then_inserts(monkeypatch):
    connection = _enable_recording_db(monkeypatch)
    frame = pd.DataFrame(
        {
            "lat_grid": [15.1],
            "lon_grid": [101.2],
            "probability": [0.42],
            "urgency_level": ["WATCH"],
        }
    )

    inserted = db.insert_predictions(frame, date(2026, 8, 24))

    assert inserted == 1
    assert len(connection.calls) == 2
    delete_sql, delete_parameters = connection.calls[0]
    insert_sql, records = connection.calls[1]
    assert "DELETE FROM risk_predictions WHERE base_date = :bd" in delete_sql
    assert delete_parameters == {"bd": date(2026, 8, 24)}
    assert "INSERT INTO risk_predictions" in insert_sql
    assert records == [
        {
            "lat_grid": 15.1,
            "lon_grid": 101.2,
            "probability": 0.42,
            "urgency_level": "WATCH",
            "base_date": date(2026, 8, 24),
        }
    ]
