"""Runtime/API trust regressions that do not load a real model or use network."""

import asyncio
import ast
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api
import db
import security


def test_api_bootstraps_dotenv_before_env_snapshot_modules():
    """`.env` must load before local modules cache their environment settings."""
    source = Path(api.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    dotenv_line = min(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_dotenv"
    )
    snapshot_modules = {"features", "security", "db", "scheduler"}
    import_lines = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in snapshot_modules:
            import_lines[node.module] = node.lineno
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in snapshot_modules:
                    import_lines[alias.name] = node.lineno
    assert snapshot_modules <= set(import_lines)
    assert all(dotenv_line < line for line in import_lines.values())


def _weather_rows(
    start: str = "2026-07-18",
    days: int = 7,
    *,
    missing_rain_index: int | None = None,
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=days, freq="D")
    rain = [1.0] * days
    if missing_rain_index is not None:
        rain[missing_rain_index] = None
    return pd.DataFrame({
        "lat_grid": [18.5] * days,
        "lon_grid": [98.5] * days,
        "date": dates,
        "temp_max": [36.0] * days,
        "temp_min": [24.0] * days,
        "precip_sum": rain,
        "wind_max": [12.0] * days,
        "et0": [3.0] * days,
    })


@pytest.fixture
def weather_cache(tmp_path, monkeypatch):
    path = tmp_path / "weather_cache.parquet"
    monkeypatch.setattr(api, "WEATHER_CACHE_PATH", str(path))
    monkeypatch.setattr(api, "_WEATHER_CACHE_DF", None)
    monkeypatch.setattr(api, "_WEATHER_CACHE_SIGNATURE", None)
    monkeypatch.setattr(api, "_WEATHER_CACHE_LOAD_ERROR", None)
    yield path
    api._WEATHER_CACHE_DF = None
    api._WEATHER_CACHE_SIGNATURE = None
    api._WEATHER_CACHE_LOAD_ERROR = None


def test_weather_path_honors_runtime_weather_dir(tmp_path, monkeypatch):
    weather_dir = tmp_path / "mounted-weather"
    monkeypatch.setenv("WEATHER_DIR", str(weather_dir))
    monkeypatch.delenv("WEATHER_PATH", raising=False)
    assert api._configured_weather_cache_path() == str(
        weather_dir / "weather_cache.parquet"
    )


def test_missing_weather_cache_is_not_cached_forever(weather_cache):
    assert api._load_weather_cache().empty
    _weather_rows().to_parquet(weather_cache, index=False)
    loaded = api._load_weather_cache()
    assert len(loaded) == 7
    assert loaded["date"].max() == pd.Timestamp("2026-07-24")


def test_cell_weather_uses_complete_seven_day_rain(weather_cache):
    _weather_rows().to_parquet(weather_cache, index=False)
    result = api.cell_weather(18.5, 98.5, "2026-07-24")
    assert result["available"] is True
    assert result["precip_7d_complete"] is True
    assert result["precip_7d_mm"] == 7.0
    assert result["fire_weather_index"]["level_key"] == "high"
    assert result["fire_weather_index"]["is_official_fwi"] is False


def test_missing_rain_stays_unknown_instead_of_becoming_zero(weather_cache):
    _weather_rows(missing_rain_index=2).to_parquet(weather_cache, index=False)
    result = api.cell_weather(18.5, 98.5, "2026-07-24")
    assert result["available"] is True
    assert result["precip_7d_complete"] is False
    assert result["precip_7d_mm"] is None
    assert result["fire_weather_index"]["level_key"] == "unknown"
    assert api._fwi(36.0, None)["level_key"] == "unknown"


def test_weather_rejects_stale_date_fallback(weather_cache, monkeypatch):
    _weather_rows(start="2026-07-09").to_parquet(weather_cache, index=False)
    monkeypatch.setenv("WEATHER_MAX_FALLBACK_DAYS", "7")
    result = api.cell_weather(18.5, 98.5, "2026-07-24")
    assert result["available"] is False
    assert result["latest_available_date"] == "2026-07-15"
    assert "9 days older" in result["reason"]


def test_weather_rejects_invalid_calendar_date(weather_cache):
    with pytest.raises(HTTPException) as exc:
        api.cell_weather(18.5, 98.5, "2026-02-30")
    assert exc.value.status_code == 422


def test_default_csp_allows_weather_fallback(monkeypatch):
    monkeypatch.delenv("CSP_CONNECT_SRC", raising=False)
    csp = security.content_security_policy()
    assert "https://api.open-meteo.com" in csp
    assert "https://archive-api.open-meteo.com" in csp


def test_untrusted_request_id_is_bounded_and_sanitized():
    with mock.patch.object(api, "_build_model_state", return_value=None):
        with TestClient(api.app) as client:
            response = client.get(
                "/health/live",
                headers={"X-Request-ID": "x" * 1000},
            )
    request_id = response.headers["X-Request-ID"]
    assert request_id != "x" * 1000
    assert len(request_id) == 16


def test_unknown_api_route_is_json_404_not_spa_html():
    with mock.patch.object(api, "_build_model_state", return_value=None):
        with TestClient(api.app) as client:
            response = client.get("/api/definitely-not-a-route")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["detail"]["error"] == "api_route_not_found"


def test_spa_candidate_blocks_similar_prefix_sibling(tmp_path, monkeypatch):
    root = tmp_path / "dist"
    root.mkdir()
    sibling = tmp_path / "dist-secret"
    sibling.mkdir()
    monkeypatch.setattr(api, "WEB_DIST_DIR", str(root))
    assert api._safe_spa_candidate("../dist-secret/file.txt") is None


def test_transboundary_window_is_exactly_24_hours(tmp_path, monkeypatch):
    latest = datetime(2026, 7, 24, 7, tzinfo=timezone.utc)
    path = tmp_path / "firms.parquet"
    pd.DataFrame({
        "acq_datetime": [
            latest - timedelta(hours=25),
            latest - timedelta(hours=23),
            latest,
        ],
        "latitude": [18.0, 18.0, 18.0],
        "longitude": [100.0, 100.0, 100.0],
    }).to_parquet(path, index=False)
    monkeypatch.setenv("FIRMS_PATH", str(path))
    payload = api._compute_transboundary_payload()
    thailand = next(
        row for row in payload["counts"] if row["country"] == "Thailand"
    )
    assert payload["available"] is True
    assert payload["source_rows_in_window"] == 2
    assert thailand["count"] == 2


def test_analytics_cache_invalidates_when_firms_file_changes(
    tmp_path, monkeypatch,
):
    path = tmp_path / "firms.parquet"
    latest = datetime(2026, 7, 24, 7, tzinfo=timezone.utc)
    monkeypatch.setenv("FIRMS_PATH", str(path))
    api._ANALYTICS_CACHE.clear()

    pd.DataFrame({
        "acq_datetime": [latest],
        "latitude": [18.0],
        "longitude": [100.0],
    }).to_parquet(path, index=False)
    first = asyncio.run(api.analytics_hotspots())
    first_count = next(
        row["count"] for row in first["transboundary"]
        if row["country"] == "Thailand"
    )

    pd.DataFrame({
        "acq_datetime": [latest, latest - timedelta(hours=1)],
        "latitude": [18.0, 18.1],
        "longitude": [100.0, 100.1],
    }).to_parquet(path, index=False)
    # Ensure a distinct signature even on filesystems with coarse mtimes.
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = asyncio.run(api.analytics_hotspots())
    second_count = next(
        row["count"] for row in second["transboundary"]
        if row["country"] == "Thailand"
    )
    assert first_count == 1
    assert second_count == 2


def test_live_sse_compound_cursor_does_not_skip_same_timestamp_batch():
    timestamp = "2026-07-24T07:00:00+00:00"
    for index in range(600):
        hotspot_id = f"hotspot-{index:04d}"
        db._MEM_LIVE[hotspot_id] = {
            "hotspot_id": hotspot_id,
            "first_seen_at": timestamp,
        }

    first = db.get_hotspots_first_seen_after(
        "2026-07-24T06:59:59+00:00", limit=500
    )
    last = first[-1]
    second = db.get_hotspots_first_seen_after(
        str(last["first_seen_at"]),
        after_id=str(last["hotspot_id"]),
        limit=500,
    )
    ids = {row["hotspot_id"] for row in first + second}
    assert len(first) == 500
    assert len(second) == 100
    assert len(ids) == 600
