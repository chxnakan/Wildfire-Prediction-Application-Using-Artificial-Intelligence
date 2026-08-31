"""Runtime/API trust regressions that do not load a real model or use network."""

import asyncio
import ast
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import State

import api
import db
import security


class _IncompatibleScoringModel:
    def predict_proba(self, _frame):
        raise ValueError("feature shape mismatch")


def _configure_incompatible_serving_artifacts(tmp_path, monkeypatch):
    """Point the real state builder at a bundle/frame that cannot score."""
    import train

    model_path = tmp_path / "model.pkl"
    feature_path = tmp_path / "features.parquet"
    model_path.write_bytes(b"model-placeholder")
    feature_path.write_bytes(b"feature-placeholder")
    frame = pd.DataFrame({
        "date": [datetime.now(timezone.utc).date()],
        "lat_grid": [18.5],
        "lon_grid": [98.5],
        "fire_count": [0],
        "feature": [1.0],
    })
    history = pd.DataFrame({
        "lat_grid": [18.5],
        "lon_grid": [98.5],
        "historical_fire_count_30d": [0],
    })
    metadata = {
        "trained_at": "incompatible-version",
        "features": ["feature"],
    }
    monkeypatch.setattr(api, "MODEL_PATH", str(model_path))
    monkeypatch.setattr(api, "FEATURE_PATH", str(feature_path))
    monkeypatch.setattr(api, "resolve_existing", lambda _path: str(feature_path))
    monkeypatch.setattr(
        train,
        "load_model_bundle",
        lambda _path: (_IncompatibleScoringModel(), metadata),
    )
    monkeypatch.setattr(
        api, "_load_minimal_state", lambda _path: (frame, history)
    )


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


def test_readiness_age_env_rejects_invalid_or_negative_values(monkeypatch):
    monkeypatch.setenv("TEST_READY_AGE", "not-a-number")
    assert api._nonnegative_float_env("TEST_READY_AGE", 7.0) == 7.0
    monkeypatch.setenv("TEST_READY_AGE", "-3")
    assert api._nonnegative_float_env("TEST_READY_AGE", 7.0) == 7.0
    monkeypatch.setenv("TEST_READY_AGE", "2.5")
    assert api._nonnegative_float_env("TEST_READY_AGE", 7.0) == 2.5


def test_incompatible_scoring_bundle_cannot_build_ready_state(
    tmp_path, monkeypatch
):
    _configure_incompatible_serving_artifacts(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="Snapshot pre-scoring failed"):
        api._build_model_state()


def test_scoring_failure_at_startup_stays_not_ready():
    with mock.patch.object(
        api,
        "_build_model_state",
        side_effect=RuntimeError("Snapshot pre-scoring failed"),
    ):
        with TestClient(api.app) as client:
            ready = client.get("/health/ready")
            prediction = client.get("/predictions/today")

    assert ready.status_code == 503
    assert ready.json()["checks"]["model_loaded"] is False
    assert prediction.status_code == 503
    assert prediction.json()["detail"]["error"] == "model_unavailable"


def test_incompatible_scoring_reload_preserves_last_good_state(
    tmp_path, monkeypatch
):
    app = SimpleNamespace(state=State())
    known_good_model = object()
    known_good = dict(
        api._DEGRADED_STATE,
        model=known_good_model,
        df=pd.DataFrame({"date": [datetime.now(timezone.utc).date()]}),
        features=["known_good_feature"],
        model_version="known-good-version",
        scored_snapshot={"trusted": True},
    )
    api._apply_model_state(app, known_good)
    _configure_incompatible_serving_artifacts(tmp_path, monkeypatch)

    assert api.reload_model_state(app) is False
    assert app.state.model is known_good_model
    assert app.state.model_version == "known-good-version"
    assert app.state.scored_snapshot == {"trusted": True}


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


def test_api_status_uses_metadata_from_served_model_state(monkeypatch):
    served = dict(api._DEGRADED_STATE)
    served["meta"] = {"imminent_days": 2}
    monkeypatch.setattr(api, "_load_metadata", lambda: {"imminent_days": 7})
    with mock.patch.object(api, "_build_model_state", return_value=None):
        with TestClient(api.app) as client:
            api._apply_model_state(client.app, served)
            response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["horizon_days"] == 2


def test_metadata_endpoint_uses_served_bundle_not_disk_metadata(monkeypatch):
    served = dict(api._DEGRADED_STATE)
    served["meta"] = {"best_model": "served-model", "imminent_days": 2}
    monkeypatch.setattr(
        api,
        "_load_metadata",
        lambda: {"best_model": "disk-model", "imminent_days": 7},
    )
    with mock.patch.object(api, "_build_model_state", return_value=None):
        with TestClient(api.app) as client:
            api._apply_model_state(client.app, served)
            response = client.get("/metadata")

    assert response.status_code == 200
    assert response.json() == {
        "best_model": "served-model",
        "imminent_days": 2,
    }


def _health_request(latest_date, *, meta=None):
    state = SimpleNamespace(
        model=object(),
        df=pd.DataFrame({"date": [latest_date]}),
        features=["feature"],
        meta=meta or {},
        model_version="served-version",
        model_loaded_at="2026-08-05T00:00:00+00:00",
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _fresh_health_files(tmp_path, monkeypatch):
    metadata = tmp_path / "dataset_info.json"
    geojson = tmp_path / "fire_dates_all.geojson"
    metadata.write_text('{"best_model":"disk-model"}', encoding="utf-8")
    geojson.write_text('{"type":"FeatureCollection"}', encoding="utf-8")
    monkeypatch.setattr(api, "META_PATH", str(metadata))
    monkeypatch.setattr(api, "GEOJSON_PATH", str(geojson))
    monkeypatch.setattr(api, "_GEOJSON_MAX_AGE_DAYS", 7)


def test_health_uses_served_bundle_metadata_not_disk_metadata(
    tmp_path, monkeypatch,
):
    _fresh_health_files(tmp_path, monkeypatch)
    request = _health_request(
        datetime.now(timezone.utc).date(),
        meta={
            "best_model": "served-model",
            "model": {"skill_check_passed": True},
        },
    )
    monkeypatch.setattr(
        api,
        "_load_metadata",
        lambda: {
            "best_model": "disk-model",
            "model": {"skill_check_passed": False},
        },
    )

    payload = api.health(request)

    assert payload["best_model"] == "served-model"
    assert payload["skill_check_passed"] is True


def test_fresh_geojson_cannot_mask_stale_actual_data(tmp_path, monkeypatch):
    _fresh_health_files(tmp_path, monkeypatch)
    monkeypatch.setattr(api, "_DATA_MAX_AGE_DAYS", 2)
    monkeypatch.setattr(api, "_DATA_FUTURE_TOLERANCE_DAYS", 1)
    stale_date = datetime.now(timezone.utc).date() - timedelta(days=3)
    request = _health_request(stale_date)

    combined = api.health(request)
    ready_response = api.health_ready(request)
    ready_body = json.loads(ready_response.body)

    assert combined["geojson_age_days"] <= 1
    assert combined["data_stale_days"] == 3
    assert combined["data_fresh"] is False
    assert combined["ready"] is False
    assert combined["status"] == "degraded"
    assert ready_response.status_code == 503
    assert ready_body["checks"]["geojson_fresh"] is True
    assert ready_body["checks"]["data_fresh"] is False
    assert ready_body["data_max_age_days"] == 2
    assert ready_body["data_future_tolerance_days"] == 1


@pytest.mark.parametrize(
    ("age_days", "expected_fresh"),
    [(-2, False), (-1, True), (0, True), (2, True), (3, False)],
)
def test_served_data_age_enforces_stale_limit_and_timezone_tolerance(
    age_days, expected_fresh, monkeypatch,
):
    monkeypatch.setattr(api, "_DATA_MAX_AGE_DAYS", 2)
    monkeypatch.setattr(api, "_DATA_FUTURE_TOLERANCE_DAYS", 1)
    latest = datetime.now(timezone.utc).date() - timedelta(days=age_days)
    state = _health_request(latest).app.state

    latest_text, actual_age, fresh = api._served_data_age(state)

    assert latest_text == latest.isoformat()
    assert actual_age == age_days
    assert fresh is expected_fresh


def test_missing_or_invalid_data_date_is_not_fresh(monkeypatch):
    monkeypatch.setattr(api, "_DATA_MAX_AGE_DAYS", 2)
    missing = SimpleNamespace(df=pd.DataFrame({"other": [1]}))
    invalid = SimpleNamespace(df=pd.DataFrame({"date": [pd.NaT]}))

    assert api._served_data_age(missing) == (None, None, False)
    assert api._served_data_age(invalid) == (None, None, False)


def test_analytics_withholds_unsourced_historical_burned_area(monkeypatch):
    monkeypatch.setattr(api, "_firms_source_signature", lambda: None)
    api._ANALYTICS_CACHE.clear()

    payload = asyncio.run(api.analytics_hotspots())

    assert payload["historical_burned_mrai"] == []
    assert payload["historical_burned_mrai_metadata"] == {
        "available": False,
        "source": None,
        "reason": "No source-backed historical burned-area dataset is configured",
    }


def test_live_hotspots_discloses_when_requested_page_is_truncated(monkeypatch):
    requested = []

    def fake_get_live_hotspots(**kwargs):
        requested.append(kwargs["limit"])
        return [{"hotspot_id": str(index)} for index in range(4)]

    monkeypatch.setattr(api.db_module, "get_live_hotspots", fake_get_live_hotspots)
    monkeypatch.setattr(api, "_live_status_payload", lambda: {})

    payload = api.live_hotspots(limit=3)

    assert requested == [4]
    assert len(payload["features"]) == 3
    assert payload["metadata"]["count"] == 3
    assert payload["metadata"]["returned_count"] == 3
    assert payload["metadata"]["requested_limit"] == 3
    assert payload["metadata"]["truncated"] is True


def test_live_hotspots_marks_short_page_complete(monkeypatch):
    monkeypatch.setattr(
        api.db_module,
        "get_live_hotspots",
        lambda **_kwargs: [{"hotspot_id": "only"}],
    )
    monkeypatch.setattr(api, "_live_status_payload", lambda: {})

    payload = api.live_hotspots(limit=3)

    assert payload["metadata"]["returned_count"] == 1
    assert payload["metadata"]["truncated"] is False


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


@pytest.mark.parametrize(
    ("age_hours", "expected_stale"),
    [(2, False), (30, True)],
)
def test_transboundary_discloses_source_age_and_staleness(
    age_hours, expected_stale, tmp_path, monkeypatch
):
    latest = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    path = tmp_path / "firms.parquet"
    pd.DataFrame({
        "acq_datetime": [latest],
        "latitude": [18.0],
        "longitude": [100.0],
    }).to_parquet(path, index=False)
    monkeypatch.setenv("FIRMS_PATH", str(path))

    payload = api._compute_transboundary_payload()

    assert payload["is_stale"] is expected_stale
    assert payload["source_age_hours"] == pytest.approx(age_hours, abs=0.05)
    assert payload["window_end_utc"] == pd.Timestamp(latest).isoformat()


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
