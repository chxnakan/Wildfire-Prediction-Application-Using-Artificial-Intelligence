"""Cached, UTC-aligned weather inputs with conservative forecast availability.

Source: Open-Meteo Previous Runs API, GFS seamless (global over Thailand).
Day-0 values describe recent weather using archived short-lead model output.
Day-5 values are forecasts made five days before their valid time, not realised
future weather. Publication is conservatively assumed six hours after that time.
"""

import gzip
import hashlib
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "gfs_seamless"
VARIABLES = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
HOURLY = [f"{name}_previous_day{lead}" for lead in (0, 5) for name in VARIABLES]
WEATHER_FEATURES = ["weather_rain_past_3d", "weather_rain_past_7d",
                    "weather_temp_max_past_3d", "weather_humidity_min_past_3d",
                    "weather_wind_max_past_3d", "weather_rain_forecast_3d",
                    "weather_temp_max_forecast_3d", "weather_humidity_min_forecast_3d",
                    "weather_wind_max_forecast_3d"]
DRYNESS_FEATURES = ["weather_vpd_mean_past_7d", "weather_vpd_max_forecast_3d",
                    "weather_rain_past_30d", "weather_dry_days_past_30d"]
POLICY = {
    "provider": "Open-Meteo Previous Runs API", "model": MODEL,
    "documentation": "https://open-meteo.com/en/docs/previous-runs-api",
    "attribution": "Weather data by Open-Meteo and NOAA GFS; CC BY 4.0",
    "timezone": "UTC", "publication_buffer_hours": 6,
    "past": "Archived day-0 model output; three/seven-day windows ending D-2, not station observations",
    "future": "Forecasts at fixed five-day lead for D+1..D+3; mixed earlier runs, not realised future weather",
    "availability": "Every included hourly value has valid_time - lead_time + 6h < D 00:00 UTC",
    "spatial_method": "Nearest configured weather sampling point; no claim of local station resolution",
}


class WeatherUnavailable(RuntimeError):
    pass


class WeatherCache:
    def __init__(self, directory, min_interval=3.0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last_request = 0.
        self._blocked = False

    def fetch(self, latitude, longitude, start, end):
        params = dict(latitude=float(latitude), longitude=float(longitude),
                      start_date=str(pd.Timestamp(start).date()), end_date=str(pd.Timestamp(end).date()),
                      hourly=",".join(HOURLY), models=MODEL, timezone="GMT", wind_speed_unit="ms")
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        path = self.directory / f"{key}.json.gz"
        if path.exists():
            with gzip.open(path, "rt") as handle:
                return json.load(handle)
        for attempt in range(3):
            with self._lock:
                if self._blocked:
                    raise WeatherUnavailable("Weather provider quota is exhausted; resume after reset")
                time.sleep(max(0., self.min_interval - (time.monotonic() - self._last_request)))
                self._last_request = time.monotonic()
            try:
                response = requests.get(ENDPOINT, params=params, timeout=(10, 90))
                if response.status_code == 429:
                    # Stop instead of bypassing a provider's quota. Cached points
                    # are retained and the same command resumes after reset.
                    self._blocked = True
                    raise WeatherUnavailable("Open-Meteo rate limit reached; cached downloads are resumable after quota reset")
                response.raise_for_status()
                data = response.json()
                # The provider returns day-0 variables without their suffix.
                if "hourly" in data:
                    for name in VARIABLES:
                        if name in data["hourly"]:
                            data["hourly"][f"{name}_previous_day0"] = data["hourly"].pop(name)
                if "hourly" not in data or any(name not in data["hourly"] for name in HOURLY):
                    raise WeatherUnavailable("Weather response lacks required variables")
                break
            except (requests.RequestException, ValueError) as error:
                if attempt == 2:
                    raise WeatherUnavailable(f"Weather download failed for {latitude}, {longitude}") from error
                time.sleep(2 ** attempt)
        record = {"request": params, "endpoint": ENDPOINT, "downloaded_at": datetime.now(timezone.utc).isoformat(),
                  "sha256": hashlib.sha256(response.content).hexdigest(), "response": data}
        with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            with gzip.open(temporary, "wt") as handle:
                json.dump(record, handle, allow_nan=False)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return record

    def _cached_segments(self):
        """Archived spans of earlier downloads, keyed by location.

        Values for the two days before a download may still be filling in, so
        only dates earlier than that are trusted for reuse.
        """
        with self._lock:
            if getattr(self, "_segments", None) is None:
                segments = {}
                for path in self.directory.glob("*.json.gz"):
                    try:
                        with gzip.open(path, "rt") as handle:
                            record = json.load(handle)
                        request = record["request"]
                        downloaded = pd.Timestamp(record["downloaded_at"]).tz_convert(None).normalize()
                    except (OSError, ValueError, KeyError, TypeError):
                        continue
                    if (request.get("hourly") != ",".join(HOURLY) or request.get("models") != MODEL
                            or request.get("timezone") != "GMT" or request.get("wind_speed_unit") != "ms"):
                        continue
                    start = pd.Timestamp(request["start_date"])
                    trusted = min(pd.Timestamp(request["end_date"]), downloaded - pd.Timedelta(days=2))
                    if trusted >= start:
                        key = (float(request["latitude"]), float(request["longitude"]))
                        segments.setdefault(key, []).append((start, trusted, path))
                self._segments = segments
            return self._segments

    def fetch_range(self, latitude, longitude, start, end):
        """Like fetch, but reuses archived spans of earlier downloads and only
        requests the missing tail, so a longer window does not re-download history."""
        start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        params = dict(latitude=float(latitude), longitude=float(longitude),
                      start_date=str(start.date()), end_date=str(end.date()),
                      hourly=",".join(HOURLY), models=MODEL, timezone="GMT", wind_speed_unit="ms")
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        if (self.directory / f"{key}.json.gz").exists():
            return self.fetch(latitude, longitude, start, end)
        available = self._cached_segments().get((float(latitude), float(longitude)), [])
        parts, cursor = [], start
        while cursor <= end:
            usable = [segment for segment in available if segment[0] <= cursor and segment[1] >= cursor]
            if not usable:
                break
            _, trusted, path = max(usable, key=lambda segment: segment[1])
            with gzip.open(path, "rt") as handle:
                record = json.load(handle)
            stop = min(trusted, end)
            parts.append((cursor, stop, record))
            cursor = stop + pd.Timedelta(days=1)
        if not parts:
            return self.fetch(latitude, longitude, start, end)
        if cursor <= end:
            parts.append((cursor, end, self.fetch(latitude, longitude, cursor, end)))
        hourly = {}
        for first, last, record in parts:
            frame = pd.DataFrame(record["response"]["hourly"])
            day = frame.time.str[:10]
            frame = frame[(day >= str(first.date())) & (day <= str(last.date()))]
            for column in ["time"] + HOURLY:
                hourly.setdefault(column, []).extend(frame[column].tolist())
        response = {**parts[-1][2]["response"], "hourly": hourly}
        digest = hashlib.sha256("".join(record["sha256"] for _, _, record in parts).encode()).hexdigest()
        return {"request": params, "endpoint": ENDPOINT, "downloaded_at": max(record["downloaded_at"] for _, _, record in parts),
                "sha256": digest, "response": response,
                "composed_from": [{"start_date": str(first.date()), "end_date": str(last.date()), "sha256": record["sha256"]}
                                  for first, last, record in parts]}


def hourly_to_features(record):
    """One row per prediction date; incomplete weather stays missing, never zero."""
    hourly = pd.DataFrame(record["response"]["hourly"])
    hourly.index = pd.DatetimeIndex(pd.to_datetime(hourly.pop("time"), utc=True)).tz_localize(None)
    if hourly.index.has_duplicates:
        raise ValueError("Duplicate weather timestamps")
    if hourly.empty:
        raise WeatherUnavailable("Empty weather response")
    hours = pd.date_range(hourly.index.min().normalize(), hourly.index.max().normalize() + pd.Timedelta(hours=23), freq="h")
    hourly = hourly.reindex(hours).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    daily = {}
    for lead in (0, 5):
        columns = [f"{v}_previous_day{lead}" for v in VARIABLES]
        values = hourly[columns]
        complete = values.resample("D").count().eq(24).all(axis=1)
        d = values.resample("D").agg({columns[0]: "max", columns[1]: "min", columns[2]: "sum", columns[3]: "max"})
        d.columns = ["temp", "humidity", "rain", "wind"]
        d.loc[~complete, :] = np.nan
        daily[lead] = d
    past, future = daily[0], daily[5]
    features = pd.DataFrame(index=past.index)
    features["weather_rain_past_3d"] = past.rain.shift(2).rolling(3, min_periods=3).sum()
    features["weather_rain_past_7d"] = past.rain.shift(2).rolling(7, min_periods=7).sum()
    for name, column, operation in [("temp", "temp", "max"), ("humidity", "humidity", "min"), ("wind", "wind", "max")]:
        features[f"weather_{name}_{operation}_past_3d"] = getattr(past[column].shift(2).rolling(3, min_periods=3), operation)()
    for name, column, operation in [("rain", "rain", "sum"), ("temp", "temp", "max"), ("humidity", "humidity", "min"), ("wind", "wind", "max")]:
        window = pd.concat([future[column].shift(-day) for day in (1, 2, 3)], axis=1)
        features[f"weather_{name}_{operation + '_' if name != 'rain' else ''}forecast_3d"] = getattr(window, operation)(axis=1).where(window.notna().all(axis=1))
    # Hourly paired temperature/humidity, not unrelated daily extrema.
    # FAO-56 saturation vapour pressure; kPa. This is atmospheric drying
    # potential, not measured fuel moisture or a locally validated FWI.
    for lead, suffix in [(0, "past"), (5, "forecast")]:
        temp = hourly[f"temperature_2m_previous_day{lead}"]
        rh = hourly[f"relative_humidity_2m_previous_day{lead}"]
        vpd = (0.6108 * np.exp(17.27 * temp / (temp + 237.3)) * (1 - rh / 100)).where(rh.between(0, 100))
        complete = vpd.resample("D").count().eq(24)
        if lead == 0:
            features["weather_vpd_mean_past_7d"] = vpd.resample("D").mean().where(complete).shift(2).rolling(7).mean()
        else:
            maxima = vpd.resample("D").max().where(complete)
            window = pd.concat([maxima.shift(-d) for d in (1, 2, 3)], axis=1)
            features["weather_vpd_max_forecast_3d"] = window.max(axis=1).where(window.notna().all(axis=1))
    features["weather_rain_past_30d"] = past.rain.shift(2).rolling(30).sum()
    dry = past.rain.lt(1).astype(float).where(past.rain.notna())
    features["weather_dry_days_past_30d"] = dry.shift(2).rolling(30).sum()
    features.index.name = "date"
    return features[WEATHER_FEATURES + DRYNESS_FEATURES].astype("float32").reset_index()


def attach_weather(panel, grid_size, cache_dir, weather_grid=.5, min_interval=3.):
    if not np.isfinite(weather_grid) or weather_grid <= 0:
        raise ValueError("Weather grid must be finite and positive")
    frame = panel.copy()
    frame["weather_lat"] = np.rint(frame.lat_cell * grid_size / weather_grid) * weather_grid
    frame["weather_lon"] = np.rint(frame.lon_cell * grid_size / weather_grid) * weather_grid
    locations = sorted(set(zip(frame.weather_lat, frame.weather_lon)))
    start, end = frame.date.min() - pd.Timedelta(days=8), frame.date.max() + pd.Timedelta(days=3)
    cache = WeatherCache(cache_dir, min_interval)
    def download(location):
        lat, lon = location
        record = cache.fetch_range(lat, lon, start, end)
        result = hourly_to_features(record)
        result["weather_lat"], result["weather_lon"] = lat, lon
        return result, {"latitude": lat, "longitude": lon, "sha256": record["sha256"], "request": record["request"]}
    tables, manifest = [], []
    with ThreadPoolExecutor(max_workers=2) as executor:
        remaining = iter(locations)
        pending = deque(executor.submit(download, location) for location in [next(remaining, None), next(remaining, None)] if location is not None)
        i = 0
        while pending:
            table, source = pending.popleft().result()
            i += 1
            tables.append(table)
            manifest.append(source)
            if i == 1 or i % 10 == 0 or i == len(locations):
                print(f"Weather: {i}/{len(locations)} locations cached", flush=True)
            location = next(remaining, None)
            if location is not None:
                pending.append(executor.submit(download, location))
    joined = frame.merge(pd.concat(tables, ignore_index=True), on=["date", "weather_lat", "weather_lon"], how="left", validate="many_to_one")
    return joined, {**POLICY, "sampling_grid_degrees": weather_grid, "locations": len(locations), "sources": manifest}
