import copy
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.forecasting import predict
from src.pipeline import FEATURES
from src.weather_features import HOURLY, WEATHER_FEATURES, WeatherCache, WeatherUnavailable, hourly_to_features


def weather_record():
    times = pd.date_range("2025-01-01", "2025-01-20 23:00", freq="h")
    values = {"time": times.strftime("%Y-%m-%dT%H:%M").tolist()}
    for column in HOURLY:
        values[column] = times.day.astype(float).tolist()
    return {"response": {"hourly": values}}


class WeatherTests(unittest.TestCase):
    def test_features_use_only_intended_past_and_archived_forecasts(self):
        record = weather_record()
        features = hourly_to_features(record).set_index("date")
        row = features.loc["2025-01-10"]
        self.assertEqual(row.weather_rain_past_3d, (6 + 7 + 8) * 24)
        self.assertEqual(row.weather_rain_past_7d, sum(range(2, 9)) * 24)
        self.assertEqual(row.weather_rain_forecast_3d, (11 + 12 + 13) * 24)
        self.assertEqual(row.weather_humidity_min_forecast_3d, 11)
        # Changing realised/short-lead future weather must not influence D.
        changed = copy.deepcopy(record)
        for column in HOURLY:
            if column.endswith("day0"):
                changed["response"]["hourly"][column][9 * 24:] = [999.] * (11 * 24)
        pd.testing.assert_series_equal(row, hourly_to_features(changed).set_index("date").loc["2025-01-10"])
        issue = pd.Timestamp("2025-01-10")
        latest_future_available = issue + pd.Timedelta(days=3, hours=23) - pd.Timedelta(days=5) + pd.Timedelta(hours=6)
        latest_past_available = issue - pd.Timedelta(days=2) + pd.Timedelta(hours=23 + 6)
        self.assertLess(latest_future_available, issue)
        self.assertLess(latest_past_available, issue)

    def test_missing_hour_is_not_silently_zero_rain(self):
        record = weather_record()
        record["response"]["hourly"]["precipitation_previous_day5"][11 * 24] = None
        features = hourly_to_features(record).set_index("date")
        self.assertTrue(pd.isna(features.loc["2025-01-10", "weather_rain_forecast_3d"]))
        self.assertFalse(pd.isna(features.loc["2025-01-10", "weather_rain_past_7d"]))

    def test_duplicate_weather_timestamps_are_rejected(self):
        record = weather_record()
        record["response"]["hourly"]["time"][1] = record["response"]["hourly"]["time"][0]
        with self.assertRaises(ValueError):
            hourly_to_features(record)

    def test_cache_reuses_exact_request_and_records_provenance(self):
        response = Mock(status_code=200, content=b"weather-response")
        response.json.return_value = weather_record()["response"]
        for name in ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]:
            response.json.return_value["hourly"][name] = response.json.return_value["hourly"].pop(name + "_previous_day0")
        with tempfile.TemporaryDirectory() as directory, patch("src.weather_features.requests.get", return_value=response) as get:
            cache = WeatherCache(directory, min_interval=0)
            first = cache.fetch(13.5, 101, "2025-01-01", "2025-01-20")
            second = cache.fetch(13.5, 101, "2025-01-01", "2025-01-20")
            self.assertEqual(first, second)
            self.assertEqual(get.call_count, 1)
            self.assertEqual(first["request"]["timezone"], "GMT")
            self.assertEqual(first["request"]["wind_speed_unit"], "ms")
            self.assertEqual(len(first["sha256"]), 64)

    def test_cache_reuses_archived_history_and_fetches_only_missing_tail(self):
        import gzip, json
        from pathlib import Path
        history = weather_record()["response"]
        params = dict(latitude=13.5, longitude=101.0, start_date="2025-01-01", end_date="2025-01-20",
                      hourly=",".join(HOURLY), models="gfs_seamless", timezone="GMT", wind_speed_unit="ms")
        old = {"request": params, "downloaded_at": "2025-01-15T00:00:00+00:00", "sha256": "a" * 64, "response": history}
        times = pd.date_range("2025-01-13", "2025-01-25 23:00", freq="h")
        tail = {"time": times.strftime("%Y-%m-%dT%H:%M").tolist()}
        for name in ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]:
            tail[name] = [100.] * len(times)
            tail[name + "_previous_day5"] = [100.] * len(times)
        response = Mock(status_code=200, content=b"tail")
        response.json.return_value = {"hourly": tail}
        with tempfile.TemporaryDirectory() as directory, patch("src.weather_features.requests.get", return_value=response) as get:
            with gzip.open(Path(directory) / "old.json.gz", "wt") as handle:
                json.dump(old, handle)
            record = WeatherCache(directory, min_interval=0).fetch_range(13.5, 101, "2025-01-01", "2025-01-25")
            # Downloaded on the 15th: only dates up to the 13th are trusted.
            self.assertEqual(get.call_args.kwargs["params"]["start_date"], "2025-01-14")
            hourly = pd.DataFrame(record["response"]["hourly"]).set_index("time")
            self.assertFalse(hourly.index.has_duplicates)
            self.assertEqual(len(hourly), 25 * 24)
            self.assertEqual(hourly.loc["2025-01-13T23:00", HOURLY[0]], 13.)
            self.assertEqual(hourly.loc["2025-01-14T00:00", HOURLY[0]], 100.)
            self.assertEqual(record["request"]["end_date"], "2025-01-25")
            self.assertEqual(len(record["composed_from"]), 2)

    def test_rate_limit_stops_subsequent_requests(self):
        with tempfile.TemporaryDirectory() as directory, patch("src.weather_features.requests.get", return_value=Mock(status_code=429)) as get:
            cache = WeatherCache(directory, min_interval=0)
            for lon in [101, 102]:
                with self.assertRaises(WeatherUnavailable):
                    cache.fetch(13.5, lon, "2025-01-01", "2025-01-20")
            self.assertEqual(get.call_count, 1)

    def test_main_inference_uses_saved_weather_feature_schema(self):
        names = FEATURES + WEATHER_FEATURES
        frame = pd.DataFrame({name: [1.] for name in names})
        frame["date"], frame["lat_cell"], frame["lon_cell"] = pd.Timestamp("2025-01-10"), 135, 1010
        classifier, regressor, calibrator = Mock(), Mock(), Mock()
        classifier.predict_proba.return_value = np.array([[.2, .8]])
        regressor.predict.return_value = np.array([2.])
        calibrator.transform.return_value = np.array([.75])
        bundle = {"schema_version": 4, "grid_size": .1, "threshold": .5, "features": names,
                  "classifier": classifier, "regressor": regressor, "calibrator": calibrator}
        self.assertEqual(predict(bundle, frame).iloc[0].probability, .75)
        self.assertEqual(classifier.predict_proba.call_args.args[0].columns.tolist(), names)
        with self.assertRaisesRegex(ValueError, "missing required features"):
            predict(bundle, frame.drop(columns=WEATHER_FEATURES[0]))
        frame[WEATHER_FEATURES[0]] = np.nan
        with self.assertRaisesRegex(ValueError, "missing values"):
            predict(bundle, frame)
