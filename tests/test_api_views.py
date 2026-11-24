import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException

import src.api as api
from src.risk_map import build_geojson
from src.detection_store import DetectionStore, save_observation_coordinates

NAME = "2025-01-08_0123456789ab.json"


def snapshot(**changes):
    rows = [
        {"lat_cell": 10, "lon_cell": 20, "lat_grid": 1.0, "lon_grid": 2.0, "detection_score": .9,
         "probability": .8, "forecast_detection": True, "estimated_days_if_detection": 2.,
         "days_until_fire": 2, "predicted_fire_date": "2025-01-10", "urgency_level": "HIGH"},
        {"lat_cell": 30, "lon_cell": 40, "lat_grid": 3.0, "lon_grid": 4.0, "detection_score": .2,
         "probability": .1, "forecast_detection": False, "estimated_days_if_detection": 5.,
         "days_until_fire": -1, "predicted_fire_date": None, "urgency_level": "NONE"},
    ]
    record = {"schema_version": 4, "issued_at": "2025-02-01T00:00:00+00:00", "base_date": "2025-01-08",
              "feature_cutoff": "2025-01-07", "horizon_days": 7, "model_version": "0123456789ab",
              "training_labels_through": "2025-01-07", "grid_size": .1, "threshold": .5,
              "prospective": False, "kind": "historical_demo", "calibrated": True, "predictions": rows}
    record.update(changes)
    return record


def detections(days):
    rows = [{"lat_cell": 50, "lon_cell": 60, "date": day, "fire_count": 1} for day in days]
    rows.append({"lat_cell": 10, "lon_cell": 20, "date": "2025-01-07", "fire_count": 3})
    rows.append({"lat_cell": 10, "lon_cell": 20, "date": "2025-01-10", "fire_count": 2})
    return pd.DataFrame(rows)


class ApiViewTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)
        for name in ("forecasts", "features", "riskmap", "metadata"):
            (self.out / name).mkdir()
        (self.out / "metadata/dataset_info.json").write_text(json.dumps({"grid_size": .1}))
        (self.out / "forecasts" / NAME).write_text(json.dumps(snapshot()))
        (self.out / "riskmap/latest.json").write_text(json.dumps({"snapshot": NAME}))
        self.write_detections(pd.date_range("2025-01-01", "2025-01-20").strftime("%Y-%m-%d"))
        patcher = patch.object(api, "OUTPUT", self.out)
        patcher.start()
        self.addCleanup(patcher.stop)
        api._cache.clear()
        self.addCleanup(api._cache.clear)

    def write_detections(self, days):
        frame = detections(days)
        frame.to_csv(self.out / "features/daily_detections.csv.gz", index=False)
        raw = frame.loc[frame.index.repeat(frame.fire_count)].copy()
        raw["latitude"], raw["longitude"] = raw.lat_cell * .1, raw.lon_cell * .1
        save_observation_coordinates(raw, self.out)
        api._cache.clear()

    def test_forecast_names_cannot_escape_the_directory(self):
        for name in ("../riskmap/latest.json", "..%2Fsecret.json", "2025-01-08_0123456789ab.json.bak",
                     "2025-01-09_0123456789ab.json"):
            with self.assertRaises(HTTPException) as caught:
                api.forecast_geojson(name)
            self.assertEqual(caught.exception.status_code, 404)

    def test_each_snapshot_keeps_its_own_outcome_source_policy(self):
        from src.pipeline import TARGET_POLICY
        path = self.out / "features/observation_coordinates.csv.gz"
        raw = pd.read_csv(path)
        raw["hotspot_type"] = 0.
        raw.loc[raw.latitude.eq(1.) & raw.date.eq("2025-01-10"), "hotspot_type"] = 2.
        save_observation_coordinates(raw, self.out)
        new_name = "2025-01-08_abcdefabcdef.json"
        (self.out / "forecasts" / new_name).write_text(json.dumps(snapshot(target_policy=TARGET_POLICY)))
        legacy = api.forecast_outcomes(NAME)
        revised = api.forecast_outcomes(new_name)
        self.assertEqual(legacy["status"], "ok")
        self.assertEqual(revised["status"], "ok")
        self.assertEqual(legacy["metrics"]["classification"]["true_positive"], 1)
        self.assertEqual(revised["metrics"]["classification"]["true_positive"], 0)
        self.assertEqual(revised["metrics"]["classification"]["false_positive"], 1)

    def test_thailand_outcomes_use_only_thai_candidates(self):
        record = snapshot()
        record["predictions"][0].update(lat_cell=135, lon_cell=1010, lat_grid=13.5, lon_grid=101.)
        (self.out / "forecasts" / NAME).write_text(json.dumps(record))
        frame = pd.DataFrame({"date": pd.date_range("2025-01-09", periods=7),
                              "lat_cell": 135, "lon_cell": 1010, "fire_count": 1})
        with patch.object(api, "detections", return_value=frame):
            regional = api.forecast_outcomes(NAME)
            scoped = api.forecast_outcomes(NAME, scope="thailand")
        self.assertEqual(regional["metrics"]["samples"], 2)
        self.assertEqual(scoped["metrics"]["samples"], 1)
        self.assertEqual(scoped["metrics"]["classification"]["true_positive"], 1)
        self.assertEqual(len(scoped["cells"]), 1)
        self.assertEqual(scoped["scope"], "thailand")

    def test_snapshot_without_thai_candidates_is_explicit(self):
        self.assertEqual(api.forecast_outcomes(NAME, scope="thailand")["status"], "no_candidates")

    def test_thailand_history_excludes_foreign_counts(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"] * 2),
                              "lat_cell": [135, 0], "lon_cell": [1010, 0], "fire_count": [2, 99]})
        with patch.object(api, "detections", return_value=frame):
            scoped = api.daily_history(scope="thailand")
        self.assertEqual(scoped["days"], [{"date": "2025-01-01", "detections": 2, "cells": 1}])

    def test_listing_marks_latest_and_counts_flagged_cells(self):
        listing = api.list_forecasts()
        self.assertEqual(listing["latest"], NAME)
        row = listing["forecasts"][0]
        self.assertEqual((row["flagged"], row["candidates"], row["latest"]), (1, 2, True))

    def test_snapshot_geojson_matches_shared_builder(self):
        frame = pd.read_csv(self.out / "features/daily_detections.csv.gz", parse_dates=["date"])
        expected = build_geojson(snapshot(), frame)
        served = api.forecast_geojson(NAME)
        self.assertCountEqual(served["features"], expected["features"])
        self.assertEqual(served["metadata"]["snapshot"], NAME)
        observed = [f for f in served["features"] if f["properties"]["source"] == "observed"]
        self.assertEqual({f["properties"]["date"] for f in observed}, {"2025-01-07"})

    def test_outcomes_match_later_detections(self):
        result = api.forecast_outcomes(NAME)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["retrospective"])
        classification = result["metrics"]["classification"]
        self.assertEqual((classification["true_positive"], classification["false_positive"]), (1, 0))
        hit = next(c for c in result["cells"] if c["flagged"])
        self.assertEqual((hit["actual_day"], hit["predicted_day"]), (2, 2))

    def test_three_day_outcomes_do_not_count_day_four_detections(self):
        name = "2025-01-08_abcdefabcdef.json"
        (self.out / "forecasts" / name).write_text(json.dumps(snapshot(horizon_days=3)))
        observed = pd.DataFrame({"date": pd.date_range("2025-01-09", periods=7),
                                 "lat_cell": 50, "lon_cell": 60, "fire_count": 1})
        observed = pd.concat([observed, pd.DataFrame([{
            "date": pd.Timestamp("2025-01-12"), "lat_cell": 10, "lon_cell": 20, "fire_count": 1}])])
        with patch.object(api, "detections", return_value=observed):
            new = api.forecast_outcomes(name)
            old = api.forecast_outcomes(NAME)
        self.assertEqual(new["window_end"], "2025-01-11")
        self.assertEqual(new["horizon_days"], 3)
        self.assertEqual(new["metrics"]["classification"]["true_positive"], 0)
        self.assertEqual(old["window_end"], "2025-01-15")
        self.assertEqual(old["metrics"]["classification"]["true_positive"], 1)

    def test_three_day_timeline_and_prediction_metadata(self):
        record = snapshot(horizon_days=3)
        record["predictions"][1]["estimated_days_if_detection"] = 3.
        frame = pd.read_csv(self.out / "features/daily_detections.csv.gz", parse_dates=["date"])
        geojson = build_geojson(record, frame)
        with patch.object(api, "get_geojson", return_value=geojson):
            forecast = api.predictions_today()
            timeline = api.predictions_timeline()
        self.assertEqual(forecast["prediction_horizon_days"], 3)
        self.assertEqual(timeline["horizon_days"], 3)
        self.assertEqual(list(timeline["timeline"]), ["2025-01-09", "2025-01-10", "2025-01-11"])

    def test_root_reports_published_horizon_during_model_migration(self):
        latest = self.out / "riskmap/latest.json"
        latest.write_text(json.dumps({"horizon_days": 7}))
        self.assertEqual(api.root()["horizon_days"], 7)
        latest.write_text(json.dumps({"horizon_days": 3}))
        self.assertEqual(api.root()["horizon_days"], 3)

    def test_outcomes_wait_for_complete_coverage(self):
        self.write_detections(pd.date_range("2025-01-01", "2025-01-12").strftime("%Y-%m-%d"))
        self.assertEqual(api.forecast_outcomes(NAME)["status"], "pending")
        days = [d for d in pd.date_range("2025-01-01", "2025-01-20").strftime("%Y-%m-%d") if d != "2025-01-11"]
        self.write_detections(days)
        result = api.forecast_outcomes(NAME)
        self.assertEqual(result["status"], "unknown_coverage")
        self.assertEqual(result["missing_dates"], ["2025-01-11"])

    def test_cell_history_uses_only_dates_up_to_cutoff(self):
        result = api.cell_history(10, 20, date(2025, 1, 7))
        self.assertEqual((result["detections_7d"], result["detections_30d"]), (3, 3))
        self.assertEqual(result["last_detection"], "2025-01-07")
        self.assertEqual(api.cell_history(99, 99, date(2025, 1, 7))["last_detection"], None)

    def test_daily_history_sums_detections(self):
        days = {row["date"]: row for row in api.daily_history()["days"]}
        self.assertEqual(days["2025-01-07"]["detections"], 4)
        self.assertEqual(days["2025-01-07"]["cells"], 2)

    def test_outcomes_wait_until_last_day_has_finished_in_utc(self):
        with patch.object(api, "utc_today", return_value=date(2025, 1, 15)):
            self.assertEqual(api.forecast_outcomes(NAME)["status"], "pending")
        with patch.object(api, "utc_today", return_value=date(2025, 1, 16)):
            self.assertEqual(api.forecast_outcomes(NAME)["status"], "ok")

    def test_archived_forecast_and_cell_history_survive_a_grid_change(self):
        (self.out / "metadata/dataset_info.json").write_text(json.dumps({"grid_size": .2}))
        # Rebuild the legacy derived table on the new grid, as retraining does.
        api.detections().to_csv(self.out / "features/daily_detections.csv.gz", index=False)
        result = api.forecast_outcomes(NAME)
        self.assertEqual(result["metrics"]["classification"]["true_positive"], 1)
        self.assertEqual(result["metrics"]["classification"]["false_positive"], 0)
        observed = [f for f in api.forecast_geojson(NAME)["features"] if f["properties"]["source"] == "observed"]
        self.assertIn([2.0, 1.0], [f["geometry"]["coordinates"] for f in observed])
        history = api.cell_history(10, 20, date(2025, 1, 7), snapshot_name=NAME)
        self.assertEqual(history["detections_7d"], 3)

    def test_regridding_uses_original_pixels_not_the_previous_cell_centres(self):
        raw = pd.DataFrame({"date": ["2025-01-10"] * 2,
                            "latitude": [1.049, 1.051], "longitude": [2., 2.]})
        save_observation_coordinates(raw, self.out)
        store = DetectionStore()
        coarse = store.daily(self.out, .2)
        fine = store.daily(self.out, .1)
        self.assertEqual(list(coarse.fire_count), [2])
        self.assertEqual(set(fine.lat_cell), {10, 11})
        self.assertEqual(list(fine.fire_count), [1, 1])
        with self.assertRaises(ValueError):
            build_geojson(snapshot(), coarse)

    def test_new_coordinates_invalidate_both_grid_caches(self):
        store = DetectionStore()
        before = store.daily(self.out, .1).fire_count.sum()
        store.daily(self.out, .2)
        raw = pd.DataFrame({"date": ["2025-01-10"], "latitude": [1.], "longitude": [2.]})
        save_observation_coordinates(raw, self.out)
        self.assertGreater(before, 1)
        self.assertEqual(store.daily(self.out, .1).fire_count.sum(), 1)
        self.assertEqual(store.daily(self.out, .2).fire_count.sum(), 1)

    def test_existing_installations_recover_coordinates_from_input_files(self):
        (self.out / "features/observation_coordinates.csv.gz").unlink()
        source = self.out / "original.csv"
        source.write_text("latitude,longitude,acq_date,acq_time,satellite,frp,brightness,confidence\n"
                          "1.049,2,2025-01-10,0030,N20,2,300,n\n")
        result = DetectionStore().daily(self.out, .1, [source])
        self.assertEqual(result.iloc[0].lat_cell, 10)
        self.assertEqual(result.iloc[0].fire_count, 1)


if __name__ == "__main__":
    unittest.main()
