import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.detection_store import DetectionStore, save_observation_coordinates
from src.geography import thailand_cells
from src.land_features import LAND_FEATURES, summarize_tile, tiles_for_cells
from src.model_selection import rolling_validation, select_features
from src.pipeline import FEATURES, TARGET_POLICY, daily_panel, load_observations, target_observations
from src.weather_features import HOURLY, hourly_to_features


def source_observations():
    dates = pd.date_range("2025-01-01", periods=60)
    return pd.DataFrame({"date": dates, "observed_at": dates, "latitude": 13., "longitude": 100.,
                         "lat_cell": 130, "lon_cell": 1000, "frp": 2., "brightness": 300.,
                         "confidence": 50., "daynight": "D", "hotspot_type": 2.})


class ResearchTests(unittest.TestCase):
    def test_typed_archive_wins_over_nrt_duplicate_and_unknown_stays_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, nrt = Path(directory) / "archive.csv", Path(directory) / "nrt.csv"
            header = "latitude,longitude,acq_date,acq_time,satellite,frp,brightness,confidence"
            row = "13,100,2025-01-01,0030,N20,2,300,n"
            archive.write_text(header + ",type,daynight\n" + row + ",2,N\n")
            nrt.write_text(header + "\n" + row + "\n13,100,2025-01-02,0030,N20,2,300,n\n")
            data, quality = load_observations([nrt, archive])
            self.assertEqual(quality["duplicates_removed"], 1)
            self.assertEqual(data.loc[data.date.eq("2025-01-01"), "hotspot_type"].iloc[0], 2)
            retained = target_observations(data, TARGET_POLICY)
            self.assertEqual(len(retained), 1)
            self.assertTrue(retained.hotspot_type.isna().all())
            self.assertEqual(retained.daynight.iloc[0], "U")

    def test_conflicting_archive_types_are_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for value in [0, 2]:
                p = Path(directory) / f"{value}.csv"
                p.write_text("latitude,longitude,acq_date,acq_time,satellite,frp,brightness,confidence,type\n"
                             f"13,100,2025-01-01,0030,N20,2,300,n,{value}\n")
                paths.append(p)
            data, quality = load_observations(paths)
            self.assertEqual(len(data), 1)
            self.assertTrue(data.hotspot_type.isna().all())
            self.assertEqual(quality["conflicting_type_rows"], 2)

    def test_source_policy_changes_outcomes_but_not_historical_features_or_coverage(self):
        data = source_observations()
        all_panel, _ = daily_panel(data)
        filtered, _ = daily_panel(data, target_policy=TARGET_POLICY)
        np.testing.assert_array_equal(all_panel[FEATURES], filtered[FEATURES])
        row = filtered[filtered.date.eq("2025-01-15")].iloc[0]
        self.assertEqual(row.target, 0)
        self.assertTrue(row.label_complete)
        self.assertTrue(row.history_complete)
        data.loc[data.date.eq("2025-01-17"), "hotspot_type"] = np.nan
        updated, _ = daily_panel(data, target_policy=TARGET_POLICY)
        row = updated[updated.date.eq("2025-01-15")].iloc[0]
        self.assertEqual(row.target, 1)
        self.assertEqual(row.days_until_detection, 2)

    def test_country_candidates_exist_before_first_detection_and_never_detected_cells_remain(self):
        data = source_observations()
        new = data.iloc[[40]].assign(lat_cell=131, latitude=13.1, hotspot_type=0.)
        data = pd.concat([data, new], ignore_index=True)
        candidates = pd.DataFrame({"lat_cell": [130, 131, 132], "lon_cell": [1000] * 3})
        panel, _ = daily_panel(data, target_policy=TARGET_POLICY, candidate_cells=candidates)
        never = panel[panel.lat_cell.eq(132)]
        self.assertTrue(never.target.eq(0).all())
        row = panel[panel.lat_cell.eq(131) & panel.date.eq("2025-02-08")].iloc[0]
        self.assertEqual(row.fire_3d, 0)
        self.assertEqual(row.target, 1)
        self.assertTrue(row.history_complete)
        thai = thailand_cells(.1)
        self.assertFalse(thai.duplicated().any())
        self.assertGreater(len(thai), 4008)

    def test_observation_archive_preserves_types_and_legacy_counts(self):
        data = source_observations()
        data.loc[0, "hotspot_type"] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            save_observation_coordinates(data, directory)
            store = DetectionStore()
            self.assertEqual(store.daily(directory, .1).fire_count.sum(), 60)
            self.assertEqual(store.daily(directory, .1, target_policy=TARGET_POLICY).fire_count.sum(), 1)
            self.assertEqual(store.daily(directory, .1).fire_count.sum(), 60)
            with self.assertRaises(ValueError):
                target_observations(data.drop(columns="hotspot_type"), TARGET_POLICY)

    def test_land_aggregation_counts_cell_areas_and_excludes_nodata(self):
        image = np.array([[10, 40], [50, np.nan]], dtype=float)
        table = summarize_tile(image, [99.95, 12.95, 100.05, 13.05], .1)
        self.assertEqual(len(table), 1)
        self.assertEqual(table.iloc[0]["count"], 3)
        fractions = table[LAND_FEATURES].div(table["count"], axis=0)
        self.assertAlmostEqual(fractions.iloc[0].land_tree_fraction, 1 / 3)
        self.assertAlmostEqual(fractions.iloc[0].sum(), 1)
        cell = pd.DataFrame({"lat_cell": [180], "lon_cell": [990]})
        self.assertEqual(set(tiles_for_cells(cell, .1, 3)), {(15, 96), (15, 99), (18, 96), (18, 99)})

    def test_terrain_flat_surface_has_zero_slope_and_correct_elevation(self):
        result = summarize_tile(np.full((5, 5), 123.), [99.96, 12.96, 100.04, 13.04], .1, terrain=True)
        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertAlmostEqual(row.elevation_sum / row["count"], 123)
        self.assertAlmostEqual(row.slope_sum, 0)

    def test_drying_windows_and_missing_data(self):
        times = pd.date_range("2025-01-01", periods=60 * 24, freq="h")
        hourly = {"time": times.strftime("%Y-%m-%dT%H:%M").tolist()}
        for name in HOURLY:
            value = 30 if name.startswith("temperature") else 50 if name.startswith("relative") else 0 if name.startswith("precipitation") else 2
            hourly[name] = [float(value)] * len(times)
        record = {"response": {"hourly": hourly}}
        features = hourly_to_features(record).set_index("date")
        row = features.loc["2025-02-05"]
        expected = .6108 * np.exp(17.27 * 30 / (30 + 237.3)) * .5
        self.assertAlmostEqual(row.weather_vpd_mean_past_7d, expected, places=5)
        self.assertEqual(row.weather_dry_days_past_30d, 30)
        changed = copy.deepcopy(record)
        # A missing rainfall hour inside the 30-day window is not a dry day.
        changed["response"]["hourly"]["precipitation_previous_day0"][10 * 24] = None
        self.assertTrue(np.isnan(hourly_to_features(changed).set_index("date").loc["2025-02-05", "weather_dry_days_past_30d"]))
        self.assertTrue(np.isnan(features.loc["2025-01-15", "weather_rain_past_30d"]))

    def test_selection_blocks_are_forward_only_and_purged(self):
        frame = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=240), "target": [0, 1] * 120})
        valid_dates = []
        for fit, valid in rolling_validation(frame):
            self.assertLess(fit.date.max() + pd.Timedelta(days=3), valid.date.min())
            valid_dates.extend(valid.date.tolist())
        self.assertEqual(len(valid_dates), len(set(valid_dates)))

    def test_feature_selection_uses_only_fitting_and_validation_frames(self):
        frame = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=4), "target": [0, 1, 0, 1],
                              "days_until_detection": [-1, 2, -1, 2], "base": [.3, .4, .5, .6], "better": [.1, .9, .2, .8]})
        def fit_model(fit, features):
            self.assertIs(fit, frame)
            classifier, regressor, calibrator = Mock(), Mock(), Mock()
            score = frame[features[-1]].to_numpy()
            classifier.predict_proba.return_value = np.column_stack([1 - score, score])
            regressor.predict.return_value = np.full(4, 2.)
            calibrator.transform.side_effect = lambda p: p
            return classifier, regressor, calibrator, {}
        with patch("src.model_selection.rolling_validation", return_value=[(frame, frame)]):
            features, result = select_features(frame, {"weather_baseline": ["base"], "new": ["base", "better"]}, fit_model, .5)
        self.assertEqual(features, ["base", "better"])
        self.assertFalse(result["holdout_used_for_selection"])
