import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.calibration import PlattCalibrator, diagnostics
from src.evaluation import evaluate
from src.forecasting import predict, save_snapshot, score_snapshots
from src.pipeline import FEATURES, HORIZON, HISTORY_DAYS, chronological_split, daily_panel, load_observations, purged_folds


def observations(days=80):
    dates = pd.date_range("2025-01-01", periods=days)
    # An independent grid ensures regional coverage on every calendar day.
    rows = [(d, 10, 20) for d in dates]
    rows += [(dates[i], 30, 40) for i in [0, 2, 9, 20, 45, 70] if i < days]
    return pd.DataFrame([{"date": d, "observed_at": d, "lat_cell": lat,
                          "lon_cell": lon, "latitude": lat * .1, "longitude": lon * .1,
                          "frp": 2., "brightness": 300., "confidence": 50.}
                         for d, lat, lon in rows])


class DailyPipelineTests(unittest.TestCase):
    def test_deduplicates_across_source_names_and_preserves_other_satellite(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "archive.csv"
            second = Path(tmp) / "nrt.csv"
            first.write_text("latitude,longitude,acq_date,acq_time,satellite,frp,brightness,confidence\n13,100,2025-01-01,0030,N20,2,300,n\n")
            second.write_text("latitude,longitude,acq_date,acq_time,dataset,frp,bright_ti4,confidence\n13,100,2025-01-01,0030,VIIRS_NOAA20_NRT,2,300,50\n13,100,2025-01-01,0030,VIIRS_NOAA21_NRT,2,300,h\n")
            data, quality = load_observations([first, second])
            self.assertEqual(quality["duplicates_removed"], 1)
            self.assertEqual(len(data), 2)
            self.assertEqual(set(data.confidence), {50., 100.})
            self.assertTrue((data.brightness == 300).all())

    def test_calendar_windows_and_features_never_use_current_or_future_day(self):
        data = observations()
        panel, _ = daily_panel(data)
        grid = panel[panel.lat_cell == 30].set_index("date")
        jan10 = grid.loc["2025-01-10"]
        self.assertEqual(jan10.fire_3d, 0)  # Jan 7–9, not last three detections
        self.assertEqual(jan10.fire_yesterday, 0)
        self.assertEqual(jan10.target, 0)  # Next detection is eleven days away
        self.assertEqual(grid.loc["2025-01-09"].days_until_detection, 1)
        changed = pd.concat([data, pd.DataFrame([dict(data.iloc[0], date=pd.Timestamp("2025-01-10"),
                                                     lat_cell=30, lon_cell=40, frp=9999)])], ignore_index=True)
        updated, _ = daily_panel(changed)
        compare = updated[(updated.lat_cell == 30) & (updated.date == pd.Timestamp("2025-01-10"))].iloc[0]
        np.testing.assert_array_equal(jan10[FEATURES], compare[FEATURES])
        self.assertEqual(grid.loc["2025-01-11"].fire_yesterday, 1)

    def test_incomplete_outcomes_and_unknown_coverage_are_excluded(self):
        data = observations()
        data = data[data.date != pd.Timestamp("2025-01-15")]
        panel, _ = daily_panel(data)
        grid = panel[panel.lat_cell == 30].set_index("date")
        self.assertTrue(grid.loc["2025-01-10"].label_complete)
        self.assertFalse(grid.loc["2025-01-12"].label_complete)
        self.assertFalse(grid.loc["2025-01-16"].history_complete)
        self.assertFalse(panel.loc[panel.date == panel.date.max(), "label_complete"].any())
        self.assertTrue((panel[panel.label_complete].days_until_detection == -1).any())

    def test_grid_enters_after_first_observation_and_split_is_purged(self):
        data = observations()
        data = pd.concat([data, pd.DataFrame([dict(data.iloc[0], date=pd.Timestamp("2025-02-01"), lat_cell=99)])], ignore_index=True)
        panel, _ = daily_panel(data)
        self.assertEqual(panel[panel.lat_cell == 99].date.min(), pd.Timestamp("2025-02-02"))
        train, test, start = chronological_split(panel)
        self.assertLess(train.date.max() + pd.Timedelta(days=HORIZON), start)
        self.assertGreaterEqual(test.date.min(), start)
        self.assertEqual(set(train.target), {0, 1})

    def test_metrics_use_negatives_and_score_timing_on_all_positive_windows(self):
        result = evaluate([1, 1, 0, 0], [2, 5, -1, -1], [.9, .1, .8, .1], [3, 2, 4, 4])
        c, t = result["classification"], result["timing"]
        self.assertEqual([c[k] for k in ["true_positive", "false_positive", "false_negative", "true_negative"]], [1, 1, 1, 1])
        self.assertEqual(c["precision"], .5)
        self.assertEqual(c["recall"], .5)
        self.assertEqual(t["mae_days"], 2.)
        self.assertEqual(t["within_one_day"], .5)
        self.assertIsNone(evaluate([0], [-1], [.1], [2])["classification"]["precision"])

    def test_negative_inference_has_no_date_or_fake_confidence(self):
        frame, _ = daily_panel(observations())
        frame = frame.tail(2)
        classifier, regressor = Mock(), Mock()
        classifier.predict_proba.return_value = np.array([[.9, .1], [.2, .8]])
        regressor.predict.return_value = np.array([3., 2.])
        calibrator = Mock()
        calibrator.transform.side_effect = lambda scores: np.asarray(scores) * .9
        bundle = {"schema_version": 4, "grid_size": .1, "threshold": .5,
                  "classifier": classifier, "regressor": regressor, "calibrator": calibrator}
        result = predict(bundle, frame)
        self.assertIsNone(result.iloc[0].predicted_fire_date)
        self.assertEqual(result.iloc[0].urgency_level, "NONE")
        self.assertEqual(result.iloc[1].days_until_fire, 2)
        self.assertAlmostEqual(result.iloc[1].probability, .72)
        self.assertAlmostEqual(result.iloc[1].detection_score, .8)
        self.assertNotIn("confidence", result)

    def test_uncalibrated_bundle_is_rejected(self):
        frame, _ = daily_panel(observations())
        bundle = {"schema_version": 3, "grid_size": .1, "threshold": .5,
                  "classifier": Mock(), "regressor": Mock()}
        with self.assertRaises(ValueError):
            predict(bundle, frame.tail(1))

    def test_cross_fitting_folds_purge_and_embargo_the_held_block(self):
        panel, _ = daily_panel(observations(120))
        train, _, _ = chronological_split(panel)
        covered = np.zeros(len(train), dtype=bool)
        for fit, held in purged_folds(train, 3):
            start, end = train.date[held].min(), train.date[held].max()
            fit_dates = train.date[fit]
            self.assertFalse(((fit_dates >= start - pd.Timedelta(days=HORIZON)) &
                              (fit_dates <= end + pd.Timedelta(days=HISTORY_DAYS))).any())
            self.assertFalse((fit & held).any())
            covered |= held
        self.assertTrue(covered.all())

    def test_only_days_one_through_three_create_a_positive_label(self):
        panel, _ = daily_panel(observations())
        grid = panel[panel.lat_cell == 30].set_index("date")
        # A detection on January 10 counts on D+3, but not D+4.
        self.assertEqual(grid.loc["2025-01-06"].days_until_detection, -1)
        self.assertEqual(grid.loc["2025-01-06"].target, 0)
        self.assertEqual(grid.loc["2025-01-07"].days_until_detection, 3)
        self.assertEqual(grid.loc["2025-01-08"].days_until_detection, 2)
        self.assertEqual(grid.loc["2025-01-09"].days_until_detection, 1)
        self.assertLessEqual(panel.days_until_detection.max(), 3)

    def test_inference_clips_to_the_bundles_own_horizon(self):
        panel, _ = daily_panel(observations())
        classifier, regressor, calibrator = Mock(), Mock(), Mock()
        classifier.predict_proba.return_value = np.array([[.1, .9]])
        regressor.predict.return_value = np.array([6.])
        calibrator.transform.side_effect = lambda scores: scores
        bundle = {"schema_version": 4, "grid_size": .1, "threshold": .5,
                  "classifier": classifier, "regressor": regressor, "calibrator": calibrator,
                  "horizon_days": 3}
        self.assertEqual(predict(bundle, panel.tail(1)).iloc[0].days_until_fire, 3)
        del bundle["horizon_days"]  # An old bundle remains a seven-day model.
        self.assertEqual(predict(bundle, panel.tail(1)).iloc[0].days_until_fire, 6)


class CalibrationTests(unittest.TestCase):
    def test_platt_is_monotonic_and_bounded(self):
        rng = np.random.default_rng(0)
        raw = rng.uniform(0, 1, 2000)
        y = rng.uniform(0, 1, 2000) < raw ** 2
        calibrated = PlattCalibrator().fit(raw, y).transform(np.linspace(0, 1, 50))
        self.assertTrue(np.all(np.diff(calibrated) >= 0))
        self.assertTrue(((calibrated >= 0) & (calibrated <= 1)).all())

    def test_calibration_requires_both_classes(self):
        with self.assertRaises(ValueError):
            PlattCalibrator().fit([.2, .3], [0, 0])

    def test_diagnostics_on_known_ranking(self):
        y = np.array([0] * 80 + [1] * 20)
        p = np.concatenate([np.full(80, .1), np.full(20, .9)])
        result = diagnostics(y, p, .5)
        self.assertEqual(result["roc_auc"], 1.0)
        self.assertAlmostEqual(result["positive_rate"], .2)
        self.assertEqual(result["precision_at_top"]["10"]["precision"], 1.0)
        self.assertAlmostEqual(result["precision_at_top"]["10"]["lift"], 5.0)
        self.assertAlmostEqual(result["ece"], .1)
        row = next(r for r in result["threshold_table"] if r["threshold"] == .5)
        self.assertEqual((row["precision"], row["recall"]), (1.0, 1.0))
        self.assertEqual(sum(b["count"] for b in result["reliability"]), 100)
        self.assertIsNone(diagnostics([0, 0], [.1, .2])["roc_auc"])


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = pd.Timestamp("2025-01-08")
        self.rows = pd.DataFrame([{"date": self.base, "lat_cell": 10, "lon_cell": 20,
                                  "detection_score": .8, "estimated_days_if_detection": 1.}])
        self.bundle = {"model_version": "test", "grid_size": .1, "threshold": .5,
                       "training_labels_through": "2025-01-07"}
        self.issued = datetime(2025, 1, 8, 12, tzinfo=timezone.utc)

    def test_snapshot_is_immutable_and_historical_issue_is_not_prospective(self):
        path = save_snapshot(self.rows, self.bundle, self.tmp.name, self.issued)
        original = path.read_text()
        self.rows["detection_score"] = .1
        save_snapshot(self.rows, self.bundle, self.tmp.name, self.issued)
        self.assertEqual(path.read_text(), original)
        self.assertTrue(json.loads(original)["prospective"])
        self.bundle["model_version"] = "later"
        later = save_snapshot(self.rows, self.bundle, self.tmp.name, datetime(2025, 2, 1, tzinfo=timezone.utc))
        self.assertFalse(json.loads(later.read_text())["prospective"])

    def test_outcome_scoring_waits_and_then_matches_location(self):
        save_snapshot(self.rows, self.bundle, self.tmp.name, self.issued)
        data = observations()
        pending = score_snapshots(self.tmp.name, data, "2025-01-14")
        self.assertEqual(pending["pending_runs"], 1)
        report = score_snapshots(self.tmp.name, data, "2025-01-15")
        metric = report["evaluated_runs"][0]["metrics"]
        self.assertEqual(metric["classification"]["true_positive"], 1)
        self.assertEqual(metric["timing"]["mae_days"], 0)
        missing = data[data.date != pd.Timestamp("2025-01-12")]
        self.assertEqual(score_snapshots(self.tmp.name, missing, "2025-01-15")["unknown_coverage_runs"], 1)

    def test_three_day_snapshot_matures_earlier_without_relabelling_legacy(self):
        save_snapshot(self.rows, self.bundle, self.tmp.name, self.issued)
        newer = {**self.bundle, "model_version": "three", "horizon_days": 3}
        path = save_snapshot(self.rows, newer, self.tmp.name, self.issued)
        self.assertEqual(json.loads(path.read_text())["horizon_days"], 3)
        report = score_snapshots(self.tmp.name, observations(), "2025-01-11")
        self.assertEqual(report["pending_runs"], 1)
        self.assertEqual(len(report["evaluated_runs"]), 1)
        self.assertEqual(report["evaluated_runs"][0]["horizon_days"], 3)

    def test_outcome_scoring_prefers_calibrated_probability(self):
        self.rows["probability"] = .3  # Below threshold although the raw score is .8
        save_snapshot(self.rows, self.bundle, self.tmp.name, self.issued)
        report = score_snapshots(self.tmp.name, observations(), "2025-01-15")
        metric = report["evaluated_runs"][0]["metrics"]["classification"]
        self.assertEqual((metric["true_positive"], metric["false_negative"]), (0, 1))

    def test_historical_and_leaking_forecasts_are_never_scored(self):
        historical = save_snapshot(self.rows, self.bundle, self.tmp.name,
                                   datetime(2025, 2, 1, tzinfo=timezone.utc))
        data = observations()
        self.assertEqual(score_snapshots(self.tmp.name, data, "2025-02-01")["historical_runs_excluded"], 1)
        record = json.loads(historical.read_text())
        record.update(prospective=True, issued_at=self.issued.isoformat(), training_labels_through="2025-01-10")
        historical.write_text(json.dumps(record))
        self.assertEqual(score_snapshots(self.tmp.name, data, "2025-02-01")["invalid_runs_excluded"], 1)

    def test_invalid_snapshot_cannot_leave_a_partial_record(self):
        self.rows["detection_score"] = np.nan
        with self.assertRaises(ValueError):
            save_snapshot(self.rows, self.bundle, self.tmp.name, self.issued)
        self.assertEqual(list(Path(self.tmp.name).glob("*.json")), [])


if __name__ == "__main__":
    unittest.main()
