import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import requests

from src.alerts import DATASETS, FireAlerts, parse_detections

NOW = datetime(2026, 11, 18, 12, tzinfo=timezone.utc)
HEADER = "latitude,longitude,acq_date,acq_time,confidence,frp\n"
BODY = HEADER + "13.5,101,2026-11-18,1130,h,12.3\n"


def response(body=BODY):
    result = Mock(text=body)
    result.raise_for_status.return_value = None
    return result


class AlertTests(unittest.TestCase):
    def test_country_filter_runs_before_limit_and_summary_counts(self):
        from src.geography import in_thailand
        service = FireAlerts()
        self.refresh(service, [response(BODY + "0,0,2026-11-18,1140,h,2\n"), response(HEADER)])
        feed = service.snapshot(NOW, limit=1, location_filter=in_thailand)
        self.assertEqual(feed["total"], 1)
        self.assertEqual(feed["by_satellite"], {"NOAA20": 1})
        self.assertEqual(feed["alerts"][0]["latitude"], 13.5)

    def setUp(self):
        self.env = patch.dict(os.environ, {"FIRMS_API_KEY": "test-private-key"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def refresh(self, service, effects):
        with patch("src.alerts.requests.get", side_effect=effects), patch("src.alerts.utc_now", return_value=NOW):
            service.refresh()

    def test_parser_deduplicates_and_interprets_utc(self):
        items = parse_detections(BODY + BODY.splitlines()[1] + "\n", DATASETS[0])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["observed_at"], "2026-11-18T11:30:00+00:00")
        self.assertEqual(items[0]["confidence"], "High")
        self.assertEqual(items[0]["id"], parse_detections(BODY, DATASETS[0])[0]["id"])

    def test_invalid_rows_and_nonfinite_values(self):
        body = HEADER + "nan,101,2026-11-18,1130,h,1\n13,101,bad,1130,h,1\n13,101,2026-11-18,0030,n,nan\n"
        items = parse_detections(body, DATASETS[0])
        self.assertEqual(len(items), 1)
        self.assertIsNone(items[0]["frp_mw"])
        with self.assertRaises(ValueError):
            parse_detections("Invalid MAP_KEY", DATASETS[0])

    def test_missing_key_never_calls_nasa(self):
        with patch.dict(os.environ, {"FIRMS_API_KEY": "YOUR_FIRMS_API_KEY_HERE"}), patch("src.alerts.requests.get") as get:
            service = FireAlerts()
            service.refresh()
            get.assert_not_called()
        self.assertEqual(service.snapshot(NOW)["status"], "missing_key")

    def test_recent_window_excludes_old_and_future_rows(self):
        service = FireAlerts()
        body = BODY + "13,101,2026-11-17,1159,n,2\n13,101,2026-11-19,1100,n,2\n"
        self.refresh(service, [response(body), response(HEADER)])
        feed = service.snapshot(NOW)
        self.assertEqual(feed["total"], 1)
        self.assertFalse(feed["stale"])
        self.assertEqual(feed["status"], "ok")
        self.assertTrue(service.snapshot(NOW + timedelta(hours=1))["stale"])
        self.assertEqual(service.snapshot(NOW + timedelta(days=2))["total"], 0)

    def test_successful_empty_response_is_not_failure(self):
        service = FireAlerts()
        self.refresh(service, [response(HEADER), response(HEADER)])
        self.assertEqual(service.snapshot(NOW)["status"], "ok")
        self.assertEqual(service.snapshot(NOW)["total"], 0)

    def test_failure_retains_cache_without_exposing_key(self):
        service = FireAlerts()
        self.refresh(service, [response(), response()])
        original = service.snapshot(NOW)
        self.refresh(service, [requests.Timeout("test-private-key"), response("Invalid MAP_KEY")])
        feed = service.snapshot(NOW)
        self.assertEqual(feed["status"], "error")
        self.assertTrue(feed["stale"])
        self.assertEqual(feed["alerts"], original["alerts"])
        self.assertNotIn("test-private-key", str(feed))

    def test_partial_failure_keeps_other_source_cache(self):
        service = FireAlerts()
        self.refresh(service, [response(), response()])
        self.refresh(service, [response(HEADER), requests.Timeout()])
        feed = service.snapshot(NOW)
        self.assertEqual(feed["status"], "partial")
        self.assertEqual(feed["failed_sources"], [DATASETS[1]])
        self.assertEqual(feed["total"], 1)
        self.assertTrue(feed["stale"])

    def test_feed_returns_newest_hundred_with_full_count(self):
        body = HEADER + "".join(f"13,{100 + i / 1000},2026-11-18,1130,n,2\n" for i in range(120))
        service = FireAlerts()
        self.refresh(service, [response(body), response(HEADER)])
        feed = service.snapshot(NOW)
        self.assertEqual(feed["total"], 120)
        self.assertEqual(len(feed["alerts"]), 100)
        self.assertEqual(feed["by_satellite"], {"NOAA20": 120})
        self.assertEqual(len(service.snapshot(NOW, limit=110)["alerts"]), 110)
        self.assertEqual(len(service.snapshot(NOW, limit=10**6)["alerts"]), 120)


if __name__ == "__main__":
    unittest.main()
