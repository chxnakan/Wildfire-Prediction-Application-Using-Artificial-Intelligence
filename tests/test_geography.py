import unittest

import pandas as pd

from src.geography import in_ring, in_thailand, thailand_detections


class GeographyTests(unittest.TestCase):
    def test_thai_cities_and_islands_are_inside(self):
        for point in [(13.7563, 100.5018), (18.7883, 98.9853), (7.88, 98.39)]:
            self.assertTrue(in_thailand(*point), point)

    def test_nearby_foreign_cities_and_sea_are_outside(self):
        # Myawaddy is very close to the border: no nearest-province snapping.
        for point in [(16.689, 98.508), (17.9757, 102.6331), (6.12, 102.24), (12, 100)]:
            self.assertFalse(in_thailand(*point), point)

    def test_ring_and_daily_cell_filter(self):
        ring = [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]
        self.assertTrue(in_ring(1, 1, ring))
        self.assertFalse(in_ring(3, 1, ring))
        frame = pd.DataFrame({"lat_cell": [135, 135, 0], "lon_cell": [1010, 1010, 0], "fire_count": [2, 3, 99]})
        self.assertEqual(thailand_detections(frame, .1).fire_count.sum(), 5)
        self.assertTrue(thailand_detections(frame.iloc[:0], .1).empty)
