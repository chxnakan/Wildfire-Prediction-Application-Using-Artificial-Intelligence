import json
import tempfile
import unittest
from pathlib import Path

from deploy.railway.build_bundle import build
from tests.test_deploy import SNAPSHOT, outputs


class RailwayBundleTests(unittest.TestCase):
    def test_copies_only_serving_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = outputs(root, "new")
            (source / "models").mkdir()
            (source / "models/model.txt").write_text("training only")
            bundle = build(source, root / "serving")
            names = {str(path.relative_to(bundle)) for path in bundle.rglob("*") if path.is_file()}
            self.assertEqual(names, {
                "metadata/dataset_info.json", "evaluation/metrics.json",
                "features/observation_coordinates.csv.gz", "riskmap/fire_dates_all.geojson",
                "riskmap/latest.json", f"forecasts/{SNAPSHOT}",
            })

    def test_incomplete_outputs_keep_previous_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = build(outputs(root / "old", "old"), root / "serving")
            broken = outputs(root / "new", "new")
            (broken / "riskmap/latest.json").unlink()
            with self.assertRaises(ValueError):
                build(broken, bundle)
            self.assertEqual(json.loads((bundle / "metadata/dataset_info.json").read_text())["release"], "old")
            self.assertEqual([path.name for path in root.iterdir() if path.name.startswith(".serving-")], [])


if __name__ == "__main__":
    unittest.main()
