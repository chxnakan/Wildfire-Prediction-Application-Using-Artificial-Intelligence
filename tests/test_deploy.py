import fcntl
import gzip
import io
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from deploy.package import write_archive
from deploy.update_forecasts import update


SNAPSHOT = "2026-01-18_0123456789ab.json"


def outputs(root, marker):
    output = root / "outputs"
    documents = {
        "metadata/dataset_info.json": {"release": marker},
        "evaluation/metrics.json": {},
        "riskmap/fire_dates_all.geojson": {"type": "FeatureCollection", "features": []},
        "riskmap/latest.json": {"snapshot": SNAPSHOT},
        f"forecasts/{SNAPSHOT}": {"release": marker},
    }
    for name, value in documents.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        path.chmod(0o600)
    coordinates = output / "features/observation_coordinates.csv.gz"
    coordinates.parent.mkdir()
    with gzip.open(coordinates, "wt") as stream:
        stream.write("date,latitude,longitude\n2026-01-17,15,100\n")
    return output


def marker(root):
    return json.loads((root / "outputs/metadata/dataset_info.json").read_text())["release"]


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "source"
        self.server = self.root / "server"
        self.source.mkdir()
        self.server.mkdir()
        outputs(self.source, "new")
        outputs(self.server, "old")
        self.archive = self.root / "release.tar.gz"

    def package(self):
        write_archive(self.archive, self.source, ("outputs",))

    def test_package_rejects_file_and_directory_symlinks_without_partial_archive(self):
        secret = self.root / "secret.txt"
        secret.write_text("must not be packaged")
        for target in (secret, self.server):
            with self.subTest(target=target):
                link = self.source / "outputs/linked"
                link.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "Refusing to package link"):
                    self.package()
                self.assertFalse(self.archive.exists())
                link.unlink()

    def test_package_rejects_hard_links(self):
        (self.source / "outputs/duplicate.json").hardlink_to(
            self.source / "outputs/metadata/dataset_info.json")
        with self.assertRaisesRegex(ValueError, "Refusing to package link"):
            self.package()
        self.assertFalse(self.archive.exists())

    def test_existing_release_is_never_overwritten_or_removed(self):
        self.archive.write_bytes(b"previous release")
        with self.assertRaises(FileExistsError):
            self.package()
        self.assertEqual(self.archive.read_bytes(), b"previous release")

    def test_package_excludes_credentials_and_normalizes_permissions(self):
        for name in (".env", ".env.production", ".cloudflare.env", "private.key"):
            (self.source / "outputs" / name).write_text("secret")
        self.package()
        with tarfile.open(self.archive) as archive:
            names = archive.getnames()
            self.assertNotIn("wildfire/outputs/.env", names)
            self.assertFalse(any("production" in name or name.endswith(".key") for name in names))
            self.assertNotIn("wildfire/outputs/.cloudflare.env", names)
            self.assertEqual(archive.getmember("wildfire/outputs/metadata/dataset_info.json").mode, 0o644)

    def test_activation_stops_readers_before_switch_and_recreates_bind_mount(self):
        self.package()
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            self.assertEqual(marker(self.server), "old" if "stop" in command else "new")

        backup = update(self.archive, self.server, run)
        self.assertEqual(marker(self.server), "new")
        self.assertEqual(marker(backup.parent), "old")
        self.assertIn("stop", calls[0])
        self.assertIn("--force-recreate", calls[1])
        self.assertIn("--wait", calls[1])
        self.assertFalse(list(self.server.glob(".forecast-stage-*")))

    def test_unhealthy_release_restores_previous_files_and_container(self):
        self.package()
        seen = []

        def run(command, **kwargs):
            if "up" in command:
                seen.append(marker(self.server))
                if len(seen) == 1:
                    raise subprocess.CalledProcessError(1, command)

        with self.assertRaises(subprocess.CalledProcessError):
            update(self.archive, self.server, run)
        self.assertEqual(seen, ["new", "old"])
        self.assertEqual(marker(self.server), "old")
        self.assertTrue(list(self.server.glob(".forecast-backups/*/failed-outputs")))

    def test_invalid_release_never_stops_the_existing_app(self):
        path = self.source / "outputs/metadata/dataset_info.json"
        path.write_text('{"release":')
        self.package()
        calls = []
        with self.assertRaises(ValueError):
            update(self.archive, self.server, lambda *args, **kwargs: calls.append(args))
        self.assertEqual(calls, [])
        self.assertEqual(marker(self.server), "old")

    def test_truncated_coordinates_never_stop_existing_app(self):
        path = self.source / "outputs/features/observation_coordinates.csv.gz"
        path.write_bytes(path.read_bytes()[:-8])
        self.package()
        calls = []
        with self.assertRaises(EOFError):
            update(self.archive, self.server, lambda *args, **kwargs: calls.append(args))
        self.assertEqual(calls, [])
        self.assertEqual(marker(self.server), "old")

    def test_unsafe_archive_members_are_rejected_before_activation(self):
        for kind in ("traversal", "symlink", "hardlink"):
            with self.subTest(kind=kind):
                with tarfile.open(self.archive, "w:gz") as archive:
                    info = tarfile.TarInfo("../escape" if kind == "traversal" else "wildfire/outputs/link")
                    if kind != "traversal":
                        info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                        info.linkname = "/etc/passwd"
                    archive.addfile(info, io.BytesIO())
                calls = []
                with self.assertRaises(ValueError):
                    update(self.archive, self.server, lambda *args, **kwargs: calls.append(args))
                self.assertEqual(calls, [])
                self.assertEqual(marker(self.server), "old")

    def test_concurrent_update_is_rejected(self):
        self.package()
        with (self.server / ".forecast-update.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                update(self.archive, self.server, lambda *args, **kwargs: self.fail("Docker was called"))


if __name__ == "__main__":
    unittest.main()
