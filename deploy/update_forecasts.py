"""Stage a release archive, then replace forecasts with a brief app restart.

Run on the Linux server: python3 deploy/update_forecasts.py /path/to/release.tar.gz
Only outputs are installed; code, data and credentials are left in place.
"""

import argparse
import fcntl
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

if __package__:
    from .package import ROOT, validate_outputs
else:
    from package import ROOT, validate_outputs


def stage_outputs(archive_path, staging):
    with tarfile.open(archive_path, "r:gz") as archive:
        # Extract only regular output assets. Reject traversal and links before
        # extraction, even for members we would otherwise skip.
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe archive entry: {member.name}")
            if name.parts[:2] != ("wildfire", "outputs"):
                continue
            member.mode = 0o755 if member.isdir() else 0o644
            archive.extract(member, staging, filter="data")
    outputs = staging / "wildfire/outputs"
    validate_outputs(outputs)
    return outputs


def install_outputs(staged, root, run):
    """Stop readers before switching directories; recreate Docker's bind mount."""
    current = root / "outputs"
    if current.is_symlink() or not current.is_dir():
        raise ValueError("Expected an existing, real outputs directory")
    backups = root / ".forecast-backups"
    backups.mkdir(exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix="release-", dir=backups)) / "outputs"
    compose = ["docker", "compose", "-f", str(root / "compose.yaml")]
    start = compose + ["up", "-d", "--no-build", "--no-deps", "--force-recreate",
                       "--wait", "--wait-timeout", "120", "app"]
    run(compose + ["stop", "app"], cwd=root, check=True)
    try:
        current.rename(backup)
        staged.rename(current)
        run(start, cwd=root, check=True)
    except BaseException:
        if backup.exists():
            run(compose + ["stop", "app"], cwd=root, check=True)
            if current.exists():
                current.rename(backup.parent / "failed-outputs")
            backup.rename(current)
        # A fresh container must mount the restored directory, not the failed one.
        run(start, cwd=root, check=True)
        raise
    return backup


def update(archive_path, root=ROOT, run=subprocess.run):
    # Lock spans validation and activation so two uploads cannot interleave.
    with (root / ".forecast-update.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(prefix=".forecast-stage-", dir=root) as tmp:
            staged = stage_outputs(archive_path, Path(tmp))
            return install_outputs(staged, root, run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    backup = update(args.archive.resolve())
    print(f"Forecast update is healthy. Previous outputs retained at {backup}")


if __name__ == "__main__":
    main()
