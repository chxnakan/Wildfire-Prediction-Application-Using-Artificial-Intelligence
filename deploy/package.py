"""Build a first-install archive including assets that Git deliberately ignores."""

import argparse
import gzip
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = (
    "Dockerfile", ".dockerignore", "compose.yaml", "compose.cloudflare.yaml",
    ".env.example", "requirements.txt", "README.md", "AGENTS.md",
    "src", "frontend", "deploy", "data", "outputs",
)
# Training-only caches and build folders are never read by the web server.
SKIP = {"__pycache__", ".git", ".venv", "backups", "research_build", "weather_build", "three_day_build", "cache"}


def archive_filter(info):
    path = Path(info.name)
    if any(part in SKIP for part in path.parts):
        return None
    if path.name.startswith(".env") and path.name != ".env.example":
        return None
    if path.name in {".cloudflare.env"} or path.suffix in {".pyc", ".pem", ".key"}:
        return None
    if info.issym() or info.islnk():
        raise ValueError(f"Refusing to package link: {info.name}")
    if not (info.isfile() or info.isdir()):
        raise ValueError(f"Refusing to package special file: {info.name}")
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    info.mode = 0o755 if info.isdir() else 0o644
    return info


def validate_outputs(output):
    """Check complete serving assets before packaging or stopping a running app."""
    required = ("metadata/dataset_info.json", "evaluation/metrics.json",
                "riskmap/fire_dates_all.geojson", "riskmap/latest.json",
                "features/observation_coordinates.csv.gz")
    for name in required:
        if not (output / name).is_file():
            raise ValueError(f"Missing deployment asset: {name}")
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Refusing linked output: {path.name}")
        if path.is_file() and path.suffix in {".json", ".geojson"}:
            json.loads(path.read_text())
    latest = json.loads((output / "riskmap/latest.json").read_text())["snapshot"]
    if not isinstance(latest, str) or Path(latest).name != latest:
        raise ValueError("Invalid latest forecast snapshot name")
    if not (output / "forecasts" / latest).is_file():
        raise ValueError("The latest forecast snapshot is missing")
    # Reading to EOF verifies gzip integrity, including an interrupted upload.
    with gzip.open(output / "features/observation_coordinates.csv.gz", "rb") as stream:
        while stream.read(1024 * 1024):
            pass


def write_archive(destination, root=ROOT, names=INCLUDE):
    # Do not dereference: the filter must see links before tar reads their targets.
    # Remove failed partial archives, but never remove an already-existing release.
    with destination.open("xb") as stream:
        try:
            with tarfile.open(fileobj=stream, mode="w:gz", dereference=False) as archive:
                for name in names:
                    archive.add(root / name, arcname=f"wildfire/{name}", filter=archive_filter)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/wildfire-deploy.tar.gz")
    args = parser.parse_args()
    try:
        validate_outputs(ROOT / "outputs")
    except (ValueError, KeyError, OSError, EOFError) as exc:
        parser.error(str(exc))
    destination = args.output.resolve()
    for name in INCLUDE:
        source = (ROOT / name).resolve()
        if destination == source or (source.is_dir() and destination.is_relative_to(source)):
            parser.error("Archive destination must be outside the packaged source directories")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_archive(destination)
    print(f"Created {destination} ({destination.stat().st_size / 1024**2:.1f} MiB)")
    print("FIRMS and Cloudflare credentials are excluded; configure them on the server.")


if __name__ == "__main__":
    main()
