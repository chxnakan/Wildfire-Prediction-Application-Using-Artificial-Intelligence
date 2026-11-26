"""Copy the forecast files the web server reads into deploy/railway/serving/.

Railway builds from the Git repository, where outputs/ is mostly ignored, so
this small serving subset is committed instead. Run after each training run:

    python deploy/railway/build_bundle.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from package import ROOT, validate_outputs  # noqa: E402

BUNDLE = Path(__file__).resolve().parent / "serving"
# Everything src/api.py reads from OUTPUT_DIR; training artefacts stay local.
FILES = ("metadata/dataset_info.json", "evaluation/metrics.json",
         "evaluation/prospective.json", "features/observation_coordinates.csv.gz")
DIRECTORIES = ("riskmap", "forecasts")
OPTIONAL = {"evaluation/prospective.json"}


def build(source=ROOT / "outputs", destination=BUNDLE):
    validate_outputs(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Assemble beside the destination, then swap, so a failure keeps the old bundle.
    staging = Path(tempfile.mkdtemp(prefix=".serving-", dir=destination.parent))
    try:
        for name in FILES:
            if name in OPTIONAL and not (source / name).is_file():
                continue
            (staging / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, staging / name)
        for name in DIRECTORIES:
            for path in sorted((source / name).iterdir()):
                if path.is_file() and not path.is_symlink():
                    (staging / name).mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, staging / name / path.name)
        validate_outputs(staging)
        if destination.exists():
            shutil.rmtree(destination)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def main():
    try:
        destination = build()
    except (ValueError, KeyError, OSError, EOFError) as exc:
        sys.exit(f"error: {exc}")
    size = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    print(f"Wrote {destination.relative_to(ROOT)} ({size / 1024**2:.1f} MiB). Commit and push it to redeploy.")


if __name__ == "__main__":
    main()
