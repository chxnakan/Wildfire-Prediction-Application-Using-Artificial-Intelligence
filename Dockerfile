# syntax=docker/dockerfile:1.7

# =============================================================================
# Stage 1 — build the Vite + React + TypeScript SPA
# =============================================================================
FROM node:22-alpine AS web-build

WORKDIR /web

# Copy dependency manifests first so Docker can cache npm dependencies.
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY web/ ./
RUN npm run build


# =============================================================================
# Stage 2 — Python runtime serving FastAPI + the built SPA
# =============================================================================
FROM python:3.12-slim-bookworm AS runtime

# LightGBM needs libgomp; rasterio/pyproj use GDAL/PROJ runtime libraries.
# curl is used by the container health check.
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
        libgomp1 libexpat1 libgdal32 libproj25 \
        ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_ENV=production \
    OUTPUT_DIR=/srv/outputs \
    DATA_DIR=/srv/data \
    RAW_DIR=/srv/data/raw \
    FIRMS_DIR=/srv/data/firms \
    FIRMS_PATH=/srv/data/firms/firms_all.parquet \
    WEATHER_DIR=/srv/data/weather \
    WEB_DIST_DIR=/app/web/dist

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY .env.example /app/.env.example

# Bootstrap artifacts let a fresh persistent volume start in a useful state.
# scripts/start.sh seeds the four promoted outputs as one atomic version set;
# it preserves a complete mounted bundle and rejects a partial one.
COPY outputs/ /app/bootstrap/outputs/
COPY data/ /app/bootstrap/data/
COPY scripts/start.sh /usr/local/bin/start.sh

# Railway's Git source snapshot does not materialise Git-LFS objects.  Its
# injected commit SHA lets us retrieve the *exact* public GitHub object that
# belongs with the model/metadata/GeoJSON copied above.  A normal local Docker
# build already has the real parquet and never makes this download.
#
# Railway documents that build-time variables require an ARG declaration, and
# RAILWAY_GIT_COMMIT_SHA is populated for GitHub-triggered deployments.  The
# URL deliberately uses the immutable commit rather than main, so a concurrent
# push can never mix feature data from a newer bundle into this image.
ARG RAILWAY_GIT_COMMIT_SHA
RUN python - <<'PY'
import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path

feature = Path("/app/bootstrap/outputs/features/full_features.parquet")
pointer_prefix = b"version https://git-lfs.github.com/spec/v1"
with feature.open("rb") as handle:
    prefix = handle.read(512)

if prefix.startswith(pointer_prefix):
    match = re.search(
        rb"oid sha256:([0-9a-f]{64})\nsize (\d+)\n?", prefix
    )
    if match is None:
        raise SystemExit("malformed Git-LFS pointer for bootstrap feature artifact")
    expected_digest = match.group(1).decode("ascii")
    expected_size = int(match.group(2))
    revision = os.environ.get("RAILWAY_GIT_COMMIT_SHA", "").strip()
    if not revision:
        raise SystemExit(
            "full_features.parquet is a Git-LFS pointer, but no "
            "RAILWAY_GIT_COMMIT_SHA was supplied to resolve it"
        )

    url = (
        "https://github.com/0Maruz/Wildfire-Prediction-Project/raw/"
        f"{revision}/outputs/features/full_features.parquet"
    )
    with tempfile.NamedTemporaryFile(
        prefix="full_features.", suffix=".parquet", dir=feature.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        subprocess.run(
            [
                "curl", "--fail", "--location", "--retry", "4",
                "--retry-all-errors", "--connect-timeout", "20",
                "--output", str(temporary), url,
            ],
            check=True,
        )
        actual_size = temporary.stat().st_size
        if actual_size != expected_size:
            raise SystemExit(
                "downloaded bootstrap feature size mismatch: "
                f"expected {expected_size}, got {actual_size}"
            )
        with temporary.open("rb") as downloaded:
            digest = hashlib.file_digest(downloaded, "sha256").hexdigest()
        if digest != expected_digest:
            raise SystemExit(
                "downloaded bootstrap feature checksum mismatch: "
                f"expected {expected_digest}, got {digest}"
            )
        os.replace(temporary, feature)
        print(f"materialised Git-LFS bootstrap feature ({actual_size:,} bytes)")
    finally:
        temporary.unlink(missing_ok=True)
PY

# Such an image would otherwise build successfully but fail only after
# deployment when PyArrow tries to open a pointer as Parquet. Validate all
# serving-critical bootstrap artifacts while dependencies are available.
RUN python - <<'PY'
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, "/app/src")
from train import load_model_bundle

root = Path("/app/bootstrap")
feature = root / "outputs/features/full_features.parquet"
if not feature.is_file():
    raise SystemExit(f"missing bootstrap feature artifact: {feature}")
with feature.open("rb") as handle:
    feature_prefix = handle.read(80)
if feature_prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
    raise SystemExit(
        "full_features.parquet remains a Git-LFS pointer after bootstrap "
        "materialisation"
    )
parquet = pq.ParquetFile(feature)
if parquet.metadata.num_rows < 1:
    raise SystemExit("bootstrap feature parquet contains no rows")

model = root / "outputs/models/lgbm_fire_date_model.pkl"
if not model.is_file():
    raise SystemExit(f"missing bootstrap model artifact: {model}")
# Use the application's compatibility loader: older promoted ensembles were
# pickled as __main__._EnsembleRegressor and deliberately cannot be opened by
# a bare joblib.load call.
load_model_bundle(str(model))

for relative in (
    "outputs/metadata/dataset_info.json",
    "outputs/riskmap/fire_dates_all.geojson",
):
    path = root / relative
    with path.open(encoding="utf-8") as handle:
        json.load(handle)

print(f"validated bootstrap artifacts ({parquet.metadata.num_rows:,} feature rows)")
PY

COPY --from=web-build /web/dist /app/web/dist

# The API does not need root. /srv remains writable for the mounted volume.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
 && mkdir -p /srv/outputs /srv/data/raw /srv/data/firms /srv/data/weather \
 && chmod +x /usr/local/bin/start.sh \
 && chown -R appuser:appuser /app /srv
USER appuser

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health/ready" || exit 1

ENTRYPOINT ["/usr/local/bin/start.sh"]
