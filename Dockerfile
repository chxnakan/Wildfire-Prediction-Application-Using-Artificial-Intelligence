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
FROM python:3.13-slim-bookworm AS runtime

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
# scripts/start.sh copies them only when the mounted destination is empty.
COPY outputs/ /app/bootstrap/outputs/
COPY data/ /app/bootstrap/data/
COPY scripts/start.sh /usr/local/bin/start.sh

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
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/start.sh"]
