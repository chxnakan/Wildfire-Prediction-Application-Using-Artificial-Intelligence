# Railway image: same runtime as ../../Dockerfile, with the forecast files
# baked in because Railway has no bind mounts. Build context: repository root.
FROM python:3.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PROJECT_ROOT=/app \
    OUTPUT_DIR=/app/outputs \
    DATA_DIR=/app/data \
    OMP_NUM_THREADS=1
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home app
COPY deploy/requirements.lock ./requirements.lock
RUN python -m pip install -r requirements.lock
COPY src ./src
COPY frontend ./frontend
COPY deploy/railway/serving ./outputs
USER app
EXPOSE 8000
# Railway sets PORT. Only Railway's proxy can reach the container, so trust its
# X-Forwarded-* headers to keep redirects on https. One process owns the
# in-memory FIRMS cache and its background poller.
CMD ["sh", "-c", "exec python -m uvicorn src.api:app --host 0.0.0.0 --port \"${PORT:-8000}\" --workers 1 --proxy-headers --forwarded-allow-ips '*' --no-server-header --limit-concurrency 100"]
