FROM python:3.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home app
COPY deploy/requirements.lock ./requirements.lock
RUN python -m pip install -r requirements.lock
COPY src ./src
COPY frontend ./frontend
USER app
EXPOSE 8000
# One process owns the in-memory FIRMS cache and its background poller.
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-proxy-headers", \
     "--no-server-header", "--limit-concurrency", "100"]
