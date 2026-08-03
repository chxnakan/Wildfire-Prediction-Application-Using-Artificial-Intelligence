# Production deployment — Railway

The supported topology is **one Railway web service**, one PostgreSQL service,
and one persistent volume:

```text
GitHub push
    │
    ▼
Docker build (Node 22 SPA build → Python 3.13 runtime)
    │
    ▼
FastAPI web service
  ├─ React SPA + API
  ├─ FIRMS sync / prediction refresh
  ├─ retraining quality-gate check
  └─ GISTDA polling
    │
    ├──────── PostgreSQL (audit data + advisory locks)
    └──────── /srv volume (data + promoted model artifacts)
```

Do **not** create the optional `railway.cron.json` service for the normal
deployment. Railway volumes attach to one service; a second service would not
see the web service's `/srv` volume and its trained artifact would not reach the
serving process. The in-process APScheduler jobs already provide guarded
automation.

## Files used

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the SPA and production Python runtime |
| `railway.json` | Canonical Railway build/health configuration |
| `railway.toml` | Mirrors the canonical web-service settings and documents variables |
| `scripts/start.sh` | Seeds a new volume, then starts Uvicorn |
| `src/scheduler.py` | Sync, predict, retrain-check, and GISTDA schedules |

## One-time Railway setup

1. Create a project from this GitHub repository.
2. Add PostgreSQL. Railway injects `DATABASE_URL`.
3. Create a volume and attach it to the **web service** at `/srv`.
4. Keep one web replica unless PostgreSQL is configured. With PostgreSQL,
   advisory locks prevent duplicate poll/train work across replicas.
5. Set:

   ```text
   FIRMS_API_KEY=<NASA FIRMS map key>
   ADMIN_TOKEN=<long random secret>
   APP_ENV=production
   ```

6. Recommended runtime variables:

   ```text
   SCHEDULER_ENABLED=true
   SCHEDULER_SYNC_ENABLED=true
   SCHEDULER_RETRAIN_ENABLED=true
   SYNC_INTERVAL_MINUTES=10
   RETRAIN_EVERY_DAYS=3
   RETRAIN_CHECK_HOURS=6
   TRAIN_ARGS=--quick
   PREDICT_MAX_HISTORY_DAYS=180

   GISTDA_ENABLED=true
   GISTDA_POLL_SECONDS=300
   GISTDA_RETENTION_DAYS=7
   GISTDA_REQUEST_TIMEOUT_SECONDS=15
   GISTDA_MAX_RETRIES=4
   GISTDA_STALE_AFTER_SECONDS=900
   ```

The two per-job switches default to `true`, so existing production deployments
keep their current behaviour. They allow a GISTDA-only presentation process
without coupling live observations to FIRMS sync or retraining; local demos
should use `./run.sh --demo` so the safe combination is applied atomically.

The Dockerfile supplies:

```text
OUTPUT_DIR=/srv/outputs
DATA_DIR=/srv/data
RAW_DIR=/srv/data/raw
FIRMS_DIR=/srv/data/firms
FIRMS_PATH=/srv/data/firms/firms_all.parquet
WEATHER_DIR=/srv/data/weather
WEB_DIST_DIR=/app/web/dist
```

`railway.json` uses `/health` with a 300-second deployment health timeout.
The endpoint intentionally returns HTTP 200 with `ready:false` when a fresh
volume has not yet produced a model. This keeps the service alive while
automation accumulates data; the dashboard must still disclose degraded state.

## Bootstrap behavior

The image includes the repository's current `data/` and `outputs/` as bootstrap
content. `scripts/start.sh` copies them only when the corresponding `/srv`
directory is empty. Later deploys do not overwrite the persistent artifact.

For an empty deployment with no committed promoted model:

- the API boots in degraded mode;
- map/prediction endpoints remain unavailable or empty;
- sync jobs collect FIRMS data;
- training promotes a candidate only after its quality gates pass.

For tomorrow's presentation, deploy an image that contains the already-tested
promoted artifact or verify that `/srv/outputs` is populated before going live.

## Pre-deploy validation

```bash
cd web
npm ci
npm run typecheck
npm test
npm run build
cd ..

source .venv/bin/activate
python -m compileall -q src scripts
SCHEDULER_ENABLED=false python -m pytest

git diff --check
docker build -t firewatch-thailand .
docker run --rm -p 8000:8000 \
  --env-file .env \
  -e APP_ENV=production \
  -e SCHEDULER_ENABLED=false \
  firewatch-thailand
```

Then verify:

```bash
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/geojson >/dev/null
curl -I http://localhost:8000/
```

The dashboard should load, navigation should survive refresh on every hash
route, and the browser console should show no blocked API/weather requests.

## Operational checks

Use these endpoints:

| Endpoint | Check |
|---|---|
| `/health` | readiness, model load, data age, scheduler status |
| `/api/analytics/hotspots` | GISTDA feed state and data availability |
| `/live-hotspots` | cached live detections and freshness metadata |
| `/geojson` | current snapshot, thresholds, metrics, provenance |

Before a demo or incident review:

1. Confirm `base_date`, `latest_data_time`, and `generated_at`.
2. Confirm live-feed state is available/stale/unavailable, not merely `0`.
3. Open one map cell and verify weather source is labelled.
4. Open Predicted vs Actual and select a mature risk window.
5. Confirm held-out metrics and snapshot-audit counts are clearly described as
   different evidence.

## Static hosting caveat

`vercel.json` builds only the React SPA. It does not deploy FastAPI, model
artifacts, database, or scheduler. If Vercel is used for the SPA, set
`VITE_API_BASE` at **build time** to the HTTPS FastAPI origin and configure
that API's CORS policy. Same-origin Railway hosting is the supported and
simplest production path.

## Rollback

Railway can redeploy an earlier image, but the `/srv` volume is not rolled back
with application code. A bad candidate should normally never replace the
incumbent because promotion is quality-gated. If artifact rollback is needed,
restore a known promoted bundle to `/srv/outputs`, restart the service, and
verify `/health` plus `/geojson` before reopening access.
