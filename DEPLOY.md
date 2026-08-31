# Production deployment — Railway

The supported topology is **one Railway web service**, one PostgreSQL service,
and one persistent volume:

```text
GitHub push
    │
    ▼
Docker build (Node 22 SPA build → Python 3.12 runtime)
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

6. Confirm the source checkout materialises Git LFS. The serving feature
   parquet is an LFS object, not a normal Git blob. The Docker build now stops
   with an explicit error if it receives the small LFS pointer instead of the
   real Parquet file.

`ADMIN_TOKEN` is for trusted server-to-server/operator tooling only. Never put
it in `VITE_*`, browser storage, or frontend code: every visitor could read it.
The public capability endpoint therefore reports the production Alerts page as
preview-only. The repository has no SMS, LINE, or email provider; the local
development queue is an in-memory stub and is not delivery.

7. Recommended runtime variables:

   ```text
   SCHEDULER_ENABLED=true
   SCHEDULER_SYNC_ENABLED=true
   SCHEDULER_RETRAIN_ENABLED=true
   SYNC_INTERVAL_MINUTES=10
   RETRAIN_EVERY_DAYS=3
   RETRAIN_CHECK_HOURS=6
   TRAIN_ARGS=--quick --max-history-days 400
   PREDICT_MAX_HISTORY_DAYS=180

   GISTDA_ENABLED=true
   GISTDA_POLL_SECONDS=300
   GISTDA_RETENTION_DAYS=7
   GISTDA_REQUEST_TIMEOUT_SECONDS=15
   GISTDA_MAX_RETRIES=4
   GISTDA_STALE_AFTER_SECONDS=900
   ```

The two per-job switches default to `true`, so existing production deployments
with a populated `/srv/data` volume keep their current behaviour. On a brand
new Railway volume, initially set `SCHEDULER_SYNC_ENABLED=false` and
`SCHEDULER_RETRAIN_ENABLED=false` until a sufficiently long FIRMS history has
been restored to `/srv/data/firms`. A clean GitHub checkout contains directory
scaffolding but not the ignored local FIRMS/weather caches; allowing the first
one-day sync to rebuild features would discard the useful history in the
bootstrap feature artifact. GISTDA polling may remain enabled because it is an
independent observation layer and is not a model feature.

After the FIRMS cache is restored and checked, enable sync first, verify one
prediction refresh, and only then enable scheduled retraining. The switches
also allow a GISTDA-only presentation process without coupling live
observations to FIRMS sync or retraining; local demos should use
`./run.sh --demo` so the safe combination is applied atomically.

The Dockerfile supplies:

```text
APP_ENV=production
OUTPUT_DIR=/srv/outputs
DATA_DIR=/srv/data
RAW_DIR=/srv/data/raw
FIRMS_DIR=/srv/data/firms
FIRMS_PATH=/srv/data/firms/firms_all.parquet
WEATHER_DIR=/srv/data/weather
WEB_DIST_DIR=/app/web/dist
```

`railway.json` uses the strict `/health/ready` endpoint with a 300-second
deployment health timeout, so Railway does not route prediction traffic until
the served model and data bundle pass readiness checks. The diagnostic
`/health` endpoint still returns HTTP 200 with `ready:false` for an observable
but degraded process. The production Docker health check also uses
`/health/ready` because this image has no separate orchestrator readiness
probe.

## Bootstrap behavior

The image includes only the repository's version-controlled bootstrap files:
the promoted model, LFS-backed feature parquet, dataset metadata, risk GeoJSON,
Thailand boundary, tree-cover aggregate, and empty cache directories. Local
raw FIRMS/weather caches and generated reports are excluded from the Docker
context so local and Railway builds use the same inputs. `scripts/start.sh`
seeds the four promoted outputs as one artifact set when none of them exists.
A volume containing only logs or state markers therefore bootstraps correctly;
a complete promoted set is preserved. If only part of that set exists, startup
fails explicitly rather than mixing model, features, metadata, and GeoJSON from
different versions. Boundary and tree-cover reference files can be filled
independently because they are not a promoted model bundle.

The supported production image must contain a complete, validated bootstrap
bundle; the Docker build fails if it is missing, malformed, or still an LFS
pointer. On an empty `/srv` volume, that complete set is seeded before Uvicorn
starts. Before tomorrow's presentation, also verify that an existing volume
contains all four promoted files and that `/health/ready` returns HTTP 200.

## Pre-deploy validation

```bash
git lfs install
git lfs pull --include=outputs/features/full_features.parquet
git lfs fsck

# Must report Parquet data, not "ASCII text" (an LFS pointer).
file outputs/features/full_features.parquet

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
curl -fsS http://localhost:8000/health/ready
curl -fsS http://localhost:8000/geojson >/dev/null
curl -I http://localhost:8000/
```

The dashboard should load, navigation should survive refresh on every hash
route, and the browser console should show no blocked API/weather requests.

## Operational checks

Use these endpoints:

| Endpoint | Check |
|---|---|
| `/health` | liveness plus diagnostic model, data-age, and scheduler state |
| `/health/ready` | strict HTTP 200/503 gate for prediction traffic |
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
6. Open Alerts and confirm it says **preview only** in production. Do not claim
   that a prepared message was sent; no delivery provider is implemented.

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
verify `/health/ready` plus `/geojson` before reopening access.
