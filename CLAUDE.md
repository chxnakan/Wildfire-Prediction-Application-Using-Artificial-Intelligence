# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Wildfire **imminence** prediction for Thailand (BBOX `96,4,107,22`). A LightGBM binary classifier answers *"will a fire occur in this 0.1° cell within the next 3 days?"* and outputs a probability. The probability is converted to a pseudo-days value (1–7) via a monotone mapping at inference time so the existing CRITICAL/HIGH/MEDIUM/LOW tier logic in `risk_map.py` keeps working unchanged.

### Why binary, not regression
Regression with MAE/MSE objectives consistently collapsed predictions toward the median (~3–4 days) — the available features (FIRMS hotspots + partial ERA5 weather) lack the signal needed to distinguish "fire in 1 day" from "fire in 7 days". Multiclass softmax did the same at probability scale. Binary "fire-in-3-days" is the framing the features can support — held-out test AUC ≈ 0.84 and the model meaningfully ranks cells. **Do not reintroduce regression or multiclass.**

### Hard rule: real data only
Every feature must come from a measurable source. Pipeline consumes:
- **NASA FIRMS VIIRS NRT** (required) — hotspots; powers fire/FRP/brightness/confidence + spatial-neighbor features.
- **Open-Meteo Archive API** (optional, no key) — real ECMWF ERA5 daily reanalysis. Activate via `python fetch_weather.py`. Cached to `data/weather/weather_cache.parquet`.
- **Hansen GFC** (optional, run-once) — tree cover baseline + recent loss via `python fetch_treecover.py`.
- **Calendar** — derived from each row's real date.

No synthetic / simulated / interpolated values. If a real source isn't available for a column, the column is simply not in `FEATURES`. Don't introduce fabricated defaults — when something is missing, it's missing.

## Common commands

Python entry points use bare imports (`from features import FEATURES`) so they must run from inside `src/`:

```bash
# 1. Pull latest VIIRS NRT hotspots → data/firms/firms_all.parquet
cd src && python fetch_firms.py [--days 1-10]

# 1b. (OPTIONAL) Pull ERA5 weather (idempotent; resume on rerun)
cd src && python fetch_weather.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--limit-cells N] [--quiet-hours 2-8]

# 1c./1d. (OPTIONAL, not wired into training) GISTDA hotspots / LULC polygons
cd src && python fetch_gistda_hotspots.py
cd src && python fetch_gistda_lulc.py    # run-once

# 2. Train (load → features → undersample → tune → ensemble → eval → persist)
cd src && python train.py [--n-iter 30] [--n-splits 5] [--n-ensemble 5]
#                          [--quick]   ← n_iter=12, n_splits=3 (~20 min on a laptop)
#                          [--fast]    ← n_iter=20, n_splits=3

# 3. Regenerate the GeoJSON map without retraining
cd src && python risk_map.py

# 4. Serve the API + SPA on :8000
cd src && uvicorn api:app --reload

# OR: end-to-end orchestrator at project root
./run.sh [--fresh] [--weather] [--quick|--fast] [--no-train] [--predict-only]
```

Frontend (React SPA) — rebuild after editing `web/src/`:
```bash
cd web
./node_modules/.bin/tsc -b           # type-check
./node_modules/.bin/vite build       # → web/dist/ (served by api.py at /)
```
**`frontend/` is a legacy vanilla-JS dashboard kept only for historical reference — do not edit it.** The React app in `web/src/` is what the FastAPI server serves.

Dependencies: `pip install -r requirements.txt` into `.venv/`. Requires `.env` with `FIRMS_API_KEY` (see `.env.example`). `scikit-optimize` and `matplotlib` are optional — train.py falls back gracefully if missing (RandomizedSearchCV instead of BayesSearchCV; PNG report skipped).

There is no Python test suite, linter, or build step.

## Architecture

### Module layout (in `src/`)

| File | Role |
|---|---|
| `fetch_firms.py` | Pulls VIIRS NRT hotspots from NASA FIRMS with retry/backoff. Accumulative parquet. |
| `fetch_weather.py` | Real ERA5 daily aggregates from Open-Meteo Archive. Idempotent, checkpoints, retries failed batches. Optional quiet-hours wait window. Fetches at ERA5-native 0.25° (collapses ~6 grid cells per API coordinate — free-tier quota is priced per coordinate × time). |
| `fetch_weather_cds.py` | Bulk ERA5 backfill via Copernicus CDS (needs free account + `~/.cdsapirc` + `pip install cdsapi xarray netcdf4`). Whole-bbox NetCDF per (variable, statistic, year) → same cache schema/units as fetch_weather.py; ET0 computed with FAO-56 from real ERA5 fields; never overwrites existing cache rows. Use for multi-year history; use fetch_weather.py for daily top-ups. |
| `fetch_treecover.py` | Hansen GFC `treecover2000` + `lossyear` rasters → per-cell 0.1° aggregates. |
| `fetch_gistda_*.py` | GISTDA hotspots / LULC. Not wired into training yet. |
| `io_utils.py` | Format-agnostic table I/O — dispatches on `.csv` ↔ `.parquet`. **Use these instead of `pd.read_csv` / `pd.to_csv`** so files stay swappable. |
| `data_loader.py` | Schema-validated load + clean + grid snap + daily aggregation + densification + urban filter + optional weather/tree-cover merge. |
| `urban_areas.py` | Curated ~40 Thai urban centers with hand-tuned exclusion radii. Used at training (drop hotspots) and inference (drop predictions + annotate nearest city). |
| `thailand_boundary.py` | 77-province GeoJSON merged → `is_in_thailand()` country mask + `find_province()` per-cell annotation. |
| `features.py` | Lag/rolling/calendar/spatial-neighbor/streak features. Owns `FEATURES_CORE` + `FEATURES_WEATHER`. Every rolling window applies `.shift(1)` BEFORE `.rolling()` (past-only — see `# CAUSAL` comments). |
| `thresholds.py` | **Reusable, leakage-safe alert-threshold selection** (shared by train.py + rolling_backtest.py). `f1_optimal_threshold` (HIGH), `recall_target_threshold` (WATCH, target `WATCH_TARGET_RECALL`=0.90, optional `WATCH_MIN_PRECISION` guard), `select_operational_thresholds` (HIGH+WATCH+MEDIUM-alias, optional per-month season-matched thresholds with insufficient-positive fallback, full calib metadata), `threshold_metrics` (P/R/F1/alert-rate/confusion), `topk_metrics` (P@K + recall captured), `prevalence_drift`. Chosen on **calib only**; enforces HIGH ≥ WATCH ≥ 0. |
| `model.py` | **Legacy from the multi-model era.** train.py no longer uses its `select_best` / multi-candidate helpers; the binary classifier path is self-contained in train.py. Kept around because some tools still import `evaluate()` / `EnsembleRegressor`. |
| `train.py` | End-to-end binary classifier pipeline (see "Training pipeline" below). |
| `inference.py` | **Single source of truth for user-facing model output.** `calibrated_probability()` (Platt-calibrated `predict_proba`), `probability_to_tier()` (High/Watch/Background from persisted `probability_tier_thresholds`), `score_cells()`, `snapshot_context()`. risk_map.py AND the API prediction endpoints score through this so `/predictions/*`, `/predict/location` and `/geojson` can never disagree. Never emits an exact fire date. |
| `security.py` | Production safety middleware: env-gated CORS allowlist (`configure_security`), in-process per-IP rate limiter, request-ID + structured JSON access logging, security headers + Leaflet/CARTO-compatible CSP. Driven by `APP_ENV` (`production` ⇒ fail-closed). |
| `risk_map.py` | Loads model + features, predicts for latest base date via `inference.calibrated_probability`, filters (history / urban / country / per-day cap), assigns absolute tiers, writes `outputs/riskmap/fire_dates_all.geojson` (append-mode; metadata carries `latest_data_time` + `generated_at`). |
| `api.py` | FastAPI. Prediction surfaces (`/predictions/today`, `/predictions/tiers`, `/predict/location`) return calibrated `probability` + `risk_tier` + shared context — **no exact fire date**; degraded model ⇒ stable `503 {"error":"model_unavailable"}`. `/predictions/timeline` + `/predictions/day/{n}` are deprecated (`410`, pseudo-date semantics removed). Split probes: `/health/live` (always 200) + `/health/ready` (503 until model+metadata+fresh GeoJSON). `/api/admin/*` + `/api/notify*` are privileged (`_check_admin`, fail-closed in production); notify stores counts only, never recipients/message. Mounts `web/dist/` at `/`. Boots degraded (no crash) with no model. |
| `db.py` | PostgreSQL persistence (hotspots / predictions / model runs / fetch log). **Every function is a silent no-op when `DATABASE_URL` is unset** — local dev needs no Postgres. psycopg3 dialect, `ON CONFLICT DO NOTHING` dedupe. |
| `sync_worker.py` | One sync cycle: fetch FIRMS → detect new rows (parquet delta + DB rowcount) → if new, subprocess `train.py --predict-only` → always write `fetch_log`. Never raises. |
| `scheduler.py` | APScheduler wiring: `sync_job` every `SYNC_INTERVAL_MINUTES` (10), `retrain_check_job` every 6 h retrains when the model is older than `RETRAIN_EVERY_DAYS` (3). Restart-safe (compares persisted timestamps, not interval clocks). `outputs/.train.lock` guards train vs. predict overlap. |

### Pipeline

```
data/raw/*.parquet ─┐
                    ├─► data_loader ─► features ─► train.py ─► outputs/models/lgbm_fire_date_model.pkl
data/firms/        ─┘    (densify)    (CAUSAL                       │
firms_all.parquet                      lag/roll)                    ▼
                                                            outputs/features/full_features.parquet
                                                            outputs/metadata/dataset_info.json
                                                                       │
                                                                       ▼
                                                                risk_map.py
                                                                       │
                                                                       ▼
                                                outputs/riskmap/fire_dates_all.geojson
                                                                       │
                                                       ┌───────────────┴───────────────┐
                                                       ▼                               ▼
                                                    api.py                    web/src/* → web/dist/
                                                  (FastAPI)            (React SPA fetches /geojson)
```

### Training pipeline (binary classifier, memory-frugal)

`train.py` is self-contained — it does **not** use `model.py`'s multi-candidate helpers. Flow:

1. **Load + features** — `load_and_prepare` → `build_features`, densified **causally**: a cell enters the universe only from its first FIRMS detection onward (pre-first-detection rows would encode hindsight).
2. **Label + downcast + censor** — drop unused columns, cast to **float32**, build binary label `y = (days_until_fire ∈ {1..IMMINENT_DAYS=3})`, then **censor the dataset tail**: rows whose 3-day label window extends past the last observed date are removed from all supervised use (`days_ahead_observed < 3` — their "negative" labels would mean "future not observed").
3. **Date-grouped 4-way split with purge** — `eval_protocol.date_grouped_split` → train/val/calib/test on **whole calendar dates** (default 60/15/10/15 by date count) with a **7-day purge** dropped before each boundary. Disjointness + purge width are hard-asserted. The old row-based split let one date land in two splits (confirmed leakage) — never reintroduce it.
4. **Undersample negatives in TRAIN ONLY** at `NEG_TO_POS_RATIO=4`. val/calib/test keep **natural prevalence** — undersampling them corrupted every prevalence-sensitive metric (precision, AP, calibration) reported previously.
5. **Sample weights** (train only) — recency decay (halflife 45 d) × inverse class frequency × gentle multi-sat boost.
6. **LightGBM tuning** — `BayesSearchCV` (fallback `RandomizedSearchCV`) with `eval_protocol.purged_date_folds` (whole-date expanding-window CV, 7-day purge; sklearn's `TimeSeriesSplit(gap=7)` purges 7 *rows*, not days — banned). **Outer CV `n_jobs=1`, inner LGBM `n_jobs=-1`** — both -1 spawns cores² threads.
7. **Ensemble fit on train, early stopping on val** — `n_ensemble` seeds at best params, `eval_metric=["binary_logloss", "auc"]`. Wrapped in `_EnsembleRegressor`.
8. **Calibration + thresholds on calib** — Platt sigmoid fit on the calib partition (model never trained on it, natural prevalence). Deployment threshold = F1-optimal **on calib**; MEDIUM tier cutoff = threshold catching ≥85% of calib fires. Persisted as `probability_tier_thresholds`. Test data is never consulted for any parameter.
9. **Single test evaluation** — natural prevalence, thresholds locked: ROC-AUC, PR-AUC, **Brier**, ECE, precision/recall/F1 @ deployment threshold, P@top-K%, plus **baselines** (persistence fired-last-3d/7d, per-cell train-period climatology) for honest uplift claims.
10. **Persist** — model pkl + history + `dataset_info.json` (includes an `evaluation_protocol` audit block: split ranges, purge, prevalences, censored rows, what was selected where) + PNG report.
11. **risk_map.run()** refreshes the GeoJSON.

**Run the complete test suite** (protocol invariants, GISTDA pipeline, sync versioning, model promotion): `cd src && ../.venv/bin/python -m pytest tests -q`. All external APIs are mocked; no network or PostgreSQL needed.

### Model artifact = single pickle bundle + quality gates

`lgbm_fire_date_model.pkl` is a **bundle-v1** pickle `{"format": "bundle-v1", "model": _EnsembleRegressor, "metadata": {...}}` — model and feature contract travel together so one `os.replace()` promotes both atomically (the old two-file scheme tore on a mid-write kill, pairing a new model with a stale feature list). `dataset_info.json` is still written for the frontend but loaders prefer the bundle's embedded metadata. Legacy bare-model pkls still load.

Training does **not** automatically replace the trusted model. `train.py` STEP 9b gates: test AUC ≥ `MODEL_GATE_MIN_AUC` (0.65), test PR-AUC ≥ `MODEL_GATE_MIN_AP_UPLIFT`× prevalence (1.5), and head-to-head vs the incumbent **on the same untouched test period** when feature contracts match (reject if AUC drop > `MODEL_GATE_MAX_AUC_DROP`). Rejected candidates land in `models/rejected/`; the incumbent + its risk map keep serving. On promotion the incumbent is kept at `models/previous.pkl` (rollback) and the scheduler fires a callback that **hot-reloads** the running API (`api.reload_model_state`; manual lever: `POST /api/admin/reload-model`). A failed reload keeps the previous in-memory model.

### Sync = input-versioned, prediction-only, idempotent

`sync_worker.run_sync()` computes a deterministic input version (latest FIRMS acq time | row count | weather mtime | schema tag). Unchanged version ⇒ **no-op**. Changed ⇒ **exactly one** `train.py --predict-only` subprocess, guarded by a PostgreSQL advisory lock (replicas can't double-predict) and recorded in `prediction_runs`. GISTDA polling is a separate job and **never triggers prediction** — the model has no GISTDA features, so a GISTDA-only change would re-produce an identical snapshot. Prediction failure ⇒ version not remembered (next cycle retries) and the last snapshot keeps serving.

### `_EnsembleRegressor` — the inference contract risk_map.py and api.py rely on

The ensemble class in train.py is named `_EnsembleRegressor` for historical reasons; it wraps **binary classifiers**.

- `predict_proba(X)` → 1-D `np.array` of **calibrated** `P(fire in next 3 days)` (raw ensemble mean → a persisted Platt or isotonic calibrator fit on the calib split). This is the model's real output and what risk_map uses for tiers, ranking, and display. Legacy Platt bundles remain pickle-compatible.
- `predict(X)` → 1-D pseudo-days via `_prob_to_days_for_compat` — monotone-in-probability **display scaffolding only** (the day-selector buckets). The binary target cannot resolve day-of-fire; never present these as date forecasts.
- `feature_importances_` → averaged across ensemble.

Tier counts follow the **absolute** `probability_tier_thresholds` persisted at training time, not the pseudo-days mapping.

### Densification (required for correctness)

`data_loader.densify_active_cells()` expands the sparse hotspot frame into a dense (cell × day) grid, no-fire days filled with zeros — **causally**: each cell's rows start at its first detection (the universe at time t = cells that had burned by t; including earlier rows leaked "this cell will eventually burn"). This makes lag features (`fire_lag_1` = literally yesterday) and rolling windows correct. Documented limitation: never-burned cells are outside the prediction universe — the model cannot flag first-ever fires.

### CAUSAL feature audit (don't break)

`features.py` has a strict no-future-data rule:
- Every rolling window uses `_past_roll()`, which applies `.shift(1)` **before** `.rolling()`. Today's value is excluded from past aggregates.
- Every lag is `.shift(lag)` with `lag >= 1`.
- Trend features are `lag_1 − lag_k`, not `today − shift(k)`.
- Streak counters use shifted fire flags.
- Every feature has a `# CAUSAL` comment marking it audited.

The label `days_until_fire` is computed strictly from FUTURE rows, so any feature that accidentally touches future data is label leakage. **Do not move features to "today" data without re-auditing the label construction.**

### Feature contract

- `FEATURES_CORE` (~135 cols) — always present (including archive-era satellite coverage).
- `FEATURES_WEATHER` (~37 cols: 30 daily/lag/roll + 7 causal drought signals) — only emitted when ERA5 weather is present in the daily frame. Auto-dropped at training time when coverage falls below `MIN_WEATHER_COVERAGE` (default 20%) — sparse weather is more harmful than absent weather.
- `resolve_features(df)` → subset actually present.
- **`dataset_info.json["features"]` is the persisted contract** — risk_map.py / api.py prefer this list at runtime. Don't hardcode feature lists in those files.

If you add or remove features, delete `outputs/models/*.pkl` before retraining or the resolver will silently mispredict.

### Urgency tiers + risk_map filtering

Tiers use **absolute calibrated-probability cutoffs** from `dataset_info.json["probability_tier_thresholds"]`, selected on the calib split at training time via `thresholds.select_operational_thresholds` (never on test):

- **HIGH** = F1-optimal cutoff → **urgent alert** (precision-oriented).
- **WATCH** = lowest cutoff catching ≥ `WATCH_TARGET_RECALL` (default 0.90) of calibration fires → **monitoring list** (recall-oriented; expect more false positives). `MEDIUM` is a persisted **compatibility alias of WATCH**.
- **LOW** = 0.0 → no operational alert.

The two-threshold split exists because of **base-rate drift**: a single F1-optimal threshold picked on a high-prevalence calibration window collapses to ~13% recall on a low-prevalence test window. WATCH is robust to that; HIGH keeps precision. Training persists a **base-rate drift flag** (`test_metrics.prevalence_drift`: calib vs recent prevalence, ratio, `drift_flag`) surfaced in `/metrics`, the GeoJSON metadata, and a dashboard warning banner. Never force a fixed % of cells into HIGH — a genuinely low-risk day shows zero HIGH cells. Top-K rankings (`precision_at_top_K` / `recall_captured_top_K`) are **relative priority**, reported separately from the absolute tiers. Fallbacks: legacy artifacts without persisted cutoffs get per-snapshot percentile **priority ranking** (labelled relative); tiny snapshots (<10 cells) get fixed-domain days thresholds.

**Activation:** the new WATCH cutoff + drift metadata are written by `train.py` — activate with a retrain (`cd src && python train.py`), or validate cross-season behaviour first with `python rolling_backtest.py` (which now shares `thresholds.py`). Existing saved models keep working (HIGH/MEDIUM read as before; WATCH falls back to the MEDIUM alias).

Output semantics: `probability` = calibrated P(≥1 hotspot within 3 days). `days_until_fire` is a probability-rank display bucket; `predicted_fire_date` is the **risk-window end** (base_date + 3d, same for all cells) — the model never claims an exact fire date.

risk_map.py then runs four filters and persists the funnel:
1. **History** — min 3 fires in last 30d, min 3 in last 90d, min 3 fire-days/year. Cells failing all three are dropped (climatology-dominated).
2. **Urban** — drop cells inside curated city polygons + annotate `nearest_urban_area`.
3. **Country** — drop cells outside Thailand's land border (training data spans the full BBOX so the model learns regional patterns; the dashboard is Thailand-focused).
4. **Display cap** — within each bucket, keep top `MAX_CELLS_PER_DAY` (default 100) ranked **by model probability** (tie-break: 30d history). The suppressed count is persisted in GeoJSON metadata (`suppressed_by_display_cap`). Set `MAX_CELLS_PER_DAY=0` to disable.

### Frontend (React SPA in `web/`)

- `App.tsx` — top-level; fetches `/geojson` via relative path so the SPA can be served by any backend that mounts the GeoJSON.
- `components/MapView.tsx` — Leaflet map, predicted/observed/live-fire layers, marker popups. The predicted-cell popup leads with the **calibrated probability %** (persisted `probability` field; legacy snapshots fall back to inverting `raw_prediction`) + a Thai risk-tier headline + ground-truth-anchored interpretation. The date line shows the **risk-window end**, not a per-cell predicted date; the day selector is labelled as probability-rank bands.
- `components/AccuracyHero.tsx` — operator-facing performance card. Headline is **Recall** (e.g. 97%) = "of fires that actually happened, what fraction did we catch?". Sub-stats: letter grade A–D from ROC-AUC, uplift vs random (P@top-20% / `test_positive_rate`), false alarm rate, watch-list precision. All technical metrics live in InfoModal. Detects task via `metrics.task === "binary_fire_in_3d"` with a legacy regression fallback view.
- `components/InfoModal.tsx` — glossary modal, task-type-aware metric definitions.
- `components/Sidebar.tsx` — base-date / province filters, day selector, hit-rate, land-cover breakdown, CSV export.
- `types.ts` — TypeScript shapes mirroring what risk_map writes. `ValidationMetrics` includes both legacy keys (`mae_days`, `accuracy_within_1day`) and binary keys (`roc_auc`, `f1`, `precision_at_best_thr`, ...).

The legacy regression keys in `test_metrics` look terrible (MAE ~5 d, acc±1 ~6%) because they're computed on pseudo-days output, not real days. **Don't report them as the model's quality**. Always check `task` first.

### Path resolution & env

All entry points resolve paths via `BASE_DIR = parent-of-src`. `train.py` reads `.env` and resolves relative env values against `BASE_DIR`. Absolute paths pass through.

| Var | Default | Purpose |
|---|---|---|
| `FIRMS_API_KEY` | (required) | NASA FIRMS map key |
| `RAW_DIR` | `./data/raw` | Bulk archive of historical FIRMS exports |
| `FIRMS_PATH` | `./data/firms/firms_all.parquet` | NRT cache |
| `OUTPUT_DIR` | `./outputs` | Models / features / metadata / geojson |
| `GRID_SIZE` | `0.1` | Lat/lon snap resolution in degrees |
| `URBAN_FILTER_ENABLED` | `true` | Drop urban hotspots at training and inference |
| `URBAN_BUFFER_KM` | `0` | Extra km beyond each city's hand-tuned radius |
| `COUNTRY_FILTER_ENABLED` | `true` | Drop predicted cells outside Thailand at risk_map time |
| `STALE_WARN_DAYS` | `5` | Warn when latest FIRMS observation is older |
| `MIN_WEATHER_COVERAGE` | `0.20` | Below this, training auto-drops weather columns |
| `MIN_HISTORICAL_FIRES_FOR_DISPLAY` | `3` | risk_map filter: min fires in last 30d |
| `MIN_LONG_HISTORICAL_FIRES` | `3` | risk_map filter: min fires in last 90d |
| `MAX_CELLS_PER_DAY` | `100` | risk_map per-day cap; 0 disables |
| `MIN_FIRE_DAYS_PER_YEAR` | `3.0` | risk_map filter: annualized fire-day rate floor |
| `MAX_TRAIN_HISTORY_DAYS` | `0` | If >0, train.py keeps only this many latest days of densified data |
| `DATABASE_URL` | (unset) | PostgreSQL archive. Unset ⇒ db.py is a no-op; parquet/GeoJSON only |
| `SCHEDULER_ENABLED` | `true` | Start APScheduler jobs in api.py's lifespan |
| `SYNC_INTERVAL_MINUTES` | `10` | Fetch-and-predict cycle interval |
| `RETRAIN_EVERY_DAYS` | `3` | Retrain when the last successful run is older than this |
| `TRAIN_ARGS` | `--quick` | Args passed to the scheduled train.py subprocess |
| `ADMIN_TOKEN` | (unset) | Secret for `/api/admin/sync` + `/api/admin/retrain` (`X-Admin-Token` header). Unset ⇒ open (dev) |

### Automation / deployment (Railway + Vercel)

- Split deploy: Railway runs FastAPI + APScheduler + PostgreSQL; Vercel serves `web/dist/` with `VITE_API_BASE` pointing at the Railway URL. Config: `railway.toml` (has the full setup checklist in comments) + `vercel.json`.
- Railway needs a **Volume** (mount e.g. `/data`, set `OUTPUT_DIR=/data/outputs`, `FIRMS_PATH=/data/firms/firms_all.parquet`, `RAW_DIR=/data/raw`) — the filesystem is otherwise wiped on each deploy, losing the model + FIRMS cache.
- GeoJSON stays the serving format; PostgreSQL is the durable archive (hotspots, prediction snapshots, model-run history, fetch audit log). A DB outage never blocks the map refresh.
- `/api/automation/status` = one-call monitoring (scheduler next-run times, last fetch, last model run, row counts).

### Gotchas

- **OOM during training** = you didn't cast to float32, kept too much history, or `n_ensemble` too high. Recipe: float32 downcast → `--max-history-days` to bound the frame → ensemble ≤ 5 on a 22 GB laptop. **Do NOT "fix" memory by undersampling before the split** — that corrupts val/calib/test prevalence and was one of the leakage bugs removed in the protocol repair.
- **Tuning hangs / extreme load average** = `n_jobs=-1` on both BayesSearchCV outer AND LGBM inner spawns cores² threads. Keep outer `n_jobs=1` (sequential CV folds, parallelism inside the fit).
- **Test metrics look terrible (MAE 5 d, acc±1 6%)** = you're reading legacy regression keys from `test_metrics` on a binary task. Read `roc_auc`, `f1`, `precision_at_best_thr` instead; check `test_metrics.task` first.
- **All predictions land in MEDIUM/LOW, no CRITICAL/HIGH** = the binary classifier's probabilities are mostly 0.3–0.5 (uncertain), which under the pseudo-days mapping land in days 4–5. To get more CRITICAL alerts, either improve features (weather completion, drought index) or tune `_prob_to_days_for_compat()`.
- **`fetch_firms.py` HTTP 400 across all datasets** = bad / rate-limited `MAP_KEY`. Check status at `https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=…`. Resets daily.
- **API boots but `/health` says `ready: false`** = no trained artifact yet (degraded mode — the old hard `RuntimeError` on startup was removed so fresh Railway deploys don't crash-loop). Run `python train.py`, or let the scheduler train once enough data accumulates.
- **Stale model after feature changes**: delete `outputs/models/*.pkl` before retraining. api.py / risk_map.py resolve features from `dataset_info.json` but a leftover `.pkl` from a different feature contract will silently mispredict.
- **`scikit-optimize` / `matplotlib` missing**: train.py falls back to `RandomizedSearchCV` / skips PNG report. Install for full functionality: `pip install scikit-optimize matplotlib scipy`.
- **`PerformanceWarning: DataFrame is highly fragmented`** during feature build is harmless — pandas warns about column-by-column `frame.insert` calls.
- **Open-Meteo 429s**: archive endpoint is per-hour and per-day rate-limited. `fetch_weather.py` is idempotent — rerun resumes. Optional `OPEN_METEO_QUIET_HOURS=2-8` (or `--quiet-hours 2-8`) waits for an off-peak window.
- **`./run.sh` ends with `exec uvicorn`** — it never returns. Use `--no-train` to skip retraining and just serve.
- **`frontend/` directory is legacy** — vanilla JS dashboard from a previous version. Edit `web/src/` instead.
