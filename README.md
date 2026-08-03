# FireWatch Thailand

FireWatch Thailand is a research-grade decision-support dashboard for
**3-day wildfire hotspot risk** across Thailand. A LightGBM classifier ranks
0.1° grid cells and estimates:

> `P(≥1 NASA FIRMS hotspot in this cell during base_date+1 … base_date+3)`

The system does **not** predict an exact ignition date and does not guarantee
that a displayed cell will burn. It combines calibrated model risk, live or
recent satellite observations, historical context, weather, and retrospective
validation so an operator can prioritise where to look first.

## What is trustworthy today

The machine-readable source of truth is
[`outputs/metadata/dataset_info.json`](outputs/metadata/dataset_info.json).
The dashboard reads metrics and thresholds from the current GeoJSON metadata;
it does not replace weak values with nicer hardcoded numbers.

Current promoted-artifact evidence (trained 2026-07-16):

| Evidence | Current value | Correct interpretation |
|---|---:|---|
| Held-out ROC-AUC | 0.9051 | Strong relative ranking on the chronological test period |
| Top-20% historical coverage | 82.93% | The top fifth of ranked cells captured this share of held-out fires |
| WATCH held-out recall | 31.55% | Absolute WATCH cutoff under the current seasonal prevalence shift |
| HIGH held-out recall | 11.68% | Stricter, precision-oriented alert tier |
| Test sample | 709,002 cell-days | Natural-prevalence held-out data; 22,992 positive cell-days |

These are different measurements, not interchangeable claims. Ranking coverage
describes **relative resource prioritisation**; WATCH/HIGH describe
**absolute calibrated cutoffs**. The current calibration period and test period
have very different fire prevalence, so the UI discloses the seasonal drift
rather than hiding it.

For the full protocol and limitations, see:

- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)
- [`docs/PRESENTATION_TH.md`](docs/PRESENTATION_TH.md)
- [`docs/DATA_APIS_TH.md`](docs/DATA_APIS_TH.md)

## Dashboard

The FastAPI process serves both the React SPA and APIs on the same origin.

| Route | Purpose |
|---|---|
| Map | Current 3-day risk cells, evidence, weather, and live observations |
| Live Fires | Recent GISTDA satellite detections with feed-health context |
| Analytics | Live-feed composition plus clearly labelled historical context |
| Predicted vs Actual | Mature archived risk windows compared with matching FIRMS dates |
| Alerts | Operator notification workflow and audit log |
| Reports | Held-out metrics, baselines, calibration, and methodology |

The map uses two separate concepts:

- **Absolute tiers:** HIGH, WATCH, and Background from fixed calibrated
  probability cutoffs.
- **Relative ranking:** top-ranked cells for finite patrol resources. A
  relative rank is not an absolute alert.

Very small non-zero probabilities display as `<0.01%`. Operators may hide those
markers for visual focus, but the probability is unchanged and exports retain
the complete selected dataset.

## Quick start for tomorrow's demo

Use the already-promoted local artifact without fetching or retraining:

```bash
cd /home/chanakan/Documents/wild-fire
source .venv/bin/activate
./run.sh --demo --open
```

Open <http://localhost:8000/> if the browser does not open automatically.
`run.sh` now rebuilds the React app whenever its source is newer than
`web/dist`, so it cannot silently serve an old UI. In `--demo` mode, the
scheduler keeps GISTDA live polling enabled but does not run FIRMS sync or the
five-minute retraining check; the promoted model and prediction artifacts stay
unchanged throughout the presentation.

`--demo` deliberately rejects `--fresh`, `--weather`, `--gistda`,
`--predict-only`, `--quick`, `--fast`, and extra training arguments. Run any
refresh before the presentation, then restart with `./run.sh --demo`.

To refresh real FIRMS inputs and create a new prediction snapshot without
retraining:

```bash
./run.sh --fresh --predict-only --open
```

This can take time and needs a valid `FIRMS_API_KEY`; do it before the
presentation, not during it. If the network is unavailable, keep the existing
promoted snapshot and explain its visible “data as of” timestamp.

## First-time setup

```bash
cd /path/to/wild-fire
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

cd web
npm ci
npm run build
cd ..
```

Set `FIRMS_API_KEY` in `.env` before requesting fresh NASA FIRMS data. Do not
commit `.env`.

For frontend development:

```bash
# terminal 1
cd src
SCHEDULER_ENABLED=false uvicorn api:app --reload --port 8000

# terminal 2
cd web
npm run dev
```

Open <http://localhost:5173/>. Vite proxies API paths to port 8000.

## Data and model pipeline

```text
NASA FIRMS ─┐
ERA5 weather├─> clean/grid/densify ─> causal features ─> train/calibrate
Tree cover ─┘                                            │
                                                         ├─> promoted model
GISTDA live feed ─> validated cache/database             └─> risk GeoJSON
                                                                  │
                                            FastAPI + React/Leaflet dashboard
```

Key commands:

```bash
./run.sh --fresh --predict-only  # daily data + prediction refresh
./run.sh --quick                 # retrain with quality-gated promotion
cd src && python risk_map.py     # rebuild GeoJSON from current artifact
```

Training uses whole-date chronological train/validation/calibration/test
partitions with 7-day purges. Negative undersampling is limited to training;
calibration and test retain natural prevalence. Candidate models pass promotion
gates before replacing the incumbent.

## Data-quality contract

- Grain: one predicted record per `(base_date, 0.1° cell)`.
- Probability domain: finite `[0, 1]`.
- Risk window: the next three days, common to every cell in a snapshot.
- `predicted_fire_date` and `days_until_fire` in legacy artifacts are
  deprecated compatibility fields and must not be presented as exact forecasts.
- Live GISTDA and historical FIRMS observations are evidence layers, not model
  predictions.
- Missing or stale feeds must be labelled unavailable/stale, never converted to
  a trustworthy-looking zero.
- Weather prefers the local ERA5 cache and uses Open-Meteo only as a visibly
  labelled fallback.

## Validation

Frontend:

```bash
cd web
npm run typecheck
npm test
npm run build
```

Backend:

```bash
source .venv/bin/activate
python -m compileall -q src scripts
SCHEDULER_ENABLED=false python -m pytest
```

CI runs these deterministic checks without retraining or network access.

## Academic figures

Generate the publication-ready evaluation, calibration, threshold, feature,
backtest, activity, correlation, and ablation figures with:

```bash
pip install -r requirements-figures.txt
.venv/bin/python scripts/academic_figures.py
```

Figures are written to `outputs/figures/academic/` as PNG (300 dpi), SVG, and
PDF. See [`docs/ACADEMIC_FIGURES_TH.md`](docs/ACADEMIC_FIGURES_TH.md) for the
recommended figures, captions, and interpretation cautions.

## Production

Railway deployment uses one Docker image and one web service. The FastAPI
lifespan starts guarded sync, prediction, retraining-check, and GISTDA polling
jobs. PostgreSQL supplies durable audit data and advisory locks; a `/srv`
volume stores model/data artifacts.

Follow [`DEPLOY.md`](DEPLOY.md). Do not create a second Railway cron service
that expects to share the web service's volume: Railway volumes attach to one
service, so that topology would train into a different filesystem.

## Intended use and limits

Use FireWatch as an operator-reviewed prioritisation aid, research prototype,
and educational system. Do not use it as the sole basis for evacuation,
life-safety dispatch, legal attribution, or operation outside its trained
geographic scope. Satellite detections can be delayed by overpass timing,
cloud, smoke, and upstream availability.
