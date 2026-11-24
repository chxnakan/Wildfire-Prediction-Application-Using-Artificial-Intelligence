# Hotspot Forecasts and Alerts

This project monitors NASA FIRMS satellite fire/thermal-anomaly detections and
forecasts whether a detection will be recorded in a Thai grid location
during the next three days (D+1 through D+3). A hotspot is not a confirmed wildfire.

## Three-day forecast horizon

New training runs target **at least one recorded satellite hotspot on D+1, D+2
or D+3**. D is the forecast base date; the current day is excluded. Targets,
conditional date estimates, evaluation splits and new saved forecasts all use
this three-day horizon. Forecast probabilities are retrained for this outcome;
they are not truncated seven-day probabilities.

Past-only inputs retain their existing windows, including seven-day hotspot
history and 30-day drying context. Archived weather forecasts already cover
D+1 through D+3. The alert cutoff remains 0.5, fixed before evaluation.

Every new model and evaluation stores `horizon_days=3`. Old seven-day snapshots
remain immutable and are scored using their own seven-day outcomes. Dashboard
forecast labels, timeline lengths and comparison dates follow the selected
snapshot; the visual style is unchanged. The historical seven-day result tables
below are retained for provenance and are not the new model's performance.

The three-day model `98bfd2a97aab` selects **weather + terrain (20 features)**.
Its chronological test covers **2025-11-06 through 2026-01-14**, with 259,860
grid-day windows, including 32,655 with a recorded hotspot within three days:

| Metric | Three-day model, threshold 0.5 |
| --- | ---: |
| ROC-AUC | 0.7800 |
| Event capture (recall) | 38.4% |
| Alert precision | 39.4% |
| False-positive rate | 8.49% |
| Average precision | 0.3625 |
| Brier score | 0.1130 |
| Conditional date MAE | 0.674 days |

Results reverified on **November 22, 2026**: all **82 software tests passed**.
All 259,860 held-out labels were independently rebuilt from the saved observations,
the metrics were recalculated, and production predictions were reproduced.
The public API was checked against the saved evaluation. This verifies the existing
historical test; it does not add new observations or retrain the model.
See [the verification results](outputs/evaluation/test_results.json) for exact
metrics, test dates and evidence hashes.

The model flags roughly 38 of every 100 positive windows; roughly 39 of every
100 flagged windows contain a recorded hotspot. The shorter target window,
changed prevalence and revised test dates mean these are not a controlled
improvement comparison with the old seven-day figures. Timing errors cover
actual-positive windows, including missed alerts. Within-one-day timing accuracy
is weak evidence for a three-day horizon: always predicting day 2 already puts
every day-1/day-2/day-3 event within one day.

Probability calibration remains a limitation: held-out ECE is 0.0885 after
calibration versus 0.0511 before it. The published map remains a historical
demonstration for **2026-01-18**, covering **January 19–21**, because recent
input history is incomplete. Live FIRMS alerts remain separate.

## Local setup

For a public website using your Cloudflare domain, follow
[the cloud hosting guide](deploy/README.md). It includes Docker startup,
uploading the existing forecast assets, and connecting Cloudflare Tunnel.

Run from the project root (Python 3.14 is verified):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
test -f .env || cp .env.example .env
python src/training.py
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Open [the dashboard](http://127.0.0.1:8000/dashboard/) or
[API documentation](http://127.0.0.1:8000/docs).
The API serves the frontend; do not open the HTML file directly.
For later starts, activate `.venv` and run the Uvicorn command.

The supplied observations reach April 22, 2026, but contain long gaps after
January 17. The dashboard displays a **historical demonstration** from the most
recent date with a complete seven-day regional history. Its historical status
is recorded in the API metadata; the dashboard's Base Date shows its actual date.
The live alert feed is independent.

## Weather in the main model

`python src/training.py` now includes weather by default. It trains on Thai
grid-cell centres and publishes through the existing model, snapshot, API and
dashboard paths. The original hotspot-only architecture is also trained on
identical rows for a controlled comparison; it is not a second dashboard.

The weather baseline keeps the eight hotspot-history features and adds nine weather features.
The research pipeline below evaluates additional inputs before selecting the main model:

- Recent three/seven-day rainfall totals.
- Recent three-day maximum temperature, minimum humidity and maximum wind.
- Forecast three-day rainfall total, maximum temperature, minimum humidity and
  maximum wind, valid on D+1 through D+3.

Weather comes from the [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api)
using NOAA GFS (`gfs_seamless`). All aggregation is UTC; wind is metres/second,
rain millimetres, temperature Celsius and humidity percent. Recent weather uses
archived short-lead model output, not station measurements. Its windows end at
D-2. Future weather uses fixed five-day-lead forecasts, so the three future days
come from different earlier runs. A six-hour publication buffer puts every
included value before the D 00:00 UTC decision cutoff. Exact historical release
timestamps are unavailable, and this inferred availability is recorded as a
limitation. Realised future weather is never used as an input.

`WEATHER_GRID_SIZE=0.5` controls weather sampling independently of the 0.1°
hotspot grid. Several hotspot cells share weather; this does not create 0.1°
weather resolution. `WEATHER_REQUEST_INTERVAL=3` spaces uncached requests.
Responses, request parameters, timestamps and hashes are cached under
`data/weather/cache/`. Rerunning the same training period reuses these files.
Provider quota failures stop downloading and preserve the cache for a later
resume. A day needs all 24 hourly values; missing weather is not converted to
zero rainfall. Rows lacking the required windows are excluded from both sides
of the comparison, and inference rejects missing weather inputs.

The main model bundle stores its feature names, and inference uses that exact
schema. Saved inputs in `outputs/features/latest_features.csv` include weather.
To refresh predictions with newly supplied hotspot observations, rerun training;
the independent live-alert poller does not train or refresh this forecast.
`USE_WEATHER_FEATURES=false` explicitly restores hotspot-only features. The revised
source-label policy and country-wide candidate geometry still apply.

`outputs/evaluation/metrics.json` includes `weather_comparison`, weather coverage
and source provenance. The comparison reports precision, recall, false-positive
rate, timing error, calibration and descriptive recall at fixed 2%/5% false-alarm
budgets. Those holdout ROC points are for assessment, not threshold selection.
The deployment threshold remains fixed before testing. Saved holdout evidence
contains predictions from both models. Weather attribution: Open-Meteo and NOAA
GFS, CC BY 4.0; API access is subject to the provider's usage terms.

The historical **seven-day** integrated run (`c5053396ab1e`) compares 216,470 Thai grid-days from
2025-11-03 through 2026-01-10, using identical fitting rows and parameters:

| Metric | Hotspot history only | History + weather |
| --- | ---: | ---: |
| Precision at 0.5 | 74.0% | 54.7% |
| Recall at 0.5 | 16.5% | 31.9% |
| False-positive rate at 0.5 | 1.85% | 8.42% |
| Average precision | 0.447 | 0.495 |
| Brier score (lower is better) | 0.1612 | 0.1624 |
| Recall at a 2% false-positive budget | 17.4% | 15.2% |
| Recall at a 5% false-positive budget | 28.4% | 24.6% |

Weather improves overall ranking and recall at the default threshold, but creates
more false positives. It does not improve recall at matched low false-positive
budgets or probability error on this holdout. This is a mixed result, not evidence
of uniformly better forecasts. The published snapshot is still a historical
demonstration for 2026-01-18; adding weather does not fill gaps in hotspot records
or turn the forecast into a current live prediction.

## Research improvements in the main pipeline

`USE_RESEARCH_FEATURES=true` is the default when weather is enabled. Training
evaluates six feature combinations: the 17-feature weather baseline, that
baseline plus detection-quality/season features, land cover, terrain, drying
features, and all groups together. It uses three expanding chronological
validation blocks inside the fitting period, with a three-day label purge.
The final chronological holdout is never used to choose a combination. Selection
maximizes mean average precision subject to mean Brier error no more than 0.002
above the weather baseline and recall at a 2% false-positive budget no worse than
that baseline. These budget comparisons do not change the deployed 0.5 threshold.
`USE_RESEARCH_FEATURES=false` skips selection and context extraction.

The data/label changes apply to the main model:

- Preserve NASA `type`, day/night status and platform identity. A typed archive
  record takes precedence over its duplicate NRT record; conflicting known
  types become unknown. Missing type is never converted to vegetation fire.
- The target policy `exclude_known_nonvegetation_v1` excludes type 1 (volcano),
  2 (other static land source), and 3 (offshore) from **outcome labels**. Unknown
  NRT types remain eligible. Type 0 is presumed vegetation fire and includes
  agricultural burning; it is not proof of wildfire. The type annotation is
  retrospective and is not used to filter past input features or infer a
  historical source mask that would have required future information.
- Include all cell centres inside Thailand's bundled province boundaries,
  including never-detected locations. Candidate selection no longer depends on
  a previous hotspot. Zero previous detections is valid input, not a reason to
  omit a place from the forecast. This does not resolve satellite coverage gaps.
- Store the target policy with each forecast. Outcome and prospective scoring
  use that forecast's policy; older snapshots retain all-hotspot semantics.
  Regional coverage checks still use all raw detections before source filtering.

The additional feature groups are:

| Group | Inputs and source | Practical limits |
| --- | --- | --- |
| Detection quality and season | Previous-seven-day low-confidence fraction, nighttime fraction, known day/night fraction; cyclic day of year | VIIRS confidence categories are not wildfire probabilities; seasonality is not a crop calendar |
| Land cover | Eleven class fractions from ESA WorldCover 2021 v200, including trees, crops, built areas, grass and water | Static 2021 baseline. Fractions summarize valid ~300 m COG overview classes, not exact native 10 m areas. No crop species or controlled-burn labels |
| Terrain | Mean elevation, elevation standard deviation and mean slope from Copernicus GLO-90 | A surface model includes vegetation/infrastructure; slopes are estimated at ~90 m with tile-edge approximations |
| Atmospheric drying | Hourly paired temperature/humidity VPD aggregated over seven prior days and three forecast days; 30-day rainfall and count of days with less than 1 mm rain | VPD is atmospheric drying potential, not measured fuel moisture. These are not a validated Thai FWI or a drought severity index |

Weather features keep the conservative availability rules above. Thirty-day
features need complete windows ending D-2; initial incomplete rows are removed
equally from every comparison. All comparison models use the same cells, dates,
labels and fitting parameters. The latest feature table retains the context
columns even if selection does not choose them, while inference uses exactly the
selected feature list saved with the model.

Land/terrain tiles and cell summaries are cached under `data/context/cache/`;
provider URLs, extracted-raster hashes, aggregation policy and attribution are
included in metadata. No Earth Engine account or API key is required. Static
context is restricted to forecast periods starting in 2023 or later, after the
source products were publicly available. Rerun training to refresh the model.

`outputs/evaluation/selection.json` records every feature combination's earlier
validation results. `metrics.json` includes `research_comparison` against a
freshly trained 17-feature baseline, `monthly_holdout`, and the target definition.
Saved holdout predictions include the selected, weather-only and hotspot-only
models. Results from the old model cannot be compared directly with the new
headline metrics because the target labels and candidate population changed.

The historical **seven-day** research run (`c402479669a9`) selected **weather + terrain (20 features)**.
Land cover improved earlier validation on its own, but the terrain combination
ranked best under the predeclared selection rule. Drying, detection-quality,
seasonal and land-cover inputs remain available in the comparison pipeline;
they are not all active features of the selected classifier for that run.
The untouched test period is 2025-11-03 through 2026-01-10 (238,205 grid-days):

| Metric | Matched 17-feature weather baseline | Selected 20-feature model |
| --- | ---: | ---: |
| Average precision | 0.490 | 0.527 |
| Recall at 2% false-positive budget | 15.6% | 18.3% |
| Recall at 5% false-positive budget | 25.8% | 30.1% |
| Precision at the fixed 0.5 threshold | 45.2% | 44.2% |
| Recall at the fixed 0.5 threshold | 48.6% | 58.7% |
| False-positive rate at the fixed 0.5 threshold | 16.7% | 21.0% |
| Brier score (lower is better) | 0.1634 | 0.1641 |

Ranking improves at matched false-positive budgets, but the fixed-threshold
false-positive rate and probability error do not improve. The ROC budget points
do not mean the deployed threshold achieves those budgets. This is still a
mixed result; it does not resolve seasonal calibration drift or establish
confirmed wildfire prediction. The production model is refit after this test,
and its latest snapshot remains a historical demonstration for 2026-01-18.

Remaining limitations require additional data: confirmed incident and cause
labels; local cloud/overpass coverage; current vegetation/fuel measurements;
roads, crop calendars and lightning; multiple independent fire seasons; and a
complete recent hotspot history. Built/crop fractions only provide limited
human-activity context. No prediction is relabelled as a confirmed wildfire,
and incomplete recent observations still produce an explicitly historical demo.

Research and data provenance:

- [Thailand seasonal fire modelling (Bihari et al., 2025)](https://www.mdpi.com/2072-4292/17/19/3378): motivation for environmental and human-access context; its seasonal task differs from this three-day forecast.
- [NASA FIRMS archive](https://firms.modaps.eosdis.nasa.gov/download/): source type annotations are inferred and require provenance checks; NASA documented a type-classification bug corrected in 2025.
- [ESA WorldCover data access](https://esa-worldcover.org/en/data-access): © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium; CC BY 4.0.
- [Copernicus DEM source and licence](https://copernicus-dem-90m.s3.amazonaws.com/readme.html): © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018; provided under COPERNICUS by the European Union and ESA.
- [FAO-56 vapour-pressure methodology](https://www.wcc.nrcs.usda.gov/ftpref/wntsc/waterMgt/irrigation/fao56.pdf): saturation vapour pressure is computed from paired hourly temperature/RH, not unrelated daily extremes.

## Dashboard

The dashboard at `/dashboard/` has six pages, an EN/ไทย language switch, a
light/dark theme switch and a LIVE indicator for the FIRMS feed:

The dashboard displays hotspots in Thailand only, using the bundled province
polygons rather than a rectangular country approximation. Both maps show the full
surrounding geography and allow panning beyond Thailand. Forecast cells are
included by their centre; live detections use their actual coordinates. The map counts, watch lists,
exports, daily history and snapshot outcome comparisons follow this scope. Existing
older model training and holdout/prospective files retain their original regional
scope; new main-model training uses Thai cells. API history, alerts and outcome routes accept `scope=thailand`;
their default regional responses remain available for existing integrations.

- **Map**: flagged forecast cells coloured by three-day hotspot probability
  (High ≥70%, Medium ≥40% and <70%, Low <40%), FIRMS detections on the feature-cutoff day, and live FIRMS
  detections. The sidebar selects a saved forecast snapshot and a province, and
  exports the visible cells as CSV. *Share* copies a link that restores the
  view. *Threshold* changes which cells are displayed, never the model's
  threshold, and shows held-out precision and recall at the chosen value.
  *Details* summarises held-out performance. Clicking a cell shows the selected snapshot's
  probability, recent detection history and weather context.
  The display threshold remains separate: at its default 50%, Low cells are
  hidden. Lower it below 40% to display them. Predicted dates remain separate
  from probability. Legacy snapshots without calibrated probabilities use explicitly
  labelled score bands (0.70 and 0.40), not probability percentages. The dashboard
  derives these levels from the probability/score; immutable snapshot files retain
  their original timing-based `urgency_level` field for compatibility.
- **Live Fires**: the last 24 hours of NOAA-20/NOAA-21 detections with filters,
  "View on map" and a link to verify each detection on the FIRMS map.
- **Analytics**: the forecast timeline, the probability distribution, the
  probability-level breakdown, the top provinces and Thailand's daily detections.
- **Predicted vs Actual**: a retrospective check of the selected snapshot
  against later records (labelled as such and never counted as prospective
  verification), the holdout comparison with the baselines, and the
  prospective scoring status.
- **Alerts**: the forecast watch list, highest probability first, and a log of new live
  detections seen during the session.
- **Reports**: training provenance, the probability-level distribution, feature
  importance, cells per province and the recorded limitations, with JSON, CSV
  and PNG downloads.

External services used by the browser:

- OpenFreeMap road maps rendered with MapLibre beneath the Leaflet fire layers; no map API key is needed. Map labels follow the English/Thai dashboard switch, using local names when a translation is unavailable. If the vector map cannot load, a basic OpenStreetMap fallback has fixed local-language labels.
- [Open-Meteo](https://open-meteo.com/) for the weather shown in cell popups.
  The request covers the seven days ending at the snapshot's feature cutoff and
  sends only the cell's coordinates. This popup request provides display context;
  the model uses the separately cached archived weather features described above.

Provinces come from bundled geoBoundaries ADM1 polygons
(`frontend/data/`, ODbL). They are assigned in the browser by cell centre and
are approximate near borders and coasts.

## What changed in evaluation

The old random-row split and positive-only regression evaluation have been
replaced. Previous MAE and accuracy figures are not comparable with this test.

- Deduplicate observations by satellite, timestamp, latitude and longitude,
  including overlap between archive and near-real-time files.
- Build one record per Thai grid per calendar day, including locations with no
  previous detections; candidate geometry does not depend on future observations.
- For a forecast with base date D, hotspot features use days before D only;
  weather inputs follow the availability policy above. The
  prediction window is **D+1 through D+3**, excluding the current day.
- Include both windows with and without a subsequent recorded detection.
  A LightGBM classifier estimates a detection score; a separate regressor
  estimates the waiting time conditional on a detection.
- Exclude immature labels at the end of the dataset. Also exclude training/test
  windows whose preceding seven days or following three days include a regional
  date with no records, conservatively treating those dates as unknown coverage.
- Test on the latest 20% of eligible dates. Purge training rows whose three-day
  labels reach the first test date. Inputs cannot use information unavailable at issuance.
- Fix `DETECTION_THRESHOLD` before testing (default 0.5). Do not choose a threshold
  based on the final test results; threshold tuning needs a separate earlier
  validation period.
- Compare with training prevalence and a repeat-recent-detection baseline.
  Both date baselines always predict the median positive waiting time from training.
- Preserve the evaluation models and every held-out prediction. Refit separate
  production models on all mature eligible rows after evaluation.
- Calibrate classifier scores with Platt scaling. Fitting rows are split into
  four contiguous date blocks. Each block is scored by a model that never saw
  it, with a three-day label purge and a seven-day history embargo. The sigmoid is fit on those
  out-of-fold scores, and the final classifier is fit on all fitting rows.
  The test period is never used. `CALIBRATION_FOLDS` changes the fold count.

The `/performance` endpoint reports the chronological holdout, not a training
score or a prospective result for the refitted production model. It also
includes calibration diagnostics: ROC-AUC, ECE, the reliability bins,
precision in the top 5/10/20 % and a precision/recall table by threshold.
The dashboard's Details dialog and its Reports and Predicted vs Actual pages
present these results.

## Metrics and their scope

Classification metrics operate on **grid-day windows**, not distinct wildfire
incidents. Positive means at least one subsequent recorded hotspot in that grid.

| Metric | Definition |
| --- | --- |
| Precision | True positive windows / all forecast-positive windows |
| Recall | True positive windows / all actual-positive windows |
| F1 | Harmonic mean of precision and recall |
| False positive rate | False positive windows / actual-negative windows |
| Average precision | Ranking quality across score thresholds |
| ROC-AUC | Probability that a random positive window outranks a random negative one (0.5 = random) |
| ECE | Mean gap between predicted probability and observed rate across ten probability bins |
| Precision @ top-K% | Share of positives among the K% highest-probability windows |
| Brier score | Mean squared difference between classifier score and binary outcome |
| Date MAE | Mean absolute waiting-time error on every actual-positive window |
| Date RMSE | Root mean squared waiting-time error on those same windows |
| Within ±1 day | Fraction of actual-positive windows whose rounded predicted waiting time is within one day |

Date metrics include classifier misses, so a low date MAE does not imply good
recall. The complete confusion matrix, denominators, dates and baselines are
available from `/performance`. Undefined metrics are `null`, not fabricated zeroes.

Forecasts store both the raw classifier score (`detection_score`) and the
calibrated `probability`. The threshold applies to the probability. It is the
probability of **at least one recorded satellite hotspot** in the cell within
three days, not of a confirmed wildfire. Calibration only covers the seasons
in the fitting period. The holdout (November–January) can differ seasonally from
the fitting period; `metrics.json` records calibration error before and after
scaling for the currently published model. Forecast
records from before calibration (schema 3) are still listed and shown as
uncalibrated scores. A below-threshold forecast has no predicted date; it does
not mean an area is safe or that fire is impossible.

## Saved forecasts and later verification

Every generated map creates an immutable record under `outputs/forecasts/`,
including issuance time, model version, feature cutoff, base date, threshold,
grid size and all candidate grids (including below-threshold forecasts).
Regenerating the same model/base date preserves the first issuance.

Only records issued on their base date, before the future outcome window, and
using models whose training labels end earlier qualify as prospective forecasts.
Historical demonstrations are excluded from prospective evaluation.

After downloading later observations, run:

```bash
python src/evaluate_forecasts.py
```

This waits for the snapshot's full outcome window (three days for new models,
seven days for legacy snapshots) and excludes windows with
unknown regional coverage. Results are written to
`outputs/evaluation/prospective.json` and exposed by `/performance`.
The latest data must be accumulated locally; the in-memory alert feed alone
is not an archive for forecast verification. The supplied data cannot establish
prospective performance for forecasts generated today.

To update observations, evaluate prior forecasts, and generate a new forecast:

```bash
python src/fetch_firms.py
python src/evaluate_forecasts.py
python src/training.py
```

Training also runs prospective verification. The API reads published files on
request, so updated forecast results do not require a server restart.

## In-app detection alerts

Set your NASA FIRMS MAP_KEY as `FIRMS_API_KEY` in `.env` and restart the server.
Never put the key in frontend code. No key is required for historical training.

The backend checks NOAA-20 and NOAA-21 VIIRS feeds every 15 minutes while running.
The browser reads the cached feed every minute (`/alerts?limit=…`, default 100,
maximum 5000; `total` and `by_satellite` always count the whole 24-hour window).
The Live Fires page shows observation times, per-satellite counts and
missing-key, failure and partial-coverage states. Select a detection to locate it.
These are in-app alerts, not email or operating-system push notifications.
The NASA email-alert subscription is separate and is not imported.

`FIRMS_BBOX=96,4,107,22` controls the alert rectangle (west, south, east, north),
covering Thailand and surrounding areas. It does not change the geographic
coverage of the historical training CSVs. Global FIRMS detections generally
arrive within three hours of satellite observation, not instantaneously.

## Limitations

- No record does not establish no fire: cloud cover, overpass timing and missing
  local observations can cause non-detections. Regional records do not prove
  that every grid was observed. These labels measure records in supplied data.
- The evaluation includes all Thai grid-cell centres, including never-detected locations.
- Historical files lack record-arrival timestamps. Backtests assume prior-day
  observations were available at issuance; saved prospective forecasts are
  needed to measure performance under actual delivery delays.
- Overlapping three-day windows are correlated; sample counts are not counts
  of independent fires. No statistical confidence intervals are claimed.
- Confirmed wildfire precision/recall require independent verified wildfire
  reports, which are not included. Unit tests verify software behavior, not
  scientific forecasting accuracy.

## Outputs and checks

- `outputs/models/lgbm_fire_date_model.pkl`: versioned classifier/regressor bundle.
- `outputs/features/latest_features.csv`: most recent eligible forecast inputs.
- `outputs/features/daily_detections.csv.gz`: deduplicated daily detections.
- `outputs/features/observation_coordinates.csv.gz`: original coordinates for
  regrouping observations onto each archived forecast's grid. Older outputs
  without this file fall back to the original input CSVs.
- `outputs/metadata/dataset_info.json`: source quality, model metadata and evaluation.
- `outputs/evaluation/metrics.json`: current chronological holdout scores.
- `outputs/evaluation/<version>/`: evaluation model, metrics and held-out predictions.
- `outputs/forecasts/`: immutable forecast records.
- `outputs/riskmap/fire_dates_all.geojson`: current map, replaced on generation.

Read-only API routes used by the dashboard:

- `/forecasts` lists the saved snapshots.
- `/forecasts/{name}/geojson` returns one snapshot as a map.
- `/forecasts/{name}/outcomes` returns the retrospective comparison. Its status
  is `ok`, `pending` or `unknown_coverage`.
- `/cell?lat_cell&lon_cell&until` returns a cell's detection history.
- `/history/daily` returns regional detections per day.

Run regression checks with:

```bash
python -m unittest discover -s tests -v
```

Sources: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/),
[FIRMS API](https://firms.modaps.eosdis.nasa.gov/content/academy/data_api/firms_api_use.html),
[time-aware evaluation guidance](https://scikit-learn.org/stable/modules/cross_validation.html).
