# Model Card — FireWatch Thailand 3-Day Hotspot Risk

This card describes the **currently promoted artifact trained on
2026-07-16**. The authoritative machine-readable record is
[`outputs/metadata/dataset_info.json`](../outputs/metadata/dataset_info.json).
If this document and that artifact ever differ, the artifact wins.

## Model details

- Task: binary classification
- Output: calibrated `P(≥1 FIRMS hotspot in a 0.1° cell during t+1 … t+3)`
- Model: five-seed LightGBM ensemble with Platt-sigmoid calibration
- Features: 164 causal satellite, history, spatial, calendar, land-cover, and
  weather features
- Grid: 0.1° (roughly 11 km north–south)
- Geographic scope: historically active cells in the Thailand study area
- Training source: NASA FIRMS VIIRS NRT plus optional measured auxiliary data
- Promoted-artifact timestamp: `2026-07-16T14:48:35Z`

The model does **not** predict an exact ignition date, fire perimeter, cause,
severity, or evacuation need.

## Intended use

Appropriate:

1. Rank historically active cells for operator review.
2. Build a daily WATCH list when patrol capacity is limited.
3. Compare archived risk windows with later FIRMS observations.
4. Research and education about operational rare-event ML.

Not appropriate:

1. Sole-source life-safety, evacuation, or automated dispatch decisions.
2. Legal attribution or enforcement against a person or community.
3. Use outside the trained geographic/population scope.
4. Treating a probability or ranking tier as proof that a fire will occur.
5. Treating a Background cell as proof that no fire can occur.

## Data and evaluation protocol

Current data:

| Item | Value |
|---|---:|
| Source date range | 2025-06-11 … 2026-07-16 |
| Total days | 401 |
| Historically active cells | 11,833 |
| Training rows after train-only sampling | 485,190 |
| Feature count | 164 |
| Held-out test cell-days | 709,002 |
| Held-out positive cell-days | 22,992 |
| Held-out positive rate | 3.24% |

The four partitions contain whole calendar dates and are separated by 7-day
purges:

| Partition | Date range | Purpose |
|---|---|---|
| Train | 2025-06-11 … 2026-01-13 | Fit models; negative sampling only here |
| Validation | 2026-01-21 … 2026-03-21 | Early stopping/model selection |
| Calibration | 2026-03-29 … 2026-05-07 | Fit calibrator and lock thresholds |
| Test | 2026-05-15 … 2026-07-13 | One final natural-prevalence evaluation |

No calendar date belongs to two partitions. The 3-day label tail is censored
when its future observation window is incomplete. Hyperparameter search uses
purged whole-date folds, not row-based `TimeSeriesSplit`.

## Current held-out evidence

Threshold-independent ranking:

| Metric | Value | Interpretation |
|---|---:|---|
| ROC-AUC | **0.9051** | Strong ordering of positive above negative cell-days |
| Average precision | **0.2175** | Precision-recall performance under a 3.24% base rate |
| Brier score | 0.0390 | Mean squared probability error |
| ECE | 0.0503 | Aggregate calibration error; see drift caveat below |

Comparison baselines on the same test population:

| Baseline | ROC-AUC | Average precision |
|---|---:|---:|
| Fired in previous 3 days | 0.5898 | 0.0920 |
| Fired in previous 7 days | 0.6403 | 0.1165 |
| Training cell climatology | 0.6078 | 0.0431 |

Absolute operational tiers:

| Tier | Threshold | Precision | Recall | Alert rate |
|---|---:|---:|---:|---:|
| HIGH | 0.5236 | 38.50% | 11.68% | 0.98% |
| WATCH | 0.4376 | 22.72% | 31.55% | 4.50% |

Relative historical coverage:

| Ranked share | Held-out fires captured |
|---|---:|
| Top 5% | 33.91% |
| Top 10% | 52.47% |
| Top 20% | **82.93%** |

Relative coverage is a resource-allocation result, **not** an absolute alert
and not a guarantee for a particular day.

## Seasonal drift

The calibration partition had 39.33% positive prevalence while the held-out
test period had 3.24%, a 12.13× mismatch. This is a major seasonal base-rate
shift. A single cutoff locked during the high-prevalence calibration period
therefore transfers poorly to the off-season test period.

Consequences:

- The model remains a strong ranker (AUC 0.9051).
- Absolute HIGH/WATCH recall is much lower than calibration-period recall.
- Probabilities and fixed-tier decisions need the visible drift context.
- The dashboard shows WATCH recall and Top-20% coverage together because they
  answer different operational questions.
- The frontend must not select a nicer precomputed threshold or invent recall.

## Features and leakage controls

Major feature groups:

- past FIRMS counts, FRP, recurrence, and active-day windows;
- 3×3 and wider spatial-neighbour aggregates;
- day-of-year/week/month cyclic seasonality;
- tree cover and recent loss;
- temperature, precipitation, wind, evapotranspiration, and past-only weather
  windows when available;
- location and distance-to-urban context.

Time-varying rolling features shift before rolling. Lags are at least one day.
Cumulative features exclude the current/future outcome. A cell enters the
population from its first observed FIRMS detection onward; first-ever fires in
previously unseen cells are outside the current prediction universe.

## Known limitations

1. **Seasonal transfer:** the current global thresholds are affected by the
   documented prevalence drift.
2. **Population scope:** never-burned cells cannot be flagged until they enter
   the historically active universe.
3. **Spatial resolution:** an 0.1° cell is an area priority, not a point-fire
   location.
4. **Satellite observation:** cloud, smoke, overpass timing, source latency,
   and non-wildfire heat can affect labels.
5. **Geographic generalisation:** no province-held-out validation has yet
   established performance in unseen regions.
6. **Seasonal feature strength:** calendar signals are important, so unusual
   seasons or policy changes may shift performance.
7. **Weather availability:** local ERA5 may lag; an Open-Meteo response is a
   labelled fallback, not hidden substitution.
8. **Live audit sample size:** a single archived snapshot can contain very few
   outcomes. Exact counts should be shown and percentages withheld for small
   denominators.

## Operational safeguards

- fixed thresholds are read from artifact metadata, never presentation
  constants;
- candidates must pass model-promotion gates before replacing the incumbent;
- data/feed freshness is exposed through health metadata;
- live feed failure is labelled unavailable rather than displayed as zero;
- prediction exports include probability and the 3-day window, not deprecated
  pseudo-date fields;
- operator review is required.

## Maintenance

- Daily: refresh FIRMS data and generate a prediction snapshot.
- Every few days: retraining check; retain incumbent if a candidate fails.
- Per release: frontend typecheck/tests/build and backend compile/tests.
- After material drift: recalibrate/retrain with broader seasonal calibration
  coverage before changing production thresholds.

Card updated: 2026-07-24.
