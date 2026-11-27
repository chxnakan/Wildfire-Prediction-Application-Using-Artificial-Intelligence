# Research poster figures

Prepared for **Wildfire Prediction Web Application Using Artificial Intelligence**.

The three paired figures use the current, real **72-hour** evaluation of model
`98bfd2a97aab`. No simulated curves or newly downloaded observations are used.
The supplied poster image was not available; styling follows the written brief.
The dashboard and model have not been changed.

## Figure files and placement

| Group | PNG | Editable SVG |
| --- | --- | --- |
| Observed daily activity | `01_observed_daily_activity.png` | `01_observed_daily_activity.svg` |
| Calibration and score distribution | `02_calibration_and_scores.png` | `02_calibration_and_scores.svg` |
| ROC and precision–recall | `03_model_discrimination.png` | `03_model_discrimination.svg` |

All groups have a **2:1 aspect ratio**, a white background, and consistent fonts
and colors. PNGs are **4200 × 2100 pixels at 300 DPI**. SVGs retain editable text
and vector lines. Install DejaVu Sans when editing SVGs to preserve typography.

For the A1 poster, place the groups vertically within the Model Performance &
Validation section at approximately **35.6 cm wide × 17.8 cm high** each. Preserve
their aspect ratios and leave space between groups. Avoid reducing them below
about 30 cm wide without increasing label sizes. Prefer SVG for final layout and
printing; inspect a print at actual size before submission. Exact fit against the
reference poster cannot be checked without that poster.

## Verified results

- Forecast target: at least one eligible satellite hotspot in a Thailand grid
  location on days D+1 through D+3. This is not confirmed-wildfire prediction.
- Test base dates: **November 6, 2025 – January 14, 2026**.
- 259,860 overlapping grid-day windows; 32,655 positive windows.
- ROC-AUC: **0.7800356902**.
- Recall at threshold 0.50: **38.4229%**.
- Precision at threshold 0.50: **39.4126%**.
- Average precision: **0.3624660998**.
- Event prevalence / PR baseline: **12.5664%**.

The earlier AUC 0.779, recall 58.7%, precision 44.2%, and November 3–January 10
test period belong to the archived **seven-day** evaluation. They must not label
these 72-hour figures. The user selected the current three-day evaluation.

## Data definitions and interpretation

Figure 1 uses January 31, 2025 through January 17, 2026: the observation history
through the end of the three-day outcome window for the final test base date.
Each hotspot is a distinct satellite/time/pixel observation, not a separate fire.
Grid centres must fall inside the project's bundled Thailand boundaries.
Known volcano/static-land/offshore types are excluded, while untyped detections
remain eligible. The two panels use exactly the same dates. A day without records
anywhere in the supplied region is unknown coverage, plotted as a gap. The
trailing seven-day mean requires all seven days; no gaps are interpolated.

Figure 2 uses the saved calibrated held-out probabilities. The reliability curve
has ten equal-width bins, positioned at each bin's actual mean probability.
`calibration_bins.csv` preserves bin counts and observed frequencies. The score
histogram uses 40 equal-width bins and logarithmic frequency. Dashed lines show
the threshold and the separate medians of positive and negative scores. The
curve below the diagonal shows overestimation of event frequency in those bins;
the figure does not claim that calibration is perfect or improved on this test.

Figure 3 uses every held-out label and probability for ROC and precision–recall
calculations. Its PR baseline is measured event prevalence. Markers show the
operating point at threshold 0.50. Average precision is scikit-learn's
`average_precision_score`, not trapezoidal PR area. No smoothing or illustrative
replacement curves are used. Overlapping windows are not independent fires.

## Reproduction

The included `plot_data/` folder contains the actual reduced plotting evidence
and source SHA-256 hashes. No network access or original project checkout is
needed to regenerate the figures from this bundle.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python generate_figures.py
```

To rebuild the plotting evidence from the original project:

```bash
.venv/bin/python generate_figures.py --project-root "/path/to/wild fire"
```

Use `--output-dir /path/to/figures` to choose a different figure destination or
`--data-dir /path/to/evidence` to select another prepared evidence directory.
The script rejects inconsistent hashes, mixed model identities, incorrect
horizons, invalid probabilities, duplicate holdout keys, and metric mismatches.

`validation.json` records the calculations, missing observation dates, and exact
library versions used. `performance_summary.csv` gives the numerical table.
The original evaluation files and source observations are never modified.
