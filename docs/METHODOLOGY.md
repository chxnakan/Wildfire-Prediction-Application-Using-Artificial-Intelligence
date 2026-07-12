# Methodology: How the Wildfire Predictor Works

This document explains the **methodology**, **safeguards against data leakage**, and **validation protocol** used in the Thailand Wildfire Imminence Predictor. It exists so that a sceptical reader (researcher, journalist, government official) can independently verify that the model's reported metrics are achievable and not fabricated.

## 🎯 What the model does

**Question answered:** Given a 0.1° grid cell in Thailand at date `t`, what is the probability that a fire (FIRMS hotspot detection) will occur in that cell within `[t+1, t+3]`?

**Output:** Calibrated probability ∈ [0, 1].

## 📊 The pipeline (end-to-end)

```
1.  fetch_firms.py     → NASA FIRMS VIIRS hotspots (raw)
2.  fetch_weather.py   → ERA5 daily reanalysis (optional)
3.  fetch_treecover.py → Hansen GFC aggregates (optional, run-once)
4.  data_loader.py     → load, clean, grid-snap, daily aggregation, densification, merge
5.  features.py        → 164 features per cell-day (lag + rolling + calendar + neighbors)
6.  train.py           → label, undersample, tune, refit, calibrate, evaluate, persist
7.  risk_map.py        → predict, filter, write GeoJSON for dashboard
8.  scripts/rolling_eval.py → monthly stability audit
```

Every step is **deterministic given inputs + random_state=42**.

## 🔬 Critical methodology: anti-leakage

The single most important safeguard is preventing **temporal data leakage** — using information from the future (relative to a prediction date) as a model feature. Without this discipline, ML models routinely score 0.99 AUC on a "predict the past" task that's useless in production.

### Rule 1: every rolling window uses `.shift(1)` BEFORE `.rolling()`

Every aggregation in `src/features.py` goes through `_past_roll()`:

```python
def _past_roll(df, col, window, agg="sum"):
    """CAUSAL: shift(1) before rolling — 'last N days, excluding today'."""
    shifted = grp[col].shift(1)
    regrouped = shifted.groupby([df["lat_grid"], df["lon_grid"]]).rolling(window, ...)
    ...
```

So `fire_sum_7d` answers *"how many fires in the past 7 days, NOT including today"*, never *"how many fires in the past 7 days including today"*. Today's value is captured separately via `fire_count_today` (a known input at inference time).

### Rule 2: lag features `shift(lag)` with `lag ≥ 1`

```python
for lag in LAG_DAYS:  # (1, 2, 3, 4, 5, 6, 7, 14, 21, 30)
    df[f"fire_lag_{lag}"] = grp["fire_count"].shift(lag).fillna(0)  # CAUSAL
```

`fire_lag_1` is literally *yesterday's* fire count, not today's.

### Rule 3: trend features use lag arithmetic, not raw differences

```python
df["frp_trend_3d"] = df["frp_lag_1"] - df["frp_lag_4"]  # CAUSAL
# (NOT df["frp_sum"].diff(3) — that would include today)
```

### Rule 4: streak counters use shifted fire flags

```python
fire_flag_past = (grp["fire_count"].shift(1).fillna(0) > 0).astype(int)
# ... streak built on fire_flag_past, never on fire_flag_today
```

### Rule 5: cumulative features `.shift(1)` after `.cumsum()`

```python
cum = fire_flag.groupby(GROUP_KEYS).cumsum() \
              .groupby(GROUP_KEYS).shift(1).fillna(0)
# cum[i] = total past fires up to YESTERDAY (not today)
```

### Verification: every feature tagged `# CAUSAL`

Every leakage-free feature line in `src/features.py` has a `# CAUSAL` comment. A grep audit confirms 134 of 134 features are tagged:

```bash
grep -c "# CAUSAL" src/features.py
# → 134
```

### The label itself

`days_until_fire` is computed **strictly from FUTURE rows**:

```python
for k in range(1, horizon + 1):
    shifted = df.groupby(GROUP_KEYS)["fire_count"].shift(-k).fillna(0)
    # shifted[i] = fire_count at i+k (future)
    future_fire = shifted > 0
    labels = np.where(future_fire & unset, k, labels)
```

So the label uses **only data from `[t+1, t+horizon]`**. By construction, no feature can equal or be derivable from the label (because all features are computed from `≤ t-1` or `t` data only).

## 📐 The split protocol

> **⚠️ Protocol revision (2026-07-12).** An external audit found the previous
> evaluation protocol invalid: the split was row-based (one calendar date
> could appear in two splits — confirmed for 2025-11-10), negatives were
> undersampled *before* splitting (val/test measured at an artificial 4:1
> base rate, not natural prevalence), calibration was fit on data the final
> model had been refit on, the deployment threshold was selected on the test
> set, tail rows carried false-negative labels, and the cell universe leaked
> hindsight. All of this is fixed below. **Every metric published before
> this revision is superseded and must not be quoted**; new numbers are
> produced by the next training run under the repaired protocol.

### Date-grouped chronological split with purge (whole dates, 4-way)

```python
from eval_protocol import date_grouped_split
split = date_grouped_split(dates, val_fraction=0.15,
                           calib_fraction=0.10, test_fraction=0.15,
                           purge_days=7)
# [ train ][7d purge][ val ][7d purge][ calib ][7d purge][ test ]
```

- Splits on **whole calendar dates** (fractions of unique-date count) — a
  date belongs to exactly one split, asserted at runtime.
- **7 calendar days are dropped entirely** before every boundary, so a
  training row's 3-day forward label window can never overlap the next
  split's period.
- Roles: **train** = fitting (undersampled, weighted) · **val** = early
  stopping · **calib** = calibration + threshold selection ·
  **test** = evaluated exactly once, last.
- Tail rows whose 3-day label window extends past the last observed date
  are **censored** (removed from all supervised use — their "no fire"
  labels would really mean "future not observed").

### Purged date-grouped cross-validation

During hyperparameter search:

```python
from eval_protocol import purged_date_folds
cv = purged_date_folds(train_dates, n_splits=5, purge_days=7)
```

sklearn's `TimeSeriesSplit(gap=7)` is **not used**: its `gap` counts *rows*,
and with thousands of cells per date a 7-row gap is a ~0-day purge. The
replacement folds are expanding windows over whole dates with a true
7-calendar-day purge.

### Negative undersampling (train partition ONLY)

Positive class is rare. To make training tractable:

```python
NEG_TO_POS_RATIO = 4   # keep 4 negatives per 1 positive — in TRAIN only
```

**val, calib, and test are never resampled** — they keep natural prevalence
so precision, PR-AUC, calibration, and Brier reflect production reality.

### Prediction universe (causal)

A grid cell enters the dataset **from its first FIRMS detection onward**.
The previous version included every historically-active cell across the full
date range, which encoded hindsight ("this cell will burn eventually") into
early negative rows. Documented limitation: cells that have never burned are
outside the universe — the model cannot flag first-ever fires.

## 🔧 Calibration

Raw probabilities from the LightGBM ensemble are **not calibrated** — "0.7" might mean "45% in practice". Platt scaling is fit on the **calib partition**: chronologically after val, disjoint from all fitting data, natural prevalence:

```python
calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
calibrator.fit(calib_raw_probabilities.reshape(-1, 1), calib_labels)
# Inference: model.predict_proba(X) → calibrator.predict_proba(raw)
```

The deployment threshold (F1-optimal) and the map's HIGH/MEDIUM tier cutoffs
are also selected on calib and **locked before test is touched**. ECE is
reported on calib (fit quality) and on test (honest generalization measure).

## 📈 Validation

### 1. Held-out test (single pass, natural prevalence)

> **Numbers pending.** The values previously published here (AUC 0.8451,
> ECE 0.0216, …) were measured under the invalidated protocol (undersampled
> test set, test-selected threshold, leaky split) and have been withdrawn.
> This table is regenerated by the first training run under the repaired
> protocol; read the live values from `dataset_info.json → model.test_metrics`.

Reported per run: ROC-AUC, PR-AUC (with the natural base rate alongside),
Brier score, ECE, precision/recall/F1 at the calib-locked deployment
threshold, precision@top-K%, and **baseline comparisons** (persistence:
fired-in-last-3d/7d; per-cell train-period climatology) so the model's
uplift over trivial strategies is explicit.

### 2. Rolling monthly evaluation (17 months)

```bash
python scripts/rolling_eval.py
```

Computes AUC, ECE, F1, precision, recall on a 1-month sliding window across the entire dataset. Output: `outputs/metadata/rolling_eval.json`. (Historical rolling numbers predating the protocol revision share the old protocol's flaws — regenerate before quoting.)

### 3. Retrospective validation (live)

Every prediction snapshot is **appended** to `outputs/riskmap/fire_dates_all.geojson` with a `validation_status` field. When `risk_map.py` runs and the prediction window has closed, it tags each past cell:

- **hit** — FIRMS hotspot observed in the predicted ±1-day window
- **miss** — no hotspot in the window
- **future** — window hasn't closed yet

The dashboard's "Past Predictions" panel displays per-snapshot hit rates so users can audit historical performance against ground truth.

## 🧮 Why the test positive rate is "low" but the model is "good"

Class imbalance is severe: globally ~5% of cell-days have a fire within the next 3 days; in the test window it's ~3.43% (some months are off-burn-season).

This means:
- **Accuracy is misleading** — always predicting "no fire" gives 96.6% accuracy. So we don't report it as a headline.
- **Precision is structurally low** — at threshold 0.05, precision is ~11%. This isn't bad; it's the expected behavior given the prior.
- **Watch-list framing is more honest** — "the top 20% highest-scored cells contain 3.4× more actual fires than random selection" — this is operationally useful and distribution-robust.

This is why the dashboard headlines **AUC**, **ECE**, **stability**, and **watch-list lift** instead of accuracy or recall.

## 🚫 Things we explicitly do NOT do

- ❌ **No tuning on test data** — deployment threshold + tier cutoffs are selected on calib and locked; test is evaluated exactly once (enforced by pipeline structure)
- ❌ **No date straddling folds** — whole-date splits with 7-day purges; disjointness is hard-asserted and unit-tested (`src/tests/test_eval_protocol.py`)
- ❌ **No resampling outside train** — val/calib/test stay at natural prevalence
- ❌ **No look-ahead in features** — `# CAUSAL` audit on every feature; cells enter the universe only after their first detection
- ❌ **No censored labels in supervision** — tail rows with unobservable label windows are excluded
- ❌ **No exact-date claims** — the output is P(fire within 3 days); day buckets on the dashboard are probability ranks, not date forecasts

## 🔁 Reproducibility

Anyone with the same code + same `FIRMS_API_KEY` can reproduce the model:

```bash
git clone https://github.com/0Maruz/Science-Project-version-3
cd Science-Project-version-3
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill FIRMS_API_KEY
./run.sh --fresh --weather --quick
```

- All randomness is seeded with `random_state=42`
- Hyperparameters are persisted in `outputs/metadata/dataset_info.json`
- Model history retained in `outputs/models/history/{TIMESTAMP}_lightgbm.pkl`

## 📜 Audit checklist for sceptical readers

If you want to verify the model is honest:

- [ ] Inspect `src/features.py` — every feature has a `# CAUSAL` comment + explanation
- [ ] Inspect `src/train.py` — chronological split, time-series CV with gap, no test-set tuning
- [ ] Inspect `scripts/post_calibrate.py` — Platt scaling fit on val only
- [ ] Inspect `scripts/rolling_eval.py` — independent monthly evaluation
- [ ] Check `outputs/metadata/dataset_info.json` — full hyperparameter trail + metrics
- [ ] Check `outputs/metadata/rolling_eval.json` — 17-month AUC time series
- [ ] Check `outputs/riskmap/fire_dates_all.geojson` — past predictions with validation_status
- [ ] Try fetching one base_date's prediction and check FIRMS for that date — it's all public data

## 📞 Disagree with anything?

Open a [GitHub issue](https://github.com/0Maruz/Science-Project-version-3/issues) with a concrete observation. The point of an open-source methodology document is precisely to be challenged.

---

*Methodology document version 0.5 · Last updated 2026-05-16*
