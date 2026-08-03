# Audit — Why the wildfire model misses its operational targets (2026-07)

Evidence-based audit of the ten suspected bottlenecks. Line references are to
the working tree at audit time. Machine-readable copy:
`outputs/metadata/audit_bottlenecks.json`.

Current locked model: trained 2026-07-16 · test AUC 0.9051 · AP 0.2175 @
prevalence 0.0324 · HIGH P/R 0.385/0.117 · WATCH P/R 0.227/0.316 · ECE 0.0503.

## 1. Seasonal / base-rate shift — CONFIRMED, the dominant failure mode

`evaluation_protocol.split_prevalence`: calib **0.3933** (Mar 29–May 7, fire
season) vs test **0.0324** (May 15–Jul 13, off-season) — **12.13×**. Thresholds
and Platt calibration are fitted on a world 12× more fire-dense than the one
they serve. High AUC (ranking survives shift) + collapsed alert recall
(absolute cutoffs do not transfer) is exactly this signature. The rolling
backtest (AUC 0.822 ± 0.021 across 2016–2025) confirms a single recent split
overstates the recent-window number.

**Response implemented**: drift flag + WATCH trade-off menu (earlier work),
relative Top-k ranking policy, and a leak-free ablation harness. Real fix
needs representative multi-season calibration data (Phase 3).

## 2. Compounded imbalance correction — CONFIRMED

Three corrections stack (train.py):
- 4:1 negative undersampling (`_undersample_negatives`, NEG_TO_POS_RATIO=4)
- inverse-class-frequency sample weights (`_compute_sample_weights`)
- `scale_pos_weight = neg/pos ≈ 4` on the undersampled frame

At 3.24% natural prevalence the effective positive up-weighting is
(29.9/4) × 4 × 4 ≈ **120×** — raw probabilities are grossly inflated
(calib ECE before Platt ≈ 0.14) and everything leans on one Platt fit in one
season to repair them.

**Response implemented**: `IMBALANCE_MODE` A/B/C/D (balancing applied exactly
once in A/B/C; C adds the analytic prior correction
`p = βp/(βp+1−p)`), plus `scripts/ablation.py` measuring all modes on identical
purged folds that END BEFORE the frozen test.

## 3. Tuning objective — CONFIRMED

Both search paths used `scoring="roc_auc"` only. At 3% prevalence AUC is
dominated by the easy negatives; PR-AUC tracks alert utility.

**Response implemented**: `TUNE_SCORING` env, default `average_precision`
(threshold selection unchanged: calib only). Both AUC and AP still reported.

## 4. Insufficient history — CONFIRMED (design provided, run deferred)

Promoted model used 400 days although `data/raw` holds 2012–2026 (4.16M
detections). Prior evidence: a naive 4-year train was auto-REJECTED (AUC 0.874
vs 0.896) because 2012–2024 is single-satellite — the multi-sat features
dilute. Memory-safe plan (Phase 3): per-year partitioned feature generation →
incremental parquet, recency weighting instead of discarding old seasons, and a
two-model design (long-history climatology core + recent weather model,
blended on calib). `rolling_backtest.py` already provides the per-fold harness.

## 5. Calibration — CONFIRMED, measured

`scripts/recalibrate.py` (chronological calib fit/eval, frozen base): isotonic
Brier 0.1281 / ECE 0.083 vs Platt 0.1673 / 0.171 on the calib eval window —
isotonic clearly better, but milestones (Brier ≤ 0.035, ECE ≤ 0.020) are NOT
met on in-season calibration data, and transfer to 3% months is unverified.
Seasonal calibrators gated on ≥6 months / ≥60 independent dates (currently 40
dates, 1 season → fallback, explicitly reported). Serving now supports both
classifier-style `predict_proba` and regressor-style `predict` calibrators;
isotonic candidates round-trip through the bundle while legacy Platt pickles
remain compatible. `recalibrate.py --write-candidate` writes the selected
isotonic bundle separately and never promotes it directly.

## 6. Missing weather → fake zeros — CONFIRMED

`features.py` weather block fills NaN with 0 (`fillna(0)`) — "no data" becomes
"0 °C / 0 mm", physically meaningful values. **Response implemented**:
`WEATHER_MISSING_INDICATORS=true` emits causal `{col}_missing` indicators
before the fill. Seven real-data drought signals were added: dry-day streak,
30/60/90-day precipitation totals, and positive FAO-56 ET0-minus-precipitation
water deficits. A pre-test ablation (2026-03-06→2026-05-11, n=772,692,
prevalence 0.3806, seed 42) improved raw AP **0.7387→0.7649** and AUC
**0.8293→0.8398** with drought features. Missingness alone was neutral on that
window; do not adopt it without cross-season evidence. VPD remains unavailable
because the cache has no humidity field; it was not fabricated.

## 7. Prediction universe — DOCUMENTED, unchanged

Cells enter only after their first FIRMS detection → first-ever fires are
unpredictable BY DESIGN. This is the honest estimand and is stated in the
metadata/UI. Bounded expansions to evaluate in Phase 3 (do not silently change
the estimand): spatial neighbors of active cells; forest/land-mask cells;
province/land-cover constrained. Report detection quality *and* geographic
coverage per universe; keep active-cell mode as default.

## 8. Feature audit — HARNESS PROVIDED, run deferred

LightGBM split importance ≠ causal usefulness. The ablation harness runs
feature-group experiments on identical purged folds; permutation importance on
out-of-time validation is the follow-up (needs a trained candidate per group —
hours, not run here).

## 9. multi_sat label weighting — CONFIRMED, made explicit

`multi_sat_confirmed` derives from the FUTURE target date and boosts training
weights 1.1–1.2×. It is not an input feature (no leakage into X) but it shifts
the estimand toward "multi-satellite-confirmed fires".
**Response implemented**: `MULTISAT_WEIGHTING` toggle (default true = legacy),
estimand note persisted in metadata.

## 10. Promotion protocol — FIXED

Earlier session fixed fail-open + wrong "candidate ≥ incumbent" wording and
added Brier/ECE/operating gates. This session adds the dedicated
**champion split** (`CHAMPION_FRACTION` > 0 ⇒ 5-way
train/val/calib/champion/test with purges): promotion compares on champion; the
final test is evaluated after — and independent of — the decision. Default off
until the next retrain adopts it.

## Experiment matrix (leak-free; frozen test never used for selection)

| # | Question | Config | Status |
|---|---|---|---|
| E1 | Does un-stacking class weights fix raw probabilities? | IMBALANCE_MODE A/B/C/D, identical folds | **ablation harness + run** |
| E2 | Does PR-AUC tuning pick better operating models? | TUNE_SCORING=average_precision vs roc_auc | next retrain |
| E3 | Do missingness indicators / drought signals help? | baseline vs missing vs drought vs both, identical pre-test split | **run — drought wins; missingness neutral** |
| E4 | Isotonic vs Platt vs beta calibration | scripts/recalibrate.py | **run — isotonic wins, candidate written, milestones unmet** |
| E5 | multi_sat estimand shift | MULTISAT_WEIGHTING=false vs true | next retrain |
| E6 | Multi-season candidate vs baselines | rolling_backtest + Phase-3 design | deferred (expensive) |

## Commands

```bash
# lightweight ablation (leak-free, ~10 min):
.venv/bin/python scripts/ablation.py --modes A,B,C,D

# next-retrain candidate with the audited fixes active:
IMBALANCE_MODE=C TUNE_SCORING=average_precision CHAMPION_FRACTION=0.08 \
WEATHER_MISSING_INDICATORS=true \
  python src/train.py --max-history-days 400 --n-ensemble 5 --fast

# full multi-season evidence (expensive, run with approval):
cd src && python rolling_backtest.py --years 2016-2026
```
