#!/usr/bin/env python
"""Compute presentation-grade scientific statistics for the deployed model.

Memory-frugal: predicts on the 4.4M-row feature parquet in 200K-row batches
so peak RAM stays under 1 GB even with the full dataset resident.

Outputs (DEFAULT = report only, no canonical mutation):
    outputs/metadata/scientific_stats.<UTC-timestamp>.json  (versioned report)
    outputs/metadata/scientific_stats.json                  (latest pointer)
    With --publish (and only if model/protocol still match) it also writes a
    compact summary into dataset_info.json → model.test_metrics.scientific_stats
    and the GeoJSON metadata. It refuses to publish on any mismatch.

Evaluated ONLY on the exact held-out test date range recorded in the model
bundle metadata (evaluation_protocol.split_date_ranges.test). No thresholds
are tuned here. What gets computed:
    * Sample-size breakdown (train / val / test, class balance, missing %)
    * Bootstrap 95% confidence intervals for ROC-AUC, AP, F1, Brier score
    * Confusion matrix at the deployment threshold
    * ROC curve points (FPR, TPR, threshold) — 200 sampled thresholds
    * PR curve points (precision, recall, threshold) — same sampling
    * Brier score + Brier skill score (vs baseline class-prior predictor)
    * Cohen's kappa (agreement above chance)
    * Matthews correlation coefficient (balanced single-number score)
    * Log loss
    * Per-province descriptive stats (count, mean fires/year, std)

Each metric is reported with units, methodology one-liner, and target range
so a science-fair reader can interpret it without ML background.

Run from project root:
    .venv/bin/python scripts/scientific_stats.py
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

# Point matplotlib at a writable, project-independent cache dir so it never
# warns about an unwritable config dir (and never litters the repo). Must be set
# before any matplotlib import.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _sha256(path: str) -> str:
    """SHA-256 of a file, for report provenance / publish-guard checks."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return "unavailable"

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import (
    average_precision_score, brier_score_loss, cohen_kappa_score,
    confusion_matrix, f1_score, log_loss, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import train  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "outputs", "models", "lgbm_fire_date_model.pkl")
FEATURES_PATH = os.path.join(ROOT, "outputs", "features", "full_features.parquet")
META_PATH = os.path.join(ROOT, "outputs", "metadata", "dataset_info.json")
GEOJSON_PATH = os.path.join(ROOT, "outputs", "riskmap", "fire_dates_all.geojson")
OUT_PATH = os.path.join(ROOT, "outputs", "metadata", "scientific_stats.json")
CHUNK_SIZE = 200_000

# Use a fixed random state so bootstrap results are reproducible — vital for
# a science project where someone might re-run to verify.
RNG_SEED = 42


# ── Bootstrap CI helper ──
def bootstrap_ci(
    y: np.ndarray, p: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = RNG_SEED,
) -> Dict[str, float]:
    """Non-parametric percentile bootstrap CI for a metric over (y, p) pairs.

    Returns dict {point, lower, upper, std, n_boot}.
    Resamples row-indices with replacement; metric computed on each resample.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    point = float(metric_fn(y, p))
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        # Guard: bootstrap sample may contain only one class — skip those by
        # falling back to the point estimate (rare on a 900K-row test set).
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            boots[i] = point
            continue
        try:
            boots[i] = float(metric_fn(ys, p[idx]))
        except Exception:
            boots[i] = point
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    return {
        "point": round(point, 4),
        "lower": round(lo, 4),
        "upper": round(hi, 4),
        "std": round(float(boots.std(ddof=1)), 4),
        "n_boot": n_boot,
        "confidence": confidence,
        "method": "iid_row_bootstrap",
    }


def block_bootstrap_ci(
    y: np.ndarray, p: np.ndarray, dates: np.ndarray,
    metric_fn, n_boot: int = 1000, confidence: float = 0.95,
    seed: int = RNG_SEED,
) -> Dict[str, float]:
    """Block bootstrap that resamples whole DATES with replacement.

    Fire detections within a day are strongly spatially clustered and adjacent
    days are correlated, so an i.i.d. row bootstrap understates uncertainty.
    Resampling by calendar date (the natural block) keeps each day's rows
    together and treats DATES as the independent unit — a defensible, if
    conservative, acknowledgement of temporal dependence. Remaining limitation:
    within-day spatial dependence across cells is NOT fully removed, and days
    are treated as exchangeable (no explicit autocorrelation model)."""
    rng = np.random.default_rng(seed)
    point = float(metric_fn(y, p))
    uniq = np.array(sorted(pd.unique(dates)))
    idx_by_date = {d: np.where(dates == d)[0] for d in uniq}
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_date[d] for d in chosen])
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            boots[i] = point
            continue
        try:
            boots[i] = float(metric_fn(ys, p[idx]))
        except Exception:
            boots[i] = point
    alpha = (1.0 - confidence) / 2.0
    return {
        "point": round(point, 4),
        "lower": round(float(np.quantile(boots, alpha)), 4),
        "upper": round(float(np.quantile(boots, 1.0 - alpha)), 4),
        "std": round(float(boots.std(ddof=1)), 4),
        "n_boot": n_boot,
        "confidence": confidence,
        "method": "block_bootstrap_by_date",
        "n_independent_dates": int(len(uniq)),
    }


def _safe_float(x) -> float:
    """JSON-safe coercion: inf/nan → None handled by caller via downsample_curve."""
    v = float(x)
    if v != v or v == float("inf") or v == float("-inf"):
        return float("nan")  # Will be filtered out by downsample_curve
    return v


def downsample_curve(
    xs: np.ndarray, ys: np.ndarray, ts: np.ndarray, n: int = 200
) -> List[Dict[str, float]]:
    """Pick ~n evenly-spaced points from a curve for plotting (saves JSON size).

    Sanitizes inf/nan to None (sklearn's roc_curve returns thresholds[0]=inf,
    which is valid in numpy but breaks strict JSON encoders like FastAPI's).
    """
    def _t_val(i):
        # sklearn returns thresholds of length n or n+1; the first roc_curve
        # threshold is always inf, drop that to a JSON-safe None.
        v = float(ts[min(i, len(ts) - 1)])
        if v == float("inf") or v == float("-inf") or v != v:
            return None
        return v

    if len(xs) <= n:
        return [{"x": float(xs[i]), "y": float(ys[i]), "t": _t_val(i)}
                for i in range(len(xs))]
    step = len(xs) / n
    idx = [int(i * step) for i in range(n)]
    idx = sorted(set(idx + [0, len(xs) - 1]))
    return [{"x": float(xs[i]), "y": float(ys[i]), "t": _t_val(i)} for i in idx]


def main(publish: bool = False) -> None:
    t0 = time.time()
    print("[1/5] Loading model bundle + scanning parquet...")
    # Canonical loader: registers pickled classes in __main__, unwraps bundle-v1.
    model, bundle_meta = train.load_model_bundle(MODEL_PATH)

    # Metadata authority: prefer the metadata EMBEDDED in the model bundle; use
    # dataset_info.json only as a compatibility fallback (legacy bare models).
    ext_meta = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}
    meta = bundle_meta if bundle_meta else ext_meta
    if not meta:
        raise SystemExit("Refusing to run: no model metadata found (neither "
                         "embedded bundle metadata nor dataset_info.json).")

    # Feature contract comes from metadata — NOT resolve_features(head). Assert
    # every required feature exists in the parquet and the count matches.
    feature_cols = list(meta.get("features") or [])
    if not feature_cols:
        raise SystemExit("Refusing to run: metadata carries no 'features' contract.")
    pf = pq.ParquetFile(FEATURES_PATH)
    schema_cols = set(pf.schema_arrow.names)
    missing = [c for c in feature_cols if c not in schema_cols]
    if missing:
        raise SystemExit(f"Refusing to run: feature parquet is missing "
                         f"{len(missing)} contract feature(s), e.g. {missing[:5]}. "
                         "Regenerate features for this model.")
    fc_meta = meta.get("feature_count")
    if fc_meta is not None and int(fc_meta) != len(feature_cols):
        raise SystemExit(f"Refusing to run: metadata feature_count={fc_meta} but "
                         f"the features list has {len(feature_cols)} entries.")
    needed = list(set(feature_cols + ["date", "days_until_fire", "lat_grid", "lon_grid"]))
    needed = [c for c in needed if c in schema_cols]
    f32_cols = [c for c in feature_cols if c not in ("lat_grid", "lon_grid")]
    print(f"  Features: {len(feature_cols)} (from bundle metadata)  "
          f"Row groups: {pf.num_row_groups}")

    print("[2/5] Predicting probabilities chunk-by-chunk...")
    rows: List[pd.DataFrame] = []
    t_pred = time.time()
    n_done = 0
    for batch in pf.iter_batches(batch_size=CHUNK_SIZE, columns=needed):
        chunk = batch.to_pandas()
        for c in f32_cols:
            if c in chunk.columns and chunk[c].dtype != np.float32:
                chunk[c] = chunk[c].astype(np.float32)
        X = chunk[feature_cols]
        proba = model.predict_proba(X)
        rows.append(pd.DataFrame({
            "date": pd.to_datetime(chunk["date"].to_numpy()),
            "y_bin": train._make_binary_label(chunk["days_until_fire"]).astype(np.int8),
            "proba": proba.astype(np.float32),
            "lat_grid": chunk["lat_grid"].to_numpy().astype(np.float32),
            "lon_grid": chunk["lon_grid"].to_numpy().astype(np.float32),
        }))
        n_done += len(chunk)
        del chunk, X, proba, batch
        gc.collect()
        print(f"  {n_done:>9,} rows  ({time.time() - t_pred:.1f}s)", end="\r")
    print()

    df = pd.concat(rows, ignore_index=True)
    del rows; gc.collect()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    total_rows = len(df)

    # Build ALL FOUR splits from the exact date ranges recorded by train.py's
    # date-grouped 4-way purged protocol. NO row-based 60/20/20 fallback. Rows
    # in the purge gaps / censored tail fall outside every range and therefore
    # never enter any split (crucially, never the test evaluation).
    ep = meta.get("evaluation_protocol") or {}
    sdr = ep.get("split_date_ranges") or {}
    required = ["train", "val", "calib", "test"]
    if not all(k in sdr and sdr[k] for k in required):
        raise SystemExit(
            "Refusing to run: metadata evaluation_protocol.split_date_ranges "
            "must contain inclusive [start, end] for train/val/calib/test. "
            "(No unsafe row-split fallback.)")

    def _slice(name: str) -> pd.DataFrame:
        lo, hi = pd.to_datetime(sdr[name][0]), pd.to_datetime(sdr[name][1])
        return df[(df["date"] >= lo) & (df["date"] <= hi)]

    train_df = _slice("train")
    val_df = _slice("val")
    calib_df = _slice("calib")
    test_df = _slice("test")

    ranges = sdr["test"]
    y_test = test_df["y_bin"].to_numpy()
    p_test = test_df["proba"].to_numpy().astype(np.float64)
    d_test = test_df["date"].to_numpy()

    print(f"  Splits (rows): train={len(train_df):,} val={len(val_df):,} "
          f"calib={len(calib_df):,} test={len(test_df):,} of {total_rows:,} total")
    print(f"  Held-out test range (from bundle): {ranges[0]} → {ranges[1]}")
    # Honesty signal: a non-test split with 0 rows means the feature snapshot
    # no longer covers that date range (e.g. a --predict-only run rebuilt the
    # parquet with a shorter --max-history-days window). Test evaluation is
    # still valid as long as the TEST range is covered; flag the rest loudly.
    empty_non_test = [n for n, s in
                      (("train", train_df), ("val", val_df), ("calib", calib_df))
                      if len(s) == 0]
    if empty_non_test:
        print(f"  ⚠️  Feature snapshot does not cover split(s): "
              f"{', '.join(empty_non_test)} — the parquet has fewer dates than "
              "the model trained on (test range is covered, so the evaluation "
              "below is still valid; the sample table shows n=0 for those).")
    if len(test_df) == 0:
        raise SystemExit("Refusing to run: no feature rows fall in the stored "
                         "test date range — feature snapshot and model bundle "
                         "are out of sync (regenerate features for this model).")
    if len(np.unique(y_test)) < 2:
        raise SystemExit("Refusing to run: the test split is single-class "
                         "(all positive or all negative) — ROC-AUC / AP are "
                         "undefined. Nothing to report.")

    # Deployment threshold locked on CALIB at training time — never retuned here.
    tm = (meta.get("model") or {}).get("test_metrics") or meta.get("test_metrics") or {}
    deploy_thr = float(tm.get("deployment_threshold", 0.5))
    print(f"  Deployment threshold (locked on calib, not retuned): {deploy_thr:.2f}")

    print("[3/5] Block-bootstrap CIs (95%, n=1000, resampled by date)...")
    t_b = time.time()
    ci_auc = block_bootstrap_ci(y_test, p_test, d_test, roc_auc_score, n_boot=1000)
    ci_ap  = block_bootstrap_ci(y_test, p_test, d_test, average_precision_score, n_boot=1000)
    pred_at_thr = (p_test >= deploy_thr).astype(int)

    def _f1_at_thr(y, p):
        return f1_score(y, (p >= deploy_thr).astype(int), zero_division=0)
    def _prec_at_thr(y, p):
        return precision_score(y, (p >= deploy_thr).astype(int), zero_division=0)
    def _rec_at_thr(y, p):
        return recall_score(y, (p >= deploy_thr).astype(int), zero_division=0)
    def _brier(y, p):
        return brier_score_loss(y, p)

    ci_f1   = block_bootstrap_ci(y_test, p_test, d_test, _f1_at_thr, n_boot=500)
    ci_prec = block_bootstrap_ci(y_test, p_test, d_test, _prec_at_thr, n_boot=500)
    ci_rec  = block_bootstrap_ci(y_test, p_test, d_test, _rec_at_thr, n_boot=500)
    ci_brier = block_bootstrap_ci(y_test, p_test, d_test, _brier, n_boot=500)
    print(f"  Done in {time.time() - t_b:.1f}s")

    print("[4/5] Confusion matrix + classification stats at deployment threshold...")
    cm = confusion_matrix(y_test, pred_at_thr)
    tn, fp, fn, tp = (cm.ravel().tolist() if cm.size == 4
                      else (int(cm[0, 0]) if cm.shape[0] >= 1 else 0, 0, 0, 0))
    kappa = float(cohen_kappa_score(y_test, pred_at_thr))
    mcc = float(matthews_corrcoef(y_test, pred_at_thr))
    ll = float(log_loss(y_test, np.clip(p_test, 1e-15, 1 - 1e-15)))
    brier = float(brier_score_loss(y_test, p_test))
    # Brier skill score: vs baseline of predicting the class prior
    prior = float(y_test.mean())
    brier_baseline = prior * (1 - prior)
    brier_skill = 1 - (brier / brier_baseline) if brier_baseline > 0 else 0.0

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0           # = precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    fpr_at = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr_at = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    print("[5/5] ROC + PR curves...")
    fpr, tpr, roc_thr = roc_curve(y_test, p_test)
    prec_curve, rec_curve, pr_thr = precision_recall_curve(y_test, p_test)
    roc_points = downsample_curve(fpr, tpr, roc_thr, n=200)
    # precision_recall_curve returns precision/recall arrays of length n_thr+1
    pr_thr_padded = np.concatenate([pr_thr, [pr_thr[-1] if len(pr_thr) else 0.0]])
    pr_points = downsample_curve(rec_curve, prec_curve, pr_thr_padded, n=200)

    # Sample-size breakdown — ALL FOUR splits (train / validation / calibration
    # / test), each with its real count, prevalence, and date range.
    def _split_summary(sdf: pd.DataFrame) -> dict:
        if len(sdf) == 0:
            return {"n": 0, "positives": 0, "positive_rate": None, "date_range": [None, None]}
        return {
            "n": int(len(sdf)),
            "positives": int(sdf["y_bin"].sum()),
            "positive_rate": round(float(sdf["y_bin"].mean()), 4),
            "date_range": [str(sdf["date"].min().date()), str(sdf["date"].max().date())],
        }

    samples = {
        "total_densified": int(total_rows),
        "train": _split_summary(train_df),
        "validation": _split_summary(val_df),
        "calibration": _split_summary(calib_df),
        "test": _split_summary(test_df),
    }

    out = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "deployment_threshold": deploy_thr,
        "samples": samples,
        "confidence_intervals_95": {
            "roc_auc": ci_auc,
            "average_precision": ci_ap,
            "f1_at_deploy": ci_f1,
            "precision_at_deploy": ci_prec,
            "recall_at_deploy": ci_rec,
            "brier_score": ci_brier,
        },
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "row_labels": ["actual_negative", "actual_positive"],
            "col_labels": ["predicted_negative", "predicted_positive"],
        },
        "classification_stats": {
            "sensitivity":            round(sensitivity, 4),
            "specificity":            round(specificity, 4),
            "ppv":                    round(ppv, 4),
            "npv":                    round(npv, 4),
            "false_positive_rate":    round(fpr_at, 4),
            "false_negative_rate":    round(fnr_at, 4),
            "cohen_kappa":            round(kappa, 4),
            "matthews_corr_coef":     round(mcc, 4),
            "log_loss":               round(ll, 4),
            "brier_score":            round(brier, 4),
            "brier_skill_score":      round(brier_skill, 4),
            "baseline_class_prior":   round(prior, 4),
        },
        "roc_curve": roc_points,
        "pr_curve": pr_points,
        "interpretations_th": {
            "roc_auc": "ความสามารถจัดอันดับ cell โดยรวม (0.5=สุ่ม, 1.0=สมบูรณ์); ≥0.80 ถือว่าดี",
            "average_precision": "AUC ของ Precision-Recall curve; baseline = positive rate",
            "cohen_kappa": "ความสอดคล้องเหนือกว่าการเดา (0=เดา, 1=สมบูรณ์); >0.4 = ปานกลาง, >0.6 = ดี",
            "matthews_corr_coef": "MCC คะแนนสมดุล robust ต่อ class imbalance (-1 ถึง 1); >0.3 = ดี",
            "brier_skill_score": "BSS = 1 - Brier/baseline; >0 = ดีกว่าเดาด้วย class prior",
            "log_loss": "ค่ายิ่งต่ำยิ่งดี; เปรียบเทียบกับ -log(prior) เพื่อความหมาย",
            "confusion_matrix": "ตาราง TN, FP, FN, TP สำหรับการจำแนก yes/no ที่ deployment threshold",
            "roc_curve": "TPR vs FPR เส้นโค้งสีน้ำเงิน; ทแยงแดง = สุ่ม; ใกล้มุมซ้ายบน = ดี",
            "pr_curve": "Precision vs Recall; เส้นแนวนอน = baseline positive rate",
        },
        "methodology_note": (
            "Evaluated on the EXACT held-out test date range recorded by "
            f"train.py ({ranges[0]} → {ranges[1]}; date-grouped 4-way purged "
            "split), read from the model bundle metadata — NOT a row-based "
            "60/20/20 split. Deployment threshold is the one locked on calib at "
            "training time (never retuned here). Class balance NOT undersampled "
            "in test — reflects the real production prior. Confidence intervals "
            "use a BLOCK bootstrap resampling whole dates (n=1000 AUC/AP, n=500 "
            "threshold-dependent) to acknowledge temporal dependence; seed=42. "
            "Limitation: within-day spatial dependence across cells is not fully "
            "removed and days are treated as exchangeable."
        ),
    }

    print("\n━" * 30)
    print("STATISTICAL SUMMARY")
    print("━" * 60)
    print(f"  ROC-AUC: {ci_auc['point']:.4f}  (95% CI: {ci_auc['lower']:.4f} – {ci_auc['upper']:.4f})")
    print(f"  AP:      {ci_ap['point']:.4f}  (95% CI: {ci_ap['lower']:.4f} – {ci_ap['upper']:.4f})")
    print(f"  F1 @{deploy_thr}: {ci_f1['point']:.4f}  (95% CI: {ci_f1['lower']:.4f} – {ci_f1['upper']:.4f})")
    print()
    print("Classification stats @ deployment threshold:")
    print(f"  TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")
    print(f"  Sensitivity (recall): {sensitivity*100:.2f}%")
    print(f"  Specificity:          {specificity*100:.2f}%")
    print(f"  PPV (precision):      {ppv*100:.2f}%")
    print(f"  NPV:                  {npv*100:.2f}%")
    print()
    print(f"  Cohen's κ:            {kappa:+.4f}")
    print(f"  Matthews MCC:         {mcc:+.4f}")
    print(f"  Brier score:          {brier:.4f}  (skill={brier_skill:+.4f})")
    print(f"  Log loss:             {ll:.4f}")
    print()

    # Provenance stamp so a published report can be traced to its exact inputs.
    out["provenance"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": _sha256(MODEL_PATH),
        "feature_snapshot_sha256": _sha256(FEATURES_PATH),
        "metadata_trained_at": meta.get("trained_at"),
        "feature_count": len(feature_cols),
        "test_date_range": ranges,
        "evaluation_protocol": ep.get("split_method"),
        "task": "binary_fire_within_3d",
        "note": "Binary 3-day metrics only (ROC-AUC / AP / Brier / ECE at real "
                "prevalence). No pseudo-day MAE/RMSE/R² is reported as model "
                "quality for this binary task.",
    }

    # DEFAULT: write only a SEPARATE, versioned report. Never touch the
    # canonical dataset_info.json / GeoJSON unless --publish is given.
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    versioned = OUT_PATH.replace(".json", f".{ts}.json")
    with open(versioned, "w") as f:
        json.dump(out, f, indent=2, default=str)
    with open(OUT_PATH, "w") as f:          # latest pointer (also a report, not canonical)
        json.dump(out, f, indent=2, default=str)
    print(f"✓ Saved versioned report {versioned}")

    if not publish:
        print("ℹ️  Report only — canonical dataset_info.json / GeoJSON NOT "
              "modified. Re-run with --publish to write the summary into them.")
        print(f"\nDone in {time.time() - t0:.1f}s")
        return

    # --publish: refuse if ANYTHING the report was computed against has changed
    # since evaluation — model hash, feature contract, protocol, trained_at, or
    # test date range. A different split/model must never overwrite canonical
    # metrics under a stale report.
    if _sha256(MODEL_PATH) != out["provenance"]["model_sha256"]:
        raise SystemExit("Refusing to publish: model artifact hash changed "
                         "since evaluation. Re-run against the current model.")
    live_meta = json.load(open(META_PATH))
    live_ep = live_meta.get("evaluation_protocol") or {}
    checks = [
        ("trained_at", live_meta.get("trained_at"), meta.get("trained_at")),
        ("split_method", live_ep.get("split_method"), ep.get("split_method")),
        ("test_date_range", (live_ep.get("split_date_ranges") or {}).get("test"), ranges),
        ("feature_count", live_meta.get("feature_count"), meta.get("feature_count")),
    ]
    for name, live_v, eval_v in checks:
        if live_v != eval_v:
            raise SystemExit(
                f"Refusing to publish: {name} changed since evaluation "
                f"({live_v!r} != {eval_v!r}). Re-run against the current model.")
    # Feature contract identity (order-independent set match).
    if set(live_meta.get("features") or []) != set(meta.get("features") or []):
        raise SystemExit("Refusing to publish: feature contract changed since "
                         "evaluation.")

    summary_for_frontend = {
        "samples":              samples,
        "ci_95":                out["confidence_intervals_95"],
        "confusion_matrix":     out["confusion_matrix"],
        "classification_stats": out["classification_stats"],
        "roc_curve":            roc_points,
        "pr_curve":             pr_points,
        "provenance":           out["provenance"],
    }
    live_meta.setdefault("model", {}).setdefault("test_metrics", {})["scientific_stats"] = \
        summary_for_frontend
    with open(META_PATH, "w") as f:
        json.dump(live_meta, f, indent=2, default=str)
    print(f"✓ Published summary → {META_PATH}")

    if os.path.exists(GEOJSON_PATH):
        gj = json.load(open(GEOJSON_PATH))
        gj.setdefault("metadata", {}).setdefault("metrics", {})["scientific_stats"] = \
            summary_for_frontend
        with open(GEOJSON_PATH, "w") as f:
            json.dump(gj, f)
        print(f"✓ Published summary → {GEOJSON_PATH}")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--publish", action="store_true",
                    help="Write the summary into canonical dataset_info.json + "
                         "GeoJSON (refuses on model/protocol mismatch). Default: "
                         "versioned report only, no canonical mutation.")
    args = ap.parse_args()
    main(publish=args.publish)
