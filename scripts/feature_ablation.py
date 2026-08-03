#!/usr/bin/env python
"""Leak-free weather-feature ablation on the incumbent pre-test snapshot.

Compares the exact same train/calibration/eval dates and seed for baseline,
weather-missingness indicators, causal drought features, and both. The frozen
final test begins after this experiment's cutoff and is never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import features
import thresholds
import train
from eval_protocol import date_grouped_split

FEATURES_PATH = os.path.join(ROOT, "outputs", "features", "full_features.parquet")
META_PATH = os.path.join(ROOT, "outputs", "metadata", "dataset_info.json")
OUT_PATH = os.path.join(ROOT, "outputs", "metadata", "ablation_weather_features.json")
SEED = 42
RAW_WEATHER = list(features.WEATHER_COLUMNS)


def _metrics(y, p, threshold):
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "average_precision": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "ece": round(train.expected_calibration_error(y, p), 4),
        "operating_point": thresholds.threshold_metrics(y, p, threshold),
    }


def main(n_estimators: int = 150) -> dict:
    started = time.time()
    meta = json.load(open(META_PATH))
    baseline = list(meta["features"])
    test_start = pd.Timestamp(meta["evaluation_protocol"]["split_date_ranges"]["test"][0])
    cutoff = test_start - pd.Timedelta(days=1)
    ds = pads.dataset(FEATURES_PATH, format="parquet")
    available = set(ds.schema.names)
    needed_weather = [c for c in RAW_WEATHER if c in available]
    cols = list(dict.fromkeys(baseline + needed_weather + ["date", "days_until_fire"]))
    df = ds.scanner(columns=cols, batch_size=200_000).to_table().to_pandas()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[df["date"] <= cutoff - pd.Timedelta(days=3)].reset_index(drop=True)
    if not df.sort_values(["lat_grid", "lon_grid", "date"]).index.equals(df.index):
        df = df.sort_values(["lat_grid", "lon_grid", "date"]).reset_index(drop=True)
    df["_y_bin"] = train._make_binary_label(df["days_until_fire"])

    for c in needed_weather:
        df[f"{c}_missing"] = df[c].isna().astype("float32")
    df = features.add_weather_drought_features(df)
    missing_cols = [f"{c}_missing" for c in needed_weather]
    drought_cols = ["consecutive_dry_days"]
    for window in features.WEATHER_DROUGHT_WINDOWS:
        drought_cols += [f"precip_sum_{window}d", f"precip_deficit_{window}d"]
    drought_cols = [c for c in drought_cols if c in df.columns]

    split = date_grouped_split(df["date"], val_fraction=0.10,
                               calib_fraction=0.10, test_fraction=0.20,
                               purge_days=7)
    tr = df[split.masks["train"]]
    cal = df[split.masks["calib"]]
    ev = df[split.masks["test"]]
    variants = {
        "baseline": baseline,
        "missingness": baseline + missing_cols,
        "drought": baseline + drought_cols,
        "missingness_plus_drought": baseline + missing_cols + drought_cols,
    }

    results = {}
    for name, feature_cols in variants.items():
        t0 = time.time()
        us = train._undersample_negatives(tr, "_y_bin", 4, random_state=SEED)
        natural_neg = int(len(tr) - tr["_y_bin"].sum())
        beta = int(len(us) - us["_y_bin"].sum()) / max(natural_neg, 1)
        weights = train._compute_sample_weights(
            pd.Series(us["_y_bin"].to_numpy()), us["date"], class_weighting=False)
        model = train._build_lgbm(random_state=SEED, scale_pos_weight=1.0)
        model.set_params(n_estimators=n_estimators)
        model.fit(us[feature_cols], us["_y_bin"].to_numpy(), sample_weight=weights)

        def corrected(frame):
            p = model.predict_proba(frame[feature_cols])[:, 1]
            return beta * p / (beta * p + 1.0 - p)

        raw_cal = corrected(cal)
        raw_ev = corrected(ev)
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, cal["_y_bin"].to_numpy())
        p_cal = iso.predict(raw_cal)
        p_ev = iso.predict(raw_ev)
        threshold = thresholds.f1_optimal_threshold(cal["_y_bin"].to_numpy(), p_cal)
        results[name] = {
            "feature_count": len(feature_cols),
            "added_features": [c for c in feature_cols if c not in baseline],
            "feature_contract_sha256": hashlib.sha256(
                "|".join(sorted(feature_cols)).encode()).hexdigest()[:16],
            "raw_ranking": {
                "roc_auc": round(float(roc_auc_score(ev["_y_bin"], raw_ev)), 4),
                "average_precision": round(float(average_precision_score(ev["_y_bin"], raw_ev)), 4),
            },
            "isotonic_calibrated": _metrics(ev["_y_bin"].to_numpy(), p_ev, threshold),
            "fit_seconds": round(time.time() - t0, 1),
        }
        print(name, results[name])

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "weather_feature_ablation",
        "selection_metric": "average_precision",
        "imbalance_mode": "C",
        "calibration": "isotonic fit on calibration split only",
        "seed": SEED,
        "n_estimators": n_estimators,
        "leakage_guard": f"data ends {cutoff.date()}, before frozen test start {test_start.date()}",
        "split": {"date_ranges": split.date_ranges,
                  "prevalence": {"train": round(float(tr['_y_bin'].mean()), 5),
                                 "calib": round(float(cal['_y_bin'].mean()), 5),
                                 "eval": round(float(ev['_y_bin'].mean()), 5)},
                  "rows": {"train": len(tr), "calib": len(cal), "eval": len(ev)}},
        "results": results,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {OUT_PATH}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-estimators", type=int, default=150)
    args = ap.parse_args()
    main(args.n_estimators)
