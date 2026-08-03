#!/usr/bin/env python
"""Lightweight, leak-free ablation runner (audit Phase 1).

Compares experiment configurations — currently the IMBALANCE_MODE class-
weighting variants (audited item 2) — on IDENTICAL purged chronological folds
built from the existing feature parquet.

LEAKAGE GUARD: the ablation window ends BEFORE the promoted bundle's final
held-out test start. Selection evidence must never touch the frozen test —
choosing a weighting mode is model selection. The transfer of these findings to
low-prevalence months therefore remains unverified until a new prospective
window exists; the report states this.

Per mode it fits ONE modest LightGBM (no hyperparameter search) on the mode's
train frame and evaluates raw + Platt-calibrated probabilities on the ablation
eval block at natural prevalence: ROC-AUC, PR-AUC, Brier, ECE. Output is
machine-readable with date ranges, prevalence, counts, contract hash, seed.

Run from project root (≈5–15 min for 4 modes):
    .venv/bin/python scripts/ablation.py [--modes A,B,C,D] [--n-estimators 150]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import train  # noqa: E402
from eval_protocol import date_grouped_split  # noqa: E402

FEATURES_PATH = os.path.join(ROOT, "outputs", "features", "full_features.parquet")
META_PATH = os.path.join(ROOT, "outputs", "metadata", "dataset_info.json")
OUT_DIR = os.path.join(ROOT, "outputs", "metadata")
SEED = 42


def _mode_train_frame(mode: str, train_df: pd.DataFrame, seed: int):
    """(frame, sample_weights, scale_pos_weight, beta) per IMBALANCE_MODE."""
    natural_neg = int(len(train_df) - train_df["_y_bin"].sum())
    if mode in ("A", "B"):
        us = train_df
        beta = 1.0
    else:
        us = train._undersample_negatives(train_df, "_y_bin", 4, random_state=seed)
        beta = int(len(us) - us["_y_bin"].sum()) / max(natural_neg, 1)
    sw = train._compute_sample_weights(
        pd.Series(us["_y_bin"].to_numpy()), us["date"],
        class_weighting=(mode == "D"))
    pos = int(us["_y_bin"].sum()); neg = int(len(us) - pos)
    spw = (neg / max(pos, 1)) if mode in ("B", "D") else 1.0
    return us, sw, spw, beta


def main(modes: List[str], n_estimators: int) -> None:
    t0 = time.time()
    meta = json.load(open(META_PATH))
    feature_cols = list(meta.get("features") or [])
    test_start = pd.to_datetime(
        meta["evaluation_protocol"]["split_date_ranges"]["test"][0])
    cutoff = test_start - pd.Timedelta(days=1)
    print(f"[1/3] Loading features < {cutoff.date()} "
          f"(final frozen test excluded from ablation evidence)")

    ds = pads.dataset(FEATURES_PATH, format="parquet")
    cols = list(set(feature_cols + ["date", "days_until_fire"]))
    tbl = ds.scanner(columns=cols, batch_size=200_000).to_table()
    df = tbl.to_pandas()
    del tbl
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[df["date"] <= cutoff].reset_index(drop=True)
    for c in feature_cols:
        if c not in ("lat_grid", "lon_grid") and df[c].dtype != np.float32:
            df[c] = df[c].astype(np.float32)
    df["_y_bin"] = train._make_binary_label(df["days_until_fire"])
    # censor the tail whose 3-day window crosses the cutoff
    df = df[df["date"] <= cutoff - pd.Timedelta(days=3)]

    split = date_grouped_split(df["date"], val_fraction=0.10,
                               calib_fraction=0.10, test_fraction=0.20,
                               purge_days=7)
    tr = df[split.masks["train"]]
    cal = df[split.masks["calib"]]
    ev = df[split.masks["test"]]        # ablation eval — NOT the frozen test
    print(f"  train={len(tr):,} calib={len(cal):,} eval={len(ev):,} "
          f"(eval prevalence={ev['_y_bin'].mean():.4f}, "
          f"range {split.date_ranges['test']})")

    contract_hash = hashlib.sha256("|".join(sorted(feature_cols)).encode()).hexdigest()[:16]
    results: Dict[str, Any] = {}
    for mode in modes:
        t_m = time.time()
        us, sw, spw, beta = _mode_train_frame(mode, tr, SEED)
        model = train._build_lgbm(random_state=SEED, scale_pos_weight=spw)
        model.set_params(n_estimators=n_estimators)
        model.fit(us[feature_cols], us["_y_bin"].to_numpy(), sample_weight=sw)

        raw_ev = model.predict_proba(ev[feature_cols])[:, 1]
        raw_cal = model.predict_proba(cal[feature_cols])[:, 1]
        y_ev = ev["_y_bin"].to_numpy(); y_cal = cal["_y_bin"].to_numpy()
        if mode == "C" and beta < 1.0:
            raw_ev = beta * raw_ev / (beta * raw_ev + (1 - raw_ev))
            raw_cal = beta * raw_cal / (beta * raw_cal + (1 - raw_cal))
        platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
        platt.fit(raw_cal.reshape(-1, 1), y_cal)
        cal_ev = platt.predict_proba(raw_ev.reshape(-1, 1))[:, 1]

        results[mode] = {
            "train_rows": int(len(us)), "train_pos": int(us["_y_bin"].sum()),
            "scale_pos_weight": round(float(spw), 3),
            "undersample_beta": round(float(beta), 5),
            "class_weighting": mode == "D",
            "raw": {
                "roc_auc": round(float(roc_auc_score(y_ev, raw_ev)), 4),
                "average_precision": round(float(average_precision_score(y_ev, raw_ev)), 4),
                "brier": round(float(brier_score_loss(y_ev, raw_ev)), 4),
                "ece": round(train.expected_calibration_error(y_ev, raw_ev), 4),
            },
            "platt_calibrated": {
                # Ranking metrics are invariant under monotone Platt scaling,
                # but persist them here so every reported candidate row is
                # self-contained and AP-first.
                "roc_auc": round(float(roc_auc_score(y_ev, cal_ev)), 4),
                "average_precision": round(float(average_precision_score(y_ev, cal_ev)), 4),
                "brier": round(float(brier_score_loss(y_ev, cal_ev)), 4),
                "ece": round(train.expected_calibration_error(y_ev, cal_ev), 4),
            },
            "fit_seconds": round(time.time() - t_m, 1),
        }
        r = results[mode]
        print(f"  mode {mode}: AUC={r['raw']['roc_auc']} AP={r['raw']['average_precision']} "
              f"raw ECE={r['raw']['ece']} → platt ECE={r['platt_calibrated']['ece']} "
              f"({r['fit_seconds']}s)")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "imbalance_mode_ablation",
        "leakage_guard": f"ablation data ends {cutoff.date()}, before frozen test "
                         f"start {test_start.date()} — frozen test untouched",
        "transfer_caveat": (
            "eval block prevalence is in-season; transfer of these findings to "
            "low-prevalence months is UNVERIFIED without a prospective window"),
        "seed": SEED,
        "n_estimators": n_estimators,
        "feature_count": len(feature_cols),
        "feature_contract_sha256": contract_hash,
        "split": {"date_ranges": split.date_ranges,
                  "eval_prevalence": round(float(ev["_y_bin"].mean()), 5),
                  "rows": {"train": len(tr), "calib": len(cal), "eval": len(ev)}},
        "results": results,
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(OUT_DIR, f"ablation_imbalance.{ts}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "ablation_imbalance.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[3/3] Wrote {path}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modes", default="A,B,C,D")
    ap.add_argument("--n-estimators", type=int, default=150)
    a = ap.parse_args()
    main([m.strip().upper() for m in a.modes.split(",") if m.strip()], a.n_estimators)
