#!/usr/bin/env python
"""Post-hoc probability recalibration WITHOUT retraining the LightGBM ensemble.

The base ensemble is FROZEN — only the final probability calibrator is re-fit,
on calibration data, split CHRONOLOGICALLY (never random, never test). We
compare calibrators, report honest metrics, and write a SEPARATE candidate
bundle. The promoted model is never overwritten here (promotion is a separate,
gated step).

Method:
  1. Load the promoted bundle via the canonical loader; freeze base weights.
  2. Take raw (pre-calibration) ensemble probabilities on the calibration split
     recorded in the bundle metadata (evaluation_protocol.split_date_ranges).
  3. Split that calibration window CHRONOLOGICALLY into fit / eval sub-windows.
  4. Fit + compare calibrators on fit, select on eval (NOT test):
       - platt      (logistic / Platt scaling)
       - isotonic   (isotonic regression)
       - beta       (beta calibration, only if the `betacal` package is present)
  5. Report Brier, ECE, log-loss, reliability bins, calibration slope/intercept,
     sample count, positives, date range, independent dates for each.
  6. Write a candidate bundle + provenance (base SHA-256, calib-data SHA-256,
     date ranges, method/config, generated_at, feature contract). Never
     overwrites the promoted bundle.

Milestone ambitions (acceptance targets, NOT permission to fabricate):
  Milestone 1: Brier ≤ 0.035, ECE ≤ 0.035
  Milestone 2: Brier ≤ 0.030, ECE ≤ 0.020
If unmet, the real numbers + reason are reported and no bundle is promoted.

Run from project root:
    .venv/bin/python scripts/recalibrate.py            # report + candidate only
    .venv/bin/python scripts/recalibrate.py --write-candidate
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import train  # noqa: E402  (canonical loader + ECE + label helpers)

MODEL_PATH = os.path.join(ROOT, "outputs", "models", "lgbm_fire_date_model.pkl")
FEATURES_PATH = os.path.join(ROOT, "outputs", "features", "full_features.parquet")
OUT_DIR = os.path.join(ROOT, "outputs", "metadata")
CANDIDATE_PATH = os.path.join(ROOT, "outputs", "models", "recalibrated_candidate.pkl")
CHUNK = 200_000
RNG_SEED = 42


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return "unavailable"


def _reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> List[Dict[str, float]]:
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if m.sum() == 0:
            continue
        out.append({"bin_lower": round(float(lo), 3), "bin_upper": round(float(hi), 3),
                    "mean_predicted": round(float(p[m].mean()), 4),
                    "actual_rate": round(float(y[m].mean()), 4),
                    "count": int(m.sum())})
    return out


def _calib_slope_intercept(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """Logistic-regression of y on logit(p): slope=1, intercept=0 is perfect."""
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(logit.reshape(-1, 1), y)
    return round(float(lr.coef_[0][0]), 4), round(float(lr.intercept_[0]), 4)


def _metrics(y: np.ndarray, p: np.ndarray, dates: np.ndarray) -> Dict[str, Any]:
    slope, intercept = _calib_slope_intercept(y, p)
    return {
        "brier": round(float(brier_score_loss(y, p)), 4),
        "ece": round(train.expected_calibration_error(y, p), 4),
        "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": round(float(y.mean()), 4),
        "date_range": [str(pd.Timestamp(dates.min()).date()),
                       str(pd.Timestamp(dates.max()).date())],
        "independent_dates": int(pd.Series(dates).nunique()),
        "reliability_bins": _reliability(y, p),
    }


# ── calibrators (fit on fit-window, apply to any raw prob) ────────────────────

def _fit_platt(raw: np.ndarray, y: np.ndarray):
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
    lr.fit(raw.reshape(-1, 1), y)
    return lambda r: lr.predict_proba(r.reshape(-1, 1))[:, 1]


def _fit_isotonic(raw: np.ndarray, y: np.ndarray):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw, y)
    return lambda r: iso.predict(r)


def _fit_beta(raw: np.ndarray, y: np.ndarray):
    try:
        from betacal import BetaCalibration
    except ImportError:
        return None
    bc = BetaCalibration()
    bc.fit(raw.reshape(-1, 1), y)
    return lambda r: bc.predict(r.reshape(-1, 1))


def main(write_candidate: bool = False) -> None:
    print("[1/4] Loading frozen base model + calibration slice…")
    model, meta = train.load_model_bundle(MODEL_PATH)
    if meta is None:
        raise SystemExit("Refusing: bundle has no embedded metadata.")
    if not hasattr(model, "_raw_proba"):
        raise SystemExit("Refusing: base model exposes no _raw_proba "
                         "(cannot recalibrate without raw scores).")
    ep = meta.get("evaluation_protocol") or {}
    sdr = ep.get("split_date_ranges") or {}
    if "calib" not in sdr:
        raise SystemExit("Refusing: no calibration split range in metadata.")
    calib_lo, calib_hi = pd.to_datetime(sdr["calib"][0]), pd.to_datetime(sdr["calib"][1])
    feature_cols = list(meta.get("features") or [])

    pf = pq.ParquetFile(FEATURES_PATH)
    schema = set(pf.schema_arrow.names)
    missing = [c for c in feature_cols if c not in schema]
    if missing:
        raise SystemExit(f"Refusing: parquet missing {len(missing)} features.")
    needed = list(set(feature_cols + ["date", "days_until_fire"]))

    raws: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ds: List[np.ndarray] = []
    for batch in pf.iter_batches(batch_size=CHUNK, columns=needed):
        ch = batch.to_pandas()
        ch["date"] = pd.to_datetime(ch["date"]).dt.normalize()
        ch = ch[(ch["date"] >= calib_lo) & (ch["date"] <= calib_hi)]
        if ch.empty:
            continue
        for c in feature_cols:
            if c not in ("lat_grid", "lon_grid") and ch[c].dtype != np.float32:
                ch[c] = ch[c].astype(np.float32)
        raws.append(np.asarray(model._raw_proba(ch[feature_cols]), dtype=float))
        ys.append(train._make_binary_label(ch["days_until_fire"]).astype(int))
        ds.append(ch["date"].to_numpy())
    if not raws:
        raise SystemExit("Refusing: no calibration rows in the parquet — the "
                         "feature snapshot does not cover the calib range.")
    raw = np.concatenate(raws); y = np.concatenate(ys); dates = np.concatenate(ds)

    # Chronological split of the calibration window: earliest dates FIT, latest
    # dates EVAL. Never random, never test.
    order = np.argsort(dates, kind="mergesort")
    raw, y, dates = raw[order], y[order], dates[order]
    uniq = np.array(sorted(pd.unique(dates)))
    if len(uniq) < 4 or len(np.unique(y)) < 2:
        raise SystemExit("Refusing: calibration window too small / single-class "
                         "for a chronological fit/eval split.")
    cut_date = uniq[int(len(uniq) * 0.7)]
    fit_m = dates < cut_date
    eval_m = ~fit_m
    if len(np.unique(y[fit_m])) < 2 or len(np.unique(y[eval_m])) < 2:
        raise SystemExit("Refusing: fit or eval sub-window is single-class.")

    print(f"[2/4] Calib rows={len(y)} dates={len(uniq)}  "
          f"fit<{pd.Timestamp(cut_date).date()} eval>=")
    seasonal_months = int(pd.Series(dates).dt.month.nunique())
    indep_dates = int(len(uniq))

    print("[3/4] Fitting + comparing calibrators on identical rows…")
    calibrators = {"platt": _fit_platt, "isotonic": _fit_isotonic, "beta": _fit_beta}
    results: Dict[str, Any] = {}
    for name, fitter in calibrators.items():
        fn = fitter(raw[fit_m], y[fit_m])
        if fn is None:
            results[name] = {"available": False,
                             "reason": "betacal package not installed"}
            continue
        p_eval = np.clip(fn(raw[eval_m]), 0.0, 1.0)
        results[name] = {"available": True, **_metrics(y[eval_m], p_eval, dates[eval_m])}

    # Select the best AVAILABLE calibrator on EVAL (never test), by ECE then Brier.
    usable = {k: v for k, v in results.items() if v.get("available")}
    best_name = min(usable, key=lambda k: (usable[k]["ece"], usable[k]["brier"]))
    best = usable[best_name]

    # Seasonal usage gate: only claim seasonal calibration when coverage is real.
    seasonal_ok = seasonal_months >= 6 and indep_dates >= 60
    m1 = best["brier"] <= 0.035 and best["ece"] <= 0.035
    m2 = best["brier"] <= 0.030 and best["ece"] <= 0.020

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "post_hoc_recalibration",
        "provenance": {
            "base_model_sha256": _sha256(MODEL_PATH),
            "feature_snapshot_sha256": _sha256(FEATURES_PATH),
            "base_trained_at": meta.get("trained_at"),
            "feature_count": len(feature_cols),
            "calibration_date_range": [str(calib_lo.date()), str(calib_hi.date())],
            "fit_eval_split_cut": str(pd.Timestamp(cut_date).date()),
            "independent_dates": indep_dates,
            "seasonal_months": seasonal_months,
            "selected_on": "chronological calib eval sub-window (never test)",
        },
        "calibrator_comparison": results,
        "selected_calibrator": best_name,
        "selected_metrics": best,
        "seasonal_calibration_supported": seasonal_ok,
        "milestones": {
            "milestone_1_brier<=0.035_ece<=0.035": bool(m1),
            "milestone_2_brier<=0.030_ece<=0.020": bool(m2),
        },
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"recalibration_report.{ts}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "recalibration_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[4/4] Report → {out_path}")
    print(f"  best={best_name}  Brier={best['brier']}  ECE={best['ece']}  "
          f"slope={best['calibration_slope']}  M1={m1}  M2={m2}")

    if not write_candidate:
        print("ℹ️  Report only. Re-run with --write-candidate to write a SEPARATE "
              "candidate bundle (the promoted model is never overwritten here).")
        return

    # Refit the SELECTED calibrator on the FULL calibration window and write a
    # candidate bundle. The ensemble accepts both classifier-style
    # predict_proba calibrators (Platt) and regressor-style predict calibrators
    # (isotonic/beta), while old Platt bundles remain readable.
    if best_name == "platt":
        fitted = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
        fitted.fit(raw.reshape(-1, 1), y)
    elif best_name == "isotonic":
        fitted = IsotonicRegression(out_of_bounds="clip")
        fitted.fit(raw, y)
    elif best_name == "beta":
        from betacal import BetaCalibration
        fitted = BetaCalibration()
        fitted.fit(raw.reshape(-1, 1), y)
    else:  # fail closed if a future comparison adds an unsupported method
        raise RuntimeError(f"unsupported selected calibrator: {best_name}")
    candidate = copy.deepcopy(model)
    candidate.calibrator = fitted
    cand_meta = dict(meta)
    cand_meta["recalibration"] = report["provenance"] | {
        "method": f"{best_name}_refit_full_calib",
        "selection_metrics": best,
    }
    from storage import write_pickle
    write_pickle({"format": "bundle-v1", "model": candidate, "metadata": cand_meta},
                 CANDIDATE_PATH)
    print(f"✓ Candidate bundle → {CANDIDATE_PATH} (NOT promoted; promote via gates)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write-candidate", action="store_true",
                    help="Write a separate candidate bundle (never overwrites promoted)")
    args = ap.parse_args()
    main(write_candidate=args.write_candidate)
