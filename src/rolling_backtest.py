"""Walk-forward (rolling-origin) backtest across fire seasons.

Why this exists
---------------
train.py's single held-out test window answers *"how good is one model on one
recent period?"*. It is a deliberately harsh number: a threshold frozen on an
early-year calib split is measured against a late-year, off-season test window,
so its **recall reads low (~13%)** even though ranking quality (AUC ≈ 0.9) is
strong. That gap is seasonal drift, not a broken model — and the deployed system
does not actually run a frozen 6-month-old threshold: sync/retrain refresh it
every few days on recent data.

This script measures what that *deployed* system would actually have delivered,
season by season. For each target fire-season year Y it:

    train on everything strictly before season Y  ── mimics "what we knew then"
      │   (last `--calib-days` of it = calib, natural prevalence)
      ▼
    pick the deployment threshold on that calib split  ── exactly like prod STEP 8
      ▼
    evaluate on season Y at natural prevalence  ── the honest per-season number

7-day purges sit between train-fit / calib / test so no label window straddles a
boundary (same invariant as train.py). Every fold reuses train.py's real
building blocks — label construction, tail censoring, train-only undersampling,
recency sample weights, Platt calibration, F1-optimal threshold — so the numbers
are directly comparable to production, just repeated over history.

Output
------
    outputs/metadata/rolling_backtest.json   per-fold + aggregate metrics
plus a printed summary table. Results are written incrementally and completed
folds are skipped on rerun (resumable).

Usage
-----
    cd src && python rolling_backtest.py                     # 2014..present
    cd src && python rolling_backtest.py --years 2019-2026
    cd src && python rolling_backtest.py --season 1-4        # Jan–Apr test window
    cd src && python rolling_backtest.py --n-estimators 400 --quick

NOTE: features are causal (past-only), so building them once over full history
and slicing by date gives each row exactly the values it would have had if built
per-fold — no leakage, far cheaper than rebuilding per fold.
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_loader import load_and_prepare
from eval_protocol import observed_label_mask
from features import MAX_PREDICTION_DAYS, build_features, resolve_features
from storage import resolve_existing, write_json
from train import (
    IMMINENT_DAYS,
    _compute_sample_weights,
    _build_lgbm,
    _make_binary_label,
    _paths,
    _undersample_negatives,
    expected_calibration_error,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("rolling_backtest")

PURGE_DAYS = 7
WEATHER_COLUMNS = ["temp_max", "temp_min", "precip_sum", "wind_max", "et0"]


# ─────────────────────────────────────────────────────────────────────────────
# Threshold selection — shared with train.py via thresholds.py (no duplication).
# The backtest mirrors deployment: pick thresholds on each fold's calib tail
# only, never on the fold's test season.
# ─────────────────────────────────────────────────────────────────────────────

from thresholds import (
    f1_optimal_threshold as _f1_optimal_threshold,
    threshold_metrics as _metrics_at,
)
import thresholds as _thr


def _recall_target_threshold(y: np.ndarray, proba: np.ndarray, target_recall: float) -> float:
    """Recall-first "watch list" cutoff (thin wrapper — drops the fallback
    reason the backtest doesn't need)."""
    thr, _reason = _thr.recall_target_threshold(y, proba, target_recall)
    return thr


def _precision_at_k(y: np.ndarray, proba: np.ndarray, k_frac: float) -> float:
    n = max(1, int(len(proba) * k_frac))
    top = np.argsort(-proba)[:n]
    return float(y[top].mean())


def _drop_sparse_weather(window: pd.DataFrame, min_coverage: float) -> pd.DataFrame:
    """Apply the production weather gate to one bounded fold window only."""
    present = [c for c in WEATHER_COLUMNS if c in window.columns]
    if not present:
        return window
    coverage = float(window[present[0]].notna().mean())
    if coverage < min_coverage:
        return window.drop(columns=present)
    return window


# ─────────────────────────────────────────────────────────────────────────────
# One walk-forward fold
# ─────────────────────────────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root,
            stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except Exception:
        return "unknown"


def _raw_data_fingerprint(raw_dir: str) -> str:
    """Cheap fingerprint of the input FIRMS parquets (name/size/mtime) — binds
    the backtest output to the exact data it ran on without hashing GBs."""
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(raw_dir, "fire_*.parquet"))):
        st = os.stat(f)
        h.update(f"{os.path.basename(f)}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def _provenance(paths: dict, feature_cols: List[str], n_estimators: int,
                max_train_years: float, calib_days: int,
                season_months: Tuple[int, int]) -> dict:
    """Bind the backtest result to the code + data + config it was produced
    with, so a stored report can be traced and reproduced."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_version": _git_commit(),
        "evaluation": "out_of_sample_rolling_origin (retrain per fold)",
        "feature_count": len(feature_cols),
        "feature_contract_sha256":
            hashlib.sha256("|".join(sorted(feature_cols)).encode()).hexdigest()[:16],
        "raw_data_fingerprint": _raw_data_fingerprint(paths.get("raw_dir", "")),
        "model_configuration": {
            "n_estimators": n_estimators,
            "max_train_years": max_train_years,
            "calib_days": calib_days,
            "purge_days": PURGE_DAYS,
            "season_months": list(season_months),
            "imminent_days": IMMINENT_DAYS,
        },
    }


def fold_windows(year: int, season_months: Tuple[int, int], calib_days: int,
                 max_train_years: float) -> Dict[str, "pd.Timestamp"]:
    """Chronological fit / calib / test date windows for one rolling-origin
    fold, with a PURGE_DAYS gap before each boundary. Pure + testable:
    guarantees fit_end precedes calib_start and calib_end precedes test_start by
    > PURGE_DAYS, so no forward label window straddles a split (no leakage)."""
    m0, m1 = season_months
    test_start = pd.Timestamp(year=year, month=m0, day=1)
    test_end = pd.Timestamp(year=year, month=m1, day=1) + pd.offsets.MonthEnd(0)
    calib_end = test_start - pd.Timedelta(days=PURGE_DAYS + 1)
    calib_start = calib_end - pd.Timedelta(days=calib_days - 1)
    fit_end = calib_start - pd.Timedelta(days=PURGE_DAYS + 1)
    fit_start = fit_end - pd.Timedelta(days=int(round(max_train_years * 365.25)))
    return {"fit_start": fit_start, "fit_end": fit_end,
            "calib_start": calib_start, "calib_end": calib_end,
            "test_start": test_start, "test_end": test_end}


def _run_fold(
    feats: pd.DataFrame,
    feature_cols: List[str],
    year: int,
    season_months: Tuple[int, int],
    calib_days: int,
    n_estimators: int,
    random_state: int,
    max_train_years: float = 3.0,
) -> Optional[dict]:
    """Train on data before season Y, calibrate + pick threshold on the tail of
    that data, evaluate on season Y. Returns a metrics dict or None when there
    isn't enough history.

    Training history is capped to the most recent `max_train_years` before the
    season — both a realistic deployment choice (a live model retrains on recent
    data, not 15-year-old fires) and a memory bound for the fold loop."""
    dates = pd.to_datetime(feats["date"])
    w = fold_windows(year, season_months, calib_days, max_train_years)
    test_mask = (dates >= w["test_start"]) & (dates <= w["test_end"])
    if test_mask.sum() == 0:
        return {"year": year, "status": "skipped", "reason": "no_rows_in_season"}

    calib_mask = (dates >= w["calib_start"]) & (dates <= w["calib_end"])
    fit_mask = (dates >= w["fit_start"]) & (dates <= w["fit_end"])

    if fit_mask.sum() < 5000 or calib_mask.sum() == 0:
        return {"year": year, "status": "skipped",
                "reason": "insufficient_history",
                "fit_rows": int(fit_mask.sum()), "calib_rows": int(calib_mask.sum())}

    y_all = _make_binary_label(feats["days_until_fire"])
    # Censor rows whose 3-day label window runs past the last observed date.
    observed = observed_label_mask(feats["date"], IMMINENT_DAYS)

    fit_idx = np.where(fit_mask.to_numpy() & observed)[0]
    calib_idx = np.where(calib_mask.to_numpy() & observed)[0]
    test_idx = np.where(test_mask.to_numpy() & observed)[0]
    if len(np.unique(y_all[calib_idx])) < 2 or len(np.unique(y_all[test_idx])) < 2:
        # Report the skip with its reason — NEVER substitute zeros for a
        # single-class fold (that would fabricate a metric).
        return {"year": year, "status": "skipped", "reason": "single_class_split",
                "calib_positives": int(y_all[calib_idx].sum()),
                "test_positives": int(y_all[test_idx].sum())}

    # ── Train (same single-mechanism IMBALANCE_MODE contract as train.py) ──
    fit_df = feats.iloc[fit_idx].copy()
    fit_df["_y_bin"] = y_all[fit_idx]
    mode = os.getenv("IMBALANCE_MODE", "D").upper()
    if mode not in {"A", "B", "C", "D"}:
        mode = "D"
    natural_neg = int(len(fit_df) - fit_df["_y_bin"].sum())
    if mode in {"C", "D"}:
        fit_df = _undersample_negatives(fit_df, random_state=random_state)

    X_fit = fit_df[feature_cols]
    y_fit = fit_df["_y_bin"].to_numpy()
    multi_sat = (fit_df["multi_sat_confirmed"].to_numpy()
                 if "multi_sat_confirmed" in fit_df.columns else None)
    sw = _compute_sample_weights(
        fit_df["_y_bin"], fit_df["date"], multi_sat=multi_sat,
        class_weighting=(mode == "D"))

    pos = int(y_fit.sum())
    neg = int(len(y_fit) - pos)
    spw = (neg / max(pos, 1)) if mode in {"B", "D"} else 1.0
    beta = neg / max(natural_neg, 1) if mode == "C" else 1.0
    model = _build_lgbm(random_state=random_state, scale_pos_weight=spw)
    model.set_params(n_estimators=n_estimators)
    model.fit(X_fit, y_fit, sample_weight=sw)

    # ── Platt calibration + threshold selection on CALIB (natural prev.) ────
    calib_raw = model.predict_proba(feats.iloc[calib_idx][feature_cols])[:, 1]
    if beta < 1.0:
        calib_raw = beta * calib_raw / (beta * calib_raw + 1.0 - calib_raw)
    y_calib = y_all[calib_idx]
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    platt.fit(calib_raw.reshape(-1, 1), y_calib)

    def _cal(raw: np.ndarray) -> np.ndarray:
        return platt.predict_proba(raw.reshape(-1, 1))[:, 1]

    calib_proba = _cal(calib_raw)
    deploy_thr = _f1_optimal_threshold(y_calib, calib_proba)
    watch_thr = _recall_target_threshold(y_calib, calib_proba, target_recall=0.90)

    # ── Evaluate on the season at natural prevalence ────────────────────────
    test_raw = model.predict_proba(feats.iloc[test_idx][feature_cols])[:, 1]
    if beta < 1.0:
        test_raw = beta * test_raw / (beta * test_raw + 1.0 - test_raw)
    y_test = y_all[test_idx]
    test_proba = _cal(test_raw)

    prevalence = float(y_test.mean())
    auc = float(roc_auc_score(y_test, test_proba))
    ap = float(average_precision_score(y_test, test_proba))
    brier = float(brier_score_loss(y_test, test_proba))
    ece = float(expected_calibration_error(y_test, test_proba))

    # Persistence baseline: rank by recent fire activity (past-only features).
    baseline_auc = None
    for col in ("fire_sum_3d", "fire_sum_7d"):
        if col in feats.columns:
            s = feats.iloc[test_idx][col].to_numpy(float)
            if len(np.unique(s)) > 1:
                baseline_auc = round(float(roc_auc_score(y_test, s)), 4)
                break

    result = {
        "year": year,
        "status": "ok",
        "test_window": [str(w["test_start"].date()), str(w["test_end"].date())],
        "train_fit_window": [str(w["fit_start"].date()), str(w["fit_end"].date())],
        "calib_window": [str(w["calib_start"].date()), str(w["calib_end"].date())],
        "purge_days": PURGE_DAYS,
        "imbalance_mode": mode,
        "n_train_fit": int(len(fit_idx)),
        "n_calib": int(len(calib_idx)),
        "n_test": int(len(test_idx)),
        "test_prevalence": round(prevalence, 4),
        "roc_auc": round(auc, 4),
        "average_precision": round(ap, 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "persistence_auc": baseline_auc,
        # Two operating points, both chosen on calib and applied to test:
        "deployment": _metrics_at(y_test, test_proba, deploy_thr),      # F1-optimal
        "watch_list": _metrics_at(y_test, test_proba, watch_thr),       # ≥90% calib recall
        "precision_at_top_5pct": round(_precision_at_k(y_test, test_proba, 0.05), 4),
        "precision_at_top_10pct": round(_precision_at_k(y_test, test_proba, 0.10), 4),
        "uplift_at_top_10pct": round(_precision_at_k(y_test, test_proba, 0.10) / max(prevalence, 1e-9), 2),
    }
    log.info(
        "%d season %s: prev=%.3f AUC=%.3f AP=%.3f | deploy recall=%.2f prec=%.2f | "
        "watch recall=%.2f prec=%.2f",
        year, result["test_window"], prevalence, auc, ap,
        result["deployment"]["recall"], result["deployment"]["precision"],
        result["watch_list"]["recall"], result["watch_list"]["precision"],
    )
    del fit_df, X_fit
    gc.collect()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate(all_folds: List[dict]) -> dict:
    """Mean/spread across the VALID (status==ok) folds only. Skipped folds are
    counted with their reasons — never zero-filled into the aggregate."""
    ok = [f for f in all_folds if f.get("status") == "ok"]
    skipped = [f for f in all_folds if f.get("status") == "skipped"]
    skip_reasons: Dict[str, int] = {}
    for f in skipped:
        skip_reasons[f.get("reason", "unknown")] = skip_reasons.get(f.get("reason", "unknown"), 0) + 1
    base = {
        "n_folds_total": len(all_folds),
        "valid_folds": len(ok),
        "skipped_folds": len(skipped),
        "skipped_reasons": skip_reasons,
    }
    if not ok:
        return base

    def _col(path: List[str]) -> np.ndarray:
        vals = []
        for f in ok:
            v = f
            for k in path:
                v = v[k]
            if v is not None:
                vals.append(float(v))
        return np.array(vals)

    aucs = _col(["roc_auc"])
    aps = _col(["average_precision"])
    deploy_recall = _col(["deployment", "recall"])
    watch_recall = _col(["watch_list", "recall"])
    watch_prec = _col(["watch_list", "precision"])
    base.update({
        "auc_mean": round(float(aucs.mean()), 4),
        "auc_std": round(float(aucs.std()), 4),
        "auc_min": round(float(aucs.min()), 4),
        "auc_max": round(float(aucs.max()), 4),
        "ap_mean": round(float(aps.mean()), 4),
        "ap_median": round(float(np.median(aps)), 4),
        "ap_std": round(float(aps.std()), 4),
        "ap_min": round(float(aps.min()), 4),
        "ap_max": round(float(aps.max()), 4),
        "deployment_recall_mean": round(float(deploy_recall.mean()), 4),
        "watch_recall_mean": round(float(watch_recall.mean()), 4),
        "watch_precision_mean": round(float(watch_prec.mean()), 4),
    })
    return base


def _print_table(folds: List[dict], agg: dict) -> None:
    print("\n" + "=" * 92)
    print("ROLLING-ORIGIN BACKTEST — one model retrained per fire season, evaluated at natural prevalence")
    print("=" * 92)
    hdr = f"{'season':>10} {'prev':>6} {'AUC':>6} {'AP':>6} {'persist':>8} " \
          f"{'deploy R/P':>14} {'watch R/P':>14} {'uplift@10%':>10}"
    print(hdr)
    print("-" * 92)
    for f in folds:
        if f.get("status") != "ok":
            print(f"{f['year']:>10}   SKIPPED — {f.get('reason', 'unknown')}")
            continue
        d, w = f["deployment"], f["watch_list"]
        print(f"{f['year']:>10} {f['test_prevalence']:>6.3f} {f['roc_auc']:>6.3f} "
              f"{f['average_precision']:>6.3f} {str(f['persistence_auc']):>8} "
              f"{d['recall']:>5.2f}/{d['precision']:<5.2f}  "
              f"{w['recall']:>5.2f}/{w['precision']:<5.2f}  "
              f"{f['uplift_at_top_10pct']:>10.1f}")
    print("-" * 92)
    if agg.get("valid_folds"):
        print(f"MEAN over {agg['valid_folds']} valid seasons "
              f"({agg['skipped_folds']} skipped):  "
              f"AUC {agg['auc_mean']:.3f} ± {agg['auc_std']:.3f}  "
              f"(min {agg['auc_min']:.3f}, max {agg['auc_max']:.3f})  |  "
              f"AP mean/median {agg['ap_mean']:.3f}/{agg['ap_median']:.3f} "
              f"(min {agg['ap_min']:.3f})  |  "
              f"deploy recall {agg['deployment_recall_mean']:.2f}  |  "
              f"watch recall {agg['watch_recall_mean']:.2f} @ precision {agg['watch_precision_mean']:.2f}")
    print("=" * 92 + "\n")


def main(
    years: List[int],
    season_months: Tuple[int, int],
    calib_days: int,
    n_estimators: int,
    random_state: int,
    out_path: str,
    max_train_years: float,
) -> dict:
    load_dotenv()
    p = _paths()

    weather_path = resolve_existing(p["weather_path"])
    tree_cover_path = p["tree_cover_path"] if os.path.exists(p["tree_cover_path"]) else None
    radd_path = resolve_existing(p["radd_path"])
    urban_filter_enabled = os.getenv("URBAN_FILTER_ENABLED", "true").lower() in ("1", "true", "yes")

    # Densify ONCE (bounded to the overall span the requested folds can touch),
    # keep the compact ~15-column daily frame in memory, then featurize each
    # fold's own window inside the loop. This caps peak feature-frame memory to
    # a single fold regardless of how many seasons are requested — building the
    # whole multi-year feature frame at once is what OOMs / stalls.
    daily = load_and_prepare(
        raw_dir=p["raw_dir"], firms_path=p["firms_path"], grid_size=0.1,
        min_confidence=0, densify=True, weather_path=weather_path,
        tree_cover_path=tree_cover_path, radd_path=radd_path,
        filter_urban=urban_filter_enabled, urban_buffer_km=float(os.getenv("URBAN_BUFFER_KM", "0.0")),
    )
    daily["date"] = pd.to_datetime(daily["date"])

    # Weather coverage is evaluated per bounded fold window below. Evaluating
    # it over the whole 2012–present daily frame would make a complete recent
    # cache look globally sparse and silently disable weather in every fold.
    min_cov = float(os.getenv("MIN_WEATHER_COVERAGE", "0.20"))

    lead_days = int(round((max_train_years + 1) * 365.25))
    m0, m1 = season_months

    # Resume: keep previously-computed folds for years not requested this run.
    existing: Dict[int, dict] = {}
    if os.path.exists(out_path):
        try:
            prev = json.load(open(out_path))
            for f in prev.get("folds", []):
                existing[int(f["year"])] = f
        except Exception:
            pass

    feature_cols_last: List[str] = []
    for year in years:
        # Featurize only this fold's window: [season − train − warm-up, season end].
        lo = pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=lead_days)
        hi = (pd.Timestamp(year=year, month=m1, day=1) + pd.offsets.MonthEnd(0)
              + pd.Timedelta(days=IMMINENT_DAYS + PURGE_DAYS + 2))
        window = daily[(daily["date"] >= lo) & (daily["date"] <= hi)]
        if window.empty:
            existing[year] = {"year": year, "status": "skipped",
                              "reason": "no_densified_rows_in_window"}
        else:
            window = window.copy()
            present = [c for c in WEATHER_COLUMNS if c in window.columns]
            coverage = float(window[present[0]].notna().mean()) if present else 0.0
            gated = _drop_sparse_weather(window, min_cov)
            if present and not any(c in gated.columns for c in present):
                log.info("%d weather coverage %.1f%% < %.0f%% — dropping weather columns",
                         year, coverage * 100, min_cov * 100)
            window = gated
            feats = build_features(window.copy(), horizon=MAX_PREDICTION_DAYS, grid_size=0.1)
            feature_cols = resolve_features(feats)
            feature_cols_last = feature_cols
            for c in feature_cols:
                if c not in ("lat_grid", "lon_grid") and feats[c].dtype != np.float32:
                    feats[c] = feats[c].astype(np.float32)
            res = _run_fold(feats, feature_cols, year, season_months,
                            calib_days, n_estimators, random_state,
                            max_train_years=max_train_years)
            del feats
            gc.collect()
            existing[year] = res     # always store (ok OR skipped) — never drop
        # Persist incrementally (resumable) after every fold.
        all_folds = [existing[y] for y in sorted(existing)]
        write_json({
            "task": "rolling_origin_backtest",
            "provenance": _provenance(p, feature_cols_last, n_estimators,
                                      max_train_years, calib_days, season_months),
            "folds": all_folds,
            "aggregate": _aggregate(all_folds),
        }, out_path)

    all_folds = [existing[y] for y in sorted(existing)]
    agg = _aggregate(all_folds)
    _print_table(all_folds, agg)
    log.info("Wrote %d folds → %s", len(all_folds), out_path)
    return {"folds": all_folds, "aggregate": agg}


def _parse_years(spec: str) -> List[int]:
    out: List[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", default="2014-2026",
                    help="Target fire-season years, e.g. 2014-2026 or 2019,2021")
    ap.add_argument("--season", default="1-4",
                    help="Test-window months START-END (default 1-4 = Jan–Apr)")
    ap.add_argument("--calib-days", type=int, default=30,
                    help="Days of pre-season data used to calibrate + pick threshold")
    ap.add_argument("--max-train-years", type=float, default=3.0,
                    help="Cap each fold's training history to the most recent N years")
    ap.add_argument("--n-estimators", type=int, default=600,
                    help="LightGBM trees per fold")
    ap.add_argument("--quick", action="store_true",
                    help="Fewer trees (250) for a fast dry run")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    m0, m1 = (int(x) for x in args.season.split("-", 1))
    n_est = 250 if args.quick else args.n_estimators
    _p = _paths()
    out = os.path.join(_p["meta_dir"], "rolling_backtest.json")
    main(_parse_years(args.years), (m0, m1), args.calib_days, n_est,
         args.random_state, out, args.max_train_years)
