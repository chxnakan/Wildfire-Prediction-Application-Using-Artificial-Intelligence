"""Train and evaluate hotspot forecasts using a purged chronological holdout."""

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import brier_score_loss

from src.calibration import PlattCalibrator, diagnostics, expected_calibration_error
from src.evaluation import evaluate, recall_at_false_alarm_budgets
from src.pipeline import FEATURES, HORIZON, HISTORY_DAYS, QUALITY_FEATURES, TARGET_POLICY, chronological_split, daily_panel, load_observations, observation_paths, purged_folds
from src.forecasting import SCHEMA_VERSION, score_snapshots
from src.detection_store import save_observation_coordinates
from src.geography import thailand_cells
from src.weather_features import WEATHER_FEATURES, DRYNESS_FEATURES, attach_weather
from src.land_features import LAND_FEATURES, TERRAIN_FEATURES, attach_context
from src.model_selection import select_features, monthly_evaluation


def fit_models(frame, params, features=FEATURES):
    classifier = LGBMClassifier(objective="binary", **params)
    regressor = LGBMRegressor(objective="regression_l1", **params)
    classifier.fit(frame[features], frame.target)
    positive = frame[frame.target == 1]
    regressor.fit(positive[features], positive.days_until_detection)
    return classifier, regressor


def fit_calibrated(frame, params, folds, features=FEATURES):
    """Cross-fit Platt scaling on purged out-of-fold scores, then fit on all rows."""
    target = frame.target.to_numpy()
    oof = np.full(len(frame), np.nan)
    for fit, held in purged_folds(frame, folds):
        if len(np.unique(target[fit])) < 2:
            raise ValueError("A calibration fold lacks positive or negative windows")
        model = LGBMClassifier(objective="binary", **params)
        model.fit(frame.loc[fit, features], target[fit])
        oof[held] = model.predict_proba(frame.loc[held, features])[:, 1]
    calibrator = PlattCalibrator().fit(oof, target)
    classifier, regressor = fit_models(frame, params, features)
    info = {"method": calibrator.method, "folds": folds,
            "description": f"Platt sigmoid fit on out-of-fold scores from {folds} contiguous date blocks of the "
                           f"fitting period, each predicted by a model that never saw that block ({HORIZON}-day label purge "
                           f"and {HISTORY_DAYS}-day history embargo); the final classifier is then fit on all fitting rows. Never fit on test.",
            "rows": len(frame), "start": str(frame.date.min().date()), "end": str(frame.date.max().date()),
            "oof_ece_raw": expected_calibration_error(target, oof),
            "oof_ece_calibrated": expected_calibration_error(target, calibrator.transform(oof))}
    return classifier, regressor, calibrator, info


def feature_importance(classifier, features=FEATURES):
    gain = classifier.booster_.feature_importance(importance_type="gain")
    splits = classifier.booster_.feature_importance(importance_type="split")
    total = float(gain.sum()) or 1.0
    rows = [{"feature": name, "gain_pct": round(100 * float(g) / total, 3), "splits": int(c)}
            for name, g, c in zip(features, gain, splits)]
    return sorted(rows, key=lambda row: row["gain_pct"], reverse=True)


def main():
    started = time.monotonic()
    load_dotenv(ROOT / ".env")
    grid = float(os.getenv("GRID", os.getenv("GRID_SIZE", "0.1")))
    output = Path(os.getenv("OUTPUT_DIR", str(ROOT / "outputs")))
    threshold = float(os.getenv("DETECTION_THRESHOLD", "0.5"))
    use_weather = os.getenv("USE_WEATHER_FEATURES", "true").lower() in {"true", "1", "yes"}
    research = use_weather and os.getenv("USE_RESEARCH_FEATURES", "true").lower() in {"true", "1", "yes"}
    features = FEATURES + WEATHER_FEATURES if use_weather else FEATURES
    required_features = features + QUALITY_FEATURES + LAND_FEATURES + TERRAIN_FEATURES + DRYNESS_FEATURES if research else features
    context_info, selection, research_comparison = None, None, None
    weather_info = None
    weather_comparison = None
    if not 0 < threshold < 1:
        raise ValueError("DETECTION_THRESHOLD must be between zero and one")
    calibration_folds = int(os.getenv("CALIBRATION_FOLDS", "4"))
    for name in ("models", "features", "metadata", "evaluation", "forecasts"):
        (output / name).mkdir(parents=True, exist_ok=True)
    params = dict(n_estimators=int(os.getenv("N_ESTIMATORS", "500")),
                  learning_rate=float(os.getenv("LEARNING_RATE", "0.05")),
                  num_leaves=int(os.getenv("NUM_LEAVES", "31")),
                  min_child_samples=int(os.getenv("MIN_CHILD_SAMPLES", "50")),
                  random_state=int(os.getenv("RANDOM_STATE", "42")),
                  n_jobs=4, verbosity=-1, force_col_wise=True)
    print("Loading and deduplicating satellite observations...", flush=True)
    paths = observation_paths(ROOT, os.getenv("RAW_DIR"), os.getenv("FIRMS_PATH"))
    observations, quality = load_observations(paths, grid)
    source_manifest = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    yesterday = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=1)
    observation_end = min(observations.date.max(), yesterday)
    print("Building calendar-day features using only dates before each forecast...", flush=True)
    panel, info = daily_panel(observations, observation_end, target_policy=TARGET_POLICY, candidate_cells=thailand_cells(grid))
    info["candidate_grids"] = len(panel[["lat_cell", "lon_cell"]].drop_duplicates())
    info["daily_rows"] = len(panel)
    info["geographic_scope"] = "Thailand grid-cell centres within bundled province boundaries"
    info["candidate_population"] = "All Thailand grid-cell centres, including cells without any previous hotspot"
    info["target_policy"] = TARGET_POLICY
    if use_weather:
        # Incomplete hotspot histories cannot be trained or forecast, so do not
        # spend weather requests on those dates.
        panel = panel[panel.history_complete].copy()
        panel, weather_info = attach_weather(
            panel, grid, ROOT / "data/weather/cache",
            weather_grid=float(os.getenv("WEATHER_GRID_SIZE", "0.5")),
            min_interval=float(os.getenv("WEATHER_REQUEST_INTERVAL", "3")))
        weather_info["complete_rows"] = int(panel[WEATHER_FEATURES].notna().all(axis=1).sum())
        weather_info["missing_rows"] = int(panel[WEATHER_FEATURES].isna().any(axis=1).sum())
    if research:
        print("Attaching historical land cover and terrain...", flush=True)
        panel, context_info = attach_context(panel, grid, ROOT / "data/context/cache")
    train, test, test_start = chronological_split(panel, float(os.getenv("TEST_SIZE", "0.2")))
    if research:
        train = train.dropna(subset=required_features)
        test = test.dropna(subset=required_features)
        base = FEATURES + WEATHER_FEATURES
        candidates = {"weather_baseline": base, "quality_and_season": base + QUALITY_FEATURES,
                      "land_cover": base + LAND_FEATURES, "terrain": base + TERRAIN_FEATURES,
                      "drying": base + DRYNESS_FEATURES, "all_context": required_features}
        features, selection = select_features(train, candidates,
            lambda frame, names: fit_calibrated(frame, params, calibration_folds, names), threshold)
        print(f"Selected on earlier validation only: {selection['selected']} ({len(features)} features)", flush=True)
        (output / "evaluation/selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False))
    if use_weather:
        # Fix date boundaries before removing incomplete weather rows. Both
        # models are evaluated on precisely the same retained grid-days.
        train = train.dropna(subset=WEATHER_FEATURES)
        test = test.dropna(subset=WEATHER_FEATURES)
        if train.empty or test.empty:
            raise ValueError("Insufficient complete archived weather for training/testing")
        print("Training matched hotspot-only baseline for the weather comparison...", flush=True)
        base_cls, base_reg, base_cal, _ = fit_calibrated(train, params, calibration_folds)
        base_raw = base_cls.predict_proba(test[FEATURES])[:, 1]
        hotspot_scores = base_cal.transform(base_raw)
        hotspot_days = np.clip(base_reg.predict(test[FEATURES]), 1, HORIZON)
        weather_comparison = {"scope": "Thailand; identical train/test rows, parameters and chronological split",
                              "hotspot_only": evaluate(test.target, test.days_until_detection, hotspot_scores, hotspot_days, threshold),
                              "hotspot_only_diagnostics": diagnostics(test.target, hotspot_scores, threshold),
                              "hotspot_only_recall_at_fpr": recall_at_false_alarm_budgets(test.target, hotspot_scores),
                              "roc_comparison_note": "Descriptive holdout ROC comparison at 2%/5% false-positive budgets; these thresholds are not deployed"}
        del base_cls, base_reg, base_cal
    if research:
        print("Training matched 17-feature weather baseline on revised labels/population...", flush=True)
        base_features = FEATURES + WEATHER_FEATURES
        base_cls, base_reg, base_cal, _ = fit_calibrated(train, params, calibration_folds, base_features)
        weather_scores = base_cal.transform(base_cls.predict_proba(test[base_features])[:, 1])
        weather_days = np.clip(base_reg.predict(test[base_features]), 1, HORIZON)
        research_comparison = {"scope": "Identical Thai cells, target policy, complete feature rows and dates; final test never used for selection",
                               "weather_baseline": evaluate(test.target, test.days_until_detection, weather_scores, weather_days, threshold),
                               "weather_baseline_diagnostics": diagnostics(test.target, weather_scores, threshold),
                               "weather_baseline_recall_at_fpr": recall_at_false_alarm_budgets(test.target, weather_scores)}
        del base_cls, base_reg, base_cal
    print(f"Training on {len(train):,} grid-days; testing on {len(test):,} later grid-days", flush=True)
    classifier, regressor, calibrator, calibration = fit_calibrated(train, params, calibration_folds, features)
    raw_scores = classifier.predict_proba(test[features])[:, 1]
    scores = calibrator.transform(raw_scores)
    days = np.clip(regressor.predict(test[features]), 1, HORIZON)
    median_days = float(train.loc[train.target == 1, "days_until_detection"].median())
    base_scores = np.full(len(test), float(train.target.mean()))
    baseline_days = np.full(len(test), median_days)
    evaluation = {
        "schema_version": 3, "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "sources": source_manifest,
        "horizon_days": HORIZON,
        "target": f"At least one recorded satellite hotspot, excluding known volcano/static-land/offshore types, in the same grid on days 1–{HORIZON}; untyped detections retained, not confirmed wildfire",
        "target_policy": TARGET_POLICY,
        "population": f"All Thailand grid centres, including never-detected locations; overlapping {HORIZON}-day windows",
        "feature_cutoff": "End of the UTC day before the forecast date",
        "split": {
            "method": f"chronological holdout with {HORIZON}-day label purge",
            "train_start": str(train.date.min().date()), "train_end": str(train.date.max().date()),
            "training_labels_through": str((train.date.max() + pd.Timedelta(days=HORIZON)).date()),
            "test_start": str(test_start.date()), "test_end": str(test.date.max().date()),
            "train_rows": len(train), "test_rows": len(test),
        },
        "calibration": {
            **calibration,
            "test_ece_raw": expected_calibration_error(test.target, raw_scores),
            "test_ece_calibrated": expected_calibration_error(test.target, scores),
            "test_brier_raw": float(brier_score_loss(test.target, raw_scores)),
            "test_brier_calibrated": float(brier_score_loss(test.target, scores)),
        },
        "model": evaluate(test.target, test.days_until_detection, scores, days, threshold),
        "baselines": {
            "training_prevalence_and_median_days": evaluate(test.target, test.days_until_detection, base_scores, baseline_days, threshold),
            "repeat_recent_detection_and_median_days": evaluate(test.target, test.days_until_detection, test.fire_days_7d.gt(0).astype(float), baseline_days, threshold),
        },
        "diagnostics": diagnostics(test.target, scores, threshold),
        "features": features,
        "weather": weather_info,
        "context": context_info,
        "feature_selection": selection,
        "monthly_holdout": monthly_evaluation(test, scores, days, threshold),
        "baseline_median_days": median_days,
        "limitations": [
            "No confirmed-wildfire labels: these metrics measure recorded hotspots.",
            "No record is not proof of no fire; local cloud cover and overpass availability are unknown.",
            "Windows crossing a regional day with zero records are excluded as unknown coverage.",
            "Historical downloads lack arrival timestamps; backtesting assumes prior-day records were available at issuance.",
            "Known non-vegetation source types are removed from outcome labels only; untyped NRT detections remain unknown and eligible. Historical input counts retain all detections to avoid hindsight from archive type classifications.",
            "Archive type annotations may contain errors; type 0 means presumed vegetation fire, not confirmed wildfire. Agricultural burns and wildfires are not separated by ground truth.",
            "Overlapping grid-day windows are correlated; sample counts are not independent wildfire events.",
            "The decision threshold is fixed in advance, not tuned on test results.",
            "Probabilities are Platt-calibrated on purged cross-fitted scores from the training period only; seasonal drift can still miscalibrate later periods (see test ECE). They are probabilities of a recorded hotspot, not of a confirmed wildfire.",
            "Production models are refit after evaluation; holdout scores apply to the separately saved evaluation model.",
        ],
    }
    if use_weather:
        weather_comparison["with_weather"] = research_comparison["weather_baseline"] if research else evaluation["model"]
        weather_comparison["with_weather_diagnostics"] = research_comparison["weather_baseline_diagnostics"] if research else evaluation["diagnostics"]
        weather_comparison["with_weather_recall_at_fpr"] = recall_at_false_alarm_budgets(test.target, weather_scores if research else scores)
        evaluation["weather_comparison"] = weather_comparison
        evaluation["limitations"].extend([
            "Weather is sampled on a 0.5-degree grid by default; local weather variation within a sampling cell is not resolved.",
            "Recent weather uses archived short-lead GFS output, not verified station observations.",
            "Future weather uses five-day-lead archived forecasts valid on days 1–3, with a conservative publication buffer; it does not use realised future weather.",
            "Weather availability is inferred from documented forecast leads plus six hours; exact historical publication timestamps are unavailable.",
        ])
    if research:
        research_comparison["selected_model"] = evaluation["model"]
        research_comparison["selected_recall_at_fpr"] = recall_at_false_alarm_budgets(test.target, scores)
        evaluation["research_comparison"] = research_comparison
        evaluation["limitations"].extend([
            context_info["limitations"],
            "The 30-day drying features require complete archived weather, removing initial warm-up rows equally from every comparison model. VPD is atmospheric drying potential, not measured fuel moisture or a validated Thai FWI.",
            "Selection uses three earlier chronological blocks; the available record does not establish performance across multiple independent fire seasons or truly unobserved fires beneath clouds/canopy.",
            "No road network, crop calendar, dynamic NDVI, lightning or confirmed incident-cause labels are currently joined. Built/crop fractions and seasonality are limited proxies for human activity.",
        ])
    evaluation_id = hashlib.sha256(json.dumps(evaluation, sort_keys=True).encode()).hexdigest()[:12]
    artifact = output / "evaluation" / evaluation_id
    artifact.mkdir()
    evidence = test[["date", "lat_cell", "lon_cell", "target", "days_until_detection"]].copy()
    evidence["detection_score"], evidence["probability"] = raw_scores, scores
    evidence["estimated_days_if_detection"] = days
    evidence["baseline_score"], evidence["baseline_days"] = base_scores, baseline_days
    evidence["repeat_recent_score"] = test.fire_days_7d.gt(0).astype(float)
    if use_weather:
        evidence["hotspot_only_probability"] = hotspot_scores
        evidence["hotspot_only_days"] = hotspot_days
    if research:
        evidence["weather_baseline_probability"] = weather_scores
        evidence["weather_baseline_days"] = weather_days
        evidence["dominant_land_cover"] = test[LAND_FEATURES].idxmax(axis=1).to_numpy()
    evidence.to_csv(artifact / "holdout_predictions.csv.gz", index=False)
    joblib.dump({"classifier": classifier, "regressor": regressor, "calibrator": calibrator, "features": features,
                 "horizon_days": HORIZON,
                 "threshold": threshold, "params": params, "split": evaluation["split"]}, artifact / "evaluation_model.pkl")
    evaluation["evidence_directory"] = str(artifact.relative_to(output))
    (artifact / "metrics.json").write_text(json.dumps(evaluation, indent=2, allow_nan=False))
    (output / "evaluation/metrics.json").write_text(json.dumps(evaluation, indent=2, allow_nan=False))
    del train, test, evidence, classifier, regressor
    eligible = panel[panel.history_complete & panel.label_complete]
    eligible = eligible.dropna(subset=required_features)
    print(f"Refitting production models on {len(eligible):,} mature grid-days...", flush=True)
    classifier, regressor, calibrator, production_calibration = fit_calibrated(eligible, params, calibration_folds, features)
    bundle = {"schema_version": SCHEMA_VERSION, "features": features, "classifier": classifier,
              "regressor": regressor, "calibrator": calibrator, "calibration": production_calibration,
              "grid_size": grid, "threshold": threshold, "horizon_days": HORIZON,
              "model_version": evaluation_id, "trained_at": datetime.now(timezone.utc).isoformat(),
              "training_labels_through": str((eligible.date.max() + pd.Timedelta(days=HORIZON)).date())}
    bundle["target_policy"] = TARGET_POLICY
    bundle["candidate_population"] = info["candidate_population"]
    if research:
        bundle["context"] = {key: value for key, value in context_info.items() if key != "sources"}
        bundle["selection"] = {key: value for key, value in selection.items() if key != "candidates"}
    if use_weather:
        bundle["weather"] = {key: value for key, value in weather_info.items() if key != "sources"}
    history_ready = panel[panel.history_complete]
    history_ready = history_ready.dropna(subset=required_features)
    if history_ready.empty:
        raise ValueError("No forecast date has seven regional history days")
    latest = history_ready[history_ready.date == history_ready.date.max()]
    if latest.date.max() < panel.date.max():
        print(f"Recent regional data are incomplete; publishing a historical demonstration for {latest.date.max().date()}.", flush=True)
    joblib.dump(bundle, output / "models/lgbm_fire_date_model.pkl")
    latest.to_csv(output / "features/latest_features.csv", index=False)
    save_observation_coordinates(observations, output)
    observations.groupby(["lat_cell", "lon_cell", "date"]).size().rename("fire_count").reset_index().to_csv(
        output / "features/daily_detections.csv.gz", index=False)
    metadata = {**info, **quality, "schema_version": 3, "latest_date": str(latest.date.max().date()),
                "prediction_type": "satellite_hotspot", "grid_size": grid, "max_prediction_days": HORIZON,
                "horizon_days": HORIZON,
                "features": features, "model_version": evaluation_id, "weather": weather_info,
                "context": context_info, "feature_selection": selection,
                "model": {"type": "LightGBM classifier (Platt-calibrated) + conditional date regressor",
                          "params": params, "calibration": production_calibration},
                "feature_importance": feature_importance(classifier, features),
                "trained_at": bundle["trained_at"],
                "training_seconds": round(time.monotonic() - started, 1),
                "evaluation": evaluation}
    (output / "metadata/dataset_info.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    from src.risk_map import run
    run(output)
    prospective = score_snapshots(output / "forecasts", observations, observation_end)
    (output / "evaluation/prospective.json").write_text(json.dumps(prospective, indent=2, allow_nan=False))
    print(json.dumps({"model": evaluation["model"], "baseline_median_days": median_days,
                      "split": evaluation["split"], "data_quality": quality}, indent=2), flush=True)
    print("Saved evaluation evidence, production model, forecasts and map.", flush=True)


if __name__ == "__main__":
    main()
