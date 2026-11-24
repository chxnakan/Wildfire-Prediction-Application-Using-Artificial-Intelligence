"""Feature selection using earlier rolling validation; final holdout stays sealed."""

import numpy as np
import pandas as pd

from src.calibration import diagnostics
from src.evaluation import evaluate, recall_at_false_alarm_budgets
from src.pipeline import HORIZON


def rolling_validation(frame):
    dates = np.sort(frame.date.unique())
    if len(dates) < 120:
        raise ValueError("Research feature selection needs at least 120 fitting dates")
    boundaries = [int(len(dates) * f) for f in (.55, .70, .85)] + [len(dates)]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        start = pd.Timestamp(dates[left])
        fit = frame[frame.date + pd.Timedelta(days=HORIZON) < start]
        valid = frame[frame.date.isin(dates[left:right])]
        if fit.target.nunique() != 2 or valid.target.nunique() != 2:
            raise ValueError("A chronological selection block lacks both outcome classes")
        yield fit, valid


def select_features(frame, candidates, fit_model, threshold):
    """Predeclared rule: AP with Brier and low-false-alarm recall guardrails."""
    results = {}
    splits = list(rolling_validation(frame))
    for name, features in candidates.items():
        records = []
        for i, (fit, valid) in enumerate(splits, 1):
            print(f"Selection {name}: chronological block {i}/{len(splits)}", flush=True)
            classifier, regressor, calibrator, _ = fit_model(fit, features)
            scores = calibrator.transform(classifier.predict_proba(valid[features])[:, 1])
            days = np.clip(regressor.predict(valid[features]), 1, HORIZON)
            records.append({"fit_start": str(fit.date.min().date()), "fit_end": str(fit.date.max().date()),
                            "validation_start": str(valid.date.min().date()), "validation_end": str(valid.date.max().date()),
                            "metrics": evaluate(valid.target, valid.days_until_detection, scores, days, threshold),
                            "ece": diagnostics(valid.target, scores, threshold)["ece"],
                            "recall_at_fpr": recall_at_false_alarm_budgets(valid.target, scores)})
        results[name] = {"features": features, "blocks": records,
                         "mean_average_precision": float(np.mean([r["metrics"]["classification"]["average_precision"] for r in records])),
                         "mean_brier": float(np.mean([r["metrics"]["classification"]["brier_score"] for r in records])),
                         "mean_recall_at_2pct_fpr": float(np.mean([r["recall_at_fpr"]["0.02"] for r in records]))}
    baseline = results["weather_baseline"]
    eligible = [name for name, row in results.items()
                if row["mean_brier"] <= baseline["mean_brier"] + .002
                and row["mean_recall_at_2pct_fpr"] >= baseline["mean_recall_at_2pct_fpr"] - 1e-12]
    selected = max(eligible, key=lambda name: (results[name]["mean_average_precision"], -len(candidates[name])))
    return candidates[selected], {
        "method": f"Three expanding chronological validation blocks within the fitting period; {HORIZON}-day label purge",
        "horizon_days": HORIZON,
        "rule": "Maximize mean average precision among candidates with mean Brier <= weather baseline + 0.002 and mean recall at 2% FPR >= weather baseline; tie favors fewer features",
        "holdout_used_for_selection": False, "selected": selected, "candidates": results,
        "note": "ROC budget points are descriptive validation metrics, not deployed thresholds; the display/model threshold is unchanged",
    }


def monthly_evaluation(frame, scores, days, threshold):
    rows = []
    months = frame.date.dt.to_period("M").astype(str)
    for month in sorted(months.unique()):
        mask = months.eq(month).to_numpy()
        rows.append({"month": month, "metrics": evaluate(frame.target.to_numpy()[mask], frame.days_until_detection.to_numpy()[mask], scores[mask], days[mask], threshold),
                     "diagnostics": diagnostics(frame.target.to_numpy()[mask], scores[mask], threshold)})
    return rows
