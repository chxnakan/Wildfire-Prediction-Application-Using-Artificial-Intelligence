"""Forecast metrics with explicit denominators, baselines and time boundaries."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score, roc_curve


def recall_at_false_alarm_budgets(y, scores):
    """Descriptive holdout ROC points, not deployment-threshold selection."""
    if len(np.unique(y)) < 2:
        return None
    fpr, recall, _ = roc_curve(y, scores)
    return {str(budget): float(recall[fpr <= budget].max()) for budget in (.02, .05)}


def evaluate(y, actual_days, scores, predicted_days, threshold=0.5):
    y = np.asarray(y, dtype=int)
    actual_days, scores, predicted_days = map(np.asarray, (actual_days, scores, predicted_days))
    if not len(y):
        return {"samples": 0, "classification": None, "timing": None}
    predicted = scores >= threshold
    tn, fp, fn, tp = map(int, confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    positive = y == 1
    errors = predicted_days[positive] - actual_days[positive]
    return {
        "samples": len(y),
        "classification": {
            "threshold": threshold, "actual_positive_windows": int(positive.sum()),
            "actual_negative_windows": int((~positive).sum()),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "average_precision": float(average_precision_score(y, scores)) if positive.any() else None,
            "roc_auc": float(roc_auc_score(y, scores)) if positive.any() and (~positive).any() else None,
            "brier_score": float(brier_score_loss(y, scores)),
            "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        },
        "timing": {
            "samples": int(positive.sum()),
            "scope": "All windows with a recorded future detection, including classifier misses",
            "mae_days": float(np.abs(errors).mean()) if len(errors) else None,
            "rmse_days": float(np.sqrt(np.square(errors).mean())) if len(errors) else None,
            "within_one_day": float((np.abs(np.rint(predicted_days[positive]) - actual_days[positive]) <= 1).mean()) if len(errors) else None,
        },
    }
