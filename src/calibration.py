"""Probability calibration and held-out diagnostics for the hotspot classifier."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def _logit(scores):
    p = np.clip(np.asarray(scores, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibrator:
    """Sigmoid (Platt) calibration fit on a held-out slice; monotonic in the raw score."""

    method = "platt"

    def fit(self, scores, y):
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("Calibration slice needs both positive and negative windows")
        self.model_ = LogisticRegression(C=1e6, max_iter=1000)
        self.model_.fit(_logit(scores), y)
        return self

    def transform(self, scores):
        return self.model_.predict_proba(_logit(scores))[:, 1]


def reliability(y, p, bins=10):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        count = int(mask.sum())
        rows.append({"lower": float(edges[b]), "upper": float(edges[b + 1]), "count": count,
                     "mean_probability": float(p[mask].mean()) if count else None,
                     "positive_rate": float(y[mask].mean()) if count else None})
    return rows


def expected_calibration_error(y, p, bins=10):
    total = len(y)
    if not total:
        return None
    return float(sum(row["count"] / total * abs(row["mean_probability"] - row["positive_rate"])
                     for row in reliability(y, p, bins) if row["count"]))


def diagnostics(y, p, threshold=0.5):
    """Ranking, calibration and watch-list quality at fixed, pre-declared cut-offs."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if not len(y):
        return None
    prevalence = float(y.mean())
    both = len(np.unique(y)) == 2
    order = np.argsort(-p, kind="stable")
    top = {}
    for share in (5, 10, 20):
        k = max(1, int(round(len(y) * share / 100)))
        precision = float(y[order[:k]].mean())
        top[str(share)] = {"k": k, "precision": precision,
                           "lift": precision / prevalence if prevalence else None}
    table = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        flagged = p >= t
        tp = int((flagged & (y == 1)).sum())
        table.append({"threshold": float(t), "flagged_share": float(flagged.mean()),
                      "precision": tp / int(flagged.sum()) if flagged.any() else None,
                      "recall": tp / int(y.sum()) if y.any() else None})
    return {"samples": int(len(y)), "positive_rate": prevalence, "threshold": threshold,
            "roc_auc": float(roc_auc_score(y, p)) if both else None,
            "ece": expected_calibration_error(y, p), "reliability": reliability(y, p),
            "precision_at_top": top, "threshold_table": table}
