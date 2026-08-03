"""Model-promotion gate correctness (Phase 4).

Reproduces + fixes the reported bug: a candidate with lower AP (or lower AUC
beyond tolerance) was promoted with a false "candidate ≥ incumbent" message.
Also covers fail-closed behaviour, feature-contract-change → pending, and the
configurable Brier/ECE/operating gates.

    cd src && ../.venv/bin/python -m pytest tests/test_promotion_gates.py -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

import train
from storage import write_pickle
from sklearn.metrics import roc_auc_score, average_precision_score


class _StubModel:
    def __init__(self, score):
        self.score = score

    def predict_proba(self, X):
        return np.clip(np.linspace(0, 1, len(X)) * self.score, 0, 1)


X = pd.DataFrame({"a": np.linspace(0, 1, 200)})
Y = (np.linspace(0, 1, 200) > 0.8).astype(int)


def _incumbent(tmp_path, features=("a",)):
    p = str(tmp_path / "model.pkl")
    write_pickle({"format": "bundle-v1", "model": _StubModel(1.0),
                  "metadata": {"features": list(features), "trained_at": "old"}}, p)
    return p


def _incumbent_scores():
    proba = _StubModel(1.0).predict_proba(X)
    return float(roc_auc_score(Y, proba)), float(average_precision_score(Y, proba))


def test_bug_equal_auc_but_lower_ap_is_rejected_not_promoted(tmp_path):
    """THE reported bug: AUC within tolerance but AP lower must be REJECTED,
    never promoted with a 'candidate ≥ incumbent' message."""
    cur_auc, cur_ap = _incumbent_scores()
    ok, reason = train._evaluate_promotion_gates(
        candidate_auc=cur_auc, candidate_ap=cur_ap - 0.05, test_prevalence=0.1,
        X_test=X, y_test_bin=Y, feature_cols=["a"],
        promoted_path=_incumbent(tmp_path))
    assert ok is False
    assert "AP" in reason and "≥ incumbent" not in reason


def test_strictly_ge_message_only_when_truly_ge(tmp_path):
    cur_auc, cur_ap = _incumbent_scores()
    ok, reason = train._evaluate_promotion_gates(
        candidate_auc=cur_auc, candidate_ap=cur_ap, test_prevalence=0.1,
        X_test=X, y_test_bin=Y, feature_cols=["a"],
        promoted_path=_incumbent(tmp_path))
    assert ok is True and "candidate ≥ incumbent" in reason


def test_within_tolerance_message_when_slightly_worse(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "MODEL_GATE_MAX_AUC_DROP", 0.01)
    cur_auc, cur_ap = _incumbent_scores()
    ok, reason = train._evaluate_promotion_gates(
        candidate_auc=cur_auc - 0.005, candidate_ap=cur_ap, test_prevalence=0.1,
        X_test=X, y_test_bin=Y, feature_cols=["a"],
        promoted_path=_incumbent(tmp_path))
    assert ok is True
    assert "within allowed regression tolerance" in reason
    assert "candidate ≥ incumbent" not in reason


def test_fail_closed_when_incumbent_unloadable(tmp_path):
    bad = str(tmp_path / "model.pkl")
    write_pickle({"format": "bundle-v1", "model": None, "metadata": {}}, bad)  # corrupt
    ok, reason = train._evaluate_promotion_gates(
        candidate_auc=0.99, candidate_ap=0.99, test_prevalence=0.1,
        X_test=X, y_test_bin=Y, feature_cols=["a"], promoted_path=bad)
    assert ok is False and "fail-closed" in reason


def test_feature_contract_change_is_pending_not_auto_promoted(tmp_path):
    ok, reason = train._evaluate_promotion_gates(
        candidate_auc=0.99, candidate_ap=0.99, test_prevalence=0.1,
        X_test=X, y_test_bin=Y, feature_cols=["a", "b"],   # differs from incumbent ["a"]
        promoted_path=_incumbent(tmp_path))
    assert ok is False and "PENDING MANUAL REVIEW" in reason


def test_feature_change_allowed_when_explicitly_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "MODEL_GATE_ALLOW_FEATURE_CHANGE", True)
    ok, reason = train._evaluate_promotion_gates(
        candidate_auc=0.99, candidate_ap=0.99, test_prevalence=0.1,
        X_test=X, y_test_bin=Y, feature_cols=["a", "b"],
        promoted_path=_incumbent(tmp_path))
    assert ok is True and "feature contract changed" in reason


def test_brier_gate_rejects_worse_calibration(tmp_path):
    cur_auc, cur_ap = _incumbent_scores()
    ok, reason = train._evaluate_promotion_gates(
        candidate_auc=cur_auc, candidate_ap=cur_ap, test_prevalence=0.1,
        X_test=X, y_test_bin=Y, feature_cols=["a"], promoted_path=_incumbent(tmp_path),
        candidate_brier=0.99)      # far worse than incumbent Brier
    assert ok is False and "Brier" in reason


def test_operating_floor_alert_rate_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "MODEL_GATE_MAX_ALERT_RATE", 0.10)
    ok, reason = train._evaluate_promotion_gates(
        candidate_auc=0.99, candidate_ap=0.99, test_prevalence=0.1,
        X_test=X, y_test_bin=Y, feature_cols=["a"], promoted_path=_incumbent(tmp_path),
        candidate_alert_rate=0.30)   # exceeds the ceiling
    assert ok is False and "alert rate" in reason
