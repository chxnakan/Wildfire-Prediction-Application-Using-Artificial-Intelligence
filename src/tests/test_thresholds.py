"""Operational threshold selection — recall-target, drift, seasonal grouping,
fallback, ordering, config validation, top-K. Pure functions, no model needed.

    cd src && ../.venv/bin/python -m pytest tests -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

import thresholds as T


def _ranked(n=2000, prevalence=0.1, sep=0.6, seed=0):
    """Synthetic well-ranked scores: positives score higher than negatives."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    proba = np.clip(0.15 + sep * y + rng.normal(0, 0.2, n), 0, 1)
    return y, proba


# ── recall-target selection ──────────────────────────────────────────────────

def test_watch_meets_target_recall():
    y, proba = _ranked(prevalence=0.1)
    thr, reason = T.recall_target_threshold(y, proba, 0.90)
    got = T.threshold_metrics(y, proba, thr)["recall"]
    assert got >= 0.90 and reason is None


def test_higher_target_recall_lowers_threshold():
    y, proba = _ranked()
    t90, _ = T.recall_target_threshold(y, proba, 0.90)
    t99, _ = T.recall_target_threshold(y, proba, 0.99)
    assert t99 <= t90        # catching more fires ⇒ lower cutoff


# ── ordering invariant: HIGH > WATCH >= 0 ────────────────────────────────────

def test_threshold_ordering_high_ge_watch_ge_zero():
    y, proba = _ranked()
    m = T.select_operational_thresholds(y, proba, target_recall=0.90)
    assert m["HIGH"] >= m["WATCH"] >= 0.0
    assert m["MEDIUM"] == m["WATCH"]          # compatibility alias


def test_watch_clamped_below_high_when_recall_target_pushes_above():
    # Very high target recall on a skewed set can push WATCH up; must stay < HIGH.
    y, proba = _ranked(prevalence=0.4, sep=0.2)
    m = T.select_operational_thresholds(y, proba, target_recall=0.999)
    assert m["HIGH"] >= m["WATCH"]


# ── minimum-precision guard ──────────────────────────────────────────────────

def test_min_precision_guard_raises_threshold_and_flags():
    y, proba = _ranked(prevalence=0.1)
    base, _ = T.recall_target_threshold(y, proba, 0.99)
    guarded, reason = T.recall_target_threshold(y, proba, 0.99, min_precision=0.6)
    assert guarded >= base
    assert reason in ("min_precision_guard", "min_precision_unreachable")


# ── no test-set tuning: selection consumes only the arrays it is given ───────

def test_selection_is_marked_calib_only():
    y, proba = _ranked()
    m = T.select_operational_thresholds(y, proba)
    assert m["selected_on"] == "calib"


# ── configuration validation ─────────────────────────────────────────────────

def test_config_defaults_and_validation(monkeypatch):
    monkeypatch.delenv("WATCH_TARGET_RECALL", raising=False)
    monkeypatch.delenv("WATCH_MIN_PRECISION", raising=False)
    assert T.watch_target_recall() == T.DEFAULT_WATCH_TARGET_RECALL
    assert T.watch_min_precision() is None

    monkeypatch.setenv("WATCH_TARGET_RECALL", "1.5")   # out of range → clamp
    assert T.watch_target_recall() == 1.0
    monkeypatch.setenv("WATCH_TARGET_RECALL", "junk")  # unparseable → default
    assert T.watch_target_recall() == T.DEFAULT_WATCH_TARGET_RECALL
    monkeypatch.setenv("WATCH_MIN_PRECISION", "0.3")
    assert T.watch_min_precision() == 0.3


def test_validate_config_returns_full_dict():
    cfg = T.validate_config()
    assert set(cfg) >= {"watch_target_recall", "watch_min_precision",
                        "drift_prevalence_ratio", "min_positives_per_group"}


# ── top-K precision + recall capture ─────────────────────────────────────────

def test_topk_precision_and_recall_capture():
    y, proba = _ranked(prevalence=0.1)
    tk = T.topk_metrics(y, proba, ks=(0.05, 0.10, 0.20))
    # a good ranker's precision@5% >> base rate, and recall grows with K
    assert tk["precision_at_top_5pct"] > 0.1
    assert (tk["recall_captured_top_5pct"] <= tk["recall_captured_top_10pct"]
            <= tk["recall_captured_top_20pct"])


def test_topk_recall_capture_never_exceeds_one():
    y, proba = _ranked()
    tk = T.topk_metrics(y, proba)
    for k, v in tk.items():
        if k.startswith("recall_captured"):
            assert 0.0 <= v <= 1.0


# ── drift monitoring ─────────────────────────────────────────────────────────

def test_prevalence_drift_flags_large_gap():
    d = T.prevalence_drift(0.3682, 0.0345, ratio_threshold=3.0)  # the real case
    assert d["drift_flag"] is True and d["ratio"] > 3.0
    assert d["direction"] == "calibration_higher"


def test_prevalence_drift_quiet_when_similar():
    d = T.prevalence_drift(0.10, 0.11, ratio_threshold=3.0)
    assert d["drift_flag"] is False


# ── seasonal grouping + insufficient-positive fallback ───────────────────────

def test_seasonal_thresholds_and_fallback():
    # Two months: Jan with many positives, Aug with almost none.
    rng = np.random.default_rng(1)
    n_jan, n_aug = 1500, 300
    yj = (rng.random(n_jan) < 0.25).astype(int)
    pj = np.clip(0.2 + 0.6 * yj + rng.normal(0, 0.2, n_jan), 0, 1)
    ya = (rng.random(n_aug) < 0.01).astype(int)          # < min_positives
    pa = np.clip(0.1 + 0.6 * ya + rng.normal(0, 0.2, n_aug), 0, 1)
    y = np.concatenate([yj, ya])
    proba = np.concatenate([pj, pa])
    dates = ([pd.Timestamp("2026-01-15")] * n_jan
             + [pd.Timestamp("2026-08-15")] * n_aug)

    m = T.select_operational_thresholds(
        y, proba, dates=dates, group_by_month=True, min_positives=20)
    seasonal = m["seasonal_thresholds"]
    assert seasonal["1"]["fallback"] is None            # January selected its own
    assert seasonal["8"]["fallback"] == "insufficient_positives"
    # the fallback month reuses the global thresholds
    assert seasonal["8"]["HIGH"] == m["HIGH"]
    assert m["calib_date_range"] == ["2026-01-15", "2026-08-15"]
    # Option B: a 2-month window is NOT enough to APPLY seasonal thresholds —
    # they are descriptive only and serving uses the global cutoffs.
    assert m["seasonal_thresholds_applied"] is False
    assert m["seasonal_thresholds_status"] == "descriptive_only"
    assert "insufficient seasonal coverage" in m["seasonal_thresholds_reason"]


# ── metadata shape (what gets persisted in dataset_info.json) ────────────────

def test_metadata_is_complete_for_persistence():
    y, proba = _ranked()
    m = T.select_operational_thresholds(
        y, proba, dates=[pd.Timestamp("2026-01-01")] * len(y))
    for key in ("HIGH", "WATCH", "MEDIUM", "LOW", "high_metrics", "watch_metrics",
                "watch_target_recall", "selection_method", "calib_prevalence",
                "calib_sample_size", "calib_positives", "calib_date_range"):
        assert key in m, f"missing persisted key: {key}"
    for mk in ("precision", "recall", "f1", "alert_rate", "confusion_matrix"):
        assert mk in m["high_metrics"] and mk in m["watch_metrics"]


# ── single-class safety ──────────────────────────────────────────────────────

def test_single_class_calib_is_safe():
    y = np.zeros(100, dtype=int)
    proba = np.linspace(0, 1, 100)
    thr, reason = T.recall_target_threshold(y, proba, 0.9)
    assert thr == 0.5 and reason in ("single_class", "no_positives")
    assert T.f1_optimal_threshold(y, proba) == 0.5


# ── WATCH invariant + edge-case hardening (finding 6) ────────────────────────

def test_high_strictly_greater_than_watch_after_rounding():
    # Nearly-separable data can round HIGH and WATCH to the same value.
    y, proba = _ranked(prevalence=0.5, sep=0.9)
    m = T.select_operational_thresholds(y, proba, target_recall=0.90)
    assert 1.0 >= m["HIGH"] > m["WATCH"] >= 0.0


def test_empty_input_is_safe():
    m = T.select_operational_thresholds(np.array([]), np.array([]))
    assert m["HIGH"] >= m["WATCH"] >= 0.0
    assert T.threshold_metrics(np.array([]), np.array([]), 0.5)["n"] == 0
    tk = T.topk_metrics(np.array([]), np.array([]))
    assert all(v == 0.0 for v in tk.values())


def test_all_zero_and_all_one_labels():
    proba = np.linspace(0, 1, 50)
    m0 = T.select_operational_thresholds(np.zeros(50, int), proba)
    m1 = T.select_operational_thresholds(np.ones(50, int), proba)
    assert m0["HIGH"] >= m0["WATCH"] >= 0.0
    assert m1["HIGH"] >= m1["WATCH"] >= 0.0


def test_nan_and_inf_probabilities_are_dropped():
    y = np.array([0, 1, 0, 1, 1])
    proba = np.array([0.1, np.nan, 0.3, np.inf, 0.9])
    # non-finite rows dropped, remaining are valid
    m = T.threshold_metrics(y, proba, 0.5)
    assert m["n"] == 3
    with pytest.raises(ValueError):
        T.threshold_metrics(np.array([0, 1]), np.array([0.5]), 0.5)  # length mismatch


def test_tied_probabilities_do_not_break_ordering():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    proba = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # all tied
    m = T.select_operational_thresholds(y, proba, target_recall=0.90)
    assert m["HIGH"] >= m["WATCH"] >= 0.0


def test_high_zero_disables_watch():
    grp = T._one_group_thresholds(
        np.array([1, 1, 0]), np.array([0.0, 0.0, 0.0]), 0.9, None)
    if grp["HIGH"] <= 0.0:
        assert grp["watch_available"] is False
        assert grp["watch_disabled_reason"] == "high_threshold_zero"


def test_min_precision_guard_actually_met_or_disabled():
    y, proba = _ranked(prevalence=0.1)
    m = T.select_operational_thresholds(y, proba, target_recall=0.95, min_precision=0.6)
    if m["watch_available"]:
        assert m["watch_metrics"]["precision"] + 1e-9 >= 0.6   # guard truly met
    else:
        assert m["watch_disabled_reason"] in (
            "min_precision_unreachable", "min_precision_not_met")


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        T.f1_optimal_threshold(np.array([0, 1, 1]), np.array([0.2, 0.8]))


# ── multi-target WATCH trade-off table (Task 3) ──────────────────────────────

def test_tradeoff_table_higher_recall_target_lowers_threshold():
    y, proba = _ranked(prevalence=0.1)
    rows = T.threshold_tradeoff_table(y, proba, recall_targets=[0.5, 0.7, 0.8, 0.9])
    assert [r["target_recall"] for r in rows] == [0.5, 0.7, 0.8, 0.9]
    thr = [r["threshold"] for r in rows]
    rec = [r["recall"] for r in rows]
    # more recall demanded ⇒ lower cutoff (monotone non-increasing) and each
    # row actually achieves at least its target recall
    assert all(thr[i] >= thr[i + 1] for i in range(len(thr) - 1))
    for r in rows:
        assert r["recall"] >= r["target_recall"] - 1e-9
        assert set(["precision", "f1", "alert_rate", "confusion_matrix"]) <= set(r)


def test_tradeoff_table_uses_only_given_calib_arrays():
    # Selection must depend only on the arrays passed in (the calib split) —
    # different data ⇒ different thresholds; no hidden global/test state.
    y1, p1 = _ranked(prevalence=0.1, seed=1)
    y2, p2 = _ranked(prevalence=0.3, seed=2, sep=0.3)
    r1 = T.threshold_tradeoff_table(y1, p1, recall_targets=[0.8])
    r2 = T.threshold_tradeoff_table(y2, p2, recall_targets=[0.8])
    assert r1[0]["threshold"] != r2[0]["threshold"]


def test_select_operational_includes_tradeoff_menu():
    y, proba = _ranked()
    m = T.select_operational_thresholds(y, proba)
    assert "watch_tradeoff" in m and len(m["watch_tradeoff"]) >= 3


def test_recall_targets_config(monkeypatch):
    monkeypatch.setenv("WATCH_RECALL_TARGETS", "0.6, junk, 0.85, 2.0")
    # invalid (junk, 2.0) dropped; valid sorted
    assert T.watch_recall_targets() == (0.6, 0.85)
    monkeypatch.delenv("WATCH_RECALL_TARGETS", raising=False)
    assert T.watch_recall_targets() == T.DEFAULT_WATCH_RECALL_TARGETS


# ── constrained selector: structured feasibility (Phase 2) ───────────────────

def test_constrained_feasible_meets_all_constraints():
    y, proba = _ranked(prevalence=0.1)
    r = T.select_constrained_threshold(y, proba, target_recall=0.5,
                                       min_precision=0.15, max_alert_rate=0.6)
    assert r["feasible"] is True and r["threshold"] is not None
    m = r["metrics"]
    assert m["recall"] >= 0.5 - 1e-9 and m["precision"] >= 0.15 - 1e-9
    assert m["alert_rate"] <= 0.6 + 1e-9


def test_constrained_infeasible_reports_unmet_not_fake_pass():
    # precision >= 0.9 AND recall >= 0.9 is impossible on overlapping classes.
    y, proba = _ranked(prevalence=0.1, sep=0.3)
    r = T.select_constrained_threshold(y, proba, target_recall=0.9, min_precision=0.9)
    assert r["feasible"] is False and r["threshold"] is None
    assert r["unmet_constraints"]                      # names the violated ones
    assert r["best_available_operating_point"] is not None
    assert r["fallback_reason"]


def test_constrained_alert_rate_ceiling_enforced():
    y, proba = _ranked(prevalence=0.2)
    r = T.select_constrained_threshold(y, proba, target_recall=0.9, max_alert_rate=0.05)
    # 90% recall needs a big alert budget; a 5% ceiling makes it infeasible
    if r["feasible"]:
        assert r["metrics"]["alert_rate"] <= 0.05 + 1e-9
    else:
        assert "alert_rate" in r["unmet_constraints"] or r["best_available_operating_point"]


def test_constrained_insufficient_calibration_is_infeasible():
    r = T.select_constrained_threshold(np.zeros(5, int), np.linspace(0, 1, 5),
                                       target_recall=0.5, min_positives=20)
    assert r["feasible"] is False
    assert "insufficient_data" in r["unmet_constraints"]


def test_selector_signature_takes_no_test_data():
    # The selection API accepts only (y, proba) calibration arrays + constraints
    # — there is no parameter through which test data could enter selection.
    import inspect
    params = set(inspect.signature(T.select_constrained_threshold).parameters)
    assert "test" not in " ".join(params).lower()
    assert params >= {"y", "proba", "target_recall", "min_precision", "max_alert_rate"}
