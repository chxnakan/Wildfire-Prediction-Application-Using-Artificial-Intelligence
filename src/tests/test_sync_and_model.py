"""Sync input-versioning + model promotion/reload lifecycle tests.

Run the complete suite:  cd src && ../.venv/bin/python -m pytest tests -q
External processes (train.py subprocess, FIRMS fetch) are mocked.
"""

import os
import sys
from datetime import datetime, timezone
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

import db
import scheduler
import sync_worker as sw


# ─── input-data versioning → no-op vs exactly-one prediction ────────────────

def _fake_firms_cache(tmp_path, n_rows, latest="2026-07-10 06:00:00"):
    df = pd.DataFrame({
        "latitude": [18.5] * n_rows,
        "longitude": [98.9] * n_rows,
        "acq_datetime": pd.to_datetime([latest] * n_rows),
        "frp": [5.0] * n_rows,
    })
    p = tmp_path / "firms_all.parquet"
    df.to_parquet(p, index=False)
    return str(p)


@pytest.fixture
def sync_env(tmp_path, monkeypatch):
    cache = _fake_firms_cache(tmp_path, 10)
    monkeypatch.setattr(sw, "OUT_FILE", cache)
    monkeypatch.setattr(sw, "_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr(sw, "_VERSION_STATE_FILE",
                        str(tmp_path / "outputs" / ".last_prediction_version"))
    monkeypatch.setattr(sw, "update_firms", lambda days: None)   # no network
    monkeypatch.setattr(sw, "_mirror_new_rows_to_db", lambda: 0)
    return tmp_path, cache


def test_version_deterministic_across_restarts(sync_env):
    v1, _ = sw.compute_input_version()
    v2, _ = sw.compute_input_version()
    assert v1 == v2 and v1.startswith("v1-")


def test_unchanged_input_is_noop(sync_env):
    calls = []
    with mock.patch.object(sw, "_refresh_predictions",
                           side_effect=lambda: calls.append(1) or True):
        first = sw.run_sync()          # version never predicted before → runs
        second = sw.run_sync()         # identical input → no-op
    assert first["noop"] is False and first["predictions_refreshed"] is True
    assert second["noop"] is True and second["predictions_refreshed"] is False
    assert len(calls) == 1             # exactly ONE prediction run — never per-row


def test_changed_input_triggers_exactly_one_prediction(sync_env):
    tmp_path, cache = sync_env
    calls = []
    with mock.patch.object(sw, "_refresh_predictions",
                           side_effect=lambda: calls.append(1) or True):
        sw.run_sync()
        _fake_firms_cache(tmp_path, 25, latest="2026-07-11 06:00:00")  # new data
        sw.run_sync()
        sw.run_sync()                  # unchanged again → no-op
    assert len(calls) == 2


def test_failed_prediction_retries_next_cycle_and_keeps_snapshot(sync_env):
    """Prediction failure: version NOT remembered → next cycle retries;
    the last successful snapshot (risk map/GeoJSON) is untouched because
    _refresh_predictions failing means risk_map never rewrote it."""
    outcomes = [False, True]
    with mock.patch.object(sw, "_refresh_predictions",
                           side_effect=lambda: outcomes.pop(0)):
        first = sw.run_sync()
        second = sw.run_sync()
    assert first["predictions_refreshed"] is False
    assert second["predictions_refreshed"] is True      # retried, then remembered
    run = db.last_prediction_run()
    assert run["status"] == "done"


def test_prediction_lock_prevents_duplicate_jobs(sync_env):
    calls = []
    with db.advisory_lock(db.PREDICTION_LOCK_KEY) as held:   # replica A predicting
        assert held
        with mock.patch.object(sw, "_refresh_predictions",
                               side_effect=lambda: calls.append(1) or True):
            out = sw.run_sync()                              # replica B
    assert calls == []                  # B skipped — no duplicate prediction
    assert out["predictions_refreshed"] is False


def test_scheduler_reload_callback_fires_only_after_successful_refresh(monkeypatch):
    callbacks = []
    scheduler.register_prediction_refreshed_callback(
        lambda: callbacks.append("reload") or True
    )
    outcomes = [
        {"predictions_refreshed": True, "noop": False},
        {"predictions_refreshed": False, "noop": True},
        {"predictions_refreshed": False, "noop": False, "error": "failed"},
    ]
    monkeypatch.setattr(sw, "run_sync", lambda: outcomes.pop(0))

    refreshed = scheduler.sync_job()
    noop = scheduler.sync_job()
    failed = scheduler.sync_job()

    assert refreshed["predictions_refreshed"] is True
    assert noop["noop"] is True
    assert failed["error"] == "failed"
    assert callbacks == ["reload"]


def test_prediction_reload_callback_failure_does_not_break_sync(monkeypatch):
    def broken_reload():
        raise RuntimeError("reload failed")

    scheduler.register_prediction_refreshed_callback(broken_reload)
    monkeypatch.setattr(
        sw,
        "run_sync",
        lambda: {"predictions_refreshed": True, "noop": False},
    )

    summary = scheduler.sync_job()

    assert summary["predictions_refreshed"] is True
    assert summary["serving_state_reload_attempted"] is True
    assert summary["serving_state_reloaded"] is False
    assert summary["serving_state_reload_pending"] is True


def test_failed_prediction_reload_retries_on_next_identical_noop(monkeypatch):
    """The input version is already committed after predict-only succeeds.

    If the in-process serving reload then fails, the next identical sync is a
    worker no-op.  It must still retry the pending reload instead of leaving
    the API pinned to the previous snapshot forever.
    """
    reload_outcomes = [False, True]
    reload_calls = []

    def flaky_reload():
        reload_calls.append("reload")
        return reload_outcomes.pop(0)

    scheduler.register_prediction_refreshed_callback(flaky_reload)
    sync_outcomes = [
        {"predictions_refreshed": True, "noop": False},
        {"predictions_refreshed": False, "noop": True},
    ]
    monkeypatch.setattr(sw, "run_sync", lambda: sync_outcomes.pop(0))

    first = scheduler.sync_job()
    second = scheduler.sync_job()

    assert first["serving_state_reloaded"] is False
    assert first["serving_state_reload_pending"] is True
    assert second["noop"] is True
    assert second["serving_state_reload_attempted"] is True
    assert second["serving_state_reloaded"] is True
    assert second["serving_state_reload_pending"] is False
    assert reload_calls == ["reload", "reload"]


# ─── model bundle: gates / atomic promotion / rollback ───────────────────────

class _StubModel:
    """Tiny picklable stand-in with the predict_proba contract."""
    def __init__(self, score):
        self.score = score

    def predict_proba(self, X):
        import numpy as np
        # higher self.score → better separation of the fixture labels
        n = len(X)
        base = np.linspace(0, 1, n)
        return np.clip(base * self.score, 0, 1)


def _train():
    import train
    return train


def test_gate_rejects_below_auc_floor():
    train = _train()
    ok, reason = train._evaluate_promotion_gates(
        candidate_auc=0.55, candidate_ap=0.5, test_prevalence=0.1,
        X_test=None, y_test_bin=None, feature_cols=["a"],
        promoted_path="/nonexistent.pkl",
    )
    assert not ok and "floor" in reason


def test_gate_rejects_ap_below_prevalence_uplift():
    train = _train()
    ok, reason = train._evaluate_promotion_gates(
        candidate_auc=0.9, candidate_ap=0.10, test_prevalence=0.10,
        X_test=None, y_test_bin=None, feature_cols=["a"],
        promoted_path="/nonexistent.pkl",
    )
    assert not ok and "prevalence" in reason


def test_gate_passes_without_incumbent():
    train = _train()
    ok, reason = train._evaluate_promotion_gates(
        candidate_auc=0.85, candidate_ap=0.4, test_prevalence=0.1,
        X_test=None, y_test_bin=None, feature_cols=["a"],
        promoted_path="/nonexistent.pkl",
    )
    assert ok and "no incumbent" in reason


def test_gate_head_to_head_rejects_worse_candidate(tmp_path):
    """Candidate materially worse than the incumbent on the SAME test
    period → rejected, incumbent bundle untouched."""
    import numpy as np
    train = _train()
    from storage import write_pickle

    X_test = pd.DataFrame({"a": np.linspace(0, 1, 200)})
    y = (np.linspace(0, 1, 200) > 0.8).astype(int)

    promoted_path = str(tmp_path / "model.pkl")
    incumbent = {"format": "bundle-v1", "model": _StubModel(1.0),
                 "metadata": {"features": ["a"], "trained_at": "old"}}
    write_pickle(incumbent, promoted_path)
    before = os.path.getmtime(promoted_path)

    ok, reason = train._evaluate_promotion_gates(
        candidate_auc=0.70, candidate_ap=0.30, test_prevalence=0.1,
        X_test=X_test, y_test_bin=y, feature_cols=["a"],
        promoted_path=promoted_path,
    )
    assert not ok and "worse than incumbent" in reason
    assert os.path.getmtime(promoted_path) == before     # trusted model preserved


def test_promotion_is_atomic_and_keeps_previous(tmp_path):
    train = _train()
    from storage import read_pickle, write_pickle

    model_dir = str(tmp_path)
    promoted = os.path.join(model_dir, "model.pkl")
    write_pickle({"format": "bundle-v1", "model": _StubModel(1.0),
                  "metadata": {"trained_at": "v1"}}, promoted)

    train._promote_bundle(
        {"format": "bundle-v1", "model": _StubModel(2.0),
         "metadata": {"trained_at": "v2"}},
        model_dir, promoted,
    )
    now = read_pickle(promoted)
    assert now["metadata"]["trained_at"] == "v2"
    prev = read_pickle(os.path.join(model_dir, "previous.pkl"))
    assert prev["metadata"]["trained_at"] == "v1"        # rollback path exists
    assert not os.path.exists(os.path.join(model_dir, ".candidate.pkl.tmp"))


def test_load_model_bundle_handles_legacy_and_bundle(tmp_path):
    train = _train()
    from storage import write_pickle

    legacy = str(tmp_path / "legacy.pkl")
    write_pickle(_StubModel(1.0), legacy)
    model, meta = train.load_model_bundle(legacy)
    assert meta is None and hasattr(model, "predict_proba")

    bundled = str(tmp_path / "bundle.pkl")
    write_pickle({"format": "bundle-v1", "model": _StubModel(1.0),
                  "metadata": {"features": ["a"]}}, bundled)
    model, meta = train.load_model_bundle(bundled)
    assert meta == {"features": ["a"]}


# ─── API model state: empty volume / reload rollback ─────────────────────────

def test_empty_volume_startup_is_degraded_not_ready(tmp_path, monkeypatch):
    """No artifact at all → API builds no model state (degraded), and /health
    semantics (ready:false, model_loaded:false, HTTP 200) follow from it."""
    api = __import__("api")
    monkeypatch.setattr(api, "MODEL_PATH", str(tmp_path / "missing.pkl"))
    monkeypatch.setattr(api, "FEATURE_PATH", str(tmp_path / "missing.parquet"))
    assert api._build_model_state() is None


def test_failed_reload_keeps_previous_model(tmp_path, monkeypatch):
    """reload_model_state must never replace a serving model with a broken
    build — the previous in-memory state survives a failed reload."""
    api = __import__("api")

    class _App:                                   # minimal Starlette-State stand-in
        class _S:
            def __init__(self):
                self._state = {}
            def __getattr__(self, k):
                try:
                    return self.__dict__["_state"][k]
                except KeyError:
                    raise AttributeError(k)
        def __init__(self):
            self.state = self._S()

    app = _App()
    served = {"model": "GOOD", "features": ["a"], "model_version": "v1",
              "model_loaded_at": "t0", "df": None, "historical_counts": None,
              "meta": {}, "thresholds": {}}
    api._apply_model_state(app, served)

    with mock.patch.object(api, "_build_model_state",
                           side_effect=RuntimeError("corrupt artifact")):
        ok = api.reload_model_state(app)
    assert ok is False
    assert app.state.model == "GOOD"              # untouched
    assert app.state.model_version == "v1"

    with mock.patch.object(api, "_build_model_state", return_value=None):
        assert api.reload_model_state(app) is False
    assert app.state.model == "GOOD"

    new_state = dict(served, model="NEW", model_version="v2")
    with mock.patch.object(api, "_build_model_state", return_value=new_state):
        assert api.reload_model_state(app) is True
    assert app.state.model == "NEW" and app.state.model_version == "v2"
