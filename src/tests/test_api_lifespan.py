"""Regression guard for the TestClient lifespan deadlock (Task 1).

Root cause: TestClient.__enter__ starts an anyio blocking portal (a background
event-loop thread) to run the ASGI lifespan. On Python 3.14 the DEPRECATED
httpx TestClient backend could deadlock there — independent of app startup,
which is why mocking `_build_model_state` did not fix it. Starlette 1.3 prefers
`httpx2`; installing it uses the maintained backend that enters/exits cleanly.

This test proves TestClient startup+shutdown completes well within a time budget
and leaks no background threads. If the deadlock ever returns, this FAILS fast
(it does NOT just raise the suite timeout) so the cause is obvious.

    cd src && ../.venv/bin/python -m pytest tests/test_api_lifespan.py -q
"""

import os
import sys
import threading
import time
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

# Startup+shutdown must complete far inside this budget on a healthy stack.
_ENTER_EXIT_BUDGET_S = 15.0


@pytest.fixture
def api_module():
    import api
    return api


def test_testclient_enter_exit_is_fast_and_leak_free(api_module):
    baseline_threads = threading.active_count()
    # Degrade fast (no real artifact load) so we time the transport/lifespan,
    # not model I/O — the deadlock was in the portal, not the model.
    with mock.patch.object(api_module, "_build_model_state", return_value=None):
        t0 = time.monotonic()
        with TestClient(api_module.app) as client:
            enter_s = time.monotonic() - t0
            assert enter_s < _ENTER_EXIT_BUDGET_S, (
                f"TestClient startup took {enter_s:.1f}s (deadlock regression?)")
            # liveness must respond while inside the context
            assert client.get("/health/live").status_code == 200
        exit_s = time.monotonic() - t0
    assert exit_s < _ENTER_EXIT_BUDGET_S, (
        f"TestClient startup+shutdown took {exit_s:.1f}s (hung shutdown?)")

    # The blocking-portal thread must be torn down on exit — allow a brief grace
    # period for the daemon thread to actually stop.
    deadline = time.monotonic() + 5.0
    while threading.active_count() > baseline_threads and time.monotonic() < deadline:
        time.sleep(0.05)
    assert threading.active_count() <= baseline_threads + 1, (
        f"thread leak: {threading.active_count()} active vs baseline "
        f"{baseline_threads} — lifespan/portal did not clean up")


def test_scheduler_is_disabled_in_tests(api_module):
    # conftest sets SCHEDULER_ENABLED=false so no background job/thread starts —
    # a real scheduler thread would both leak and (via network) risk hanging.
    assert os.environ.get("SCHEDULER_ENABLED") == "false"


def test_repeated_testclients_do_not_accumulate_callbacks(api_module):
    import scheduler
    scheduler._on_model_promoted.clear()
    scheduler._on_prediction_refreshed.clear()
    with mock.patch.object(api_module, "_build_model_state", return_value=None):
        for _ in range(3):
            with TestClient(api_module.app):
                pass
    # conftest's autouse cleaner clears these between tests; within one test the
    # lifespan registers one callback per enter — assert it doesn't explode.
    assert len(scheduler._on_model_promoted) <= 3
    assert len(scheduler._on_prediction_refreshed) <= 3


def test_prediction_refresh_callback_reloads_api_state(api_module):
    import scheduler

    with mock.patch.object(api_module, "_build_model_state", return_value=None):
        with mock.patch.object(
            api_module, "reload_model_state", return_value=True
        ) as reload_state:
            with TestClient(api_module.app):
                scheduler._fire_prediction_refreshed()

    reload_state.assert_called_once_with(api_module.app)


def test_successful_sync_atomically_advances_in_memory_feature_snapshot(api_module):
    import scheduler
    import sync_worker

    refreshed_state = dict(
        api_module._DEGRADED_STATE,
        model="same-promoted-model",
        df="new-feature-snapshot",
        features=["feature"],
        meta={"trained_at": "unchanged-model-version"},
        model_version="unchanged-model-version",
        model_loaded_at="reload-time",
    )
    with mock.patch.object(api_module, "_build_model_state", return_value=None):
        with TestClient(api_module.app):
            assert api_module.app.state.df is None
            with mock.patch.object(
                api_module, "_build_model_state", return_value=refreshed_state
            ):
                with mock.patch.object(
                    sync_worker,
                    "run_sync",
                    return_value={
                        "predictions_refreshed": True,
                        "noop": False,
                    },
                ):
                    summary = scheduler.sync_job()

            assert summary["predictions_refreshed"] is True
            assert api_module.app.state.model == "same-promoted-model"
            assert api_module.app.state.df == "new-feature-snapshot"
            assert api_module.app.state.model_version == "unchanged-model-version"


def test_failed_sync_reload_recovers_on_next_identical_noop(api_module):
    """A transient artifact-read failure must not strand the old API state.

    sync_worker has already remembered the successful input version, so the
    next cycle is intentionally a prediction no-op.  The scheduler's pending
    serving-state reload is nevertheless retried and atomically applied.
    """
    import scheduler
    import sync_worker

    refreshed_state = dict(
        api_module._DEGRADED_STATE,
        model="same-promoted-model",
        df="recovered-feature-snapshot",
        features=["feature"],
        meta={"trained_at": "unchanged-model-version"},
        model_version="unchanged-model-version",
        model_loaded_at="retry-time",
    )
    with mock.patch.object(api_module, "_build_model_state", return_value=None):
        with TestClient(api_module.app):
            sync_outcomes = [
                {"predictions_refreshed": True, "noop": False},
                {"predictions_refreshed": False, "noop": True},
            ]
            with mock.patch.object(
                sync_worker, "run_sync", side_effect=sync_outcomes
            ):
                with mock.patch.object(
                    api_module,
                    "_build_model_state",
                    side_effect=[RuntimeError("partially visible artifact"), refreshed_state],
                ):
                    first = scheduler.sync_job()
                    assert first["serving_state_reloaded"] is False
                    assert first["serving_state_reload_pending"] is True
                    assert api_module.app.state.model is None

                    second = scheduler.sync_job()

            assert second["noop"] is True
            assert second["serving_state_reloaded"] is True
            assert second["serving_state_reload_pending"] is False
            assert api_module.app.state.model == "same-promoted-model"
            assert api_module.app.state.df == "recovered-feature-snapshot"
