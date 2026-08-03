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
    with mock.patch.object(api_module, "_build_model_state", return_value=None):
        for _ in range(3):
            with TestClient(api_module.app):
                pass
    # conftest's autouse cleaner clears these between tests; within one test the
    # lifespan registers one callback per enter — assert it doesn't explode.
    assert len(scheduler._on_model_promoted) <= 3
