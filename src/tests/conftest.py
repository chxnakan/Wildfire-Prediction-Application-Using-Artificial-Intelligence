"""pytest configuration for the whole suite.

Run the COMPLETE suite from src/ with:

    ../.venv/bin/python -m pytest tests -q

Every test uses mocked external APIs — no NASA/GISTDA network access — and
forces the in-memory store (DATABASE_URL removed) so no PostgreSQL is
needed either.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must happen before any project import: force in-memory store + no scheduler.
os.environ.pop("DATABASE_URL", None)
os.environ.setdefault("SCHEDULER_ENABLED", "false")
os.environ.setdefault("GISTDA_ENABLED", "true")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_memory_stores():
    """Isolate module-level in-memory state between tests: the DB stores AND
    the scheduler's model-promoted callback registry (each TestClient lifespan
    registers api's hot-reload callback; without clearing it they accumulate
    and leak across TestClient instances)."""
    import db
    def _reset():
        db._MEM_LIVE.clear()
        db._MEM_POLL_LOG.clear()
        db._MEM_LOCKS.clear()
        db._MEM_PRED_RUNS.clear()
        try:
            import scheduler
            scheduler._on_model_promoted.clear()
        except Exception:
            pass
    _reset()
    yield
    _reset()
