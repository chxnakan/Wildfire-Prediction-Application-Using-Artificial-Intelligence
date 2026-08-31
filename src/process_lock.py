"""Small process-lock helpers shared by scheduler and prediction sync."""

from __future__ import annotations

import os
import time


def process_lock_active(path: str, *, max_age_seconds: int) -> bool:
    """Return True only when a lock points to a live, reasonably fresh PID.

    A plain ``exists()`` check turns a crash-left lock into a permanent outage.
    Malformed locks, dead PIDs, and locks older than the operation timeout are
    stale and may be safely overwritten by the next run.
    """
    try:
        stat = os.stat(path)
        with open(path, encoding="utf-8") as handle:
            pid = int(handle.read().strip())
    except (OSError, TypeError, ValueError):
        return False

    if pid <= 0 or time.time() - stat.st_mtime > max(1, max_age_seconds):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The PID exists but belongs to another user; fail closed.
        return True
    except OSError:
        return False
    return True
