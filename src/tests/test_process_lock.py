from __future__ import annotations

import os
import time

from process_lock import process_lock_active


def test_process_lock_requires_live_pid(tmp_path):
    lock = tmp_path / "train.lock"
    lock.write_text(str(os.getpid()))
    assert process_lock_active(str(lock), max_age_seconds=60)

    lock.write_text("999999999")
    assert not process_lock_active(str(lock), max_age_seconds=60)


def test_old_or_malformed_lock_is_stale(tmp_path):
    lock = tmp_path / "train.lock"
    lock.write_text(str(os.getpid()))
    old = time.time() - 120
    os.utime(lock, (old, old))
    assert not process_lock_active(str(lock), max_age_seconds=60)

    lock.write_text("not-a-pid")
    assert not process_lock_active(str(lock), max_age_seconds=60)
