"""Focused guards for production and presentation scheduler policies."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import scheduler


class _FakeScheduler:
    def __init__(self, *, timezone: str):
        self.timezone = timezone
        self.jobs = []
        self.started = False
        self.stopped = False

    def add_job(self, _fn, _trigger, **kwargs):
        self.jobs.append(
            SimpleNamespace(
                id=kwargs["id"],
                next_run_time=kwargs.get("next_run_time"),
            )
        )

    def start(self):
        self.started = True

    def shutdown(self, *, wait: bool):
        self.stopped = True

    def get_jobs(self):
        return list(self.jobs)


def _start_with_fake_scheduler():
    with mock.patch(
        "apscheduler.schedulers.background.BackgroundScheduler",
        _FakeScheduler,
    ):
        assert scheduler.start() is True
    return scheduler._scheduler


def test_scheduler_defaults_keep_all_production_jobs(monkeypatch, tmp_path):
    """Omitting the new flags preserves the previous production behaviour."""
    scheduler._scheduler = None
    monkeypatch.setattr(scheduler, "TRAIN_LOCK", str(tmp_path / ".train.lock"))
    monkeypatch.delenv("SCHEDULER_SYNC_ENABLED", raising=False)
    monkeypatch.delenv("SCHEDULER_RETRAIN_ENABLED", raising=False)
    monkeypatch.delenv("GISTDA_ENABLED", raising=False)

    fake = _start_with_fake_scheduler()

    assert fake.started is True
    assert [job.id for job in fake.jobs] == [
        "sync",
        "retrain_check",
        "gistda_poll",
    ]
    scheduler.shutdown()


def test_demo_policy_schedules_only_gistda_and_preserves_train_lock(
    monkeypatch, tmp_path
):
    scheduler._scheduler = None
    train_lock = tmp_path / ".train.lock"
    train_lock.write_text("another-process", encoding="utf-8")
    monkeypatch.setattr(scheduler, "TRAIN_LOCK", str(train_lock))
    monkeypatch.setenv("SCHEDULER_SYNC_ENABLED", "false")
    monkeypatch.setenv("SCHEDULER_RETRAIN_ENABLED", "false")
    monkeypatch.setenv("GISTDA_ENABLED", "true")

    fake = _start_with_fake_scheduler()

    assert [job.id for job in fake.jobs] == ["gistda_poll"]
    assert train_lock.read_text(encoding="utf-8") == "another-process"
    assert scheduler.status()["jobs"][0]["id"] == "gistda_poll"
    scheduler.shutdown()


def test_run_sh_demo_rejects_mutating_pipeline_flags():
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        ["bash", str(repo_root / "run.sh"), "--demo", "--fresh"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert proc.returncode == 2
    assert "--demo cannot be combined" in proc.stderr


def test_run_sh_help_documents_demo_safety_mode():
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        ["bash", str(repo_root / "run.sh"), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    assert "--demo" in proc.stdout
    assert "only the live GISTDA scheduler job" in proc.stdout
