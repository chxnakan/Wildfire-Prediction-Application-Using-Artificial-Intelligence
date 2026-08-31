"""Focused guards for production and presentation scheduler policies."""

from __future__ import annotations

import os
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


def test_default_scheduled_retrain_uses_bounded_history():
    assert scheduler.TRAIN_ARGS == [
        "--quick", "--max-history-days", "400",
    ]


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


def _fake_runner_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Small hermetic run.sh fixture that never trains or starts a server."""
    repo_root = Path(__file__).resolve().parents[2]
    runner = tmp_path / "run.sh"
    runner.write_text(
        (repo_root / "run.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "if [ \"$#\" -eq 0 ]; then\n"
        "  script=$(cat)\n"
        "  if printf '%s' \"$script\" | grep -q FIREWATCH_OUTPUT_DIR_LOOKUP; then\n"
        "    sed -n 's/^OUTPUT_DIR=//p' .env 2>/dev/null | tail -n 1\n"
        "  else\n"
        "    echo unknown\n"
        "  fi\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return runner, fake_bin


def _write_demo_snapshot(
    root: Path,
    *,
    output_name: str = "custom-output",
    include_features: bool = True,
    include_geojson: bool = True,
) -> Path:
    output = root / output_name
    (output / "models").mkdir(parents=True)
    (output / "models" / "lgbm_fire_date_model.pkl").write_bytes(b"model")
    (output / "metadata").mkdir(parents=True)
    (output / "metadata" / "dataset_info.json").write_text(
        "{}", encoding="utf-8"
    )
    if include_features:
        (output / "features").mkdir(parents=True)
        (output / "features" / "full_features.parquet").write_bytes(b"features")
    if include_geojson:
        (output / "riskmap").mkdir(parents=True)
        (output / "riskmap" / "fire_dates_all.geojson").write_text(
            '{"type":"FeatureCollection","features":[]}', encoding="utf-8"
        )
    (root / "src").mkdir()
    (root / "web" / "dist").mkdir(parents=True)
    (root / "web" / "dist" / "index.html").write_text("ok", encoding="utf-8")
    return output


def _run_fake_demo(
    tmp_path: Path,
    fake_bin: Path,
    runner: Path,
    *,
    output_dir_env: str | None = "./custom-output",
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update({"PATH": f"{fake_bin}:{env['PATH']}", "STALE_DAYS": "999"})
    if output_dir_env is None:
        env.pop("OUTPUT_DIR", None)
    else:
        env["OUTPUT_DIR"] = output_dir_env
    return subprocess.run(
        ["bash", str(runner), "--demo"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_run_sh_demo_honors_configured_output_dir(tmp_path):
    runner, fake_bin = _fake_runner_repo(tmp_path)
    _write_demo_snapshot(tmp_path)

    proc = _run_fake_demo(tmp_path, fake_bin, runner)

    assert proc.returncode == 0, proc.stderr
    assert "Ready" in proc.stdout


def test_run_sh_demo_reads_output_dir_from_dotenv(tmp_path):
    runner, fake_bin = _fake_runner_repo(tmp_path)
    _write_demo_snapshot(tmp_path, output_name="dotenv-output")
    (tmp_path / ".env").write_text(
        "OUTPUT_DIR=./dotenv-output\n", encoding="utf-8"
    )

    proc = _run_fake_demo(
        tmp_path, fake_bin, runner, output_dir_env=None
    )

    assert proc.returncode == 0, proc.stderr
    assert "Ready" in proc.stdout


def test_run_sh_demo_refuses_missing_feature_frame(tmp_path):
    runner, fake_bin = _fake_runner_repo(tmp_path)
    _write_demo_snapshot(tmp_path, include_features=False)

    proc = _run_fake_demo(tmp_path, fake_bin, runner)

    assert proc.returncode == 1
    assert "No serving feature frame" in proc.stderr


def test_run_sh_demo_never_generates_missing_geojson(tmp_path):
    runner, fake_bin = _fake_runner_repo(tmp_path)
    output = _write_demo_snapshot(tmp_path, include_geojson=False)

    proc = _run_fake_demo(tmp_path, fake_bin, runner)

    assert proc.returncode == 1
    assert "Presentation mode will not generate or modify" in proc.stderr
    assert not (output / "riskmap" / "fire_dates_all.geojson").exists()
