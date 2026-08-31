"""Container-entrypoint bootstrap tests (no Docker daemon required)."""

import os
import json
import re
import subprocess
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_production_container_defaults_to_fail_closed_app_env():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"\bAPP_ENV=production\b", dockerfile)
    assert 'http://127.0.0.1:${PORT}/health/ready' in dockerfile


def test_railway_deployments_use_strict_readiness_probe():
    railway_json = json.loads(
        (REPO_ROOT / "railway.json").read_text(encoding="utf-8")
    )
    with (REPO_ROOT / "railway.toml").open("rb") as handle:
        railway_toml = tomllib.load(handle)

    assert railway_json["deploy"]["healthcheckPath"] == "/health/ready"
    assert railway_toml["deploy"]["healthcheckPath"] == "/health/ready"


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _bootstrap_tree(root: Path) -> None:
    sources = {
        "outputs/models/lgbm_fire_date_model.pkl": "bootstrap-model",
        "outputs/features/full_features.parquet": "bootstrap-features",
        "outputs/metadata/dataset_info.json": "bootstrap-metadata",
        "outputs/riskmap/fire_dates_all.geojson": "bootstrap-riskmap",
        "data/boundaries/thailand.geojson": "bootstrap-boundary",
        "data/static/tree_cover_per_cell.parquet": "bootstrap-tree-cover",
    }
    for relative, value in sources.items():
        _write(root / relative, value)


def _run_start(repo_root: Path, bootstrap: Path, output: Path, data: Path):
    env = {
        **os.environ,
        "BOOTSTRAP_DIR": str(bootstrap),
        "OUTPUT_DIR": str(output),
        "DATA_DIR": str(data),
    }
    return subprocess.run(
        ["sh", str(repo_root / "scripts/start.sh"), "true"],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )


def test_logs_only_volume_seeds_complete_bundle(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    bootstrap = tmp_path / "bootstrap"
    output = tmp_path / "volume" / "outputs"
    data = tmp_path / "volume" / "data"

    _bootstrap_tree(bootstrap)

    # A state marker made the former all-or-nothing seeder skip the entire
    # directory even though no promoted artifact existed.
    _write(output / ".last_prediction_version", "v1-existing")

    result = _run_start(repo_root, bootstrap, output, data)

    assert result.returncode == 0, result.stderr
    assert (output / "models/lgbm_fire_date_model.pkl").read_text() == "bootstrap-model"
    assert (output / "features/full_features.parquet").read_text() == "bootstrap-features"
    assert (output / "metadata/dataset_info.json").read_text() == "bootstrap-metadata"
    assert (output / "riskmap/fire_dates_all.geojson").read_text() == "bootstrap-riskmap"
    assert (data / "boundaries/thailand.geojson").read_text() == "bootstrap-boundary"
    assert (data / "static/tree_cover_per_cell.parquet").read_text() == "bootstrap-tree-cover"
    assert (data / "raw").is_dir()
    assert (data / "firms").is_dir()
    assert (data / "weather").is_dir()


def test_partial_promoted_bundle_fails_without_mixing_versions(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    bootstrap = tmp_path / "bootstrap"
    output = tmp_path / "volume" / "outputs"
    data = tmp_path / "volume" / "data"
    _bootstrap_tree(bootstrap)
    _write(output / "models/lgbm_fire_date_model.pkl", "promoted-model")

    result = _run_start(repo_root, bootstrap, output, data)

    assert result.returncode != 0
    assert "partial output bundle (1/4 files)" in result.stderr
    assert (output / "models/lgbm_fire_date_model.pkl").read_text() == "promoted-model"
    assert not (output / "features/full_features.parquet").exists()
    assert not (output / "metadata/dataset_info.json").exists()
    assert not (output / "riskmap/fire_dates_all.geojson").exists()


def test_complete_promoted_bundle_is_preserved(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    bootstrap = tmp_path / "bootstrap"
    output = tmp_path / "volume" / "outputs"
    data = tmp_path / "volume" / "data"
    _bootstrap_tree(bootstrap)
    promoted = {
        "models/lgbm_fire_date_model.pkl": "promoted-model",
        "features/full_features.parquet": "promoted-features",
        "metadata/dataset_info.json": "promoted-metadata",
        "riskmap/fire_dates_all.geojson": "promoted-riskmap",
    }
    for relative, value in promoted.items():
        _write(output / relative, value)

    result = _run_start(repo_root, bootstrap, output, data)

    assert result.returncode == 0, result.stderr
    assert "complete promoted output bundle present" in result.stdout
    for relative, value in promoted.items():
        assert (output / relative).read_text() == value
