from pathlib import Path

import pytest
import yaml

from src.utils.run_logger import (
    get_artifact_paths,
    initialize_artifact_dirs,
    initialize_run,
)


def write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def make_mvp_config(tmp_path: Path) -> dict:
    return {
        "paths": {
            "run_dir": str(tmp_path / "runs"),
        },
        "experiment": {
            "name": "diffdock_baseline",
            "model": "diffdock",
            "mode": "baseline",
            "seed": 42,
        },
        "reward": {
            "enabled": False,
        },
    }


def test_initialize_run_creates_run_directory(tmp_path):
    config = make_mvp_config(tmp_path)

    run_dir = initialize_run(config)

    assert run_dir.exists()
    assert run_dir.is_dir()
    assert "diffdock" in run_dir.name
    assert "baseline" in run_dir.name
    assert "seed42" in run_dir.name


def test_initialize_run_saves_config_snapshot(tmp_path):
    config = make_mvp_config(tmp_path)

    run_dir = initialize_run(config)

    config_snapshot = run_dir / "config_snapshot.json"

    assert config_snapshot.exists()


def test_initialize_run_copies_config_file_when_provided(tmp_path):
    config = make_mvp_config(tmp_path)
    config_path = tmp_path / "baseline.yaml"

    write_yaml(config_path, config)

    run_dir = initialize_run(
        config=config,
        config_path=config_path,
    )

    copied_config = run_dir / "config.yaml"

    assert copied_config.exists()


def test_initialize_run_creates_errors_log(tmp_path):
    config = make_mvp_config(tmp_path)

    run_dir = initialize_run(config)

    errors_log = run_dir / "errors.log"

    assert errors_log.exists()

    with errors_log.open("r", encoding="utf-8") as f:
        text = f.read()

    assert text == ""


def test_initialize_run_fails_if_run_already_exists(tmp_path):
    config = make_mvp_config(tmp_path)

    initialize_run(config)

    with pytest.raises(FileExistsError):
        initialize_run(config)


def test_initialize_run_allows_existing_run_when_exist_ok_true(tmp_path):
    config = make_mvp_config(tmp_path)

    first_run_dir = initialize_run(config)

    second_run_dir = initialize_run(
        config,
        exist_ok=True,
    )

    assert first_run_dir == second_run_dir
    assert second_run_dir.exists()


def test_get_artifact_paths_returns_standard_paths(tmp_path):
    run_dir = tmp_path / "runs" / "test_run"

    paths = get_artifact_paths(run_dir)

    assert paths["config"] == run_dir / "config_snapshot.json"
    assert paths["generated_samples"] == run_dir / "generated_samples_manifest.json"
    assert paths["rewards"] == run_dir / "rewards.csv"
    assert paths["metrics"] == run_dir / "metrics.json"
    assert paths["summary"] == run_dir / "summary.md"
    assert paths["errors"] == run_dir / "errors.log"


def test_initialize_artifact_dirs_creates_subdirectories(tmp_path):
    run_dir = tmp_path / "runs" / "test_run"

    initialize_artifact_dirs(run_dir)

    assert (run_dir / "generated_samples").exists()
    assert (run_dir / "generated_samples").is_dir()
    assert (run_dir / "logs").exists()
    assert (run_dir / "logs").is_dir()