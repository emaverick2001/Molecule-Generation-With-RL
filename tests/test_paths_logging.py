from pathlib import Path

import pytest

from src.utils.paths import (
    create_run_dir,
    ensure_dir,
    make_run_id,
    resolve_project_path,
)


def test_make_run_id_without_reward():
    run_id = make_run_id(
        model="diffdock",
        experiment="baseline",
        seed=42,
    )

    assert "diffdock" in run_id
    assert "baseline" in run_id
    assert "seed42" in run_id
    assert "None" not in run_id


def test_make_run_id_with_reward():
    run_id = make_run_id(
        model="diffdock",
        experiment="reward_filtering",
        reward="negative_rmsd",
        seed=42,
    )

    assert "diffdock" in run_id
    assert "reward_filtering" in run_id
    assert "negative_rmsd" in run_id
    assert "seed42" in run_id


def test_create_run_dir_creates_directory(tmp_path):
    run_id = "test_run_seed42"

    run_dir = create_run_dir(tmp_path, run_id)

    assert run_dir.exists()
    assert run_dir.is_dir()
    assert run_dir == tmp_path / run_id


def test_create_run_dir_fails_if_directory_exists(tmp_path):
    run_id = "existing_run"
    existing_dir = tmp_path / run_id
    existing_dir.mkdir()

    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, run_id)


def test_create_run_dir_allows_existing_directory_when_exist_ok_true(tmp_path):
    run_id = "existing_run"
    existing_dir = tmp_path / run_id
    existing_dir.mkdir()

    run_dir = create_run_dir(tmp_path, run_id, exist_ok=True)

    assert run_dir.exists()
    assert run_dir.is_dir()


def test_ensure_dir_creates_directory(tmp_path):
    target_dir = tmp_path / "artifacts" / "runs"

    created_dir = ensure_dir(target_dir)

    assert created_dir.exists()
    assert created_dir.is_dir()
    assert created_dir == target_dir


def test_resolve_project_path_resolves_relative_path(tmp_path):
    resolved = resolve_project_path(
        "artifacts/runs",
        root_dir=tmp_path,
    )

    assert resolved == (tmp_path / "artifacts/runs").resolve()


def test_resolve_project_path_preserves_absolute_path(tmp_path):
    absolute_path = tmp_path / "artifacts" / "runs"

    resolved = resolve_project_path(absolute_path)

    assert resolved == absolute_path