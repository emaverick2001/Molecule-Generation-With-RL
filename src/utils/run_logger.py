# Run lifecycle utilities for experiment tracking and artifact organization.

# Use this module to create and standardize per-run directories and file layout:
# - `initialize_run(...)`:
#   - builds a unique run ID (model/mode/seed/reward)
#   - creates the run directory
#   - saves `config_snapshot.json`
#   - optionally copies the source YAML config
#   - initializes an empty `errors.log`
# - `get_artifact_paths(run_dir)`:
#   - returns canonical paths for common artifacts
#     (config, generated samples, rewards, metrics, summary, errors)
# - `initialize_artifact_dirs(run_dir)`:
#   - creates optional subfolders used during execution
#     (e.g., `generated_samples/`, `logs/`)

# Why this exists:
# - keeps run outputs consistent across baseline/rerank/reward-filtering/post-training
# - improves reproducibility by snapshotting config per run
# - simplifies downstream code by using predictable artifact paths

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from src.utils.artifact_logger import save_json, save_text
from src.utils.paths import create_run_dir, ensure_dir, make_run_id


def initialize_run(
    config: dict[str, Any],
    config_path: str | Path | None = None,
    exist_ok: bool = False,
) -> Path:
    """
    Initialize a run directory and save basic run artifacts.

    Creates:
    - run directory
    - config_snapshot.json
    - optional copied config.yaml
    - errors.log
    """
    experiment = config["experiment"]

    reward_type = None
    if config.get("reward", {}).get("enabled", False):
        reward_type = config["reward"].get("type")

    run_id = make_run_id(
        model=experiment["model"],
        experiment=experiment["mode"],
        seed=experiment["seed"],
        reward=reward_type,
    )

    base_run_dir = config.get("paths", {}).get("run_dir", "artifacts/runs")
    run_dir = create_run_dir(base_run_dir, run_id, exist_ok=exist_ok)

    save_json(config, run_dir / "config_snapshot.json")

    if config_path is not None:
        shutil.copy(config_path, run_dir / "config.yaml")

    save_text("", run_dir / "errors.log")

    return run_dir


def get_artifact_paths(run_dir: str | Path) -> dict[str, Path]:
    """
    Return standard artifact paths for one experiment run.
    """
    run_dir = Path(run_dir)

    return {
        "config": run_dir / "config_snapshot.json",
        "generated_samples": run_dir / "generated_samples_manifest.json",
        "rewards": run_dir / "rewards.csv",
        "metrics": run_dir / "metrics.json",
        "summary": run_dir / "summary.md",
        "errors": run_dir / "errors.log",
    }


def initialize_artifact_dirs(run_dir: str | Path) -> None:
    """
    Create optional subdirectories inside a run directory.
    """
    run_dir = Path(run_dir)

    ensure_dir(run_dir / "generated_samples")
    ensure_dir(run_dir / "logs")