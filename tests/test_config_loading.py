from pathlib import Path

import pytest
import yaml

from src.utils.config import load_yaml, deep_merge, load_experiment_config


def write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def test_load_yaml_reads_config(tmp_path):
    config_path = tmp_path / "config.yaml"

    write_yaml(
        config_path,
        {
            "experiment": {
                "name": "diffdock_baseline",
                "seed": 42,
            }
        },
    )

    config = load_yaml(config_path)

    assert config["experiment"]["name"] == "diffdock_baseline"
    assert config["experiment"]["seed"] == 42


def test_load_yaml_missing_file_raises_error(tmp_path):
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        load_yaml(missing_path)


def test_deep_merge_preserves_nested_global_values():
    global_config = {
        "evaluation": {
            "rmsd_threshold": 2.0,
            "top_k": [1, 5, 10],
            "compute_diversity": True,
        }
    }

    experiment_config = {
        "evaluation": {
            "top_k": [1, 5],
        }
    }

    merged = deep_merge(global_config, experiment_config)

    assert merged["evaluation"]["rmsd_threshold"] == 2.0
    assert merged["evaluation"]["top_k"] == [1, 5]
    assert merged["evaluation"]["compute_diversity"] is True


def test_load_experiment_config_merges_global_and_experiment_configs(tmp_path):
    global_path = tmp_path / "global.yaml"
    experiment_path = tmp_path / "baseline.yaml"

    write_yaml(
        global_path,
        {
            "runtime": {
                "seed": 42,
                "device": "cuda",
            },
            "evaluation": {
                "rmsd_threshold": 2.0,
                "top_k": [1, 5, 10],
            },
        },
    )

    write_yaml(
        experiment_path,
        {
            "experiment": {
                "name": "diffdock_baseline",
                "mode": "baseline",
            },
            "evaluation": {
                "top_k": [1, 5],
            },
        },
    )

    config = load_experiment_config(global_path, experiment_path)

    assert config["experiment"]["name"] == "diffdock_baseline"
    assert config["experiment"]["mode"] == "baseline"
    assert config["runtime"]["seed"] == 42
    assert config["runtime"]["device"] == "cuda"
    assert config["evaluation"]["rmsd_threshold"] == 2.0
    assert config["evaluation"]["top_k"] == [1, 5]