from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML config file as a dictionary.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two dictionaries.

    Values in `override` replace values in `base`.
    Nested dictionaries are merged instead of fully replaced.
    """
    merged = base.copy()

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_experiment_config(
    global_path: str | Path,
    experiment_path: str | Path,
) -> dict[str, Any]:
    """
    Load global config + experiment config and merge them.

    Example:
        global.yaml provides shared paths/settings.
        posttraining.yaml overrides experiment-specific settings.
    """
    global_config = load_yaml(global_path)
    experiment_config = load_yaml(experiment_path)

    return deep_merge(global_config, experiment_config)