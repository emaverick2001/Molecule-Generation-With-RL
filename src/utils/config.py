from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(global_config: dict, experiment_config: dict) -> dict:
    merged = global_config.copy()
    merged.update(experiment_config)
    return merged


def load_experiment_config(global_path: str, experiment_path: str) -> dict:
    global_config = load_yaml(global_path)
    experiment_config = load_yaml(experiment_path)
    return merge_configs(global_config, experiment_config)