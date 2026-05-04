from __future__ import annotations

import argparse
from pathlib import Path
import re

import yaml

from src.rl.config import RLConfig, parse_rl_config, resolve_run_paths
from src.rl.train import run_training
from src.utils.artifact_logger import save_json, save_text
from src.utils.paths import create_run_dir, make_run_id


def _normalize_run_tag(value: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return tag.strip("-_.")


def _load_config_with_overrides(
    config_path: str | Path,
    *,
    source_run_dir: str | None = None,
    run_tag: str | None = None,
    seed: int | None = None,
) -> RLConfig:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Posttraining config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if source_run_dir is not None:
        raw.setdefault("data", {})["source_run_dir"] = source_run_dir
    if run_tag is not None:
        raw.setdefault("artifacts", {})["run_tag"] = _normalize_run_tag(run_tag)
    if seed is not None:
        raw.setdefault("experiment", {})["seed"] = seed

    return resolve_run_paths(parse_rl_config(raw))


def create_run_skeleton(
    cfg: RLConfig,
    *,
    config_path: str | Path,
    exist_ok: bool = False,
) -> Path:
    mode = cfg.experiment.mode
    if cfg.artifacts.run_tag:
        mode = f"{mode}_{cfg.artifacts.run_tag}"

    run_id = make_run_id(
        model=cfg.experiment.model,
        experiment=mode,
        seed=cfg.experiment.seed,
    )
    run_dir = create_run_dir(cfg.artifacts.run_root, run_id, exist_ok=exist_ok)

    for child in [
        "checkpoints",
        "rollouts",
        "logs",
        "eval",
    ]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)

    config_text = Path(config_path).read_text(encoding="utf-8")
    save_text(config_text, run_dir / "config.yaml")
    save_json(cfg.to_dict(), run_dir / "config_snapshot.json")

    return run_dir


def run_posttraining(
    config_path: str | Path,
    *,
    source_run_dir: str | None = None,
    run_tag: str | None = None,
    seed: int | None = None,
    exist_ok: bool = False,
) -> Path:
    cfg = _load_config_with_overrides(
        config_path,
        source_run_dir=source_run_dir,
        run_tag=run_tag,
        seed=seed,
    )
    run_dir = create_run_skeleton(
        cfg,
        config_path=config_path,
        exist_ok=exist_ok,
    )
    summary = run_training(cfg, run_dir)
    save_json(summary.to_dict(), run_dir / "posttraining_summary.json")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DiffDock posttraining/RL setup workflows.",
    )
    parser.add_argument(
        "--config",
        default="configs/rl/offline_reward_debug.yaml",
        help="Path to RL/posttraining config.",
    )
    parser.add_argument(
        "--source-run-dir",
        default=None,
        help=(
            "Completed baseline run directory to use for offline reward debug. "
            "Overrides data.source_run_dir in the config."
        ),
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Append a tag to the posttraining run ID.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override experiment.seed from the config.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow writing to an existing posttraining run directory.",
    )

    args = parser.parse_args()
    run_dir = run_posttraining(
        args.config,
        source_run_dir=args.source_run_dir,
        run_tag=args.run_tag,
        seed=args.seed,
        exist_ok=args.exist_ok,
    )
    print(f"Posttraining setup complete: {run_dir}")


if __name__ == "__main__":
    main()
