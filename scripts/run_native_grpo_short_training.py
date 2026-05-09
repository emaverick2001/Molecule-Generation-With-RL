#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rl.checkpoint_export import export_native_grpo_checkpoint_for_diffdock_inference
from src.utils.artifact_logger import save_json
from src.utils.config import load_yaml
from src.utils.paths import create_run_dir, make_run_id


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-_.") or "native_grpo_short"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _remove_command_option(command: list[str], option: str) -> list[str]:
    cleaned = []
    index = 0
    while index < len(command):
        if command[index] == option:
            index += 2
            continue
        cleaned.append(command[index])
        index += 1
    return cleaned


def _config_for_diffdock_checkpoint(
    *,
    base_config: dict[str, Any],
    mode: str,
    seed: int,
    group_size: int,
    model_dir: Path,
    ckpt: str,
) -> dict[str, Any]:
    config = dict(base_config)
    config["experiment"] = dict(config.get("experiment", {}))
    config["experiment"]["name"] = "diffdock_native_grpo_short_training"
    config["experiment"]["mode"] = mode
    config["experiment"]["seed"] = seed

    config["generation"] = dict(config.get("generation", {}))
    config["generation"]["backend"] = "diffdock"
    config["generation"]["num_samples"] = group_size

    diffdock_config = dict(config.get("diffdock", {}))
    command_template = list(diffdock_config["command_template"])
    command_template = _remove_command_option(command_template, "--model_dir")
    command_template = _remove_command_option(command_template, "--ckpt")
    command_template.extend(["--model_dir", str(model_dir), "--ckpt", ckpt])
    diffdock_config["command_template"] = command_template
    config["diffdock"] = diffdock_config

    return config


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True, env=env)


def _run_dirs() -> set[Path]:
    root = Path("artifacts/runs")
    if not root.is_dir():
        return set()
    return {path for path in root.iterdir() if path.is_dir()}


def _find_created_run(before: set[Path], after: set[Path]) -> Path:
    created = sorted(after - before, key=lambda path: path.stat().st_mtime, reverse=True)
    if not created:
        raise RuntimeError("Could not detect newly created run directory")
    return created[0]


def _subprocess_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    if env.get("CONDA_PREFIX") and not env.get("DIFFDOCK_PYTHON"):
        env["DIFFDOCK_PYTHON"] = str(Path(env["CONDA_PREFIX"]) / "bin" / "python")
    return env


def _write_config(config: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _run_rollout(
    *,
    config_path: Path,
    seed: int,
    run_tag: str,
    env: dict[str, str],
) -> Path:
    before = _run_dirs()
    _run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.pipeline.run_baseline",
            "--config",
            str(config_path),
            "--seed",
            str(seed),
            "--run-tag",
            run_tag,
        ],
        env=env,
    )
    run_dir = _find_created_run(before, _run_dirs())
    _run(["./scripts/run_evaluation.sh", str(run_dir)])
    _run(["./scripts/diagnose_run_structures.py", str(run_dir)])
    return run_dir


def _generated_sample_count(run_dir: Path) -> int:
    manifest_path = run_dir / "generated_samples_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Generated manifest not found: {manifest_path}")
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"Generated manifest is empty: {manifest_path}")
    return len(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small native DiffDock GRPO training loop: generate grouped "
            "rollouts from the current model, apply one GRPO update, export the "
            "checkpoint for inference, and repeat."
        ),
    )
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--repo-root", default="external/DiffDock")
    parser.add_argument("--base-model-dir", default="external/DiffDock/workdir/v1.1/score_model")
    parser.add_argument("--base-ckpt", default="best_ema_inference_epoch_model.pt")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", default="native_grpo_short")
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--lm-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive")
    if args.group_size <= 0:
        raise ValueError("--group-size must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.eval_every < 0:
        raise ValueError("--eval-every must be non-negative")

    run_tag = _slug(args.run_tag)
    controller_run_id = make_run_id(
        model="diffdock",
        experiment=f"posttraining_native_grpo_short_{run_tag}",
        seed=args.seed,
    )
    controller_dir = create_run_dir(args.run_root, controller_run_id, exist_ok=False)
    config_dir = controller_dir / "configs"
    exported_root = controller_dir / "exported_models"
    save_json(vars(args), controller_dir / "args.json")

    base_config = load_yaml(args.train_config)
    env = _subprocess_env()
    current_model_dir = Path(args.base_model_dir)
    current_ckpt = args.base_ckpt
    step_summaries = []

    for step in range(args.num_steps):
        step_name = f"step_{step:03d}"
        rollout_tag = f"{run_tag}_{step_name}_rollout"
        rollout_config = _config_for_diffdock_checkpoint(
            base_config=base_config,
            mode=f"native_grpo_short_rollout_{run_tag}_{step_name}",
            seed=args.seed,
            group_size=args.group_size,
            model_dir=current_model_dir,
            ckpt=current_ckpt,
        )
        rollout_config_path = _write_config(
            rollout_config,
            config_dir / f"{step_name}_rollout.yaml",
        )

        rollout_run_dir = _run_rollout(
            config_path=rollout_config_path,
            seed=args.seed,
            run_tag=rollout_tag,
            env=env,
        )
        generated_count = _generated_sample_count(rollout_run_dir)

        before = _run_dirs()
        grpo_command = [
            sys.executable,
            "scripts/run_native_grpo_smoke.py",
            str(rollout_run_dir),
            "--repo-root",
            args.repo_root,
            "--model-dir",
            str(current_model_dir),
            "--ckpt",
            current_ckpt,
            "--limit",
            str(generated_count),
            "--samples-per-complex",
            str(args.group_size),
            "--learning-rate",
            str(args.learning_rate),
            "--clip-epsilon",
            str(args.clip_epsilon),
            "--max-grad-norm",
            str(args.max_grad_norm),
            "--seed",
            str(args.seed),
            "--run-tag",
            f"{run_tag}_{step_name}_grpo",
        ]
        if args.lm_embeddings:
            grpo_command.append("--lm-embeddings")
        else:
            grpo_command.append("--no-lm-embeddings")
        if args.no_parallel:
            grpo_command.append("--no-parallel")
        _run(grpo_command, env=env)
        grpo_run_dir = _find_created_run(before, _run_dirs())

        native_checkpoint = grpo_run_dir / "checkpoints" / "native_grpo_step_000.pt"
        exported_model_dir = exported_root / step_name / "score_model"
        exported_ckpt = f"native_grpo_{step_name}_inference.pt"
        export = export_native_grpo_checkpoint_for_diffdock_inference(
            checkpoint_path=native_checkpoint,
            source_model_dir=args.base_model_dir,
            output_model_dir=exported_model_dir,
            checkpoint_name=exported_ckpt,
        )
        save_json(export.to_dict(), exported_model_dir / "export_metadata.json")

        eval_run_dir = None
        if args.eval_every and ((step + 1) % args.eval_every == 0 or step == args.num_steps - 1):
            eval_tag = f"{run_tag}_{step_name}_eval"
            eval_config = _config_for_diffdock_checkpoint(
                base_config=base_config,
                mode=f"native_grpo_short_eval_{run_tag}_{step_name}",
                seed=args.seed,
                group_size=args.group_size,
                model_dir=exported_model_dir,
                ckpt=exported_ckpt,
            )
            eval_config_path = _write_config(
                eval_config,
                config_dir / f"{step_name}_eval.yaml",
            )
            eval_run_dir = _run_rollout(
                config_path=eval_config_path,
                seed=args.seed,
                run_tag=eval_tag,
                env=env,
            )

        summary = {
            "step": step,
            "rollout_run_dir": str(rollout_run_dir),
            "grpo_run_dir": str(grpo_run_dir),
            "native_checkpoint": str(native_checkpoint),
            "exported_model_dir": str(exported_model_dir),
            "exported_ckpt": exported_ckpt,
            "eval_run_dir": str(eval_run_dir) if eval_run_dir else None,
            "generated_count": generated_count,
        }
        step_summaries.append(summary)
        save_json(step_summaries, controller_dir / "training_steps.json")

        current_model_dir = exported_model_dir
        current_ckpt = exported_ckpt

    save_json(
        {
            "controller_run_dir": str(controller_dir),
            "num_steps": args.num_steps,
            "group_size": args.group_size,
            "final_model_dir": str(current_model_dir),
            "final_ckpt": current_ckpt,
            "steps": step_summaries,
        },
        controller_dir / "posttraining_summary.json",
    )

    print("Native DiffDock GRPO short training complete")
    print(f"run_dir={controller_dir}")
    print(f"final_model_dir={current_model_dir}")
    print(f"final_ckpt={current_ckpt}")


if __name__ == "__main__":
    main()
