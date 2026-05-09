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


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-_.") or "checkpoint_eval"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_source_run_dir(grpo_run_dir: Path, explicit_source_run_dir: str | None) -> Path:
    if explicit_source_run_dir:
        return Path(explicit_source_run_dir)

    summary_path = grpo_run_dir / "posttraining_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            "Could not infer source rollout run. Pass --source-run-dir or provide "
            f"{summary_path}."
        )

    summary = _read_json(summary_path)
    source_run_dir = summary.get("source_run_dir")
    if not source_run_dir:
        raise ValueError(
            "posttraining_summary.json does not contain source_run_dir. "
            "Pass --source-run-dir."
        )

    return Path(source_run_dir)


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


def _checkpoint_eval_config(
    *,
    source_config: dict[str, Any],
    export_model_dir: Path,
    checkpoint_name: str,
    run_tag: str,
    seed: int,
) -> dict[str, Any]:
    config = dict(source_config)
    config["experiment"] = dict(config.get("experiment", {}))
    config["experiment"]["name"] = "diffdock_native_grpo_checkpoint_eval"
    config["experiment"]["mode"] = "native_grpo_checkpoint_eval"
    config["experiment"]["seed"] = seed

    diffdock_config = dict(config.get("diffdock", {}))
    command_template = list(diffdock_config["command_template"])
    command_template = _remove_command_option(command_template, "--model_dir")
    command_template = _remove_command_option(command_template, "--ckpt")
    command_template.extend(
        [
            "--model_dir",
            str(export_model_dir),
            "--ckpt",
            checkpoint_name,
        ]
    )
    diffdock_config["command_template"] = command_template
    config["diffdock"] = diffdock_config

    config.setdefault("paths", {})
    config["metadata"] = {
        **dict(config.get("metadata", {})),
        "native_grpo_checkpoint_eval_run_tag": run_tag,
        "native_grpo_export_model_dir": str(export_model_dir),
        "native_grpo_export_checkpoint": checkpoint_name,
    }
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a native GRPO checkpoint by exporting it for DiffDock "
            "inference, generating poses for the same rollout manifest, and "
            "running evaluation/diagnostics."
        ),
    )
    parser.add_argument("grpo_run_dir")
    parser.add_argument("--source-run-dir", default=None)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Native GRPO checkpoint path. Defaults to checkpoints/native_grpo_step_000.pt.",
    )
    parser.add_argument(
        "--source-model-dir",
        default="external/DiffDock/workdir/v1.1/score_model",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--skip-reranking", action="store_true")
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--include-inputs", action="store_true")
    parser.add_argument("--package-output-dir", default="packaged_runs")
    args = parser.parse_args()

    grpo_run_dir = Path(args.grpo_run_dir)
    if not grpo_run_dir.is_dir():
        raise FileNotFoundError(f"GRPO run directory not found: {grpo_run_dir}")

    source_run_dir = _infer_source_run_dir(grpo_run_dir, args.source_run_dir)
    if not source_run_dir.is_dir():
        raise FileNotFoundError(f"Source rollout run directory not found: {source_run_dir}")

    source_config_path = source_run_dir / "config.yaml"
    if not source_config_path.is_file():
        source_config_path = source_run_dir / "config_snapshot.json"
    if not source_config_path.is_file():
        raise FileNotFoundError(
            f"Source rollout config not found in {source_run_dir}"
        )

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else grpo_run_dir / "checkpoints" / "native_grpo_step_000.pt"
    )
    run_tag = _slug(args.run_tag or f"{grpo_run_dir.name}_checkpoint_eval")
    checkpoint_name = "native_grpo_step_000_inference.pt"
    export_model_dir = (
        Path("artifacts/checkpoints/diffdock/native_grpo")
        / run_tag
        / "score_model"
    )

    export = export_native_grpo_checkpoint_for_diffdock_inference(
        checkpoint_path=checkpoint_path,
        source_model_dir=args.source_model_dir,
        output_model_dir=export_model_dir,
        checkpoint_name=checkpoint_name,
    )
    save_json(export.to_dict(), export_model_dir / "export_metadata.json")

    source_config = load_yaml(source_config_path)
    eval_config = _checkpoint_eval_config(
        source_config=source_config,
        export_model_dir=export_model_dir,
        checkpoint_name=checkpoint_name,
        run_tag=run_tag,
        seed=args.seed,
    )
    eval_config_dir = Path("artifacts/tmp/native_grpo_checkpoint_eval")
    eval_config_dir.mkdir(parents=True, exist_ok=True)
    eval_config_path = eval_config_dir / f"{run_tag}.yaml"
    eval_config_path.write_text(yaml.safe_dump(eval_config, sort_keys=False), encoding="utf-8")

    before = _run_dirs()
    baseline_command = [
        "uv",
        "run",
        "python",
        "-m",
        "src.pipeline.run_baseline",
        "--config",
        str(eval_config_path),
        "--seed",
        str(args.seed),
        "--run-tag",
        run_tag,
    ]
    if args.exist_ok:
        baseline_command.append("--exist-ok")
    _run(baseline_command, env=_subprocess_env())
    eval_run_dir = _find_created_run(before, _run_dirs())

    _run(["./scripts/run_evaluation.sh", str(eval_run_dir)])
    _run(["./scripts/diagnose_run_structures.py", str(eval_run_dir)])
    if not args.skip_reranking:
        _run(
            [
                "./scripts/run_reranking.sh",
                str(eval_run_dir),
                "--config",
                "configs/diffdock/rerank_baseline.yaml",
            ]
        )

    package_path = None
    if args.package:
        package_dir = Path(args.package_output_dir)
        before_packages = set(package_dir.glob("*.tar.gz")) if package_dir.is_dir() else set()
        package_command = [
            "./scripts/package_run_artifacts.sh",
            str(eval_run_dir),
            "--key",
            "--output-dir",
            args.package_output_dir,
        ]
        if args.include_inputs:
            package_command.append("--include-inputs")
        _run(package_command)
        after_packages = set(package_dir.glob("*.tar.gz")) if package_dir.is_dir() else set()
        new_packages = sorted(
            after_packages - before_packages,
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if new_packages:
            package_path = str(new_packages[0])

    source_metrics_path = source_run_dir / "metrics.json"
    eval_metrics_path = eval_run_dir / "metrics.json"
    summary = {
        "source_run_dir": str(source_run_dir),
        "grpo_run_dir": str(grpo_run_dir),
        "checkpoint_path": str(checkpoint_path),
        "export": export.to_dict(),
        "eval_config_path": str(eval_config_path),
        "eval_run_dir": str(eval_run_dir),
        "source_metrics": _read_json(source_metrics_path) if source_metrics_path.is_file() else None,
        "eval_metrics": _read_json(eval_metrics_path) if eval_metrics_path.is_file() else None,
        "package_path": package_path,
    }
    save_json(summary, grpo_run_dir / "checkpoint_eval_summary.json")

    print("Native GRPO checkpoint evaluation complete")
    print(f"eval_run_dir={eval_run_dir}")
    print(f"export_model_dir={export.output_model_dir}")
    print(f"ckpt={export.checkpoint_name}")


if __name__ == "__main__":
    main()
