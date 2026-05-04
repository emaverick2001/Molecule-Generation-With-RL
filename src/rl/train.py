from __future__ import annotations

from pathlib import Path
from shutil import copyfile
from typing import Any

from src.rl.config import RLConfig
from src.rl.data import (
    load_offline_rl_examples,
    write_rollout_manifest,
)
from src.rl.grpo import (
    LinearSurrogateState,
    build_surrogate_score_rows,
    save_linear_surrogate_checkpoint,
    train_linear_grpo_step,
)
from src.rl.rewards import build_reward_rows
from src.rl.rollouts import (
    build_rollout_records,
    compute_group_advantages,
    summarize_rollout_groups,
)
from src.rl.types import TrainSummary
from src.rl.utils import summarize_rewards, write_jsonl
from src.utils.artifact_logger import save_csv, save_json, save_text


def _copy_if_present(source: str | None, destination: Path) -> None:
    if source is None:
        return
    source_path = Path(source)
    if source_path.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source_path, destination)


def _build_offline_rollout_records(cfg: RLConfig) -> tuple[list, list]:
    if cfg.data.input_manifest is None or cfg.data.generated_manifest is None:
        raise ValueError("Resolved input and generated manifests are required")

    source_run_id = (
        Path(cfg.data.source_run_dir).name if cfg.data.source_run_dir else None
    )
    examples = load_offline_rl_examples(
        cfg.data.input_manifest,
        cfg.data.generated_manifest,
        source_run_id=source_run_id,
        source_run_dir=cfg.data.source_run_dir,
    )
    rollout_records = build_rollout_records(examples, reward_cfg=cfg.reward)
    rollout_records = compute_group_advantages(
        rollout_records,
        rollout_cfg=cfg.rollout,
    )

    return examples, rollout_records


def _write_rollout_artifacts(
    cfg: RLConfig,
    run_dir: Path,
    rollout_dir: Path,
    rollout_records: list,
) -> dict[str, Any]:
    write_rollout_manifest(rollout_records, rollout_dir / "rollout.jsonl")
    save_csv(
        build_reward_rows(
            [(record.example, record.reward) for record in rollout_records]
        ),
        rollout_dir / "rewards.csv",
    )

    _copy_if_present(
        cfg.data.generated_manifest,
        rollout_dir / "generated_samples_manifest.json",
    )
    _copy_if_present(cfg.data.input_manifest, run_dir / "input_train_manifest.json")
    if cfg.data.val_manifest:
        _copy_if_present(cfg.data.val_manifest, run_dir / "input_val_manifest.json")

    reward_summary = summarize_rewards(rollout_records)
    group_summary = summarize_rollout_groups(rollout_records)

    save_json(reward_summary, rollout_dir / "reward_summary.json")
    save_json(group_summary, rollout_dir / "group_summary.json")

    return {
        "reward": reward_summary,
        "groups": group_summary,
    }


def run_offline_reward_debug(cfg: RLConfig, run_dir: str | Path) -> TrainSummary:
    run_dir = Path(run_dir)
    rollout_dir = run_dir / "rollouts" / "offline"
    logs_dir = run_dir / "logs"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    examples, rollout_records = _build_offline_rollout_records(cfg)
    rollout_metrics = _write_rollout_artifacts(
        cfg,
        run_dir,
        rollout_dir,
        rollout_records,
    )
    reward_summary = rollout_metrics["reward"]
    group_summary = rollout_metrics["groups"]
    metrics: dict[str, Any] = {
        "algorithm": cfg.algorithm.name,
        "num_examples": len(examples),
        "reward": reward_summary,
        "groups": group_summary,
    }

    write_jsonl([metrics], logs_dir / "train_metrics.jsonl")

    summary = "\n".join(
        [
            "# Offline Reward Debug",
            "",
            f"- Examples: {len(examples)}",
            f"- Rollout records: {len(rollout_records)}",
            f"- Groups: {group_summary['num_groups']}",
            f"- Valid rewards: {reward_summary['num_valid_rewards']}",
            f"- Reward mean: {reward_summary['reward_mean']}",
            f"- Reward std: {reward_summary['reward_std']}",
            "",
            "This run validates reward computation and group-relative advantages. "
            "It does not update DiffDock weights.",
            "",
        ]
    )
    save_text(summary, run_dir / "summary.md")

    return TrainSummary(
        run_dir=str(run_dir),
        algorithm=cfg.algorithm.name,
        num_examples=len(examples),
        num_rollout_records=len(rollout_records),
        metrics=metrics,
    )


def run_grpo_surrogate(cfg: RLConfig, run_dir: str | Path) -> TrainSummary:
    run_dir = Path(run_dir)
    rollout_dir = run_dir / "rollouts" / "grpo_step_000"
    logs_dir = run_dir / "logs"
    checkpoint_dir = run_dir / "checkpoints"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    examples, rollout_records = _build_offline_rollout_records(cfg)
    rollout_metrics = _write_rollout_artifacts(
        cfg,
        run_dir,
        rollout_dir,
        rollout_records,
    )

    state = LinearSurrogateState.initialized()
    train_metrics = []

    for epoch in range(cfg.algorithm.grpo_epochs):
        state, step_metrics = train_linear_grpo_step(
            rollout_records,
            state,
            learning_rate=cfg.algorithm.learning_rate,
        )
        step_metrics["epoch"] = epoch
        step_metrics["algorithm"] = cfg.algorithm.name
        step_metrics["surrogate_backend"] = cfg.algorithm.surrogate_backend
        train_metrics.append(step_metrics)

    save_csv(
        build_surrogate_score_rows(rollout_records, state),
        rollout_dir / "surrogate_scores.csv",
    )
    save_json(train_metrics, rollout_dir / "grpo_train_metrics.json")
    write_jsonl(train_metrics, logs_dir / "train_metrics.jsonl")
    checkpoint_path = checkpoint_dir / "grpo_debug_linear_step1.json"
    save_linear_surrogate_checkpoint(
        state,
        checkpoint_path,
        metadata={
            "algorithm": cfg.algorithm.name,
            "surrogate_backend": cfg.algorithm.surrogate_backend,
            "num_examples": len(examples),
            "num_rollout_records": len(rollout_records),
            "grpo_epochs": cfg.algorithm.grpo_epochs,
        },
    )

    reward_summary = rollout_metrics["reward"]
    group_summary = rollout_metrics["groups"]
    final_metrics: dict[str, Any] = {
        "algorithm": cfg.algorithm.name,
        "surrogate_backend": cfg.algorithm.surrogate_backend,
        "num_examples": len(examples),
        "num_rollout_records": len(rollout_records),
        "reward": reward_summary,
        "groups": group_summary,
        "training": {
            "num_epochs": cfg.algorithm.grpo_epochs,
            "initial_loss": train_metrics[0]["loss_before"],
            "final_loss": train_metrics[-1]["loss_after"],
            "checkpoint_path": str(checkpoint_path),
        },
    }

    save_json(final_metrics, rollout_dir / "grpo_summary.json")
    summary = "\n".join(
        [
            "# GRPO Surrogate Smoke",
            "",
            f"- Examples: {len(examples)}",
            f"- Rollout records: {len(rollout_records)}",
            f"- Groups: {group_summary['num_groups']}",
            f"- Valid rewards: {reward_summary['num_valid_rewards']}",
            f"- Initial loss: {final_metrics['training']['initial_loss']}",
            f"- Final loss: {final_metrics['training']['final_loss']}",
            f"- Checkpoint: {checkpoint_path}",
            "",
            "This run validates the GRPO objective shape over grouped rollouts. "
            "The current surrogate backend is a trainable debug-linear scorer; "
            "the DiffDock loss backend remains the next implementation hook.",
            "",
        ]
    )
    save_text(summary, run_dir / "summary.md")

    return TrainSummary(
        run_dir=str(run_dir),
        algorithm=cfg.algorithm.name,
        num_examples=len(examples),
        num_rollout_records=len(rollout_records),
        metrics=final_metrics,
    )


def run_training(cfg: RLConfig, run_dir: str | Path) -> TrainSummary:
    if cfg.algorithm.name == "offline_reward_debug":
        return run_offline_reward_debug(cfg, run_dir)
    if cfg.algorithm.name == "grpo_surrogate":
        return run_grpo_surrogate(cfg, run_dir)

    raise NotImplementedError(
        f"{cfg.algorithm.name} is scaffolded but not implemented yet."
    )
