#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from shutil import copyfile
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rl.data import load_offline_rl_examples, write_rollout_manifest
from src.rl.diffdock_batch_builder import DiffDockGeneratedPoseBatchBuilder
from src.rl.diffdock_loss import import_diffdock_loss_function
from src.rl.diffdock_model import (
    load_diffdock_score_model,
    load_score_model_args,
    score_model_uses_lm_embeddings,
)
from src.rl.native_grpo import (
    run_native_diffdock_grpo_step,
    run_native_diffdock_grpo_step_batched,
)
from src.rl.rewards import build_reward_rows
from src.rl.rollouts import build_rollout_records, compute_group_advantages
from src.rl.config import RewardConfig, RolloutConfig
from src.rl.utils import summarize_rewards, write_jsonl
from src.utils.artifact_logger import save_csv, save_json, save_text
from src.utils.paths import create_run_dir, make_run_id


def _select_device(requested_device: str):
    import torch

    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    return device


def _copy_if_present(source: Path, destination: Path) -> None:
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, destination)


def _load_smoke_rollout_records(
    *,
    source_run_dir: Path,
    limit: int,
    samples_per_complex: int,
    min_valid_samples: int,
) -> tuple[list, list]:
    examples = load_offline_rl_examples(
        source_run_dir / "input_manifest.json",
        source_run_dir / "generated_samples_manifest.json",
        source_run_id=source_run_dir.name,
        source_run_dir=source_run_dir,
    )[:limit]
    if not examples:
        raise ValueError(f"No generated examples found in run: {source_run_dir}")

    rollout_records = build_rollout_records(
        examples,
        reward_cfg=RewardConfig(weights={"rmsd": 1.0, "confidence": 0.0}),
    )
    rollout_records = compute_group_advantages(
        rollout_records,
        rollout_cfg=RolloutConfig(
            samples_per_complex=samples_per_complex,
            advantage_normalization="zscore",
            min_valid_samples_per_complex=min_valid_samples,
            invalid_group_action="zero",
        ),
    )
    valid_records = [
        record
        for record in rollout_records
        if record.reward.valid and record.advantage is not None
    ]
    if len(valid_records) < min_valid_samples:
        raise ValueError(
            "Not enough valid rollout records for native GRPO smoke: "
            f"{len(valid_records)} < {min_valid_samples}"
        )

    valid_examples = [record.example for record in valid_records]
    return valid_examples, valid_records


def _score_rows(records, result) -> list[dict]:
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "complex_id": record.example.complex_id,
                "sample_id": record.example.sample_id,
                "rank": record.example.sample_rank,
                "reward": record.reward.total,
                "advantage": record.advantage,
                "old_surrogate_score": result.old_scores[index],
                "surrogate_score_before": result.scores_before[index],
                "surrogate_score_after": result.scores_after[index],
                "surrogate_delta": (
                    result.scores_after[index] - result.scores_before[index]
                ),
                "ratio_before": result.ratios_before[index],
                "clipped_ratio_before": result.clipped_ratios_before[index],
                "objective_term_before": result.objective_terms_before[index],
                "tr_loss_before": result.tr_loss_before[index],
                "rot_loss_before": result.rot_loss_before[index],
                "tor_loss_before": result.tor_loss_before[index],
                "diffdock_loss_before": result.total_loss_before[index],
                "tr_loss_after": result.tr_loss_after[index],
                "rot_loss_after": result.rot_loss_after[index],
                "tor_loss_after": result.tor_loss_after[index],
                "diffdock_loss_after": result.total_loss_after[index],
            }
        )
    return rows


def _chunked(items, chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def _prepare_batch_for_model(batch, *, no_parallel: bool, device):
    if not no_parallel or not isinstance(batch, list):
        return batch

    from torch_geometric.data import Batch

    return Batch.from_data_list(batch).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a one-step native DiffDock GRPO smoke update from an existing "
            "one-complex, four-pose rollout."
        ),
    )
    parser.add_argument("source_run_dir", help="Completed one-complex rollout run.")
    parser.add_argument("--repo-root", default="external/DiffDock")
    parser.add_argument("--model-dir", default="external/DiffDock/workdir/v1.1/score_model")
    parser.add_argument("--ckpt", default="best_ema_inference_epoch_model.pt")
    parser.add_argument("--model-args", default=None)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument(
        "--samples-per-complex",
        type=int,
        default=None,
        help=(
            "Grouped rollout size for advantage normalization. Defaults to "
            "--limit for one-complex smoke runs."
        ),
    )
    parser.add_argument("--min-valid-samples", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--max-score-delta", type=float, default=20.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--score-batch-size",
        type=int,
        default=0,
        help=(
            "If positive, build/scoring DiffDock graphs in chunks and accumulate "
            "gradients before one optimizer step. Use this to avoid GPU OOM on "
            "multi-complex rollouts."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--work-dir", default="artifacts/tmp/diffdock_rl_batches")
    parser.add_argument(
        "--lm-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable DiffDock ESM language-model embeddings during graph construction. "
            "Default is true because the v1.1 score model commonly expects them."
        ),
    )
    parser.add_argument("--old-score-model", action="store_true")
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable DiffDock/PyG DataParallel.",
    )
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    samples_per_complex = args.samples_per_complex or args.limit
    if samples_per_complex <= 0:
        raise ValueError("--samples-per-complex must be positive")
    if args.score_batch_size < 0:
        raise ValueError("--score-batch-size must be non-negative")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")

    import torch

    torch.manual_seed(args.seed)

    source_run_dir = Path(args.source_run_dir)
    source_run_tag = args.run_tag or f"native_grpo_{source_run_dir.name}"
    run_id = make_run_id(
        model="diffdock",
        experiment=f"posttraining_native_grpo_smoke_{source_run_tag}",
        seed=args.seed,
    )
    run_dir = create_run_dir(args.run_root, run_id, exist_ok=args.exist_ok)
    rollout_dir = run_dir / "rollouts" / "native_grpo_step_000"
    checkpoint_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    for directory in [rollout_dir, checkpoint_dir, logs_dir, run_dir / "eval"]:
        directory.mkdir(parents=True, exist_ok=True)

    examples, rollout_records = _load_smoke_rollout_records(
        source_run_dir=source_run_dir,
        limit=args.limit,
        samples_per_complex=samples_per_complex,
        min_valid_samples=args.min_valid_samples,
    )
    device = _select_device(args.device)
    score_model_args = load_score_model_args(
        model_dir=args.model_dir,
        model_args=args.model_args,
    )
    model_needs_lm_embeddings = score_model_uses_lm_embeddings(score_model_args)
    if model_needs_lm_embeddings and not args.lm_embeddings:
        raise ValueError(
            "The score model appears to require ESM/LM embeddings, but "
            "--no-lm-embeddings was passed."
        )

    no_parallel = args.no_parallel or device.type != "cuda"
    bundle = load_diffdock_score_model(
        repo_root=args.repo_root,
        model_dir=args.model_dir,
        ckpt=args.ckpt,
        device=device,
        score_model_args=score_model_args,
        old_score_model=args.old_score_model,
        no_parallel=no_parallel,
        strict=args.strict,
    )
    batch_builder = DiffDockGeneratedPoseBatchBuilder.from_score_model_args(
        repo_root=args.repo_root,
        score_model_args=bundle.score_model_args,
        device=device,
        work_dir=args.work_dir,
        lm_embeddings=args.lm_embeddings,
    )
    loss_function = import_diffdock_loss_function(args.repo_root)
    optimizer = torch.optim.Adam(
        [parameter for parameter in bundle.model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )

    if args.score_batch_size and args.score_batch_size < len(examples):
        example_batches = list(_chunked(examples, args.score_batch_size))
        record_batches = list(_chunked(rollout_records, args.score_batch_size))

        def build_model_batch(batch_index: int):
            return _prepare_batch_for_model(
                batch_builder(example_batches[batch_index]),
                no_parallel=no_parallel,
                device=device,
            )

        result, model_state_dict = run_native_diffdock_grpo_step_batched(
            model=bundle.model,
            batches=build_model_batch,
            advantages_by_batch=[
                [float(record.advantage) for record in batch_records]
                for batch_records in record_batches
            ],
            loss_function=loss_function,
            t_to_sigma=bundle.t_to_sigma,
            device=device,
            optimizer=optimizer,
            torch_module=torch,
            no_torsion=getattr(bundle.score_model_args, "no_torsion", False),
            clip_epsilon=args.clip_epsilon,
            max_score_delta=args.max_score_delta,
            max_grad_norm=args.max_grad_norm,
        )
    else:
        batch = _prepare_batch_for_model(
            batch_builder(examples),
            no_parallel=no_parallel,
            device=device,
        )
        result, model_state_dict = run_native_diffdock_grpo_step(
            model=bundle.model,
            batch=batch,
            advantages=[float(record.advantage) for record in rollout_records],
            loss_function=loss_function,
            t_to_sigma=bundle.t_to_sigma,
            device=device,
            optimizer=optimizer,
            torch_module=torch,
            no_torsion=getattr(bundle.score_model_args, "no_torsion", False),
            clip_epsilon=args.clip_epsilon,
            max_score_delta=args.max_score_delta,
            max_grad_norm=args.max_grad_norm,
        )

    checkpoint_path = checkpoint_dir / "native_grpo_step_000.pt"
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": {
                "source_run_dir": str(source_run_dir),
                "repo_root": args.repo_root,
                "model_dir": args.model_dir,
                "ckpt": args.ckpt,
                "seed": args.seed,
                "samples_per_complex": samples_per_complex,
                "score_batch_size": args.score_batch_size,
                "learning_rate": args.learning_rate,
                "clip_epsilon": args.clip_epsilon,
                "max_score_delta": args.max_score_delta,
                "max_grad_norm": args.max_grad_norm,
                "lm_embeddings": args.lm_embeddings,
                "model_needs_lm_embeddings": model_needs_lm_embeddings,
            },
        },
        checkpoint_path,
    )

    _copy_if_present(source_run_dir / "input_manifest.json", run_dir / "input_train_manifest.json")
    _copy_if_present(
        source_run_dir / "generated_samples_manifest.json",
        rollout_dir / "generated_samples_manifest.json",
    )
    write_rollout_manifest(rollout_records, rollout_dir / "rollout.jsonl")
    save_csv(
        build_reward_rows([(record.example, record.reward) for record in rollout_records]),
        rollout_dir / "rewards.csv",
    )
    save_csv(_score_rows(rollout_records, result), rollout_dir / "native_surrogate_scores.csv")

    reward_summary = summarize_rewards(rollout_records)
    metrics = {
        "algorithm": "grpo_surrogate",
        "surrogate_backend": "diffdock_native_loss",
        "source_run_dir": str(source_run_dir),
        "num_examples": len(examples),
        "num_rollout_records": len(rollout_records),
        "samples_per_complex": samples_per_complex,
        "score_batch_size": args.score_batch_size,
        "reward": reward_summary,
        "training": {
            "loss": result.loss,
            "grad_norm": result.grad_norm,
            "learning_rate": args.learning_rate,
            "clip_epsilon": args.clip_epsilon,
            "max_score_delta": args.max_score_delta,
            "checkpoint_path": str(checkpoint_path),
            "mean_score_before": sum(result.scores_before) / len(result.scores_before),
            "mean_score_after": sum(result.scores_after) / len(result.scores_after),
            "mean_diffdock_loss_before": (
                sum(result.total_loss_before) / len(result.total_loss_before)
            ),
            "mean_diffdock_loss_after": (
                sum(result.total_loss_after) / len(result.total_loss_after)
            ),
        },
    }
    save_json(metrics, rollout_dir / "native_grpo_summary.json")
    save_json(metrics, run_dir / "posttraining_summary.json")
    write_jsonl([metrics], logs_dir / "train_metrics.jsonl")

    summary = "\n".join(
        [
            "# Native DiffDock GRPO Smoke",
            "",
            f"- Source run: {source_run_dir}",
            f"- Examples: {len(examples)}",
            f"- Loss: {result.loss}",
            f"- Grad norm: {result.grad_norm}",
            f"- Mean score before: {metrics['training']['mean_score_before']}",
            f"- Mean score after: {metrics['training']['mean_score_after']}",
            f"- Checkpoint: {checkpoint_path}",
            "",
        ]
    )
    save_text(summary, run_dir / "summary.md")

    print("Native DiffDock GRPO smoke complete")
    print(f"run_dir={run_dir}")
    print(f"checkpoint={checkpoint_path}")
    print(f"loss={result.loss:.6f}")
    print(f"grad_norm={result.grad_norm}")
    print(
        "mean_score_before="
        f"{metrics['training']['mean_score_before']:.6f} "
        "mean_score_after="
        f"{metrics['training']['mean_score_after']:.6f}"
    )


if __name__ == "__main__":
    main()
