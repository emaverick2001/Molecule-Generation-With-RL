from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.evaluation.metrics import (
    PoseMetricRecord,
    aggregate_topk_metrics,
    evaluate_generated_poses,
    save_pose_metrics_csv,
)
from src.evaluation.reranking_comparison import (
    compare_reranking_strategies,
    save_reranking_comparison_csv,
    summarize_reranking_comparison,
)
from src.utils.artifact_logger import read_json, save_csv, save_json, save_text
from src.utils.config import load_experiment_config
from src.utils.schemas import ComplexInput, GeneratedPose, RewardRecord


def _load_input_records(path: str | Path) -> list[ComplexInput]:
    return [ComplexInput.from_dict(item) for item in read_json(path)]


def _load_generated_records(path: str | Path) -> list[GeneratedPose]:
    return [GeneratedPose.from_dict(item) for item in read_json(path)]


def _get_evaluation_config(config: dict[str, Any]) -> dict[str, Any]:
    evaluation_config = config.get("evaluation", {})
    return {
        "rmsd_threshold": evaluation_config.get(
            "rmsd_threshold",
            evaluation_config.get("rmsd_success_threshold", 2.0),
        ),
        "top_k": evaluation_config.get(
            "top_k",
            evaluation_config.get("top_k_values", [1, 5, 10]),
        ),
        "remove_hs": evaluation_config.get("remove_hs", True),
    }


def build_negative_rmsd_reward_records(
    metric_records: list[PoseMetricRecord],
) -> list[RewardRecord]:
    return [
        RewardRecord(
            complex_id=record.complex_id,
            sample_id=record.sample_id,
            reward=-float(record.rmsd),
            reward_type="negative_rmsd",
            valid=True,
        )
        for record in metric_records
        if record.valid and record.rmsd is not None
    ]


def run_evaluation(
    input_manifest_path: str | Path,
    generated_manifest_path: str | Path,
    pose_metrics_path: str | Path,
    metrics_path: str | Path,
    summary_path: str | Path,
    config: dict[str, Any],
    rewards_path: str | Path | None = None,
    reranking_comparison_csv_path: str | Path | None = None,
    reranking_comparison_json_path: str | Path | None = None,
) -> dict[str, Any]:
    evaluation_config = _get_evaluation_config(config)
    input_records = _load_input_records(input_manifest_path)
    generated_records = _load_generated_records(generated_manifest_path)

    pose_metric_records = evaluate_generated_poses(
        input_records=input_records,
        generated_records=generated_records,
        remove_hs=evaluation_config["remove_hs"],
    )
    aggregate = aggregate_topk_metrics(
        pose_metric_records,
        top_k=evaluation_config["top_k"],
        success_threshold=evaluation_config["rmsd_threshold"],
        attempted_complex_ids=[record.complex_id for record in input_records],
    )
    metrics = {
        "stage": "evaluation",
        "input_manifest": str(input_manifest_path),
        "generated_manifest": str(generated_manifest_path),
        "rmsd_threshold": evaluation_config["rmsd_threshold"],
        "top_k": evaluation_config["top_k"],
        "remove_hs": evaluation_config["remove_hs"],
        "aggregate": aggregate,
    }

    save_pose_metrics_csv(pose_metric_records, pose_metrics_path)
    reward_records = build_negative_rmsd_reward_records(pose_metric_records)

    if rewards_path is not None:
        rewards_path = Path(rewards_path)
        if reward_records:
            save_csv(
                [record.to_dict() for record in reward_records],
                rewards_path,
            )
        elif rewards_path.exists():
            rewards_path.unlink()

    metrics["reward_records"] = {
        "path": str(rewards_path) if rewards_path is not None else None,
        "reward_type": "negative_rmsd",
        "num_records": len(reward_records),
    }
    reranking_comparison_records = compare_reranking_strategies(
        metric_records=pose_metric_records,
        generated_records=generated_records,
        success_threshold=evaluation_config["rmsd_threshold"],
    )
    reranking_comparison_summary = summarize_reranking_comparison(
        reranking_comparison_records
    )
    metrics["reranking_comparison"] = reranking_comparison_summary

    if reranking_comparison_csv_path is not None:
        save_reranking_comparison_csv(
            reranking_comparison_records,
            reranking_comparison_csv_path,
        )
    if reranking_comparison_json_path is not None:
        save_json(
            {
                "stage": "reranking_comparison",
                "description": (
                    "Comparison of original DiffDock rank-1, confidence-selected "
                    "top-1, and oracle best-of-n. Confidence is a diagnostic "
                    "baseline, not assumed to be the final reranker."
                ),
                "aggregate": reranking_comparison_summary,
                "per_complex": [
                    record.to_dict() for record in reranking_comparison_records
                ],
            },
            reranking_comparison_json_path,
        )
    save_json(metrics, metrics_path)
    save_text(
        (
            "# Evaluation\n\n"
            f"- Input complexes: {len(input_records)}\n"
            f"- Generated complexes: {aggregate['num_generated_complexes']}\n"
            f"- Generation coverage: {aggregate['generation_coverage']}\n"
            f"- Generated poses: {len(generated_records)}\n"
            f"- Valid poses: {aggregate['num_valid_poses']}\n"
            f"- Invalid poses: {aggregate['num_invalid_poses']}\n"
            f"- Mean RMSD: {aggregate['mean_rmsd']}\n"
            f"- Success@1: {aggregate.get('success_at_1')}\n"
            f"- Strict Success@1: {aggregate.get('strict_success_at_1')}\n"
        ),
        summary_path,
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated docking poses against ground-truth SDFs."
    )
    parser.add_argument(
        "--global-config",
        default="configs/global.yaml",
        help="Path to global config file.",
    )
    parser.add_argument(
        "--config",
        default="configs/diffdock/evaluation.yaml",
        help="Path to evaluation config file.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing input and generated-sample manifests.",
    )
    parser.add_argument(
        "--input-manifest",
        default=None,
        help="Optional override for input_manifest.json.",
    )
    parser.add_argument(
        "--generated-manifest",
        default=None,
        help="Optional override for generated_samples_manifest.json.",
    )

    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config = load_experiment_config(
        global_path=args.global_config,
        experiment_path=args.config,
    )

    input_manifest_path = (
        Path(args.input_manifest)
        if args.input_manifest is not None
        else run_dir / "input_manifest.json"
    )
    generated_manifest_path = (
        Path(args.generated_manifest)
        if args.generated_manifest is not None
        else run_dir / "generated_samples_manifest.json"
    )

    run_evaluation(
        input_manifest_path=input_manifest_path,
        generated_manifest_path=generated_manifest_path,
        pose_metrics_path=run_dir / "pose_metrics.csv",
        metrics_path=run_dir / "metrics.json",
        summary_path=run_dir / "evaluation_summary.md",
        config=config,
        rewards_path=run_dir / "rewards.csv",
        reranking_comparison_csv_path=run_dir / "reranking_comparison.csv",
        reranking_comparison_json_path=run_dir / "reranking_comparison.json",
    )

    print(f"Evaluation complete: {run_dir}")


if __name__ == "__main__":
    main()
