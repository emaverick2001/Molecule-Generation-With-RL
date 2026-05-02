from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.evaluation.metrics import (
    aggregate_topk_metrics,
    evaluate_generated_poses,
    save_pose_metrics_csv,
)
from src.utils.artifact_logger import read_json, save_json, save_text
from src.utils.config import load_experiment_config
from src.utils.schemas import ComplexInput, GeneratedPose


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


def run_evaluation(
    input_manifest_path: str | Path,
    generated_manifest_path: str | Path,
    pose_metrics_path: str | Path,
    metrics_path: str | Path,
    summary_path: str | Path,
    config: dict[str, Any],
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
    save_json(metrics, metrics_path)
    save_text(
        (
            "# Evaluation\n\n"
            f"- Input complexes: {len(input_records)}\n"
            f"- Generated poses: {len(generated_records)}\n"
            f"- Valid poses: {aggregate['num_valid_poses']}\n"
            f"- Invalid poses: {aggregate['num_invalid_poses']}\n"
            f"- Mean RMSD: {aggregate['mean_rmsd']}\n"
            f"- Success@1: {aggregate.get('success_at_1')}\n"
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
    )

    print(f"Evaluation complete: {run_dir}")


if __name__ == "__main__":
    main()
