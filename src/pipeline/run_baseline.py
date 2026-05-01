# src/pipeline/run_baseline.py

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.data.loaders import load_complex_manifest
from src.data.validation import EXPECTED_EXTENSIONS, validate_manifest_records
from src.generation.dry_run_generator import generate_dry_run_poses
from src.utils.artifact_logger import save_csv, save_json, save_records_json, save_text
from src.utils.config import load_experiment_config
from src.utils.run_logger import get_artifact_paths, initialize_run
from src.utils.schemas import ComplexInput, GeneratedPose, MetricRecord, RewardRecord
from src.utils.seeds import set_seed


def build_dataset_summary(records: list[ComplexInput], config: dict) -> dict:
    split_counts = Counter(record.split for record in records)

    return {
        "dataset_name": config["dataset"]["name"],
        "stage": config["dataset"]["split"],
        "num_complexes": len(records),
        "splits": dict(sorted(split_counts.items())),
        "file_types": {
            "protein": EXPECTED_EXTENSIONS["protein_path"],
            "ligand": EXPECTED_EXTENSIONS["ligand_path"],
            "ground_truth_pose": EXPECTED_EXTENSIONS["ground_truth_pose_path"],
        },
    }


def run_baseline_dry_run(
    config: dict,
    complexes: list[ComplexInput],
    generated_output_dir: str | Path,
) -> tuple[list[GeneratedPose], list[RewardRecord], dict]:
    """
    MVP placeholder for DiffDock baseline.

    This simulates:
    - loading complexes from the configured manifest
    - multiple generated samples per loaded complex
    - one reward per generated sample
    - aggregate evaluation metrics
    """
    if not complexes:
        raise ValueError("Cannot run baseline dry run with an empty manifest.")

    num_samples = config["generation"]["num_samples"]
    rmsd_threshold = config["evaluation"]["rmsd_threshold"]
    generated_samples = generate_dry_run_poses(
        records=complexes,
        output_dir=generated_output_dir,
        num_samples=num_samples,
    )
    reward_records = []
    metric_records = []

    for complex_index, complex_input in enumerate(complexes):
        top1_rmsd = round(1.5 + (0.1 * complex_index), 3)

        metric_records.append(
            MetricRecord(
                complex_id=complex_input.complex_id,
                top1_rmsd=top1_rmsd,
                success_at_1=top1_rmsd <= rmsd_threshold,
                success_at_5=True,
                success_at_10=True,
            )
        )

        complex_generated_samples = [
            pose
            for pose in generated_samples
            if pose.complex_id == complex_input.complex_id
        ]

        for pose in complex_generated_samples:
            simulated_rmsd = round(top1_rmsd + (0.15 * pose.sample_id), 3)
            reward_records.append(
                RewardRecord(
                    complex_id=pose.complex_id,
                    sample_id=pose.sample_id,
                    reward=-simulated_rmsd,
                    reward_type="negative_rmsd",
                    valid=True,
                )
            )

    mean_top1_rmsd = round(
        sum(record.top1_rmsd for record in metric_records) / len(metric_records),
        3,
    )
    success_at_1 = round(
        sum(record.success_at_1 for record in metric_records) / len(metric_records),
        3,
    )

    metrics = {
        "experiment_name": config["experiment"]["name"],
        "mode": config["experiment"]["mode"],
        "dataset_manifest": config["dataset"]["manifest_path"],
        "num_complexes": len(metric_records),
        "num_generated_samples": len(generated_samples),
        "per_complex": [record.to_dict() for record in metric_records],
        "aggregate": {
            "mean_top1_rmsd": mean_top1_rmsd,
            "success_at_1": success_at_1,
            "success_at_5": 1.0,
            "success_at_10": 1.0,
        },
    }

    return generated_samples, reward_records, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DiffDock baseline MVP dry run.")
    parser.add_argument(
        "--global-config",
        default="configs/global.yaml",
        help="Path to global config file.",
    )
    parser.add_argument(
        "--config",
        default="configs/diffdock/baseline.yaml",
        help="Path to baseline experiment config file.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow writing to an existing run directory.",
    )

    args = parser.parse_args()

    config = load_experiment_config(
        global_path=args.global_config,
        experiment_path=args.config,
    )

    set_seed(config["experiment"]["seed"])

    run_dir = initialize_run(
        config=config,
        config_path=Path(args.config),
        exist_ok=args.exist_ok,
    )

    artifact_paths = get_artifact_paths(run_dir)
    complexes = load_complex_manifest(config["dataset"]["manifest_path"], validate=True)
    validation_report = validate_manifest_records(complexes)
    dataset_summary = build_dataset_summary(complexes, config)

    generated_samples, reward_records, metrics = run_baseline_dry_run(
        config=config,
        complexes=complexes,
        generated_output_dir=run_dir / "generated_samples",
    )

    save_records_json(
        complexes,
        run_dir / "input_manifest.json",
    )

    save_json(
        validation_report,
        run_dir / "validation_report.json",
    )

    save_json(
        dataset_summary,
        run_dir / "dataset_summary.json",
    )

    save_records_json(
        generated_samples,
        artifact_paths["generated_samples"],
    )

    save_csv(
        [record.to_dict() for record in reward_records],
        artifact_paths["rewards"],
    )

    save_json(
        metrics,
        artifact_paths["metrics"],
    )

    save_text(
        (
            f"# Baseline MVP Dry Run\n\n"
            f"- Dataset: {dataset_summary['dataset_name']}\n"
            f"- Split: {dataset_summary['stage']}\n"
            f"- Complexes: {dataset_summary['num_complexes']}\n"
            f"- Generated samples: {metrics['num_generated_samples']}\n"
        ),
        artifact_paths["summary"],
    )

    print(f"Baseline dry run complete: {run_dir}")


if __name__ == "__main__":
    main()
