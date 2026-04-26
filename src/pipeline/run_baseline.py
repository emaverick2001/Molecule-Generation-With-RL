# src/pipeline/run_baseline.py

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.artifact_logger import save_json, save_records_json, save_csv
from src.utils.config import load_experiment_config
from src.utils.run_logger import get_artifact_paths, initialize_run
from src.utils.schemas import GeneratedPose, MetricRecord, RewardRecord
from src.utils.seeds import set_seed


def run_baseline_dry_run(config: dict) -> tuple[list[GeneratedPose], list[RewardRecord], dict]:
    """
    MVP placeholder for DiffDock baseline.

    This simulates:
    - multiple complexes
    - multiple generated samples per complex
    - one reward per generated sample
    - aggregate evaluation metrics
    """

    generated_samples = [
        GeneratedPose(
            complex_id="1abc",
            sample_id=0,
            pose_path="artifacts/generated_samples/diffdock/1abc_sample_0.sdf",
            confidence_score=None,
        ),
        GeneratedPose(
            complex_id="1abc",
            sample_id=1,
            pose_path="artifacts/generated_samples/diffdock/1abc_sample_1.sdf",
            confidence_score=None,
        ),
        GeneratedPose(
            complex_id="2xyz",
            sample_id=0,
            pose_path="artifacts/generated_samples/diffdock/2xyz_sample_0.sdf",
            confidence_score=None,
        ),
        GeneratedPose(
            complex_id="2xyz",
            sample_id=1,
            pose_path="artifacts/generated_samples/diffdock/2xyz_sample_1.sdf",
            confidence_score=None,
        ),
    ]

    reward_records = [
        RewardRecord(
            complex_id="1abc",
            sample_id=0,
            reward=-1.7,
            reward_type="negative_rmsd",
            valid=True,
        ),
        RewardRecord(
            complex_id="1abc",
            sample_id=1,
            reward=-2.4,
            reward_type="negative_rmsd",
            valid=True,
        ),
        RewardRecord(
            complex_id="2xyz",
            sample_id=0,
            reward=-3.1,
            reward_type="negative_rmsd",
            valid=True,
        ),
        RewardRecord(
            complex_id="2xyz",
            sample_id=1,
            reward=-1.9,
            reward_type="negative_rmsd",
            valid=True,
        ),
    ]

    metric_records = [
        MetricRecord(
            complex_id="1abc",
            top1_rmsd=1.7,
            success_at_1=True,
            success_at_5=True,
            success_at_10=True,
        ),
        MetricRecord(
            complex_id="2xyz",
            top1_rmsd=1.9,
            success_at_1=True,
            success_at_5=True,
            success_at_10=True,
        ),
    ]

    metrics = {
        "experiment_name": config["experiment"]["name"],
        "mode": config["experiment"]["mode"],
        "num_complexes": len(metric_records),
        "num_generated_samples": len(generated_samples),
        "per_complex": [record.to_dict() for record in metric_records],
        "aggregate": {
            "mean_top1_rmsd": 1.8,
            "success_at_1": 1.0,
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

    generated_samples, reward_records, metrics = run_baseline_dry_run(config)

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


    print(f"Baseline dry run complete: {run_dir}")


if __name__ == "__main__":
    main()