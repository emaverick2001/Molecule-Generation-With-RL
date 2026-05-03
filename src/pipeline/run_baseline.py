# src/pipeline/run_baseline.py

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.data.loaders import load_complex_manifest
from src.data.structure_checks import filter_complexes_by_preflight
from src.data.validation import EXPECTED_EXTENSIONS, validate_manifest_records
from src.generation.dry_run_generator import generate_dry_run_poses
from src.generation.generate_diffdock import generate_diffdock_poses
from src.utils.artifact_logger import save_csv, save_json, save_records_json, save_text
from src.utils.config import load_experiment_config
from src.utils.error_logger import append_error
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


def generate_baseline_poses(
    config: dict,
    complexes: list[ComplexInput],
    run_dir: str | Path,
) -> list[GeneratedPose]:
    generation_config = config["generation"]
    backend = generation_config.get("backend", "dry_run")
    num_samples = generation_config["num_samples"]
    run_dir = Path(run_dir)

    if backend == "dry_run":
        return generate_dry_run_poses(
            records=complexes,
            output_dir=run_dir / "generated_samples",
            num_samples=num_samples,
        )

    if backend == "diffdock":
        diffdock_config = config["diffdock"]
        error_handling_config = config.get("error_handling", {})
        preflight_config = diffdock_config.get("preflight_filter", {})
        generation_records = complexes

        if preflight_config.get("enabled", False):
            generation_records, preflight_results = filter_complexes_by_preflight(
                complexes,
                remove_hs=preflight_config.get("remove_hs", True),
                min_protein_atoms=preflight_config.get("min_protein_atoms", 1),
                min_protein_residues=preflight_config.get("min_protein_residues", 1),
                min_protein_chains=preflight_config.get("min_protein_chains", 1),
                max_ligand_protein_centroid_distance=preflight_config.get(
                    "max_ligand_protein_centroid_distance",
                    80.0,
                ),
                max_input_reference_centroid_distance=preflight_config.get(
                    "max_input_reference_centroid_distance",
                    2.0,
                ),
                fail_on_unsupported_residues=preflight_config.get(
                    "fail_on_unsupported_residues",
                    False,
                ),
            )
            save_json(
                [result.to_dict() for result in preflight_results],
                run_dir / "preflight_report.json",
            )

            for result in preflight_results:
                if not result.valid:
                    append_error(
                        run_dir / "errors.log",
                        "Skipping complex after DiffDock preflight validation.",
                        context={
                            "stage": "diffdock_preflight",
                            "complex_id": result.complex_id,
                            "reasons": "; ".join(result.reasons),
                        },
                    )

            if not generation_records:
                raise ValueError("DiffDock preflight filtered out every complex.")

        return generate_diffdock_poses(
            records=generation_records,
            output_dir=run_dir / "generated_samples",
            num_samples=num_samples,
            command_template=diffdock_config["command_template"],
            raw_output_dir=run_dir / "raw_diffdock_outputs",
            repo_dir=diffdock_config["repo_dir"],
            config_path=diffdock_config["config_path"],
            log_dir=run_dir / "logs",
            timeout_seconds=diffdock_config.get("timeout_seconds"),
            skip_failed_complexes=error_handling_config.get(
                "skip_invalid_samples",
                False,
            ),
            errors_log_path=run_dir / "errors.log",
            max_failed_complexes=error_handling_config.get(
                "max_failed_samples_before_stop"
            ),
        )

    raise ValueError(f"Unsupported generation backend: {backend}")


def run_baseline_dry_run(
    config: dict,
    complexes: list[ComplexInput],
    run_dir: str | Path,
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
    eval_cfg = config.get("evaluation", {})
    rmsd_threshold = eval_cfg.get("rmsd_threshold", 2.0)
    generated_samples = generate_baseline_poses(
        config=config,
        complexes=complexes,
        run_dir=run_dir,
    )
    backend = config["generation"].get("backend", "dry_run")
    generated_complex_ids = {pose.complex_id for pose in generated_samples}
    successful_complexes = [
        complex_input
        for complex_input in complexes
        if complex_input.complex_id in generated_complex_ids
    ]
    failed_complex_ids = [
        complex_input.complex_id
        for complex_input in complexes
        if complex_input.complex_id not in generated_complex_ids
    ]
    reward_records = []
    metric_records = []

    if backend == "diffdock":
        metrics = {
            "stage": "generation",
            "experiment_name": config["experiment"]["name"],
            "mode": config["experiment"]["mode"],
            "generation_backend": backend,
            "dataset_manifest": config["dataset"]["manifest_path"],
            "num_requested_complexes": len(complexes),
            "num_complexes": len(successful_complexes),
            "num_successful_complexes": len(successful_complexes),
            "num_failed_complexes": len(failed_complex_ids),
            "failed_complex_ids": failed_complex_ids,
            "num_generated_samples": len(generated_samples),
            "aggregate": {},
        }
        return generated_samples, reward_records, metrics

    for complex_index, complex_input in enumerate(successful_complexes):
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
        "generation_backend": config["generation"].get("backend", "dry_run"),
        "dataset_manifest": config["dataset"]["manifest_path"],
        "num_requested_complexes": len(complexes),
        "num_complexes": len(metric_records),
        "num_successful_complexes": len(successful_complexes),
        "num_failed_complexes": len(failed_complex_ids),
        "failed_complex_ids": failed_complex_ids,
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override experiment.seed from the config. Useful for scripted "
            "multi-seed runs without creating one config file per seed."
        ),
    )

    args = parser.parse_args()

    config = load_experiment_config(
        global_path=args.global_config,
        experiment_path=args.config,
    )
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed

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
        run_dir=run_dir,
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

    if reward_records:
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
            f"- Generation backend: {metrics['generation_backend']}\n"
            f"- Generated samples: {metrics['num_generated_samples']}\n"
        ),
        artifact_paths["summary"],
    )

    print(f"Baseline dry run complete: {run_dir}")


if __name__ == "__main__":
    main()
