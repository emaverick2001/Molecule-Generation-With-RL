"""
Thin DiffDock inference wrapper.

This module only handles:
- ComplexInput records
- DiffDock inference command execution
- raw DiffDock output collection
- standardized GeneratedPose records

Reward scoring, RMSD, evaluation, and reranking belong in later pipeline stages.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from src.generation.contract import validate_generated_pose_records
from src.utils.schemas import ComplexInput, GeneratedPose


Runner = Callable[..., Any]


def _format_command(
    command_template: Sequence[str],
    record: ComplexInput,
    raw_output_dir: Path,
    num_samples: int,
) -> list[str]:
    values = {
        "complex_id": record.complex_id,
        "protein_path": record.protein_path,
        "ligand_path": record.ligand_path,
        "ground_truth_pose_path": record.ground_truth_pose_path,
        "output_dir": str(raw_output_dir),
        "num_samples": str(num_samples),
    }

    return [part.format(**values) for part in command_template]


def _standardize_diffdock_outputs(
    record: ComplexInput,
    raw_output_dir: Path,
    output_dir: Path,
    num_samples: int,
) -> list[GeneratedPose]:
    raw_pose_paths = sorted(raw_output_dir.glob("*.sdf"))

    if len(raw_pose_paths) < num_samples:
        raise FileNotFoundError(
            f"DiffDock produced {len(raw_pose_paths)} SDF files for "
            f"{record.complex_id}, expected at least {num_samples}: {raw_output_dir}"
        )

    generated = []

    for sample_id, raw_pose_path in enumerate(raw_pose_paths[:num_samples]):
        standardized_path = output_dir / f"{record.complex_id}_sample_{sample_id}.sdf"
        shutil.copyfile(raw_pose_path, standardized_path)

        generated.append(
            GeneratedPose(
                complex_id=record.complex_id,
                sample_id=sample_id,
                pose_path=str(standardized_path),
                confidence_score=None,
            )
        )

    return generated


def generate_diffdock_poses(
    records: list[ComplexInput],
    output_dir: str | Path,
    num_samples: int,
    command_template: Sequence[str],
    raw_output_dir: str | Path | None = None,
    runner: Runner = subprocess.run,
) -> list[GeneratedPose]:
    """
    Run DiffDock once per complex and return standardized pose records.

    `command_template` supports these placeholders:
    `{complex_id}`, `{protein_path}`, `{ligand_path}`,
    `{ground_truth_pose_path}`, `{output_dir}`, `{num_samples}`.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than 0")

    if not command_template:
        raise ValueError("command_template must not be empty")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_output_root = (
        Path(raw_output_dir)
        if raw_output_dir is not None
        else output_dir.parent / "diffdock_raw_outputs"
    )
    raw_output_root.mkdir(parents=True, exist_ok=True)

    generated: list[GeneratedPose] = []

    for record in records:
        complex_raw_output_dir = raw_output_root / record.complex_id
        complex_raw_output_dir.mkdir(parents=True, exist_ok=True)

        command = _format_command(
            command_template=command_template,
            record=record,
            raw_output_dir=complex_raw_output_dir,
            num_samples=num_samples,
        )

        runner(command, check=True)

        generated.extend(
            _standardize_diffdock_outputs(
                record=record,
                raw_output_dir=complex_raw_output_dir,
                output_dir=output_dir,
                num_samples=num_samples,
            )
        )

    validate_generated_pose_records(
        records=records,
        generated=generated,
        num_samples=num_samples,
        output_dir=output_dir,
    )

    return generated
