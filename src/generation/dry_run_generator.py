# src/generation/dry_run_generator.py

from __future__ import annotations

from pathlib import Path

from src.generation.contract import validate_generated_pose_records
from src.utils.schemas import ComplexInput, GeneratedPose


DRY_RUN_SDF = """DryRunGeneratedPose
  MVP

  0  0  0  0  0  0            999 V2000
M  END
$$$$
"""


def generate_dry_run_poses(
    records: list[ComplexInput],
    output_dir: str | Path,
    num_samples: int,
) -> list[GeneratedPose]:
    """
    Create placeholder generated pose files and GeneratedPose records.

    This does not run DiffDock. It only creates the same output shape
    expected from real baseline generation.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than 0")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[GeneratedPose] = []

    for record in records:
        for sample_id in range(num_samples):
            pose_path = output_dir / f"{record.complex_id}_sample_{sample_id}.sdf"

            pose_path.write_text(DRY_RUN_SDF, encoding="utf-8")

            generated.append(
                GeneratedPose(
                    complex_id=record.complex_id,
                    sample_id=sample_id,
                    pose_path=str(pose_path),
                    confidence_score=None,
                )
            )

    validate_generated_pose_records(
        records=records,
        generated=generated,
        num_samples=num_samples,
        output_dir=output_dir,
    )

    return generated
