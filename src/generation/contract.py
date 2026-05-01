"""
Generation output contract checks.

These helpers validate the shape shared by dry-run generation and real DiffDock
generation before downstream reward and evaluation stages consume outputs.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.schemas import ComplexInput, GeneratedPose


def validate_generated_pose_records(
    records: list[ComplexInput],
    generated: list[GeneratedPose],
    num_samples: int,
    output_dir: str | Path | None = None,
    require_files: bool = True,
) -> None:
    """
    Validate generated pose records for a fixed-sample generation run.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than 0")

    expected_count = len(records) * num_samples
    if len(generated) != expected_count:
        raise ValueError(
            f"Expected {expected_count} generated poses, got {len(generated)}"
        )

    valid_complex_ids = {record.complex_id for record in records}
    seen_keys: set[tuple[str, int]] = set()
    resolved_output_dir = Path(output_dir).resolve() if output_dir is not None else None

    for pose in generated:
        if pose.complex_id not in valid_complex_ids:
            raise ValueError(f"Unknown generated complex_id: {pose.complex_id}")

        if pose.sample_id >= num_samples:
            raise ValueError(
                f"sample_id out of range for {pose.complex_id}: {pose.sample_id}"
            )

        key = (pose.complex_id, pose.sample_id)
        if key in seen_keys:
            raise ValueError(
                f"Duplicate generated pose key: {pose.complex_id}, {pose.sample_id}"
            )
        seen_keys.add(key)

        pose_path = Path(pose.pose_path)

        if resolved_output_dir is not None:
            try:
                pose_path.resolve().relative_to(resolved_output_dir)
            except ValueError as error:
                raise ValueError(
                    f"Generated pose path is outside output_dir: {pose_path}"
                ) from error

        if require_files and not pose_path.is_file():
            raise FileNotFoundError(f"Generated pose file not found: {pose_path}")
