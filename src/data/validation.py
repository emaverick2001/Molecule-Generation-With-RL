"""
Validation helpers for dataset manifests.

This module performs the stricter checks needed before a manifest is used by
baseline generation, reward filtering, evaluation, or post-training.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.data.manifests import load_manifest
from src.utils.schemas import ComplexInput


DEFAULT_VALID_SPLITS = frozenset({"mini", "train", "val", "test"})

EXPECTED_EXTENSIONS = {
    "protein_path": ".pdb",
    "ligand_path": ".sdf",
    "ground_truth_pose_path": ".sdf",
}


def validate_file_exists(path: str | Path, field_name: str) -> Path:
    """
    Ensure a required manifest path points to an existing file.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")

    return path


def validate_file_not_empty(path: str | Path, field_name: str) -> None:
    """
    Ensure a required manifest path points to a non-empty file.
    """
    path = Path(path)

    if path.stat().st_size == 0:
        raise ValueError(f"{field_name} is empty: {path}")


def find_duplicate_complex_ids(records: Iterable[ComplexInput]) -> set[str]:
    """
    Return complex IDs that appear more than once in a manifest.
    """
    counts = Counter(record.complex_id for record in records)

    return {complex_id for complex_id, count in counts.items() if count > 1}


def _validate_file_extension(
    path: str | Path,
    field_name: str,
    expected_extension: str,
) -> None:
    path = Path(path)

    if path.suffix.lower() != expected_extension:
        raise ValueError(
            f"{field_name} must use extension {expected_extension}: {path}"
        )


def validate_record(
    record: ComplexInput,
    duplicate_ids: set[str],
    valid_splits: Iterable[str] = DEFAULT_VALID_SPLITS,
) -> list[str]:
    """
    Validate one manifest record and return all detected issues.
    """
    valid_split_set = set(valid_splits)
    errors = []

    if record.complex_id in duplicate_ids:
        errors.append(f"duplicate complex_id: {record.complex_id}")

    if record.split not in valid_split_set:
        errors.append(f"invalid split: {record.split}")

    for field_name, expected_extension in EXPECTED_EXTENSIONS.items():
        path = getattr(record, field_name)
        file_exists = True

        try:
            validate_file_exists(path, field_name)
        except FileNotFoundError as error:
            errors.append(str(error))
            file_exists = False

        if file_exists:
            try:
                validate_file_not_empty(path, field_name)
            except ValueError as error:
                errors.append(str(error))

        try:
            _validate_file_extension(path, field_name, expected_extension)
        except ValueError as error:
            errors.append(str(error))

    return errors


def validate_manifest_records(
    records: Iterable[ComplexInput],
    valid_splits: Iterable[str] = DEFAULT_VALID_SPLITS,
) -> dict[str, Any]:
    """
    Validate manifest records and return a report suitable for artifact logging.
    """
    records = list(records)
    duplicate_ids = find_duplicate_complex_ids(records)
    invalid_complexes = []

    for record in records:
        errors = validate_record(
            record=record,
            duplicate_ids=duplicate_ids,
            valid_splits=valid_splits,
        )

        if errors:
            invalid_complexes.append(
                {
                    "complex_id": record.complex_id,
                    "reason": "; ".join(errors),
                }
            )

    num_invalid = len(invalid_complexes)

    return {
        "num_complexes": len(records),
        "num_valid": len(records) - num_invalid,
        "num_invalid": num_invalid,
        "invalid_complexes": invalid_complexes,
    }


def validate_manifest_file(
    path: str | Path,
    valid_splits: Iterable[str] = DEFAULT_VALID_SPLITS,
) -> dict[str, Any]:
    """
    Load and validate a manifest JSON file.
    """
    records = load_manifest(path)

    return validate_manifest_records(records, valid_splits=valid_splits)
