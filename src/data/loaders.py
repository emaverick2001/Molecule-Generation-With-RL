"""
Dataset loading helpers.

Loaders convert manifest artifacts into typed records. They intentionally avoid
molecular preprocessing, featurization, DiffDock conversion, reward computation,
and RMSD computation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.data.manifests import read_complex_ids
from src.data.validation import validate_manifest_records
from src.utils.schemas import ComplexInput


def _format_invalid_complexes(invalid_complexes: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"{item['complex_id']}: {item['reason']}" for item in invalid_complexes
    )


def load_complex_manifest(
    manifest_path: str | Path,
    validate: bool = True,
) -> list[ComplexInput]:
    """
    Load a dataset manifest JSON file into ComplexInput records.
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Manifest JSON root must be a list: {manifest_path}")

    try:
        records = [ComplexInput.from_dict(item) for item in data]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid manifest record in {manifest_path}: {error}") from error

    if validate:
        validation_report = validate_manifest_records(records)

        if validation_report["num_invalid"] > 0:
            details = _format_invalid_complexes(validation_report["invalid_complexes"])
            raise ValueError(
                f"Manifest validation failed for {manifest_path}: {details}"
            )

    return records


def filter_records_by_split(
    records: Iterable[ComplexInput],
    split: str,
) -> list[ComplexInput]:
    """
    Return records matching the requested split.
    """
    return [record for record in records if record.split == split]


def load_split_ids(path: str | Path) -> list[str]:
    """
    Load one complex ID per line from a split file.
    """
    return read_complex_ids(path)
