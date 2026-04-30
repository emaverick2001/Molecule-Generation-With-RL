"""
Manifest utilities for dataset indexing and split management.

This module defines helpers to create, validate, load, and save dataset
manifest files used across the pipeline (baseline, rerank, reward-filtering,
and post-training).

In this project, a manifest is the canonical index of samples (e.g., complexes)
and their resolved file paths/metadata. Using manifests ensures:
- reproducible train/val/test splits
- consistent sample selection across runs
- simple config-driven execution via manifest_path entries
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.utils.schemas import ComplexInput


def read_complex_ids(path: str | Path) -> list[str]:
    """
    Read one complex ID per line from a text file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Complex ID file not found: {path}")

    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_manifest_records(
    complex_ids: Iterable[str],
    raw_root: str | Path,
    split: str,
) -> list[ComplexInput]:
    """
    Convert complex IDs into standardized PDBBind-style manifest records.
    """
    raw_root = Path(raw_root)

    return [
        ComplexInput(
            complex_id=complex_id,
            protein_path=str(raw_root / complex_id / "protein.pdb"),
            ligand_path=str(raw_root / complex_id / "ligand.sdf"),
            ground_truth_pose_path=str(raw_root / complex_id / "ligand_gt.sdf"),
            split=split,
        )
        for complex_id in complex_ids
    ]


def validate_manifest_records(records: Iterable[ComplexInput]) -> None:
    """
    Ensure every manifest record points to required input files.
    """
    required_fields = {
        "protein_path": "protein",
        "ligand_path": "ligand",
        "ground_truth_pose_path": "ground-truth pose",
    }

    for record in records:
        for field_name, label in required_fields.items():
            path = Path(getattr(record, field_name))

            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {label} file for complex "
                    f"{record.complex_id}: {path}"
                )


def save_manifest(records: Iterable[ComplexInput], path: str | Path) -> None:
    """
    Save manifest records as indented JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump([record.to_dict() for record in records], f, indent=2)


def load_manifest(path: str | Path) -> list[ComplexInput]:
    """
    Load manifest JSON and convert records back into ComplexInput objects.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return [ComplexInput.from_dict(record) for record in data]


def build_and_save_manifest(
    ids_path: str | Path,
    raw_root: str | Path,
    split: str,
    output_path: str | Path,
) -> list[ComplexInput]:
    """
    Build, validate, and save a manifest in one call.
    """
    complex_ids = read_complex_ids(ids_path)
    records = build_manifest_records(
        complex_ids=complex_ids,
        raw_root=raw_root,
        split=split,
    )
    validate_manifest_records(records)
    save_manifest(records, output_path)

    return records
