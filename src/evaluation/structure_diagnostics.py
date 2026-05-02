from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.data.structure_checks import parse_protein_structure_stats
from src.evaluation.rmsd import (
    compute_centroid_distance,
    count_sdf_atoms,
)
from src.utils.artifact_logger import read_json
from src.utils.schemas import ComplexInput, GeneratedPose


@dataclass(frozen=True)
class StructureDiagnosticRecord:
    complex_id: str
    sample_id: int | None
    protein_atom_count: int | None
    protein_residue_count: int | None
    protein_chain_count: int | None
    unsupported_residue_count: int | None
    unsupported_residue_names: str
    ligand_atom_count: int | None
    reference_atom_count: int | None
    generated_atom_count: int | None
    input_reference_centroid_distance: float | None
    generated_reference_centroid_distance: float | None
    valid: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_input_records(path: str | Path) -> list[ComplexInput]:
    return [ComplexInput.from_dict(item) for item in read_json(path)]


def _load_generated_records(path: str | Path) -> list[GeneratedPose]:
    return [GeneratedPose.from_dict(item) for item in read_json(path)]


def _safe_round(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def build_structure_diagnostics(
    input_records: list[ComplexInput],
    generated_records: list[GeneratedPose],
    *,
    remove_hs: bool = True,
) -> list[StructureDiagnosticRecord]:
    generated_by_complex: dict[str, list[GeneratedPose]] = {}

    for pose in generated_records:
        generated_by_complex.setdefault(pose.complex_id, []).append(pose)

    diagnostics: list[StructureDiagnosticRecord] = []

    for record in input_records:
        protein_atom_count = None
        protein_residue_count = None
        protein_chain_count = None
        unsupported_residue_count = None
        unsupported_residue_names = ""
        ligand_atom_count = None
        reference_atom_count = None
        input_reference_centroid_distance = None
        base_errors: list[str] = []

        try:
            protein_stats = parse_protein_structure_stats(record.protein_path)
            protein_atom_count = protein_stats.atom_count
            protein_residue_count = protein_stats.residue_count
            protein_chain_count = protein_stats.chain_count
            unsupported_residue_count = protein_stats.unsupported_residue_count
            unsupported_residue_names = ",".join(protein_stats.unsupported_residue_names)
        except (FileNotFoundError, ValueError) as error:
            base_errors.append(str(error))

        try:
            ligand_atom_count = count_sdf_atoms(record.ligand_path, remove_hs=remove_hs)
            reference_atom_count = count_sdf_atoms(
                record.ground_truth_pose_path,
                remove_hs=remove_hs,
            )
            input_reference_centroid_distance = compute_centroid_distance(
                record.ligand_path,
                record.ground_truth_pose_path,
                remove_hs=remove_hs,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            base_errors.append(str(error))

        complex_generated = sorted(
            generated_by_complex.get(record.complex_id, []),
            key=lambda pose: pose.sample_id,
        )

        if not complex_generated:
            diagnostics.append(
                StructureDiagnosticRecord(
                    complex_id=record.complex_id,
                    sample_id=None,
                    protein_atom_count=protein_atom_count,
                    protein_residue_count=protein_residue_count,
                    protein_chain_count=protein_chain_count,
                    unsupported_residue_count=unsupported_residue_count,
                    unsupported_residue_names=unsupported_residue_names,
                    ligand_atom_count=ligand_atom_count,
                    reference_atom_count=reference_atom_count,
                    generated_atom_count=None,
                    input_reference_centroid_distance=_safe_round(
                        input_reference_centroid_distance
                    ),
                    generated_reference_centroid_distance=None,
                    valid=not base_errors,
                    error="; ".join(base_errors) if base_errors else "no generated pose",
                )
            )
            continue

        for pose in complex_generated:
            errors = list(base_errors)
            generated_atom_count = None
            generated_reference_centroid_distance = None

            try:
                generated_atom_count = count_sdf_atoms(
                    pose.pose_path,
                    remove_hs=remove_hs,
                )
                generated_reference_centroid_distance = compute_centroid_distance(
                    pose.pose_path,
                    record.ground_truth_pose_path,
                    remove_hs=remove_hs,
                )
            except (FileNotFoundError, ValueError, RuntimeError) as error:
                errors.append(str(error))

            diagnostics.append(
                StructureDiagnosticRecord(
                    complex_id=record.complex_id,
                    sample_id=pose.sample_id,
                    protein_atom_count=protein_atom_count,
                    protein_residue_count=protein_residue_count,
                    protein_chain_count=protein_chain_count,
                    unsupported_residue_count=unsupported_residue_count,
                    unsupported_residue_names=unsupported_residue_names,
                    ligand_atom_count=ligand_atom_count,
                    reference_atom_count=reference_atom_count,
                    generated_atom_count=generated_atom_count,
                    input_reference_centroid_distance=_safe_round(
                        input_reference_centroid_distance
                    ),
                    generated_reference_centroid_distance=_safe_round(
                        generated_reference_centroid_distance
                    ),
                    valid=not errors,
                    error="; ".join(errors) if errors else None,
                )
            )

    return diagnostics


def run_structure_diagnostics(
    run_dir: str | Path,
    output_csv_path: str | Path | None = None,
    remove_hs: bool = True,
) -> list[StructureDiagnosticRecord]:
    run_dir = Path(run_dir)
    input_records = _load_input_records(run_dir / "input_manifest.json")
    generated_records = _load_generated_records(
        run_dir / "generated_samples_manifest.json"
    )
    diagnostics = build_structure_diagnostics(
        input_records=input_records,
        generated_records=generated_records,
        remove_hs=remove_hs,
    )

    if output_csv_path is not None:
        save_structure_diagnostics_csv(diagnostics, output_csv_path)

    return diagnostics


def save_structure_diagnostics_csv(
    records: list[StructureDiagnosticRecord],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(StructureDiagnosticRecord.__dataclass_fields__.keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record.to_dict() for record in records)
