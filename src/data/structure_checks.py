from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Any

from src.evaluation.rmsd import (
    compute_centroid_distance,
    compute_sdf_centroid,
    count_sdf_atoms,
)
from src.utils.schemas import ComplexInput


STANDARD_AMINO_ACIDS = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


@dataclass(frozen=True)
class ProteinStructureStats:
    atom_count: int
    residue_count: int
    chain_count: int
    unsupported_residue_count: int
    unsupported_residue_names: list[str]
    centroid: tuple[float, float, float] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComplexPreflightResult:
    complex_id: str
    valid: bool
    reasons: list[str]
    protein_atom_count: int | None
    protein_residue_count: int | None
    protein_chain_count: int | None
    unsupported_residue_count: int | None
    unsupported_residue_names: list[str]
    ligand_atom_count: int | None
    reference_atom_count: int | None
    input_reference_centroid_distance: float | None
    ligand_protein_centroid_distance: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return sqrt(
        sum(
            (first_value - second_value) ** 2
            for first_value, second_value in zip(first, second)
        )
    )


def parse_protein_structure_stats(path: str | Path) -> ProteinStructureStats:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Protein PDB file not found: {path}")

    atom_count = 0
    residues: set[tuple[str, str, str, str]] = set()
    chains: set[str] = set()
    unsupported_residue_names: set[str] = set()
    coordinate_sums = [0.0, 0.0, 0.0]

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            residue_name = line[17:20].strip().upper()
            chain_id = line[21].strip() or "_"
            residue_sequence = line[22:26].strip()
            insertion_code = line[26].strip()

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            atom_count += 1
            coordinate_sums[0] += x
            coordinate_sums[1] += y
            coordinate_sums[2] += z
            chains.add(chain_id)
            residues.add((chain_id, residue_sequence, insertion_code, residue_name))

            if residue_name not in STANDARD_AMINO_ACIDS:
                unsupported_residue_names.add(residue_name)

    centroid = (
        tuple(value / atom_count for value in coordinate_sums)
        if atom_count > 0
        else None
    )

    return ProteinStructureStats(
        atom_count=atom_count,
        residue_count=len(residues),
        chain_count=len(chains),
        unsupported_residue_count=len(unsupported_residue_names),
        unsupported_residue_names=sorted(unsupported_residue_names),
        centroid=centroid,
    )


def preflight_complex_structure(
    record: ComplexInput,
    *,
    remove_hs: bool = True,
    min_protein_atoms: int = 1,
    min_protein_residues: int = 1,
    min_protein_chains: int = 1,
    max_ligand_protein_centroid_distance: float | None = 80.0,
    max_input_reference_centroid_distance: float | None = 2.0,
    fail_on_unsupported_residues: bool = False,
) -> ComplexPreflightResult:
    reasons: list[str] = []
    protein_stats: ProteinStructureStats | None = None
    ligand_atom_count: int | None = None
    reference_atom_count: int | None = None
    input_reference_centroid_distance: float | None = None
    ligand_protein_centroid_distance: float | None = None

    try:
        protein_stats = parse_protein_structure_stats(record.protein_path)

        if protein_stats.atom_count < min_protein_atoms:
            reasons.append(
                f"protein atom count {protein_stats.atom_count} < {min_protein_atoms}"
            )
        if protein_stats.residue_count < min_protein_residues:
            reasons.append(
                "protein residue count "
                f"{protein_stats.residue_count} < {min_protein_residues}"
            )
        if protein_stats.chain_count < min_protein_chains:
            reasons.append(
                f"protein chain count {protein_stats.chain_count} < {min_protein_chains}"
            )
        if fail_on_unsupported_residues and protein_stats.unsupported_residue_names:
            reasons.append(
                "unsupported protein residues: "
                + ",".join(protein_stats.unsupported_residue_names)
            )
    except (FileNotFoundError, ValueError) as error:
        reasons.append(str(error))

    try:
        ligand_atom_count = count_sdf_atoms(record.ligand_path, remove_hs=remove_hs)
        reference_atom_count = count_sdf_atoms(
            record.ground_truth_pose_path,
            remove_hs=remove_hs,
        )

        if ligand_atom_count <= 0:
            reasons.append("ligand has no atoms")
        if reference_atom_count <= 0:
            reasons.append("reference ligand has no atoms")
        if ligand_atom_count != reference_atom_count:
            reasons.append(
                f"ligand/reference atom count mismatch: "
                f"{ligand_atom_count} != {reference_atom_count}"
            )

        input_reference_centroid_distance = compute_centroid_distance(
            record.ligand_path,
            record.ground_truth_pose_path,
            remove_hs=remove_hs,
        )
        if (
            max_input_reference_centroid_distance is not None
            and input_reference_centroid_distance > max_input_reference_centroid_distance
        ):
            reasons.append(
                "input/reference centroid distance "
                f"{input_reference_centroid_distance:.3f} "
                f"> {max_input_reference_centroid_distance:.3f}"
            )

        if protein_stats is not None and protein_stats.centroid is not None:
            ligand_centroid = compute_sdf_centroid(record.ligand_path, remove_hs=remove_hs)
            ligand_protein_centroid_distance = _distance(
                ligand_centroid,
                protein_stats.centroid,
            )

            if (
                max_ligand_protein_centroid_distance is not None
                and ligand_protein_centroid_distance
                > max_ligand_protein_centroid_distance
            ):
                reasons.append(
                    "ligand/protein centroid distance "
                    f"{ligand_protein_centroid_distance:.3f} "
                    f"> {max_ligand_protein_centroid_distance:.3f}"
                )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        reasons.append(str(error))

    return ComplexPreflightResult(
        complex_id=record.complex_id,
        valid=not reasons,
        reasons=reasons,
        protein_atom_count=protein_stats.atom_count if protein_stats else None,
        protein_residue_count=protein_stats.residue_count if protein_stats else None,
        protein_chain_count=protein_stats.chain_count if protein_stats else None,
        unsupported_residue_count=(
            protein_stats.unsupported_residue_count if protein_stats else None
        ),
        unsupported_residue_names=(
            protein_stats.unsupported_residue_names if protein_stats else []
        ),
        ligand_atom_count=ligand_atom_count,
        reference_atom_count=reference_atom_count,
        input_reference_centroid_distance=(
            round(input_reference_centroid_distance, 6)
            if input_reference_centroid_distance is not None
            else None
        ),
        ligand_protein_centroid_distance=(
            round(ligand_protein_centroid_distance, 6)
            if ligand_protein_centroid_distance is not None
            else None
        ),
    )


def filter_complexes_by_preflight(
    records: list[ComplexInput],
    **kwargs: Any,
) -> tuple[list[ComplexInput], list[ComplexPreflightResult]]:
    results = [preflight_complex_structure(record, **kwargs) for record in records]
    valid_complex_ids = {result.complex_id for result in results if result.valid}
    filtered_records = [
        record for record in records if record.complex_id in valid_complex_ids
    ]

    return filtered_records, results
