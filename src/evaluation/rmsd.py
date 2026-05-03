from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign
except ImportError:  # pragma: no cover - exercised only when RDKit is installed
    Chem = None
    rdMolAlign = None


@dataclass(frozen=True)
class SimpleMol:
    atoms: tuple[str, ...]
    coordinates: tuple[tuple[float, float, float], ...]


def _load_simple_sdf(path: str | Path) -> SimpleMol:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"SDF file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 4:
        raise ValueError(f"No valid molecules found in SDF: {path}")

    try:
        atom_count = int(lines[3][0:3])
    except ValueError as error:
        raise ValueError(f"Could not parse SDF atom count: {path}") from error

    atoms = []
    coordinates = []

    for line in lines[4 : 4 + atom_count]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Could not parse SDF atom line: {path}")

        try:
            coordinates.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError as error:
            raise ValueError(f"Could not parse SDF coordinates: {path}") from error

        atoms.append(parts[3])

    if not atoms:
        raise ValueError(f"No valid molecules found in SDF: {path}")

    return SimpleMol(atoms=tuple(atoms), coordinates=tuple(coordinates))


def _remove_hydrogens_simple(mol: SimpleMol) -> SimpleMol:
    kept = [
        (atom, coordinate)
        for atom, coordinate in zip(mol.atoms, mol.coordinates)
        if atom.upper() != "H"
    ]

    if not kept:
        raise ValueError("Cannot remove hydrogens: molecule has no heavy atoms")

    atoms, coordinates = zip(*kept)
    return SimpleMol(atoms=tuple(atoms), coordinates=tuple(coordinates))


def load_single_sdf(path: str | Path) -> Any:
    """
    Load one molecule from an SDF file.

    Uses RDKit when available. A small coordinate-only parser is used as a
    fallback so pipeline tests do not require RDKit.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"SDF file not found: {path}")

    if Chem is None:
        return _load_simple_sdf(path)

    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    molecules = [mol for mol in supplier if mol is not None]

    if not molecules:
        raise ValueError(f"No valid molecules found in SDF: {path}")

    if len(molecules) > 1:
        raise ValueError(f"Expected one molecule in SDF, found {len(molecules)}: {path}")

    return molecules[0]


def _simple_rmsd(predicted: SimpleMol, reference: SimpleMol) -> float:
    if len(predicted.coordinates) != len(reference.coordinates):
        raise ValueError(
            "Cannot compute RMSD for molecules with different atom counts: "
            f"{len(predicted.coordinates)} != {len(reference.coordinates)}"
        )

    squared_distance_sum = 0.0

    for pred_coord, ref_coord in zip(
        predicted.coordinates,
        reference.coordinates,
    ):
        squared_distance_sum += sum(
            (pred_value - ref_value) ** 2
            for pred_value, ref_value in zip(pred_coord, ref_coord)
        )

    return sqrt(squared_distance_sum / len(predicted.coordinates))


def compute_symmetry_corrected_rmsd(
    predicted_pose_path: str | Path,
    reference_pose_path: str | Path,
    remove_hs: bool = True,
) -> float:
    """
    Compute ligand pose RMSD without pre-aligning the predicted coordinates.
    """
    predicted = load_single_sdf(predicted_pose_path)
    reference = load_single_sdf(reference_pose_path)

    if Chem is None:
        if remove_hs:
            predicted = _remove_hydrogens_simple(predicted)
            reference = _remove_hydrogens_simple(reference)
        return float(_simple_rmsd(predicted, reference))

    if remove_hs:
        predicted = Chem.RemoveHs(predicted)
        reference = Chem.RemoveHs(reference)

    return float(rdMolAlign.CalcRMS(predicted, reference))


def _simple_centroid(mol: SimpleMol) -> tuple[float, float, float]:
    atom_count = len(mol.coordinates)
    return tuple(
        sum(coordinate[axis] for coordinate in mol.coordinates) / atom_count
        for axis in range(3)
    )


def count_sdf_atoms(path: str | Path, remove_hs: bool = True) -> int:
    mol = load_single_sdf(path)

    if Chem is None:
        if remove_hs:
            mol = _remove_hydrogens_simple(mol)

        return len(mol.atoms)

    if remove_hs:
        mol = Chem.RemoveHs(mol)

    return int(mol.GetNumAtoms())


def compute_sdf_centroid(
    path: str | Path,
    remove_hs: bool = True,
) -> tuple[float, float, float]:
    mol = load_single_sdf(path)

    if Chem is None:
        if remove_hs:
            mol = _remove_hydrogens_simple(mol)

        return _simple_centroid(mol)

    if remove_hs:
        mol = Chem.RemoveHs(mol)

    conformer = mol.GetConformer()

    return tuple(
        sum(conformer.GetAtomPosition(i)[axis] for i in range(mol.GetNumAtoms()))
        / mol.GetNumAtoms()
        for axis in range(3)
    )


def compute_centroid_distance(
    predicted_pose_path: str | Path,
    reference_pose_path: str | Path,
    remove_hs: bool = True,
) -> float:
    predicted_centroid = compute_sdf_centroid(predicted_pose_path, remove_hs=remove_hs)
    reference_centroid = compute_sdf_centroid(reference_pose_path, remove_hs=remove_hs)

    return sqrt(
        sum(
            (predicted_value - reference_value) ** 2
            for predicted_value, reference_value in zip(
                predicted_centroid,
                reference_centroid,
            )
        )
    )
