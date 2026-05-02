import pytest

from src.data.structure_checks import (
    filter_complexes_by_preflight,
    parse_protein_structure_stats,
)
from src.evaluation.structure_diagnostics import (
    build_structure_diagnostics,
    save_structure_diagnostics_csv,
)
from src.utils.schemas import ComplexInput, GeneratedPose


def _sdf(atom_coordinates):
    atom_lines = "\n".join(
        f"{x:10.4f}{y:10.4f}{z:10.4f} {atom:<3} 0  0  0  0  0  0  0  0  0  0  0  0"
        for atom, x, y, z in atom_coordinates
    )
    return f"""TestMol
  MVP

{len(atom_coordinates):3d}  0  0  0  0  0            999 V2000
{atom_lines}
M  END
$$$$
"""


def _pdb_atom(serial, atom_name, residue_name, chain_id, residue_number, x, y, z):
    return (
        f"ATOM  {serial:5d} {atom_name:<4}{residue_name:>3} {chain_id}"
        f"{residue_number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        "  1.00 20.00           C\n"
    )


def _write_complex(tmp_path, complex_id="1abc", ligand_offset=0.0):
    complex_dir = tmp_path / complex_id
    complex_dir.mkdir()
    protein_path = complex_dir / "protein.pdb"
    ligand_path = complex_dir / "ligand.sdf"
    reference_path = complex_dir / "ligand_gt.sdf"
    generated_path = complex_dir / "generated.sdf"

    protein_path.write_text(
        "".join(
            [
                _pdb_atom(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0),
                _pdb_atom(2, "CB", "ALA", "A", 1, 1.0, 0.0, 0.0),
            ]
        ),
        encoding="utf-8",
    )
    ligand_path.write_text(
        _sdf([("C", ligand_offset, 0.0, 0.0), ("O", ligand_offset + 1.0, 0.0, 0.0)]),
        encoding="utf-8",
    )
    reference_path.write_text(
        _sdf([("C", 0.0, 0.0, 0.0), ("O", 1.0, 0.0, 0.0)]),
        encoding="utf-8",
    )
    generated_path.write_text(
        _sdf([("C", 2.0, 0.0, 0.0), ("O", 3.0, 0.0, 0.0)]),
        encoding="utf-8",
    )

    record = ComplexInput(
        complex_id=complex_id,
        protein_path=str(protein_path),
        ligand_path=str(ligand_path),
        ground_truth_pose_path=str(reference_path),
        split="tiny_real",
    )

    return record, generated_path


def test_parse_protein_structure_stats_counts_atoms_residues_and_chains(tmp_path):
    record, _ = _write_complex(tmp_path)

    stats = parse_protein_structure_stats(record.protein_path)

    assert stats.atom_count == 2
    assert stats.residue_count == 1
    assert stats.chain_count == 1
    assert stats.centroid == pytest.approx((0.5, 0.0, 0.0))


def test_preflight_filters_complex_with_far_input_reference_centroids(tmp_path):
    valid_record, _ = _write_complex(tmp_path, complex_id="valid")
    invalid_record, _ = _write_complex(tmp_path, complex_id="far", ligand_offset=10.0)

    filtered, results = filter_complexes_by_preflight(
        [valid_record, invalid_record],
        max_input_reference_centroid_distance=2.0,
    )

    assert [record.complex_id for record in filtered] == ["valid"]
    invalid_result = next(result for result in results if result.complex_id == "far")
    assert invalid_result.valid is False
    assert "input/reference centroid distance" in "; ".join(invalid_result.reasons)


def test_build_structure_diagnostics_reports_counts_and_centroids(tmp_path):
    record, generated_path = _write_complex(tmp_path)

    diagnostics = build_structure_diagnostics(
        input_records=[record],
        generated_records=[GeneratedPose("1abc", 0, str(generated_path))],
    )

    assert len(diagnostics) == 1
    row = diagnostics[0]
    assert row.protein_atom_count == 2
    assert row.ligand_atom_count == 2
    assert row.reference_atom_count == 2
    assert row.generated_atom_count == 2
    assert row.input_reference_centroid_distance == 0.0
    assert row.generated_reference_centroid_distance == 2.0
    assert row.valid is True


def test_save_structure_diagnostics_csv(tmp_path):
    record, generated_path = _write_complex(tmp_path)
    diagnostics = build_structure_diagnostics(
        input_records=[record],
        generated_records=[GeneratedPose("1abc", 0, str(generated_path))],
    )
    output_path = tmp_path / "structure_diagnostics.csv"

    save_structure_diagnostics_csv(diagnostics, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "protein_atom_count" in text
    assert "generated_reference_centroid_distance" in text
