import pytest

from src.evaluation.rmsd import (
    compute_centroid_distance,
    compute_symmetry_corrected_rmsd,
    load_single_sdf,
)


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


def test_identical_sdfs_produce_zero_rmsd(tmp_path):
    reference_path = tmp_path / "reference.sdf"
    predicted_path = tmp_path / "predicted.sdf"
    contents = _sdf([("C", 0.0, 0.0, 0.0), ("O", 1.0, 0.0, 0.0)])

    reference_path.write_text(contents, encoding="utf-8")
    predicted_path.write_text(contents, encoding="utf-8")

    assert compute_symmetry_corrected_rmsd(predicted_path, reference_path) == pytest.approx(0.0)


def test_moved_conformer_produces_nonzero_rmsd_and_centroid_distance(tmp_path):
    reference_path = tmp_path / "reference.sdf"
    predicted_path = tmp_path / "predicted.sdf"

    reference_path.write_text(
        _sdf([("C", 0.0, 0.0, 0.0), ("O", 1.0, 0.0, 0.0)]),
        encoding="utf-8",
    )
    predicted_path.write_text(
        _sdf([("C", 1.0, 0.0, 0.0), ("O", 2.0, 0.0, 0.0)]),
        encoding="utf-8",
    )

    assert compute_symmetry_corrected_rmsd(predicted_path, reference_path) == pytest.approx(1.0)
    assert compute_centroid_distance(predicted_path, reference_path) == pytest.approx(1.0)


def test_load_single_sdf_raises_for_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="SDF file not found"):
        load_single_sdf(tmp_path / "missing.sdf")


def test_load_single_sdf_raises_for_invalid_contents(tmp_path):
    invalid_path = tmp_path / "invalid.sdf"
    invalid_path.write_text("not an sdf\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_single_sdf(invalid_path)
