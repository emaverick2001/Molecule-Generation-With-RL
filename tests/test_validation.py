import pytest

from src.data.manifests import build_manifest_records, save_manifest
from src.data.validation import (
    find_duplicate_complex_ids,
    validate_file_exists,
    validate_file_not_empty,
    validate_manifest_file,
    validate_manifest_records,
    validate_record,
)
from src.utils.schemas import ComplexInput


def _create_complex(raw_root, complex_id, *, ligand_filename="ligand.sdf"):
    complex_dir = raw_root / complex_id
    complex_dir.mkdir(parents=True)

    files = {
        "protein.pdb": "protein\n",
        ligand_filename: "ligand\n",
        "ligand_gt.sdf": "ground truth\n",
    }

    for filename, contents in files.items():
        (complex_dir / filename).write_text(contents, encoding="utf-8")


def test_validate_file_exists_passes_for_existing_file(tmp_path):
    path = tmp_path / "protein.pdb"
    path.write_text("protein\n", encoding="utf-8")

    assert validate_file_exists(path, "protein_path") == path


def test_validate_file_exists_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="protein_path does not exist"):
        validate_file_exists(tmp_path / "missing.pdb", "protein_path")


def test_validate_file_not_empty_passes_for_non_empty_file(tmp_path):
    path = tmp_path / "ligand.sdf"
    path.write_text("ligand\n", encoding="utf-8")

    validate_file_not_empty(path, "ligand_path")


def test_validate_file_not_empty_raises_for_empty_file(tmp_path):
    path = tmp_path / "ligand.sdf"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="ligand_path is empty"):
        validate_file_not_empty(path, "ligand_path")


def test_find_duplicate_complex_ids_returns_duplicates():
    records = [
        ComplexInput("1abc", "p1.pdb", "l1.sdf", "g1.sdf", "mini"),
        ComplexInput("2xyz", "p2.pdb", "l2.sdf", "g2.sdf", "mini"),
        ComplexInput("1abc", "p3.pdb", "l3.sdf", "g3.sdf", "mini"),
    ]

    assert find_duplicate_complex_ids(records) == {"1abc"}


def test_validate_record_returns_no_errors_for_valid_record(tmp_path):
    raw_root = tmp_path / "pdbbind"
    _create_complex(raw_root, "1abc")
    record = build_manifest_records(["1abc"], raw_root, split="mini")[0]

    assert validate_record(record, duplicate_ids=set()) == []


def test_validate_record_catches_empty_file_wrong_extension_duplicate_and_split(tmp_path):
    raw_root = tmp_path / "pdbbind"
    _create_complex(raw_root, "1abc", ligand_filename="ligand.mol2")
    ligand_path = raw_root / "1abc" / "ligand.mol2"
    ligand_path.write_text("", encoding="utf-8")

    record = ComplexInput(
        complex_id="1abc",
        protein_path=str(raw_root / "1abc" / "protein.pdb"),
        ligand_path=str(ligand_path),
        ground_truth_pose_path=str(raw_root / "1abc" / "ligand_gt.sdf"),
        split="bad",
    )

    errors = validate_record(record, duplicate_ids={"1abc"})
    error_text = "; ".join(errors)

    assert "duplicate complex_id: 1abc" in error_text
    assert "invalid split: bad" in error_text
    assert "ligand_path is empty" in error_text
    assert "ligand_path must use extension .sdf" in error_text


def test_validate_manifest_records_reports_missing_files(tmp_path):
    raw_root = tmp_path / "pdbbind"
    complex_dir = raw_root / "1abc"
    complex_dir.mkdir(parents=True)
    (complex_dir / "protein.pdb").write_text("protein\n", encoding="utf-8")
    (complex_dir / "ligand.sdf").write_text("ligand\n", encoding="utf-8")
    records = build_manifest_records(["1abc"], raw_root, split="mini")

    report = validate_manifest_records(records)

    assert report["num_complexes"] == 1
    assert report["num_valid"] == 0
    assert report["num_invalid"] == 1
    assert report["invalid_complexes"][0]["complex_id"] == "1abc"
    assert "ground_truth_pose_path does not exist" in report["invalid_complexes"][0]["reason"]


def test_validate_manifest_records_reports_duplicate_complex_ids(tmp_path):
    raw_root = tmp_path / "pdbbind"
    _create_complex(raw_root, "1abc")
    records = build_manifest_records(["1abc", "1abc"], raw_root, split="mini")

    report = validate_manifest_records(records)

    assert report["num_complexes"] == 2
    assert report["num_valid"] == 0
    assert report["num_invalid"] == 2
    assert all(
        "duplicate complex_id: 1abc" in invalid["reason"]
        for invalid in report["invalid_complexes"]
    )


def test_validate_manifest_records_accepts_custom_valid_splits(tmp_path):
    raw_root = tmp_path / "pdbbind"
    _create_complex(raw_root, "1abc")
    records = build_manifest_records(["1abc"], raw_root, split="holdout")

    report = validate_manifest_records(records, valid_splits={"holdout"})

    assert report["num_valid"] == 1
    assert report["num_invalid"] == 0


def test_validate_manifest_file_loads_and_validates_manifest(tmp_path):
    raw_root = tmp_path / "pdbbind"
    manifest_path = tmp_path / "processed" / "manifest.json"
    _create_complex(raw_root, "1abc")
    records = build_manifest_records(["1abc"], raw_root, split="mini")
    save_manifest(records, manifest_path)

    report = validate_manifest_file(manifest_path)

    assert report == {
        "num_complexes": 1,
        "num_valid": 1,
        "num_invalid": 0,
        "invalid_complexes": [],
    }
