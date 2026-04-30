import json
from pathlib import Path

import pytest

from src.data.loaders import (
    filter_records_by_split,
    load_complex_manifest,
    load_split_ids,
)
from src.data.manifests import build_manifest_records, save_manifest
from src.data.validation import find_duplicate_complex_ids
from src.utils.schemas import ComplexInput


def _create_complex(raw_root: Path, complex_id: str) -> None:
    complex_dir = raw_root / complex_id
    complex_dir.mkdir(parents=True)

    for filename in ["protein.pdb", "ligand.sdf", "ligand_gt.sdf"]:
        (complex_dir / filename).write_text(f"{complex_id} {filename}\n", encoding="utf-8")


def test_load_complex_manifest_raises_when_manifest_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Manifest file not found"):
        load_complex_manifest(tmp_path / "missing_manifest.json")


def test_load_complex_manifest_loads_valid_records(tmp_path):
    raw_root = tmp_path / "pdbbind"
    manifest_path = tmp_path / "manifest.json"
    _create_complex(raw_root, "1abc")
    records = build_manifest_records(["1abc"], raw_root, split="mini")
    save_manifest(records, manifest_path)

    loaded_records = load_complex_manifest(manifest_path)

    assert loaded_records == records
    assert all(isinstance(record, ComplexInput) for record in loaded_records)


def test_load_complex_manifest_records_have_required_fields(tmp_path):
    raw_root = tmp_path / "pdbbind"
    manifest_path = tmp_path / "manifest.json"
    _create_complex(raw_root, "1abc")
    save_manifest(build_manifest_records(["1abc"], raw_root, split="mini"), manifest_path)

    record = load_complex_manifest(manifest_path)[0]

    assert record.complex_id == "1abc"
    assert record.protein_path
    assert record.ligand_path
    assert record.ground_truth_pose_path
    assert record.split == "mini"


def test_load_complex_manifest_validates_all_paths_exist(tmp_path):
    raw_root = tmp_path / "pdbbind"
    manifest_path = tmp_path / "manifest.json"
    _create_complex(raw_root, "1abc")
    save_manifest(build_manifest_records(["1abc"], raw_root, split="mini"), manifest_path)

    record = load_complex_manifest(manifest_path)[0]

    assert Path(record.protein_path).is_file()
    assert Path(record.ligand_path).is_file()
    assert Path(record.ground_truth_pose_path).is_file()


def test_load_complex_manifest_validates_complex_ids_are_unique(tmp_path):
    raw_root = tmp_path / "pdbbind"
    manifest_path = tmp_path / "manifest.json"
    _create_complex(raw_root, "1abc")
    records = build_manifest_records(["1abc"], raw_root, split="mini")
    save_manifest(records, manifest_path)

    loaded_records = load_complex_manifest(manifest_path)

    assert find_duplicate_complex_ids(loaded_records) == set()


def test_load_complex_manifest_raises_clear_error_for_invalid_manifest(tmp_path):
    raw_root = tmp_path / "pdbbind"
    manifest_path = tmp_path / "manifest.json"
    complex_dir = raw_root / "1abc"
    complex_dir.mkdir(parents=True)
    (complex_dir / "protein.pdb").write_text("protein\n", encoding="utf-8")
    (complex_dir / "ligand.sdf").write_text("ligand\n", encoding="utf-8")
    save_manifest(build_manifest_records(["1abc"], raw_root, split="mini"), manifest_path)

    with pytest.raises(ValueError, match="Manifest validation failed"):
        load_complex_manifest(manifest_path)


def test_load_complex_manifest_can_skip_validation(tmp_path):
    raw_root = tmp_path / "pdbbind"
    manifest_path = tmp_path / "manifest.json"
    records = build_manifest_records(["1abc"], raw_root, split="mini")
    save_manifest(records, manifest_path)

    assert load_complex_manifest(manifest_path, validate=False) == records


def test_load_complex_manifest_raises_when_json_root_is_not_list(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"complex_id": "1abc"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Manifest JSON root must be a list"):
        load_complex_manifest(manifest_path, validate=False)


def test_load_complex_manifest_raises_for_missing_required_field(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "complex_id": "1abc",
                    "protein_path": "protein.pdb",
                    "ligand_path": "ligand.sdf",
                    "split": "mini",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid manifest record"):
        load_complex_manifest(manifest_path, validate=False)


def test_filter_records_by_split_returns_matching_records():
    records = [
        ComplexInput("1abc", "p1.pdb", "l1.sdf", "g1.sdf", "train"),
        ComplexInput("2xyz", "p2.pdb", "l2.sdf", "g2.sdf", "val"),
        ComplexInput("3def", "p3.pdb", "l3.sdf", "g3.sdf", "train"),
    ]

    filtered = filter_records_by_split(records, "train")

    assert [record.complex_id for record in filtered] == ["1abc", "3def"]


def test_load_split_ids_reads_non_empty_ids(tmp_path):
    split_path = tmp_path / "mini.txt"
    split_path.write_text("\n1abc\n2xyz\n\n", encoding="utf-8")

    assert load_split_ids(split_path) == ["1abc", "2xyz"]
