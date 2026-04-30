import json

import pytest

from src.data.manifests import (
    build_and_save_manifest,
    build_manifest_records,
    load_manifest,
    read_complex_ids,
    save_manifest,
    validate_manifest_records,
)
from src.utils.schemas import ComplexInput


def _create_complex(raw_root, complex_id):
    complex_dir = raw_root / complex_id
    complex_dir.mkdir(parents=True)

    for filename in ["protein.pdb", "ligand.sdf", "ligand_gt.sdf"]:
        (complex_dir / filename).write_text(f"{complex_id} {filename}\n", encoding="utf-8")


def test_read_complex_ids_reads_non_empty_ids(tmp_path):
    ids_path = tmp_path / "mini_ids.txt"
    ids_path.write_text("\n1abc\n\n2xyz\n  3def  \n", encoding="utf-8")

    assert read_complex_ids(ids_path) == ["1abc", "2xyz", "3def"]


def test_read_complex_ids_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Complex ID file not found"):
        read_complex_ids(tmp_path / "missing.txt")


def test_build_manifest_records_creates_correct_paths(tmp_path):
    records = build_manifest_records(
        complex_ids=["1abc"],
        raw_root=tmp_path / "pdbbind",
        split="mini",
    )

    assert records == [
        ComplexInput(
            complex_id="1abc",
            protein_path=str(tmp_path / "pdbbind" / "1abc" / "protein.pdb"),
            ligand_path=str(tmp_path / "pdbbind" / "1abc" / "ligand.sdf"),
            ground_truth_pose_path=str(tmp_path / "pdbbind" / "1abc" / "ligand_gt.sdf"),
            split="mini",
        )
    ]


def test_validate_manifest_records_passes_when_files_exist(tmp_path):
    raw_root = tmp_path / "pdbbind"
    _create_complex(raw_root, "1abc")
    records = build_manifest_records(["1abc"], raw_root, split="mini")

    validate_manifest_records(records)


def test_validate_manifest_records_fails_when_files_are_missing(tmp_path):
    raw_root = tmp_path / "pdbbind"
    complex_dir = raw_root / "1abc"
    complex_dir.mkdir(parents=True)
    (complex_dir / "protein.pdb").write_text("protein\n", encoding="utf-8")
    (complex_dir / "ligand.sdf").write_text("ligand\n", encoding="utf-8")

    records = build_manifest_records(["1abc"], raw_root, split="mini")

    with pytest.raises(FileNotFoundError, match="Missing ground-truth pose file"):
        validate_manifest_records(records)


def test_save_manifest_writes_json(tmp_path):
    output_path = tmp_path / "processed" / "manifest.json"
    records = [
        ComplexInput(
            complex_id="1abc",
            protein_path="data/raw/pdbbind/1abc/protein.pdb",
            ligand_path="data/raw/pdbbind/1abc/ligand.sdf",
            ground_truth_pose_path="data/raw/pdbbind/1abc/ligand_gt.sdf",
            split="mini",
        )
    ]

    save_manifest(records, output_path)

    with output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data == [records[0].to_dict()]


def test_load_manifest_reads_json_into_complex_inputs(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_data = [
        {
            "complex_id": "1abc",
            "protein_path": "data/raw/pdbbind/1abc/protein.pdb",
            "ligand_path": "data/raw/pdbbind/1abc/ligand.sdf",
            "ground_truth_pose_path": "data/raw/pdbbind/1abc/ligand_gt.sdf",
            "split": "mini",
        }
    ]
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    records = load_manifest(manifest_path)

    assert records == [ComplexInput.from_dict(manifest_data[0])]


def test_load_manifest_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Manifest file not found"):
        load_manifest(tmp_path / "missing_manifest.json")


def test_build_and_save_manifest_runs_full_workflow(tmp_path):
    ids_path = tmp_path / "mini_ids.txt"
    raw_root = tmp_path / "pdbbind"
    output_path = tmp_path / "processed" / "diffdock" / "manifests" / "mini.json"

    ids_path.write_text("1abc\n2xyz\n", encoding="utf-8")
    _create_complex(raw_root, "1abc")
    _create_complex(raw_root, "2xyz")

    records = build_and_save_manifest(
        ids_path=ids_path,
        raw_root=raw_root,
        split="mini",
        output_path=output_path,
    )

    assert [record.complex_id for record in records] == ["1abc", "2xyz"]
    assert output_path.exists()
    assert load_manifest(output_path) == records
