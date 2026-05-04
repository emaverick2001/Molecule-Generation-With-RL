import json

import pytest

from scripts.freeze_evaluation_protocol import freeze_evaluation_protocol


def test_freeze_evaluation_protocol_writes_manifest_split_and_protocol(tmp_path):
    input_manifest = tmp_path / "input_manifest.json"
    input_manifest.write_text(
        json.dumps(
            [
                {
                    "complex_id": "1abc",
                    "protein_path": "data/raw/pdbbind_real/1abc/protein.pdb",
                    "ligand_path": "data/raw/pdbbind_real/1abc/ligand.sdf",
                    "ground_truth_pose_path": "data/raw/pdbbind_real/1abc/ligand_gt.sdf",
                    "split": "tiny_real",
                }
            ]
        ),
        encoding="utf-8",
    )
    output_manifest = tmp_path / "main_eval_manifest.json"
    output_split = tmp_path / "main_eval.txt"
    output_protocol = tmp_path / "main_eval_protocol.json"

    protocol = freeze_evaluation_protocol(
        input_manifest=input_manifest,
        output_manifest=output_manifest,
        output_split=output_split,
        output_protocol=output_protocol,
        source_run_dir="artifacts/runs/source",
        exist_ok=False,
    )

    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert manifest[0]["split"] == "main_eval"
    assert output_split.read_text(encoding="utf-8") == "1abc\n"
    assert protocol["num_complexes"] == 1
    assert protocol["complex_ids"] == ["1abc"]
    assert json.loads(output_protocol.read_text(encoding="utf-8"))["name"] == "main_eval"


def test_freeze_evaluation_protocol_refuses_overwrite_without_exist_ok(tmp_path):
    input_manifest = tmp_path / "input_manifest.json"
    input_manifest.write_text("[]", encoding="utf-8")
    output_manifest = tmp_path / "main_eval_manifest.json"
    output_manifest.write_text("[]", encoding="utf-8")

    with pytest.raises(FileExistsError):
        freeze_evaluation_protocol(
            input_manifest=input_manifest,
            output_manifest=output_manifest,
            output_split=tmp_path / "main_eval.txt",
            output_protocol=tmp_path / "main_eval_protocol.json",
        )
