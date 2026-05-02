import json
import subprocess
import sys

from scripts.create_tiny_real_pdbbind import create_tiny_real_pdbbind


def _write_complex(root, complex_id):
    complex_dir = root / complex_id
    complex_dir.mkdir(parents=True)
    (complex_dir / f"{complex_id}_protein.pdb").write_text(
        f"HEADER {complex_id} protein\n",
        encoding="utf-8",
    )
    (complex_dir / f"{complex_id}_ligand.sdf").write_text(
        f"""{complex_id}_ligand
  PDBBind

  1  0  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
""",
        encoding="utf-8",
    )


def test_create_tiny_real_pdbbind_writes_split_manifest_and_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "pdbbind"
    for complex_id in ["1a30", "2xyz", "3def"]:
        _write_complex(source, complex_id)

    output_dirs = create_tiny_real_pdbbind(
        source=source,
        complex_ids=["1a30", "2xyz", "3def"],
        output_root=tmp_path / "data" / "raw" / "pdbbind_real",
    )

    assert len(output_dirs) == 3
    assert all((path / "protein.pdb").is_file() for path in output_dirs)
    assert all((path / "ligand.sdf").is_file() for path in output_dirs)
    assert all((path / "ligand_gt.sdf").is_file() for path in output_dirs)

    split_path = tmp_path / "data" / "processed" / "diffdock" / "splits" / "tiny_real.txt"
    manifest_path = tmp_path / "data" / "processed" / "diffdock" / "manifests" / "tiny_real_manifest.json"
    validation_path = (
        tmp_path
        / "data"
        / "processed"
        / "diffdock"
        / "manifests"
        / "tiny_real_validation_report.json"
    )

    assert split_path.read_text(encoding="utf-8").splitlines() == [
        "1a30",
        "2xyz",
        "3def",
    ]

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    with validation_path.open("r", encoding="utf-8") as f:
        validation = json.load(f)

    assert [record["complex_id"] for record in manifest] == ["1a30", "2xyz", "3def"]
    assert {record["split"] for record in manifest} == {"tiny_real"}
    assert validation["num_valid"] == 3


def test_create_tiny_real_pdbbind_script_accepts_ids_file(tmp_path):
    source = tmp_path / "pdbbind"
    for complex_id in ["1a30", "2xyz"]:
        _write_complex(source, complex_id)

    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("1a30\n2xyz\n", encoding="utf-8")
    output_root = tmp_path / "output" / "pdbbind_real"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/create_tiny_real_pdbbind.py",
            "--source",
            str(source),
            "--ids-file",
            str(ids_file),
            "--output-root",
            str(output_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Prepared 2 real PDBBind complexes" in result.stdout
    assert (output_root / "1a30" / "protein.pdb").is_file()
    assert (output_root / "2xyz" / "protein.pdb").is_file()
