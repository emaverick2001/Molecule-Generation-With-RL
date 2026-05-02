import json

from scripts.extract_pdbbind_smoke_complex import extract_smoke_complex


def test_extract_smoke_complex_copies_real_layout_and_writes_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source_root = tmp_path / "pdbbind_refined"
    complex_dir = source_root / "1a30"
    complex_dir.mkdir(parents=True)
    (complex_dir / "1a30_protein.pdb").write_text("HEADER real protein\n", encoding="utf-8")
    (complex_dir / "1a30_ligand.sdf").write_text(
        """RealLigand
  PDBBind

  1  0  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
""",
        encoding="utf-8",
    )

    output_dir = extract_smoke_complex(
        source=source_root,
        complex_id="1a30",
        output_root=tmp_path / "data" / "raw" / "pdbbind_real",
    )

    assert output_dir == tmp_path / "data" / "raw" / "pdbbind_real" / "1a30"
    assert (output_dir / "protein.pdb").read_text(encoding="utf-8").startswith("HEADER")
    assert (output_dir / "ligand.sdf").is_file()
    assert (output_dir / "ligand_gt.sdf").is_file()

    manifest_path = tmp_path / "data" / "processed" / "diffdock" / "manifests" / "smoke_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest == [
        {
            "complex_id": "1a30",
            "protein_path": str(output_dir / "protein.pdb"),
            "ligand_path": str(output_dir / "ligand.sdf"),
            "ground_truth_pose_path": str(output_dir / "ligand_gt.sdf"),
            "split": "smoke",
        }
    ]
