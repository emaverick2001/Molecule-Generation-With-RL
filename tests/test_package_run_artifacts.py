import json
import subprocess
import sys
from pathlib import Path

from scripts.list_run_input_files import list_run_input_files


def _write_manifest_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "artifacts" / "runs" / "test_run"
    input_dir = tmp_path / "data" / "raw" / "pdbbind_real" / "1abc"
    run_dir.mkdir(parents=True)
    input_dir.mkdir(parents=True)

    for filename in ["protein.pdb", "ligand.sdf", "ligand_gt.sdf"]:
        (input_dir / filename).write_text(f"{filename}\n", encoding="utf-8")

    manifest = [
        {
            "complex_id": "1abc",
            "protein_path": "data/raw/pdbbind_real/1abc/protein.pdb",
            "ligand_path": "data/raw/pdbbind_real/1abc/ligand.sdf",
            "ground_truth_pose_path": "data/raw/pdbbind_real/1abc/ligand_gt.sdf",
            "split": "tiny_real",
        }
    ]
    (run_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text('{"ok": true}\n', encoding="utf-8")

    return run_dir


def test_list_run_input_files_returns_manifest_structure_paths(tmp_path):
    run_dir = _write_manifest_run(tmp_path)

    assert list_run_input_files(run_dir, repo_root=tmp_path) == [
        "data/raw/pdbbind_real/1abc/protein.pdb",
        "data/raw/pdbbind_real/1abc/ligand.sdf",
        "data/raw/pdbbind_real/1abc/ligand_gt.sdf",
    ]


def test_list_run_input_files_accepts_posttraining_train_manifest(tmp_path):
    run_dir = _write_manifest_run(tmp_path)
    input_manifest = run_dir / "input_manifest.json"
    train_manifest = run_dir / "input_train_manifest.json"
    train_manifest.write_text(input_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    input_manifest.unlink()

    assert list_run_input_files(run_dir, repo_root=tmp_path) == [
        "data/raw/pdbbind_real/1abc/protein.pdb",
        "data/raw/pdbbind_real/1abc/ligand.sdf",
        "data/raw/pdbbind_real/1abc/ligand_gt.sdf",
    ]


def test_package_run_artifacts_can_include_input_structures(tmp_path):
    run_dir = _write_manifest_run(tmp_path)
    output_dir = tmp_path / "packaged_runs"
    script_path = Path("scripts/package_run_artifacts.sh").resolve()

    subprocess.run(
        [
            "bash",
            str(script_path),
            str(run_dir.relative_to(tmp_path)),
            "--key",
            "--include-inputs",
            "--output-dir",
            str(output_dir),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    archive_path = next(output_dir.glob("*.tar.gz"))
    listing = subprocess.run(
        [sys.executable, "-m", "tarfile", "-l", str(archive_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert "artifacts/runs/test_run/metrics.json" in listing
    assert "data/raw/pdbbind_real/1abc/protein.pdb" in listing
    assert "data/raw/pdbbind_real/1abc/ligand.sdf" in listing
    assert "data/raw/pdbbind_real/1abc/ligand_gt.sdf" in listing
