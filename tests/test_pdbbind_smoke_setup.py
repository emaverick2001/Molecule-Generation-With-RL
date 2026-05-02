import subprocess
import sys


def _write_complex(root, complex_id="1a30"):
    complex_dir = root / complex_id
    complex_dir.mkdir(parents=True)
    (complex_dir / f"{complex_id}_protein.pdb").write_text("HEADER real protein\n", encoding="utf-8")
    (complex_dir / f"{complex_id}_ligand.sdf").write_text(
        """RealLigand
  PDBBind

  1  0  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
""",
        encoding="utf-8",
    )


def test_setup_pdbbind_smoke_complex_with_explicit_source(tmp_path):
    source = tmp_path / "refined-set"
    output_root = tmp_path / "real"
    _write_complex(source)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/setup_pdbbind_smoke_complex.py",
            "--source",
            str(source),
            "--complex-id",
            "1a30",
            "--output-root",
            str(output_root),
            "--non-interactive",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Prepared real PDBBind smoke complex" in result.stdout
    assert (output_root / "1a30" / "protein.pdb").is_file()


def test_setup_pdbbind_smoke_complex_prints_download_instructions_when_missing(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/setup_pdbbind_smoke_complex.py",
            "--complex-id",
            "1a30",
            "--search-root",
            str(tmp_path / "missing"),
            "--non-interactive",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "No local PDBBind complex/archive was found" in result.stdout
    assert "https://www.pdbbind.org.cn/" in result.stdout
