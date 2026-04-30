# scripts/create_tiny_pdbbind_mvp.py

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifests import build_and_save_manifest
from src.data.validation import validate_manifest_file
from src.utils.artifact_logger import save_json


COMPLEX_IDS = ["1abc", "2xyz", "3def", "4ghi", "5jkl"]
RAW_ROOT = Path("data/raw/pdbbind")
SPLIT_PATH = Path("data/processed/diffdock/splits/mini.txt")
MANIFEST_PATH = Path("data/processed/diffdock/manifests/mini_manifest.json")
VALIDATION_REPORT_PATH = Path(
    "data/processed/diffdock/manifests/mini_validation_report.json"
)


FAKE_PROTEIN_PDB = """HEADER    TINY MVP PROTEIN
ATOM      1  N   ALA A   1      11.104  13.207   8.678  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.300   8.400  1.00 20.00           C
ATOM      3  C   ALA A   1      13.100  12.000   7.800  1.00 20.00           C
ATOM      4  O   ALA A   1      12.500  10.950   7.900  1.00 20.00           O
END
"""


FAKE_LIGAND_SDF = """TinyLigand
  MVP

  3  2  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    1.5000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
M  END
$$$$
"""


def create_raw_dataset(raw_root: Path = RAW_ROOT) -> None:
    for complex_id in COMPLEX_IDS:
        complex_dir = raw_root / complex_id
        complex_dir.mkdir(parents=True, exist_ok=True)

        (complex_dir / "protein.pdb").write_text(FAKE_PROTEIN_PDB, encoding="utf-8")
        (complex_dir / "ligand.sdf").write_text(FAKE_LIGAND_SDF, encoding="utf-8")
        (complex_dir / "ligand_gt.sdf").write_text(FAKE_LIGAND_SDF, encoding="utf-8")


def create_mini_split(split_path: Path = SPLIT_PATH) -> None:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text("\n".join(COMPLEX_IDS) + "\n", encoding="utf-8")


def create_manifest(
    ids_path: Path = SPLIT_PATH,
    raw_root: Path = RAW_ROOT,
    manifest_path: Path = MANIFEST_PATH,
) -> None:
    build_and_save_manifest(
        ids_path=ids_path,
        raw_root=raw_root,
        split="mini",
        output_path=manifest_path,
    )


def create_validation_report(
    manifest_path: Path = MANIFEST_PATH,
    report_path: Path = VALIDATION_REPORT_PATH,
) -> None:
    report = validate_manifest_file(manifest_path)
    save_json(report, report_path)


def run_baseline(exist_ok: bool) -> None:
    command = [
        sys.executable,
        "-m",
        "src.pipeline.run_baseline",
    ]

    if exist_ok:
        command.append("--exist-ok")

    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a tiny PDBBind-style MVP dataset and manifest."
    )
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="Run the baseline dry-run pipeline after creating the dataset.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow the baseline run directory to already exist.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    create_raw_dataset()
    create_mini_split()
    create_manifest()
    create_validation_report()

    print(
        f"Created tiny MVP PDBBind-style dataset with {len(COMPLEX_IDS)} complexes.",
        flush=True,
    )
    print(f"Wrote split file: {SPLIT_PATH}", flush=True)
    print(f"Wrote manifest: {MANIFEST_PATH}", flush=True)
    print(f"Wrote validation report: {VALIDATION_REPORT_PATH}", flush=True)

    if args.run_baseline:
        run_baseline(exist_ok=args.exist_ok)


if __name__ == "__main__":
    main()
