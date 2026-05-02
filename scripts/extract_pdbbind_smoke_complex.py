from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifests import build_and_save_manifest
from src.data.validation import validate_manifest_file
from src.utils.artifact_logger import save_json


DEFAULT_OUTPUT_ROOT = Path("data/raw/pdbbind_real")
SMOKE_SPLIT_PATH = Path("data/processed/diffdock/splits/smoke.txt")
SMOKE_MANIFEST_PATH = Path("data/processed/diffdock/manifests/smoke_manifest.json")
SMOKE_VALIDATION_REPORT_PATH = Path(
    "data/processed/diffdock/manifests/smoke_validation_report.json"
)


def _is_archive(path: Path) -> bool:
    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)


def _extract_archive(archive_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
        return destination

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            archive.extractall(destination)
        return destination

    raise ValueError(f"Unsupported archive format: {archive_path}")


def _find_complex_dir(source_root: Path, complex_id: str) -> Path:
    exact_matches = [
        source_root / complex_id,
        source_root / complex_id.lower(),
        source_root / complex_id.upper(),
    ]

    for path in exact_matches:
        if path.is_dir():
            return path

    matches = [
        path
        for path in source_root.rglob("*")
        if path.is_dir() and path.name.lower() == complex_id.lower()
    ]

    if not matches:
        raise FileNotFoundError(
            f"Could not find complex directory for {complex_id} under {source_root}"
        )

    return matches[0]


def _find_file(complex_dir: Path, candidates: list[str], suffix: str) -> Path:
    for candidate in candidates:
        path = complex_dir / candidate
        if path.is_file():
            return path

    matches = sorted(
        path
        for path in complex_dir.iterdir()
        if path.is_file() and path.suffix.lower() == suffix
    )

    if not matches:
        raise FileNotFoundError(
            f"Could not find required {suffix} file in {complex_dir}"
        )

    return matches[0]


def _copy_complex(complex_dir: Path, complex_id: str, output_root: Path) -> Path:
    output_dir = output_root / complex_id
    output_dir.mkdir(parents=True, exist_ok=True)

    protein_path = _find_file(
        complex_dir,
        candidates=[
            f"{complex_id}_protein.pdb",
            f"{complex_id.lower()}_protein.pdb",
            "protein.pdb",
        ],
        suffix=".pdb",
    )
    ligand_path = _find_file(
        complex_dir,
        candidates=[
            f"{complex_id}_ligand.sdf",
            f"{complex_id.lower()}_ligand.sdf",
            "ligand.sdf",
        ],
        suffix=".sdf",
    )

    shutil.copyfile(protein_path, output_dir / "protein.pdb")
    shutil.copyfile(ligand_path, output_dir / "ligand.sdf")
    shutil.copyfile(ligand_path, output_dir / "ligand_gt.sdf")

    return output_dir


def _write_smoke_manifest(complex_id: str, output_root: Path) -> None:
    SMOKE_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_SPLIT_PATH.write_text(f"{complex_id}\n", encoding="utf-8")

    build_and_save_manifest(
        ids_path=SMOKE_SPLIT_PATH,
        raw_root=output_root,
        split="smoke",
        output_path=SMOKE_MANIFEST_PATH,
    )

    report = validate_manifest_file(SMOKE_MANIFEST_PATH)
    save_json(report, SMOKE_VALIDATION_REPORT_PATH)


def extract_smoke_complex(
    source: Path,
    complex_id: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    complex_id = complex_id.lower()
    source = source.expanduser().resolve()
    output_root = output_root.expanduser()

    if not source.exists():
        raise FileNotFoundError(f"Source path not found: {source}")

    if source.is_file():
        if not _is_archive(source):
            raise ValueError(f"Source file is not a supported archive: {source}")

        with tempfile.TemporaryDirectory() as tmpdir:
            extracted_root = _extract_archive(source, Path(tmpdir))
            complex_dir = _find_complex_dir(extracted_root, complex_id)
            output_dir = _copy_complex(complex_dir, complex_id, output_root)
    else:
        complex_dir = _find_complex_dir(source, complex_id)
        output_dir = _copy_complex(complex_dir, complex_id, output_root)

    _write_smoke_manifest(complex_id, output_root)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one real PDBBind complex into data/raw/pdbbind_real and "
            "write a smoke manifest."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Path to an extracted PDBBind root directory or a zip/tar archive "
            "containing PDBBind complex folders."
        ),
    )
    parser.add_argument(
        "--complex-id",
        required=True,
        help="PDBBind/PDB complex ID to extract, for example 1a30.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root for real smoke-test complexes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    output_dir = extract_smoke_complex(
        source=Path(args.source),
        complex_id=args.complex_id,
        output_root=Path(args.output_root),
    )

    print(f"Extracted real PDBBind smoke complex: {output_dir}", flush=True)
    print(f"Wrote split file: {SMOKE_SPLIT_PATH}", flush=True)
    print(f"Wrote manifest: {SMOKE_MANIFEST_PATH}", flush=True)
    print(f"Wrote validation report: {SMOKE_VALIDATION_REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
