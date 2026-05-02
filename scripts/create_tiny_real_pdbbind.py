from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_pdbbind_smoke_complex import (
    DEFAULT_OUTPUT_ROOT,
    TINY_REAL_MANIFEST_PATH,
    TINY_REAL_SPLIT_PATH,
    TINY_REAL_VALIDATION_REPORT_PATH,
    _extract_archive,
    _is_archive,
    extract_complex_to_real_root,
    write_real_manifest,
)


def _parse_complex_ids(values: list[str] | None, ids_file: str | None) -> list[str]:
    complex_ids = []

    if values:
        complex_ids.extend(values)

    if ids_file:
        path = Path(ids_file)
        if not path.is_file():
            raise FileNotFoundError(f"Complex ID file not found: {path}")
        complex_ids.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    normalized = []
    seen = set()

    for complex_id in complex_ids:
        normalized_id = complex_id.lower()
        if normalized_id not in seen:
            normalized.append(normalized_id)
            seen.add(normalized_id)

    if not normalized:
        raise ValueError("Provide at least one complex ID via --complex-id or --ids-file")

    return normalized


def _prepare_from_source_root(
    source_root: Path,
    complex_ids: list[str],
    output_root: Path,
) -> list[Path]:
    return [
        extract_complex_to_real_root(
            source_root=source_root,
            complex_id=complex_id,
            output_root=output_root,
        )
        for complex_id in complex_ids
    ]


def create_tiny_real_pdbbind(
    source: Path,
    complex_ids: list[str],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> list[Path]:
    source = source.expanduser().resolve()
    output_root = output_root.expanduser()

    if not source.exists():
        raise FileNotFoundError(f"Source path not found: {source}")

    if source.is_file():
        if not _is_archive(source):
            raise ValueError(f"Source file is not a supported archive: {source}")

        with tempfile.TemporaryDirectory() as tmpdir:
            extracted_root = _extract_archive(source, Path(tmpdir))
            output_dirs = _prepare_from_source_root(
                source_root=extracted_root,
                complex_ids=complex_ids,
                output_root=output_root,
            )
    else:
        output_dirs = _prepare_from_source_root(
            source_root=source,
            complex_ids=complex_ids,
            output_root=output_root,
        )

    write_real_manifest(
        complex_ids=complex_ids,
        output_root=output_root,
        split="tiny_real",
        split_path=TINY_REAL_SPLIT_PATH,
        manifest_path=TINY_REAL_MANIFEST_PATH,
        validation_report_path=TINY_REAL_VALIDATION_REPORT_PATH,
    )

    return output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a tiny real PDBBind split and manifest from selected complex IDs."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Extracted PDBBind root directory or archive path.",
    )
    parser.add_argument(
        "--complex-id",
        action="append",
        default=[],
        help="Complex ID to include. Repeat for multiple IDs.",
    )
    parser.add_argument(
        "--ids-file",
        default=None,
        help="Optional text file with one complex ID per line.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root for normalized real PDBBind complexes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    complex_ids = _parse_complex_ids(
        values=args.complex_id,
        ids_file=args.ids_file,
    )
    output_dirs = create_tiny_real_pdbbind(
        source=Path(args.source),
        complex_ids=complex_ids,
        output_root=Path(args.output_root),
    )

    print(f"Prepared {len(output_dirs)} real PDBBind complexes:")
    for output_dir in output_dirs:
        print(f"  {output_dir}")
    print(f"Wrote split file: {TINY_REAL_SPLIT_PATH}")
    print(f"Wrote manifest: {TINY_REAL_MANIFEST_PATH}")
    print(f"Wrote validation report: {TINY_REAL_VALIDATION_REPORT_PATH}")
    print(
        "Run baseline with:\n"
        "  uv run python -m src.pipeline.run_baseline "
        "--config configs/diffdock/tiny_real.yaml"
    )


if __name__ == "__main__":
    main()
