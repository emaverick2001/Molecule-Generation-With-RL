from __future__ import annotations

import argparse
import os
import sys
import tarfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_pdbbind_smoke_complex import extract_smoke_complex


DEFAULT_SEARCH_ROOTS = [
    Path("~/datasets/pdbbind"),
    Path("~/datasets"),
    Path("~/Downloads"),
    PROJECT_ROOT / "data" / "downloads",
]


def _is_archive(path: Path) -> bool:
    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)


def _iter_existing_roots(paths: list[Path]):
    for path in paths:
        resolved = path.expanduser()
        if resolved.exists():
            yield resolved


def _find_complex_dirs(search_roots: list[Path], complex_id: str) -> list[Path]:
    complex_id = complex_id.lower()
    matches = []

    for root in _iter_existing_roots(search_roots):
        if root.is_dir():
            matches.extend(
                path
                for path in root.rglob("*")
                if path.is_dir() and path.name.lower() == complex_id
            )

    return sorted(set(matches))


def _find_archives(search_roots: list[Path]) -> list[Path]:
    matches = []

    for root in _iter_existing_roots(search_roots):
        if root.is_file() and _is_archive(root):
            matches.append(root)
        elif root.is_dir():
            matches.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and (
                    path.name.endswith(".zip")
                    or ".tar" in "".join(path.suffixes)
                )
            )

    return sorted(set(matches))


def _choose_path(paths: list[Path], prompt: str) -> Path | None:
    if not paths:
        return None

    print(prompt)
    for index, path in enumerate(paths, start=1):
        print(f"  {index}. {path}")

    while True:
        choice = input("Choose a number, or press Enter to skip: ").strip()

        if not choice:
            return None

        try:
            selected_index = int(choice)
        except ValueError:
            print("Please enter a number.")
            continue

        if 1 <= selected_index <= len(paths):
            return paths[selected_index - 1]

        print(f"Please choose a number from 1 to {len(paths)}.")


def _print_download_instructions() -> None:
    print(
        "\nNo local PDBBind complex/archive was found.\n\n"
        "Download a PDBBind protein-ligand package from:\n"
        "  https://www.pdbbind.org.cn/\n\n"
        "Recommended local location on ICRN:\n"
        "  ~/datasets/pdbbind/\n\n"
        "After downloading/extracting, rerun for example:\n"
        "  uv run python scripts/setup_pdbbind_smoke_complex.py --complex-id 1a30\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find or extract one real PDBBind complex and create the smoke "
            "manifest used by DiffDock."
        )
    )
    parser.add_argument(
        "--complex-id",
        required=True,
        help="PDBBind/PDB complex ID to prepare, for example 1a30.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional explicit extracted PDBBind root or archive path.",
    )
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Additional root directory/archive to search. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-root",
        default="data/raw/pdbbind_real",
        help="Output root for the normalized real smoke complex.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Do not prompt; fail with instructions if no clear source is found.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    if args.source is not None:
        source = Path(args.source).expanduser()
    else:
        search_roots = [Path(path) for path in args.search_root] + DEFAULT_SEARCH_ROOTS
        complex_dirs = _find_complex_dirs(search_roots, args.complex_id)

        if len(complex_dirs) == 1:
            source = complex_dirs[0].parent
        elif len(complex_dirs) > 1 and not args.non_interactive:
            selected_complex_dir = _choose_path(
                complex_dirs,
                "Found matching complex directories:",
            )
            source = selected_complex_dir.parent if selected_complex_dir else None
        else:
            source = None

        if source is None:
            archives = _find_archives(search_roots)

            if len(archives) == 1:
                source = archives[0]
            elif len(archives) > 1 and not args.non_interactive:
                source = _choose_path(
                    archives,
                    "Found candidate PDBBind archives:",
                )

    if source is None:
        _print_download_instructions()
        raise SystemExit(1)

    output_dir = extract_smoke_complex(
        source=source,
        complex_id=args.complex_id,
        output_root=Path(args.output_root),
    )

    print(f"Prepared real PDBBind smoke complex: {output_dir}")
    print(
        "Run DiffDock smoke test with:\n"
        f"  SMOKE_COMPLEX_ID={args.complex_id.lower()} ./scripts/run_diffdock_smoke.sh"
    )


if __name__ == "__main__":
    main()
