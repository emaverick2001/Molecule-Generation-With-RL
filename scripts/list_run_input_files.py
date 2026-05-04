#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


INPUT_FIELDS = ("protein_path", "ligand_path", "ground_truth_pose_path")
MANIFEST_CANDIDATES = (
    "input_manifest.json",
    "input_train_manifest.json",
    "input_val_manifest.json",
)


def _repo_relative_path(path_text: str, repo_root: Path) -> str:
    path = Path(path_text).expanduser()

    if path.is_absolute():
        try:
            path = path.resolve().relative_to(repo_root.resolve())
        except ValueError as error:
            raise ValueError(
                f"Input path is outside the repository and cannot be packaged "
                f"with a repo-relative archive path: {path_text}"
            ) from error

    return path.as_posix()


def list_run_input_files(run_dir: str | Path, repo_root: str | Path = ".") -> list[str]:
    run_dir = Path(run_dir)
    repo_root = Path(repo_root)

    paths: list[str] = []
    seen: set[str] = set()
    manifest_paths = [
        run_dir / filename
        for filename in MANIFEST_CANDIDATES
        if (run_dir / filename).is_file()
    ]

    if not manifest_paths:
        candidates = ", ".join(MANIFEST_CANDIDATES)
        raise FileNotFoundError(
            f"No input manifest found in {run_dir}; expected one of: {candidates}"
        )

    for manifest_path in manifest_paths:
        with manifest_path.open("r", encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError(f"Input manifest must contain a list: {manifest_path}")

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Manifest record {index} is not an object.")

            for field in INPUT_FIELDS:
                if field not in record:
                    raise ValueError(f"Manifest record {index} is missing {field}.")

                relative_path = _repo_relative_path(str(record[field]), repo_root)
                path = repo_root / relative_path

                if not path.is_file():
                    raise FileNotFoundError(f"{field} not found: {relative_path}")

                if relative_path not in seen:
                    seen.add(relative_path)
                    paths.append(relative_path)

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List input structure files referenced by a run manifest.",
    )
    parser.add_argument("run_dir", help="Run directory containing input_manifest.json.")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used to validate and relativize paths.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for path in list_run_input_files(args.run_dir, repo_root=args.repo_root):
        print(path)


if __name__ == "__main__":
    main()
