#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.structure_diagnostics import run_structure_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report structure sanity diagnostics for one generated run.",
    )
    parser.add_argument("run_dir", help="Run directory to diagnose.")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "CSV output path. Default: "
            "<run_dir>/structure_diagnostics.csv"
        ),
    )
    parser.add_argument(
        "--keep-hs",
        action="store_true",
        help="Keep hydrogens when counting SDF atoms and centroids.",
    )
    parser.add_argument(
        "--centroid-warning-threshold",
        type=float,
        default=10.0,
        help="Warn when generated/reference centroid distance exceeds this value.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_path = (
        Path(args.output)
        if args.output is not None
        else run_dir / "structure_diagnostics.csv"
    )
    records = run_structure_diagnostics(
        run_dir=run_dir,
        output_csv_path=output_path,
        remove_hs=not args.keep_hs,
        generated_centroid_warning_threshold=args.centroid_warning_threshold,
    )

    fieldnames = list(records[0].to_dict().keys()) if records else []
    if fieldnames:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record.to_dict() for record in records)

    print(f"Wrote structure diagnostics: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
