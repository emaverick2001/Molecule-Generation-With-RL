#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifests import save_manifest
from src.utils.artifact_logger import save_json
from src.utils.schemas import ComplexInput


DEFAULT_OUTPUT_MANIFEST = Path(
    "data/processed/diffdock/manifests/main_eval_manifest.json"
)
DEFAULT_OUTPUT_SPLIT = Path("data/processed/diffdock/splits/main_eval.txt")
DEFAULT_OUTPUT_PROTOCOL = Path(
    "data/processed/diffdock/manifests/main_eval_protocol.json"
)


def load_manifest(path: str | Path) -> list[ComplexInput]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Input manifest root must be a list: {path}")

    return [ComplexInput.from_dict(item) for item in data]


def with_split(records: list[ComplexInput], split: str) -> list[ComplexInput]:
    return [
        ComplexInput(
            complex_id=record.complex_id,
            protein_path=record.protein_path,
            ligand_path=record.ligand_path,
            ground_truth_pose_path=record.ground_truth_pose_path,
            split=split,
        )
        for record in records
    ]


def write_split_file(records: list[ComplexInput], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(record.complex_id for record in records) + "\n",
        encoding="utf-8",
    )


def _require_can_write(path: Path, exist_ok: bool) -> None:
    if path.exists() and not exist_ok:
        raise FileExistsError(f"Output already exists, pass --exist-ok: {path}")


def freeze_evaluation_protocol(
    *,
    input_manifest: str | Path,
    output_manifest: str | Path = DEFAULT_OUTPUT_MANIFEST,
    output_split: str | Path = DEFAULT_OUTPUT_SPLIT,
    output_protocol: str | Path = DEFAULT_OUTPUT_PROTOCOL,
    split: str = "main_eval",
    name: str = "main_eval",
    source_run_dir: str | Path | None = None,
    num_samples: int = 10,
    exist_ok: bool = False,
) -> dict:
    output_manifest = Path(output_manifest)
    output_split = Path(output_split)
    output_protocol = Path(output_protocol)

    for path in [output_manifest, output_split, output_protocol]:
        _require_can_write(path, exist_ok=exist_ok)

    records = with_split(load_manifest(input_manifest), split=split)
    save_manifest(records, output_manifest)
    write_split_file(records, output_split)

    protocol = {
        "name": name,
        "stage": split,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(source_run_dir) if source_run_dir else None,
        "source_input_manifest": str(input_manifest),
        "manifest_path": str(output_manifest),
        "split_path": str(output_split),
        "num_complexes": len(records),
        "num_samples_per_complex": num_samples,
        "metrics_policy": {
            "coverage": "generated complexes / attempted complexes",
            "generated_only_quality": [
                "success_at_1",
                "success_at_5",
                "success_at_10",
                "mean_rmsd",
                "median_rmsd",
                "best_of_n_mean_rmsd",
                "median_best_rmsd",
                "num_rmsd_gt_10",
                "fraction_rmsd_gt_10",
            ],
            "strict_attempted_set": [
                "strict_success_at_1",
                "strict_success_at_5",
                "strict_success_at_10",
            ],
        },
        "complex_ids": [record.complex_id for record in records],
    }
    save_json(protocol, output_protocol)
    return protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a fixed evaluation manifest for all main experiments.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--source-run-dir",
        help="Completed run directory containing input_manifest.json.",
    )
    source.add_argument(
        "--input-manifest",
        help="Manifest to freeze directly.",
    )
    parser.add_argument("--name", default="main_eval")
    parser.add_argument("--split", default="main_eval")
    parser.add_argument(
        "--output-manifest",
        default=str(DEFAULT_OUTPUT_MANIFEST),
    )
    parser.add_argument(
        "--output-split",
        default=str(DEFAULT_OUTPUT_SPLIT),
    )
    parser.add_argument(
        "--output-protocol",
        default=str(DEFAULT_OUTPUT_PROTOCOL),
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_manifest = (
        Path(args.source_run_dir) / "input_manifest.json"
        if args.source_run_dir
        else Path(args.input_manifest)
    )
    protocol = freeze_evaluation_protocol(
        input_manifest=input_manifest,
        output_manifest=args.output_manifest,
        output_split=args.output_split,
        output_protocol=args.output_protocol,
        split=args.split,
        name=args.name,
        source_run_dir=args.source_run_dir,
        num_samples=args.num_samples,
        exist_ok=args.exist_ok,
    )

    print("Frozen evaluation protocol:")
    print(f"  name:      {protocol['name']}")
    print(f"  complexes: {protocol['num_complexes']}")
    print(f"  manifest:  {protocol['manifest_path']}")
    print(f"  split:     {protocol['split_path']}")
    print(f"  protocol:  {args.output_protocol}")


if __name__ == "__main__":
    main()
