from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from src.data.loaders import load_complex_manifest
from src.rl.types import RLExample, RolloutRecord
from src.rl.utils import read_jsonl, write_jsonl
from src.utils.schemas import ComplexInput, GeneratedPose


def load_generated_samples_manifest(path: str | Path) -> list[GeneratedPose]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Generated samples manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Generated samples manifest root must be a list: {path}")

    return [GeneratedPose.from_dict(item) for item in data]


def export_complexes_to_diffdock_csv(
    records: Sequence[ComplexInput],
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["complex_name", "protein_path", "ligand_description"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "complex_name": record.complex_id,
                    "protein_path": record.protein_path,
                    "ligand_description": record.ligand_path,
                }
            )

    return out_path


def join_samples_with_complex_manifest(
    samples: Sequence[GeneratedPose],
    complexes: Sequence[ComplexInput],
    *,
    source_run_id: str | None = None,
    source_checkpoint: str | None = None,
) -> list[RLExample]:
    complexes_by_id = {record.complex_id: record for record in complexes}
    examples = []

    for sample in samples:
        complex_record = complexes_by_id.get(sample.complex_id)
        if complex_record is None:
            raise ValueError(
                "Generated sample references unknown complex_id: "
                f"{sample.complex_id}"
            )

        examples.append(
            RLExample(
                complex_id=sample.complex_id,
                protein_path=complex_record.protein_path,
                ligand_input_path=complex_record.ligand_path,
                predicted_pose_path=sample.pose_path,
                ground_truth_pose_path=complex_record.ground_truth_pose_path,
                sample_rank=sample.sample_id + 1,
                sample_id=sample.sample_id,
                confidence_score=sample.confidence_score,
                source_run_id=source_run_id,
                source_checkpoint=source_checkpoint,
                metadata={"split": complex_record.split},
            )
        )

    return examples


def group_examples_by_complex(
    examples: Sequence[RLExample],
    expected_group_size: int | None = None,
) -> dict[str, list[RLExample]]:
    groups: dict[str, list[RLExample]] = defaultdict(list)
    for example in examples:
        groups[example.complex_id].append(example)

    grouped = {key: sorted(value, key=lambda item: item.sample_rank) for key, value in groups.items()}

    if expected_group_size is not None:
        short_groups = [
            group_id
            for group_id, group_examples in grouped.items()
            if len(group_examples) != expected_group_size
        ]
        if short_groups:
            preview = ", ".join(short_groups[:5])
            raise ValueError(
                "Unexpected rollout group size for complexes: "
                f"{preview}; expected {expected_group_size}"
            )

    return grouped


def write_rollout_manifest(
    records: Sequence[RolloutRecord],
    out_path: str | Path,
) -> None:
    write_jsonl((record.to_dict() for record in records), out_path)


def load_rollout_manifest(path: str | Path) -> list[RolloutRecord]:
    return [RolloutRecord.from_dict(row) for row in read_jsonl(path)]


def load_offline_rl_examples(
    input_manifest: str | Path,
    generated_manifest: str | Path,
    *,
    source_run_id: str | None = None,
) -> list[RLExample]:
    complexes = load_complex_manifest(input_manifest, validate=False)
    samples = load_generated_samples_manifest(generated_manifest)
    return join_samples_with_complex_manifest(
        samples,
        complexes,
        source_run_id=source_run_id,
    )
