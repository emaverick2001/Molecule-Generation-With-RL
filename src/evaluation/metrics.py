from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.evaluation.rmsd import compute_centroid_distance, compute_symmetry_corrected_rmsd
from src.utils.schemas import ComplexInput, GeneratedPose


@dataclass(frozen=True)
class PoseMetricRecord:
    complex_id: str
    sample_id: int
    rank: int
    pose_path: str
    reference_pose_path: str
    rmsd: float | None
    centroid_distance: float | None
    rmsd_below_2: bool
    rmsd_below_5: bool
    valid: bool
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PoseMetricRecord":
        return cls(
            complex_id=data["complex_id"],
            sample_id=int(data["sample_id"]),
            rank=int(data["rank"]),
            pose_path=data["pose_path"],
            reference_pose_path=data["reference_pose_path"],
            rmsd=float(data["rmsd"]) if data.get("rmsd") not in [None, ""] else None,
            centroid_distance=(
                float(data["centroid_distance"])
                if data.get("centroid_distance") not in [None, ""]
                else None
            ),
            rmsd_below_2=_parse_bool(data["rmsd_below_2"]),
            rmsd_below_5=_parse_bool(data["rmsd_below_5"]),
            valid=_parse_bool(data["valid"]),
            error=data.get("error") or None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def evaluate_generated_poses(
    input_records: list[ComplexInput],
    generated_records: list[GeneratedPose],
    rmsd_thresholds: tuple[float, float] = (2.0, 5.0),
    remove_hs: bool = True,
) -> list[PoseMetricRecord]:
    reference_by_complex_id = {
        record.complex_id: record.ground_truth_pose_path for record in input_records
    }
    rank_by_complex_id = defaultdict(int)
    records = []

    for pose in generated_records:
        reference_pose_path = reference_by_complex_id.get(pose.complex_id)
        rank_by_complex_id[pose.complex_id] += 1
        rank = rank_by_complex_id[pose.complex_id]

        if reference_pose_path is None:
            records.append(
                PoseMetricRecord(
                    complex_id=pose.complex_id,
                    sample_id=pose.sample_id,
                    rank=rank,
                    pose_path=pose.pose_path,
                    reference_pose_path="",
                    rmsd=None,
                    centroid_distance=None,
                    rmsd_below_2=False,
                    rmsd_below_5=False,
                    valid=False,
                    error=f"Unknown complex_id: {pose.complex_id}",
                )
            )
            continue

        try:
            rmsd = compute_symmetry_corrected_rmsd(
                predicted_pose_path=pose.pose_path,
                reference_pose_path=reference_pose_path,
                remove_hs=remove_hs,
            )
            centroid_distance = compute_centroid_distance(
                predicted_pose_path=pose.pose_path,
                reference_pose_path=reference_pose_path,
                remove_hs=remove_hs,
            )
            records.append(
                PoseMetricRecord(
                    complex_id=pose.complex_id,
                    sample_id=pose.sample_id,
                    rank=rank,
                    pose_path=pose.pose_path,
                    reference_pose_path=reference_pose_path,
                    rmsd=round(rmsd, 6),
                    centroid_distance=round(centroid_distance, 6),
                    rmsd_below_2=rmsd < rmsd_thresholds[0],
                    rmsd_below_5=rmsd < rmsd_thresholds[1],
                    valid=True,
                )
            )
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            records.append(
                PoseMetricRecord(
                    complex_id=pose.complex_id,
                    sample_id=pose.sample_id,
                    rank=rank,
                    pose_path=pose.pose_path,
                    reference_pose_path=reference_pose_path,
                    rmsd=None,
                    centroid_distance=None,
                    rmsd_below_2=False,
                    rmsd_below_5=False,
                    valid=False,
                    error=str(error),
                )
            )

    return records


def _success_at_k(records: list[PoseMetricRecord], k: int, threshold: float) -> float | None:
    eligible_complexes = [
        complex_records
        for complex_records in _group_valid_by_complex(records).values()
        if len(complex_records) >= k
    ]

    if not eligible_complexes:
        return None

    successes = [
        any(record.rmsd is not None and record.rmsd < threshold for record in complex_records[:k])
        for complex_records in eligible_complexes
    ]

    return round(sum(successes) / len(successes), 6)


def _strict_success_at_k(
    records: list[PoseMetricRecord],
    *,
    attempted_complex_ids: list[str],
    k: int,
    threshold: float,
) -> float | None:
    if not attempted_complex_ids:
        return None

    grouped = _group_valid_by_complex(records)
    successes = []

    for complex_id in attempted_complex_ids:
        complex_records = grouped.get(complex_id, [])
        successes.append(
            len(complex_records) >= k
            and any(
                record.rmsd is not None and record.rmsd < threshold
                for record in complex_records[:k]
            )
        )

    return round(sum(successes) / len(attempted_complex_ids), 6)


def _group_valid_by_complex(
    records: list[PoseMetricRecord],
) -> dict[str, list[PoseMetricRecord]]:
    grouped = defaultdict(list)

    for record in records:
        if record.valid:
            grouped[record.complex_id].append(record)

    return {
        complex_id: sorted(complex_records, key=lambda record: record.rank)
        for complex_id, complex_records in grouped.items()
    }


def aggregate_topk_metrics(
    metric_records: list[PoseMetricRecord],
    top_k: list[int] | None = None,
    success_threshold: float = 2.0,
    attempted_complex_ids: list[str] | None = None,
) -> dict[str, Any]:
    top_k = top_k or [1, 5, 10]
    attempted_complex_ids = attempted_complex_ids or []
    generated_complex_ids = sorted({record.complex_id for record in metric_records})
    missing_generated_complex_ids = sorted(
        set(attempted_complex_ids) - set(generated_complex_ids)
    )
    valid_records = [record for record in metric_records if record.valid and record.rmsd is not None]
    invalid_records = [record for record in metric_records if not record.valid]
    grouped = _group_valid_by_complex(metric_records)
    rmsd_values = [record.rmsd for record in valid_records if record.rmsd is not None]
    centroid_values = [
        record.centroid_distance
        for record in valid_records
        if record.centroid_distance is not None
    ]
    best_rmsd_by_complex = [
        min(record.rmsd for record in complex_records if record.rmsd is not None)
        for complex_records in grouped.values()
    ]
    outlier_threshold = 10.0
    num_rmsd_gt_10 = sum(
        record.rmsd is not None and record.rmsd > outlier_threshold
        for record in valid_records
    )

    aggregate = {
        "num_attempted_complexes": (
            len(attempted_complex_ids) if attempted_complex_ids else None
        ),
        "num_complexes": len({record.complex_id for record in metric_records}),
        "num_generated_complexes": len(generated_complex_ids),
        "generation_coverage": (
            round(len(generated_complex_ids) / len(attempted_complex_ids), 6)
            if attempted_complex_ids
            else None
        ),
        "num_missing_generated_complexes": (
            len(missing_generated_complex_ids) if attempted_complex_ids else None
        ),
        "missing_generated_complexes": (
            missing_generated_complex_ids if attempted_complex_ids else []
        ),
        "num_valid_complexes": len(grouped),
        "num_poses": len(metric_records),
        "num_valid_poses": len(valid_records),
        "num_invalid_poses": len(invalid_records),
        "mean_rmsd": round(mean(rmsd_values), 6) if rmsd_values else None,
        "median_rmsd": round(median(rmsd_values), 6) if rmsd_values else None,
        "mean_centroid_distance": (
            round(mean(centroid_values), 6) if centroid_values else None
        ),
        "best_of_n_mean_rmsd": (
            round(mean(best_rmsd_by_complex), 6) if best_rmsd_by_complex else None
        ),
        "median_best_rmsd": (
            round(median(best_rmsd_by_complex), 6) if best_rmsd_by_complex else None
        ),
        "num_rmsd_gt_10": num_rmsd_gt_10,
        "fraction_rmsd_gt_10": (
            round(num_rmsd_gt_10 / len(valid_records), 6) if valid_records else None
        ),
    }

    for k in top_k:
        aggregate[f"success_at_{k}"] = _success_at_k(
            metric_records,
            k=k,
            threshold=success_threshold,
        )
        aggregate[f"strict_success_at_{k}"] = _strict_success_at_k(
            metric_records,
            attempted_complex_ids=attempted_complex_ids,
            k=k,
            threshold=success_threshold,
        )

    return aggregate


def save_pose_metrics_csv(records: list[PoseMetricRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(PoseMetricRecord.__dataclass_fields__.keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record.to_dict() for record in records)


def load_pose_metrics_csv(path: str | Path) -> list[PoseMetricRecord]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Pose metrics CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        return [PoseMetricRecord.from_dict(row) for row in csv.DictReader(f)]
