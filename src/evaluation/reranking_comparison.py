from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from src.evaluation.metrics import PoseMetricRecord
from src.utils.schemas import GeneratedPose


@dataclass(frozen=True)
class RerankingComparisonRecord:
    complex_id: str
    num_valid_poses: int
    original_top_sample_id: int
    original_top_rmsd: float
    original_top_success: bool
    confidence_top_sample_id: int | None
    confidence_top_rmsd: float | None
    confidence_top_score: float | None
    confidence_top_success: bool | None
    oracle_sample_id: int
    oracle_original_rank: int
    oracle_rmsd: float
    oracle_success: bool
    confidence_selected_oracle: bool | None
    confidence_improved_over_original: bool | None
    confidence_delta_rmsd_vs_original: float | None
    oracle_delta_rmsd_vs_original: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _group_metrics_by_complex(
    metric_records: list[PoseMetricRecord],
) -> dict[str, list[PoseMetricRecord]]:
    grouped = defaultdict(list)

    for record in metric_records:
        if record.valid and record.rmsd is not None:
            grouped[record.complex_id].append(record)

    return {
        complex_id: sorted(records, key=lambda record: record.rank)
        for complex_id, records in grouped.items()
    }


def _generated_lookup(
    generated_records: list[GeneratedPose],
) -> dict[tuple[str, int], GeneratedPose]:
    return {
        (record.complex_id, record.sample_id): record for record in generated_records
    }


def _metric_lookup(
    metric_records: list[PoseMetricRecord],
) -> dict[tuple[str, int], PoseMetricRecord]:
    return {
        (record.complex_id, record.sample_id): record
        for record in metric_records
        if record.valid and record.rmsd is not None
    }


def compare_reranking_strategies(
    metric_records: list[PoseMetricRecord],
    generated_records: list[GeneratedPose],
    success_threshold: float = 2.0,
) -> list[RerankingComparisonRecord]:
    grouped_metrics = _group_metrics_by_complex(metric_records)
    generated_by_key = _generated_lookup(generated_records)
    metric_by_key = _metric_lookup(metric_records)
    comparison_records = []

    for complex_id, records in grouped_metrics.items():
        original_top = records[0]
        oracle = min(records, key=lambda record: record.rmsd)
        generated_candidates = [
            generated_by_key[(record.complex_id, record.sample_id)]
            for record in records
            if (record.complex_id, record.sample_id) in generated_by_key
            and generated_by_key[(record.complex_id, record.sample_id)].confidence_score
            is not None
        ]
        confidence_top = (
            max(
                generated_candidates,
                key=lambda record: (record.confidence_score, -record.sample_id),
            )
            if generated_candidates
            else None
        )
        confidence_top_metric = (
            metric_by_key.get((confidence_top.complex_id, confidence_top.sample_id))
            if confidence_top is not None
            else None
        )
        confidence_top_rmsd = (
            confidence_top_metric.rmsd
            if confidence_top_metric is not None
            else None
        )

        comparison_records.append(
            RerankingComparisonRecord(
                complex_id=complex_id,
                num_valid_poses=len(records),
                original_top_sample_id=original_top.sample_id,
                original_top_rmsd=original_top.rmsd,
                original_top_success=original_top.rmsd < success_threshold,
                confidence_top_sample_id=(
                    confidence_top.sample_id if confidence_top is not None else None
                ),
                confidence_top_rmsd=confidence_top_rmsd,
                confidence_top_score=(
                    confidence_top.confidence_score
                    if confidence_top is not None
                    else None
                ),
                confidence_top_success=(
                    confidence_top_rmsd < success_threshold
                    if confidence_top_rmsd is not None
                    else None
                ),
                oracle_sample_id=oracle.sample_id,
                oracle_original_rank=oracle.rank,
                oracle_rmsd=oracle.rmsd,
                oracle_success=oracle.rmsd < success_threshold,
                confidence_selected_oracle=(
                    confidence_top.sample_id == oracle.sample_id
                    if confidence_top is not None
                    else None
                ),
                confidence_improved_over_original=(
                    confidence_top_rmsd < original_top.rmsd
                    if confidence_top_rmsd is not None
                    else None
                ),
                confidence_delta_rmsd_vs_original=(
                    round(confidence_top_rmsd - original_top.rmsd, 6)
                    if confidence_top_rmsd is not None
                    else None
                ),
                oracle_delta_rmsd_vs_original=round(
                    oracle.rmsd - original_top.rmsd,
                    6,
                ),
            )
        )

    return comparison_records


def summarize_reranking_comparison(
    records: list[RerankingComparisonRecord],
) -> dict[str, Any]:
    confidence_records = [
        record for record in records if record.confidence_top_rmsd is not None
    ]
    original_rmsds = [record.original_top_rmsd for record in records]
    confidence_rmsds = [
        record.confidence_top_rmsd for record in confidence_records
        if record.confidence_top_rmsd is not None
    ]
    oracle_rmsds = [record.oracle_rmsd for record in records]

    return {
        "num_complexes": len(records),
        "num_with_confidence": len(confidence_records),
        "original_top1_mean_rmsd": (
            round(mean(original_rmsds), 6) if original_rmsds else None
        ),
        "confidence_top1_mean_rmsd": (
            round(mean(confidence_rmsds), 6) if confidence_rmsds else None
        ),
        "oracle_best_of_n_mean_rmsd": (
            round(mean(oracle_rmsds), 6) if oracle_rmsds else None
        ),
        "original_success_at_1": (
            round(
                sum(record.original_top_success for record in records) / len(records),
                6,
            )
            if records
            else None
        ),
        "confidence_success_at_1": (
            round(
                sum(
                    record.confidence_top_success is True
                    for record in confidence_records
                )
                / len(confidence_records),
                6,
            )
            if confidence_records
            else None
        ),
        "oracle_success_at_1": (
            round(sum(record.oracle_success for record in records) / len(records), 6)
            if records
            else None
        ),
        "confidence_selected_oracle_rate": (
            round(
                sum(
                    record.confidence_selected_oracle is True
                    for record in confidence_records
                )
                / len(confidence_records),
                6,
            )
            if confidence_records
            else None
        ),
        "confidence_improved_over_original_rate": (
            round(
                sum(
                    record.confidence_improved_over_original is True
                    for record in confidence_records
                )
                / len(confidence_records),
                6,
            )
            if confidence_records
            else None
        ),
    }


def save_reranking_comparison_csv(
    records: list[RerankingComparisonRecord],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RerankingComparisonRecord.__dataclass_fields__.keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record.to_dict() for record in records)
