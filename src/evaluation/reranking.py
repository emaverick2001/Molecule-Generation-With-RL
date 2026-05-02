from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from src.utils.artifact_logger import read_json, save_records_json
from src.utils.schemas import GeneratedPose, RewardRecord


@dataclass(frozen=True)
class RerankedPose:
    complex_id: str
    sample_id: int
    pose_path: str
    original_rank: int
    reranked_rank: int
    reward: float
    reward_type: str
    confidence_score: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RerankedPose":
        return cls(
            complex_id=data["complex_id"],
            sample_id=int(data["sample_id"]),
            pose_path=data["pose_path"],
            original_rank=int(data["original_rank"]),
            reranked_rank=int(data["reranked_rank"]),
            reward=float(data["reward"]),
            reward_type=data["reward_type"],
            confidence_score=(
                float(data["confidence_score"])
                if data.get("confidence_score") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_reward_lookup(
    reward_records: list[RewardRecord],
) -> dict[tuple[str, int], RewardRecord]:
    lookup = {}

    for reward_record in reward_records:
        key = (reward_record.complex_id, reward_record.sample_id)

        if key in lookup:
            raise ValueError(
                f"Duplicate reward row for complex_id/sample_id: {key[0]}, {key[1]}"
            )

        if not reward_record.valid:
            raise ValueError(
                f"Invalid reward row cannot be used for reranking: {key[0]}, {key[1]}"
            )

        lookup[key] = reward_record

    return lookup


def _tie_break_value(pose: GeneratedPose, original_rank: int, tie_breaker: str) -> int:
    if tie_breaker == "sample_id":
        return pose.sample_id
    if tie_breaker == "original_rank":
        return original_rank
    raise ValueError(f"Unsupported tie_breaker: {tie_breaker}")


def rerank_generated_poses(
    generated_records: list[GeneratedPose],
    reward_records: list[RewardRecord],
    descending: bool = True,
    tie_breaker: str = "sample_id",
) -> list[RerankedPose]:
    reward_lookup = build_reward_lookup(reward_records)
    grouped_poses = defaultdict(list)
    complex_order = []

    for pose in generated_records:
        if pose.complex_id not in grouped_poses:
            complex_order.append(pose.complex_id)
        grouped_poses[pose.complex_id].append(pose)

    reranked_records = []

    for complex_id in complex_order:
        original_entries = [
            (original_rank, pose)
            for original_rank, pose in enumerate(grouped_poses[complex_id], start=1)
        ]

        for _, pose in original_entries:
            key = (pose.complex_id, pose.sample_id)
            if key not in reward_lookup:
                raise ValueError(
                    "Missing reward row for generated pose: "
                    f"{pose.complex_id}, {pose.sample_id}"
                )

        sorted_entries = sorted(
            original_entries,
            key=lambda item: (
                -reward_lookup[(item[1].complex_id, item[1].sample_id)].reward
                if descending
                else reward_lookup[(item[1].complex_id, item[1].sample_id)].reward,
                _tie_break_value(item[1], item[0], tie_breaker),
            ),
        )

        for reranked_rank, (original_rank, pose) in enumerate(sorted_entries, start=1):
            reward_record = reward_lookup[(pose.complex_id, pose.sample_id)]
            reranked_records.append(
                RerankedPose(
                    complex_id=pose.complex_id,
                    sample_id=pose.sample_id,
                    pose_path=pose.pose_path,
                    original_rank=original_rank,
                    reranked_rank=reranked_rank,
                    reward=reward_record.reward,
                    reward_type=reward_record.reward_type,
                    confidence_score=pose.confidence_score,
                )
            )

    return reranked_records


def summarize_reranking(reranked_records: list[RerankedPose]) -> dict[str, Any]:
    rank_deltas = [
        abs(record.original_rank - record.reranked_rank)
        for record in reranked_records
    ]
    changed_rank_count = sum(delta > 0 for delta in rank_deltas)

    return {
        "num_complexes": len({record.complex_id for record in reranked_records}),
        "num_poses": len(reranked_records),
        "num_rank_changed": changed_rank_count,
        "mean_absolute_rank_delta": (
            round(mean(rank_deltas), 6) if rank_deltas else None
        ),
    }


def save_reranked_manifest(records: list[RerankedPose], path: str | Path) -> None:
    save_records_json(records, path)


def load_reranked_manifest(path: str | Path) -> list[RerankedPose]:
    return [RerankedPose.from_dict(item) for item in read_json(path)]
