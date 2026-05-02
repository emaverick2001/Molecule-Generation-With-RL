from __future__ import annotations

from math import exp

from src.utils.schemas import GeneratedPose, RewardRecord


def require_confidence_scores(poses: list[GeneratedPose]) -> None:
    missing = [
        (pose.complex_id, pose.sample_id)
        for pose in poses
        if pose.confidence_score is None
    ]

    if missing:
        preview = ", ".join(
            f"{complex_id}:{sample_id}" for complex_id, sample_id in missing[:5]
        )
        raise ValueError(f"Missing confidence_score for generated poses: {preview}")


def transform_confidence_score(
    score: float,
    transform: str = "identity",
    temperature: float = 1.0,
) -> float:
    if transform == "identity":
        return float(score)

    if transform == "sigmoid":
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")
        return 1.0 / (1.0 + exp(-float(score) / temperature))

    raise ValueError(f"Unsupported confidence transform: {transform}")


def build_confidence_reward_records(
    poses: list[GeneratedPose],
    transform: str = "identity",
    temperature: float = 1.0,
) -> list[RewardRecord]:
    require_confidence_scores(poses)

    return [
        RewardRecord(
            complex_id=pose.complex_id,
            sample_id=pose.sample_id,
            reward=transform_confidence_score(
                score=pose.confidence_score,
                transform=transform,
                temperature=temperature,
            ),
            reward_type="confidence",
            valid=True,
        )
        for pose in poses
    ]
