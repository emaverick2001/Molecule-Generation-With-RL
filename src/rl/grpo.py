from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp
from pathlib import Path
from statistics import mean
from typing import Sequence

from src.rl.types import RolloutRecord
from src.utils.artifact_logger import read_json, save_json


FEATURE_NAMES = ("bias", "confidence", "inverse_rank")


@dataclass(frozen=True)
class LinearSurrogateState:
    weights: dict[str, float]

    @classmethod
    def initialized(cls) -> "LinearSurrogateState":
        return cls(weights={name: 0.0 for name in FEATURE_NAMES})

    @classmethod
    def from_file(cls, path: str | Path) -> "LinearSurrogateState":
        data = read_json(path)
        if not isinstance(data, dict) or "weights" not in data:
            raise ValueError(f"Invalid linear surrogate checkpoint: {path}")
        return cls(weights={str(key): float(value) for key, value in data["weights"].items()})

    def to_dict(self) -> dict:
        return asdict(self)


def extract_linear_surrogate_features(record: RolloutRecord) -> dict[str, float]:
    confidence = (
        float(record.example.confidence_score)
        if record.example.confidence_score is not None
        else 0.0
    )

    return {
        "bias": 1.0,
        "confidence": confidence,
        "inverse_rank": 1.0 / float(record.example.sample_rank),
    }


def linear_surrogate_score(
    record: RolloutRecord,
    state: LinearSurrogateState,
) -> float:
    features = extract_linear_surrogate_features(record)
    return sum(state.weights.get(name, 0.0) * value for name, value in features.items())


def _valid_grpo_records(records: Sequence[RolloutRecord]) -> list[RolloutRecord]:
    valid_records = [
        record
        for record in records
        if record.reward.valid and record.advantage is not None
    ]

    if not valid_records:
        raise ValueError("No valid rollout records with advantages for GRPO training")

    return valid_records


def _old_surrogate_score(record: RolloutRecord) -> float:
    if record.old_surrogate_score is None:
        return 0.0
    return float(record.old_surrogate_score)


def compute_surrogate_ratio(
    new_score: float,
    old_score: float,
    *,
    max_score_delta: float = 20.0,
) -> float:
    delta = max(-max_score_delta, min(max_score_delta, new_score - old_score))
    return exp(delta)


def clip_surrogate_ratio(
    ratio: float,
    *,
    clip_epsilon: float = 0.2,
) -> float:
    return max(1.0 - clip_epsilon, min(1.0 + clip_epsilon, ratio))


def clipped_grpo_objective_term(
    *,
    advantage: float,
    ratio: float,
    clip_epsilon: float = 0.2,
) -> float:
    clipped_ratio = clip_surrogate_ratio(ratio, clip_epsilon=clip_epsilon)
    return min(ratio * advantage, clipped_ratio * advantage)


def _ratio_has_gradient(
    *,
    new_score: float,
    old_score: float,
    advantage: float,
    ratio: float,
    clip_epsilon: float,
    max_score_delta: float,
) -> bool:
    delta = new_score - old_score
    if delta <= -max_score_delta or delta >= max_score_delta:
        return False
    if advantage >= 0.0 and ratio > 1.0 + clip_epsilon:
        return False
    if advantage < 0.0 and ratio < 1.0 - clip_epsilon:
        return False
    return True


def compute_grpo_surrogate_loss(
    records: Sequence[RolloutRecord],
    state: LinearSurrogateState,
    *,
    clip_epsilon: float = 0.2,
    max_score_delta: float = 20.0,
) -> float:
    valid_records = _valid_grpo_records(records)
    terms = [
        clipped_grpo_objective_term(
            advantage=float(record.advantage),
            ratio=compute_surrogate_ratio(
                linear_surrogate_score(record, state),
                _old_surrogate_score(record),
                max_score_delta=max_score_delta,
            ),
            clip_epsilon=clip_epsilon,
        )
        for record in valid_records
    ]

    return -mean(terms)


def train_linear_grpo_step(
    records: Sequence[RolloutRecord],
    state: LinearSurrogateState,
    *,
    learning_rate: float,
    clip_epsilon: float = 0.2,
    max_score_delta: float = 20.0,
) -> tuple[LinearSurrogateState, dict]:
    valid_records = _valid_grpo_records(records)
    gradients = {name: 0.0 for name in FEATURE_NAMES}

    for record in valid_records:
        features = extract_linear_surrogate_features(record)
        advantage = float(record.advantage)
        old_score = _old_surrogate_score(record)
        new_score = linear_surrogate_score(record, state)
        ratio = compute_surrogate_ratio(
            new_score,
            old_score,
            max_score_delta=max_score_delta,
        )

        if _ratio_has_gradient(
            new_score=new_score,
            old_score=old_score,
            advantage=advantage,
            ratio=ratio,
            clip_epsilon=clip_epsilon,
            max_score_delta=max_score_delta,
        ):
            for name in FEATURE_NAMES:
                gradients[name] += -advantage * ratio * features[name]

    gradients = {
        name: value / len(valid_records)
        for name, value in gradients.items()
    }
    updated_weights = {
        name: state.weights.get(name, 0.0) - learning_rate * gradients[name]
        for name in FEATURE_NAMES
    }
    updated_state = LinearSurrogateState(weights=updated_weights)

    metrics = {
        "num_records": len(valid_records),
        "learning_rate": learning_rate,
        "clip_epsilon": clip_epsilon,
        "max_score_delta": max_score_delta,
        "loss_before": compute_grpo_surrogate_loss(
            valid_records,
            state,
            clip_epsilon=clip_epsilon,
            max_score_delta=max_score_delta,
        ),
        "loss_after": compute_grpo_surrogate_loss(
            valid_records,
            updated_state,
            clip_epsilon=clip_epsilon,
            max_score_delta=max_score_delta,
        ),
        "gradients": gradients,
        "weights_before": state.weights,
        "weights_after": updated_weights,
    }

    return updated_state, metrics


def build_surrogate_score_rows(
    records: Sequence[RolloutRecord],
    state: LinearSurrogateState,
    *,
    clip_epsilon: float = 0.2,
    max_score_delta: float = 20.0,
) -> list[dict]:
    rows = []

    for record in records:
        features = extract_linear_surrogate_features(record)
        surrogate_score = linear_surrogate_score(record, state)
        old_score = _old_surrogate_score(record)
        ratio = compute_surrogate_ratio(
            surrogate_score,
            old_score,
            max_score_delta=max_score_delta,
        )
        clipped_ratio = clip_surrogate_ratio(ratio, clip_epsilon=clip_epsilon)
        objective_term = (
            clipped_grpo_objective_term(
                advantage=float(record.advantage),
                ratio=ratio,
                clip_epsilon=clip_epsilon,
            )
            if record.advantage is not None
            else None
        )
        rows.append(
            {
                "complex_id": record.example.complex_id,
                "sample_id": record.example.sample_id,
                "rank": record.example.sample_rank,
                "reward": record.reward.total,
                "advantage": record.advantage,
                "old_surrogate_score": old_score,
                "surrogate_score": surrogate_score,
                "surrogate_ratio": ratio,
                "clipped_surrogate_ratio": clipped_ratio,
                "clipped_objective_term": objective_term,
                "confidence": features["confidence"],
                "inverse_rank": features["inverse_rank"],
            }
        )

    return rows


def save_linear_surrogate_checkpoint(
    state: LinearSurrogateState,
    path: str | Path,
    *,
    metadata: dict | None = None,
) -> None:
    payload = state.to_dict()
    payload["metadata"] = metadata or {}
    save_json(payload, path)
