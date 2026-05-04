from __future__ import annotations

from dataclasses import asdict, dataclass
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


def compute_grpo_surrogate_loss(
    records: Sequence[RolloutRecord],
    state: LinearSurrogateState,
) -> float:
    valid_records = _valid_grpo_records(records)
    terms = [
        float(record.advantage) * linear_surrogate_score(record, state)
        for record in valid_records
    ]

    return -mean(terms)


def train_linear_grpo_step(
    records: Sequence[RolloutRecord],
    state: LinearSurrogateState,
    *,
    learning_rate: float,
) -> tuple[LinearSurrogateState, dict]:
    valid_records = _valid_grpo_records(records)
    gradients = {name: 0.0 for name in FEATURE_NAMES}

    for record in valid_records:
        features = extract_linear_surrogate_features(record)
        for name in FEATURE_NAMES:
            gradients[name] += -float(record.advantage) * features[name]

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
        "loss_before": compute_grpo_surrogate_loss(valid_records, state),
        "loss_after": compute_grpo_surrogate_loss(valid_records, updated_state),
        "gradients": gradients,
        "weights_before": state.weights,
        "weights_after": updated_weights,
    }

    return updated_state, metrics


def build_surrogate_score_rows(
    records: Sequence[RolloutRecord],
    state: LinearSurrogateState,
) -> list[dict]:
    rows = []

    for record in records:
        features = extract_linear_surrogate_features(record)
        rows.append(
            {
                "complex_id": record.example.complex_id,
                "sample_id": record.example.sample_id,
                "rank": record.example.sample_rank,
                "reward": record.reward.total,
                "advantage": record.advantage,
                "surrogate_score": linear_surrogate_score(record, state),
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
