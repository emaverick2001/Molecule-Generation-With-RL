from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any


def _require_nonempty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_nonnegative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _require_finite_float(value: float, field_name: str) -> None:
    if not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number")


@dataclass(frozen=True)
class RLExample:
    complex_id: str
    protein_path: str
    ligand_input_path: str
    predicted_pose_path: str
    ground_truth_pose_path: str | None
    sample_rank: int
    sample_id: int
    confidence_score: float | None = None
    source_run_id: str | None = None
    source_checkpoint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_str(self.complex_id, "complex_id")
        _require_nonempty_str(self.protein_path, "protein_path")
        _require_nonempty_str(self.ligand_input_path, "ligand_input_path")
        _require_nonempty_str(self.predicted_pose_path, "predicted_pose_path")
        _require_nonnegative_int(self.sample_rank, "sample_rank")
        _require_nonnegative_int(self.sample_id, "sample_id")

        if self.ground_truth_pose_path is not None:
            _require_nonempty_str(self.ground_truth_pose_path, "ground_truth_pose_path")
        if self.confidence_score is not None:
            _require_finite_float(self.confidence_score, "confidence_score")
        if self.source_run_id is not None:
            _require_nonempty_str(self.source_run_id, "source_run_id")
        if self.source_checkpoint is not None:
            _require_nonempty_str(self.source_checkpoint, "source_checkpoint")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RLExample":
        return cls(
            complex_id=data["complex_id"],
            protein_path=data["protein_path"],
            ligand_input_path=data["ligand_input_path"],
            predicted_pose_path=data["predicted_pose_path"],
            ground_truth_pose_path=data.get("ground_truth_pose_path"),
            sample_rank=data["sample_rank"],
            sample_id=data["sample_id"],
            confidence_score=data.get("confidence_score"),
            source_run_id=data.get("source_run_id"),
            source_checkpoint=data.get("source_checkpoint"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RewardComponent:
    name: str
    value: float
    raw_value: float | None = None
    valid: bool = True
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.name, "name")
        _require_finite_float(self.value, "value")
        if self.raw_value is not None:
            _require_finite_float(self.raw_value, "raw_value")
        if not isinstance(self.valid, bool):
            raise ValueError("valid must be a boolean")
        if self.reason is not None:
            _require_nonempty_str(self.reason, "reason")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RewardComponent":
        return cls(
            name=data["name"],
            value=data["value"],
            raw_value=data.get("raw_value"),
            valid=data.get("valid", True),
            reason=data.get("reason"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    components: dict[str, RewardComponent] = field(default_factory=dict)
    valid: bool = True
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_finite_float(self.total, "total")
        if not isinstance(self.components, dict):
            raise ValueError("components must be a dictionary")
        for name, component in self.components.items():
            _require_nonempty_str(name, "component name")
            if not isinstance(component, RewardComponent):
                raise ValueError("components must contain RewardComponent values")
        if not isinstance(self.valid, bool):
            raise ValueError("valid must be a boolean")
        if self.reason is not None:
            _require_nonempty_str(self.reason, "reason")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RewardBreakdown":
        return cls(
            total=data["total"],
            components={
                name: RewardComponent.from_dict(component)
                for name, component in data.get("components", {}).items()
            },
            valid=data.get("valid", True),
            reason=data.get("reason"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "components": {
                name: component.to_dict()
                for name, component in sorted(self.components.items())
            },
            "valid": self.valid,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RolloutRecord:
    group_id: str
    example: RLExample
    reward: RewardBreakdown
    advantage: float | None = None
    old_surrogate_score: float | None = None
    old_logprob: float | None = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.group_id, "group_id")
        if not isinstance(self.example, RLExample):
            raise ValueError("example must be an RLExample")
        if not isinstance(self.reward, RewardBreakdown):
            raise ValueError("reward must be a RewardBreakdown")
        for field_name in ["advantage", "old_surrogate_score", "old_logprob"]:
            value = getattr(self, field_name)
            if value is not None:
                _require_finite_float(value, field_name)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RolloutRecord":
        return cls(
            group_id=data["group_id"],
            example=RLExample.from_dict(data["example"]),
            reward=RewardBreakdown.from_dict(data["reward"]),
            advantage=data.get("advantage"),
            old_surrogate_score=data.get("old_surrogate_score"),
            old_logprob=data.get("old_logprob"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "example": self.example.to_dict(),
            "reward": self.reward.to_dict(),
            "advantage": self.advantage,
            "old_surrogate_score": self.old_surrogate_score,
            "old_logprob": self.old_logprob,
        }


@dataclass(frozen=True)
class TrainSummary:
    run_dir: str
    algorithm: str
    num_examples: int
    num_rollout_records: int
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

