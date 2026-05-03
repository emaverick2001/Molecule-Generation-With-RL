# Schema definitions and validation helpers used across the pipeline.

# This module centralizes structured data contracts for:
# - configuration objects
# - dataset manifest records
# - generation outputs
# - reward/evaluation results

# Why this exists:
# - keeps input/output formats consistent between modules
# - validates data early to fail fast with clear errors
# - reduces parsing/shape bugs in baseline, rerank, reward-filtering,
#   and post-training workflows

from __future__ import annotations

from dataclasses import asdict, dataclass
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
class ComplexInput:
    complex_id: str
    protein_path: str
    ligand_path: str
    ground_truth_pose_path: str
    split: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.complex_id, "complex_id")
        _require_nonempty_str(self.protein_path, "protein_path")
        _require_nonempty_str(self.ligand_path, "ligand_path")
        _require_nonempty_str(self.ground_truth_pose_path, "ground_truth_pose_path")
        _require_nonempty_str(self.split, "split")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ComplexInput":
        return cls(
            complex_id=data["complex_id"],
            protein_path=data["protein_path"],
            ligand_path=data["ligand_path"],
            ground_truth_pose_path=data["ground_truth_pose_path"],
            split=data["split"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeneratedPose:
    complex_id: str
    sample_id: int
    pose_path: str
    confidence_score: float | None = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.complex_id, "complex_id")
        _require_nonnegative_int(self.sample_id, "sample_id")
        _require_nonempty_str(self.pose_path, "pose_path")

        if self.confidence_score is not None:
            _require_finite_float(self.confidence_score, "confidence_score")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneratedPose":
        return cls(
            complex_id=data["complex_id"],
            sample_id=data["sample_id"],
            pose_path=data["pose_path"],
            confidence_score=data.get("confidence_score"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RewardRecord:
    complex_id: str
    sample_id: int
    reward: float
    reward_type: str
    valid: bool

    def __post_init__(self) -> None:
        _require_nonempty_str(self.complex_id, "complex_id")
        _require_nonnegative_int(self.sample_id, "sample_id")
        _require_finite_float(self.reward, "reward")
        _require_nonempty_str(self.reward_type, "reward_type")

        if not isinstance(self.valid, bool):
            raise ValueError("valid must be a boolean")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RewardRecord":
        return cls(
            complex_id=data["complex_id"],
            sample_id=data["sample_id"],
            reward=data["reward"],
            reward_type=data["reward_type"],
            valid=data["valid"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricRecord:
    complex_id: str
    top1_rmsd: float
    success_at_1: bool
    success_at_5: bool
    success_at_10: bool

    def __post_init__(self) -> None:
        _require_nonempty_str(self.complex_id, "complex_id")
        _require_finite_float(self.top1_rmsd, "top1_rmsd")

        for field_name in ["success_at_1", "success_at_5", "success_at_10"]:
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetricRecord":
        return cls(
            complex_id=data["complex_id"],
            top1_rmsd=data["top1_rmsd"],
            success_at_1=data["success_at_1"],
            success_at_5=data["success_at_5"],
            success_at_10=data["success_at_10"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
