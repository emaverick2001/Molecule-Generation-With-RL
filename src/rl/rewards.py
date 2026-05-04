from __future__ import annotations

from math import exp, isfinite
from pathlib import Path

from src.evaluation.rmsd import compute_symmetry_corrected_rmsd
from src.rl.config import RewardConfig
from src.rl.types import RLExample, RewardBreakdown, RewardComponent


def compute_rmsd_reward(
    predicted_pose_path: str | Path,
    ground_truth_pose_path: str | Path | None,
    *,
    sigma_angstrom: float = 2.0,
    max_rmsd: float = 10.0,
    invalid_reward: float = -1.0,
    remove_hs: bool = True,
) -> RewardComponent:
    if ground_truth_pose_path is None:
        return RewardComponent(
            name="rmsd",
            value=invalid_reward,
            valid=False,
            reason="missing ground-truth pose",
        )

    try:
        rmsd = compute_symmetry_corrected_rmsd(
            predicted_pose_path,
            ground_truth_pose_path,
            remove_hs=remove_hs,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        return RewardComponent(
            name="rmsd",
            value=invalid_reward,
            valid=False,
            reason=str(error),
        )

    clipped_rmsd = min(float(rmsd), max_rmsd)
    return RewardComponent(
        name="rmsd",
        value=exp(-clipped_rmsd / sigma_angstrom),
        raw_value=float(rmsd),
        valid=True,
    )


def compute_confidence_reward(
    confidence_value: float | None,
    *,
    mode: str = "logit",
    temperature: float = 1.0,
    invalid_reward: float = -1.0,
) -> RewardComponent:
    if confidence_value is None:
        return RewardComponent(
            name="confidence",
            value=invalid_reward,
            valid=False,
            reason="missing confidence score",
        )

    score = float(confidence_value)
    if not isfinite(score):
        return RewardComponent(
            name="confidence",
            value=invalid_reward,
            valid=False,
            reason="non-finite confidence score",
        )

    if mode == "identity":
        value = score
    elif mode == "probability":
        value = max(0.0, min(1.0, score))
    elif mode == "predicted_rmsd":
        value = exp(-max(score, 0.0) / temperature)
    elif mode == "logit":
        value = 2.0 / (1.0 + exp(-score / temperature)) - 1.0
    else:
        raise ValueError(f"Unsupported confidence reward mode: {mode}")

    return RewardComponent(
        name="confidence",
        value=float(value),
        raw_value=score,
        valid=True,
    )


def combine_rewards(
    components: list[RewardComponent],
    *,
    weights: dict[str, float],
    invalid_reward: float = -1.0,
) -> RewardBreakdown:
    component_map = {component.name: component for component in components}
    weighted_sum = 0.0
    active_weight_sum = 0.0
    invalid_reasons = []

    for name, weight in weights.items():
        if weight <= 0:
            continue

        component = component_map.get(name)
        if component is None:
            invalid_reasons.append(f"missing {name} component")
            continue
        if not component.valid:
            invalid_reasons.append(component.reason or f"invalid {name} component")
            continue

        weighted_sum += weight * component.value
        active_weight_sum += weight

    if active_weight_sum <= 0:
        return RewardBreakdown(
            total=invalid_reward,
            components=component_map,
            valid=False,
            reason="; ".join(invalid_reasons) or "no valid reward components",
        )

    return RewardBreakdown(
        total=weighted_sum / active_weight_sum,
        components=component_map,
        valid=True,
        reason=None,
    )


def score_example(example: RLExample, cfg: RewardConfig) -> RewardBreakdown:
    components = [
        compute_rmsd_reward(
            example.predicted_pose_path,
            example.ground_truth_pose_path,
            sigma_angstrom=cfg.sigma_angstrom,
            max_rmsd=cfg.max_rmsd,
            invalid_reward=cfg.invalid_reward,
        ),
        compute_confidence_reward(
            example.confidence_score,
            mode=cfg.confidence_mode,
            temperature=cfg.confidence_temperature,
            invalid_reward=cfg.invalid_reward,
        ),
    ]
    return combine_rewards(
        components,
        weights=cfg.weights,
        invalid_reward=cfg.invalid_reward,
    )


def build_reward_rows(records: list[tuple[RLExample, RewardBreakdown]]) -> list[dict]:
    rows = []
    for example, reward in records:
        row = {
            "complex_id": example.complex_id,
            "sample_id": example.sample_id,
            "rank": example.sample_rank,
            "reward": reward.total,
            "valid": reward.valid,
            "reason": reward.reason,
        }
        for name, component in sorted(reward.components.items()):
            row[f"{name}_reward"] = component.value
            row[f"{name}_raw"] = component.raw_value
            row[f"{name}_valid"] = component.valid
            row[f"{name}_reason"] = component.reason
        rows.append(row)

    return rows
