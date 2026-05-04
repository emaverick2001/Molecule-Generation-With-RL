from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Sequence

from src.rl.config import RewardConfig, RolloutConfig
from src.rl.rewards import score_example
from src.rl.types import RLExample, RolloutRecord
from src.rl.utils import safe_zscore


def build_rollout_records(
    examples: Sequence[RLExample],
    *,
    reward_cfg: RewardConfig,
) -> list[RolloutRecord]:
    return [
        RolloutRecord(
            group_id=example.complex_id,
            example=example,
            reward=score_example(example, reward_cfg),
        )
        for example in examples
    ]


def _rank_advantages(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [0.0 for _ in values]

    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    for rank, (index, _) in enumerate(indexed):
        ranks[index] = rank / (len(values) - 1)
    return [2.0 * rank - 1.0 for rank in ranks]


def _compute_advantages_for_group(
    rewards: list[float],
    normalization: str,
) -> list[float]:
    if normalization == "zscore":
        return safe_zscore(rewards)
    if normalization == "center":
        group_mean = mean(rewards)
        return [reward - group_mean for reward in rewards]
    if normalization == "rank":
        return _rank_advantages(rewards)

    raise ValueError(f"Unsupported advantage normalization: {normalization}")


def compute_group_advantages(
    records: Sequence[RolloutRecord],
    *,
    rollout_cfg: RolloutConfig,
) -> list[RolloutRecord]:
    groups: dict[str, list[RolloutRecord]] = defaultdict(list)
    for record in records:
        groups[record.group_id].append(record)

    updated_records = []

    for group_id in sorted(groups):
        group_records = sorted(
            groups[group_id],
            key=lambda record: record.example.sample_rank,
        )
        valid_records = [record for record in group_records if record.reward.valid]

        if len(valid_records) < rollout_cfg.min_valid_samples_per_complex:
            if rollout_cfg.invalid_group_action == "drop":
                continue
            advantages_by_key = {
                (record.example.complex_id, record.example.sample_id): 0.0
                for record in group_records
            }
        else:
            rewards = [record.reward.total for record in valid_records]
            advantages = _compute_advantages_for_group(
                rewards,
                normalization=rollout_cfg.advantage_normalization,
            )
            advantages_by_key = {
                (record.example.complex_id, record.example.sample_id): advantage
                for record, advantage in zip(valid_records, advantages)
            }
            for record in group_records:
                key = (record.example.complex_id, record.example.sample_id)
                advantages_by_key.setdefault(key, 0.0)

        for record in group_records:
            key = (record.example.complex_id, record.example.sample_id)
            updated_records.append(
                RolloutRecord(
                    group_id=record.group_id,
                    example=record.example,
                    reward=record.reward,
                    advantage=advantages_by_key[key],
                    old_surrogate_score=record.old_surrogate_score,
                    old_logprob=record.old_logprob,
                )
            )

    return updated_records


def summarize_rollout_groups(records: Sequence[RolloutRecord]) -> dict:
    groups: dict[str, list[RolloutRecord]] = defaultdict(list)
    for record in records:
        groups[record.group_id].append(record)

    valid_group_count = 0
    group_sizes = []
    valid_group_sizes = []

    for group_records in groups.values():
        group_sizes.append(len(group_records))
        valid_count = sum(1 for record in group_records if record.reward.valid)
        valid_group_sizes.append(valid_count)
        if valid_count > 0:
            valid_group_count += 1

    return {
        "num_groups": len(groups),
        "num_valid_groups": valid_group_count,
        "group_size_min": min(group_sizes) if group_sizes else 0,
        "group_size_max": max(group_sizes) if group_sizes else 0,
        "valid_group_size_min": min(valid_group_sizes) if valid_group_sizes else 0,
        "valid_group_size_max": max(valid_group_sizes) if valid_group_sizes else 0,
    }


def compute_surrogate_ratios(
    new_scores: Sequence[float],
    old_scores: Sequence[float],
    *,
    max_score_delta: float = 20.0,
) -> list[float]:
    from math import exp

    if len(new_scores) != len(old_scores):
        raise ValueError("new_scores and old_scores must have the same length")

    ratios = []
    for new_score, old_score in zip(new_scores, old_scores):
        delta = max(-max_score_delta, min(max_score_delta, new_score - old_score))
        ratios.append(exp(delta))

    return ratios
