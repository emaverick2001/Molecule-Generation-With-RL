from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Sequence

from src.rl.types import RolloutRecord
from src.utils.seeds import set_seed


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def safe_zscore(values: Sequence[float], eps: float = 1e-6) -> list[float]:
    if not values:
        return []

    mu = mean(values)
    sigma = pstdev(values)

    if sigma <= eps:
        return [0.0 for _ in values]

    return [(value - mu) / (sigma + eps) for value in values]


def replace_nan_inf(value: float, replacement: float = 0.0) -> float:
    if value != value or value in (float("inf"), float("-inf")):
        return replacement
    return value


def summarize_rewards(records: Sequence[RolloutRecord]) -> dict[str, Any]:
    valid_rewards = [
        record.reward.total for record in records if record.reward.valid
    ]
    advantages = [
        record.advantage for record in records if record.advantage is not None
    ]

    if not valid_rewards:
        return {
            "num_records": len(records),
            "num_valid_rewards": 0,
            "reward_mean": None,
            "reward_std": None,
            "reward_min": None,
            "reward_max": None,
            "advantage_std": pstdev(advantages) if len(advantages) > 1 else 0.0,
        }

    return {
        "num_records": len(records),
        "num_valid_rewards": len(valid_rewards),
        "valid_reward_fraction": len(valid_rewards) / len(records),
        "reward_mean": mean(valid_rewards),
        "reward_std": pstdev(valid_rewards) if len(valid_rewards) > 1 else 0.0,
        "reward_min": min(valid_rewards),
        "reward_max": max(valid_rewards),
        "advantage_std": pstdev(advantages) if len(advantages) > 1 else 0.0,
    }


__all__ = [
    "ensure_dir",
    "read_jsonl",
    "replace_nan_inf",
    "safe_zscore",
    "set_seed",
    "summarize_rewards",
    "write_jsonl",
]
