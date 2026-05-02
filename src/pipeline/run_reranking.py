from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from src.evaluation.reranking import (
    rerank_generated_poses,
    save_reranked_manifest,
    summarize_reranking,
)
from src.rewards.confidence_reward import build_confidence_reward_records
from src.utils.artifact_logger import read_json, save_csv, save_json, save_text
from src.utils.config import load_experiment_config
from src.utils.schemas import GeneratedPose, RewardRecord


def _load_generated_records(path: str | Path) -> list[GeneratedPose]:
    return [GeneratedPose.from_dict(item) for item in read_json(path)]


def _load_reward_records(path: str | Path) -> list[RewardRecord]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Reward CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        return [
            RewardRecord(
                complex_id=row["complex_id"],
                sample_id=int(row["sample_id"]),
                reward=float(row["reward"]),
                reward_type=row["reward_type"],
                valid=row["valid"].lower() == "true",
            )
            for row in csv.DictReader(f)
        ]


def _get_reranking_config(config: dict[str, Any]) -> dict[str, Any]:
    reranking_config = config.get("reranking", {})
    return {
        "reward_source": reranking_config.get("reward_source", "rewards_csv"),
        "descending": reranking_config.get("descending", True),
        "tie_breaker": reranking_config.get("tie_breaker", "sample_id"),
        "confidence_transform": reranking_config.get("confidence_transform", "identity"),
        "confidence_temperature": reranking_config.get("confidence_temperature", 1.0),
    }


def _build_reward_records(
    generated_records: list[GeneratedPose],
    reward_path: Path,
    reranking_config: dict[str, Any],
) -> list[RewardRecord]:
    reward_source = reranking_config["reward_source"]

    if reward_source == "rewards_csv":
        return _load_reward_records(reward_path)

    if reward_source == "confidence_score":
        return build_confidence_reward_records(
            poses=generated_records,
            transform=reranking_config["confidence_transform"],
            temperature=reranking_config["confidence_temperature"],
        )

    raise ValueError(f"Unsupported reranking reward_source: {reward_source}")


def run_reranking(
    generated_manifest_path: str | Path,
    reward_path: str | Path,
    reranked_manifest_path: str | Path,
    summary_json_path: str | Path,
    summary_text_path: str | Path,
    config: dict[str, Any],
    confidence_rewards_path: str | Path | None = None,
) -> dict[str, Any]:
    reranking_config = _get_reranking_config(config)
    generated_records = _load_generated_records(generated_manifest_path)
    reward_records = _build_reward_records(
        generated_records=generated_records,
        reward_path=Path(reward_path),
        reranking_config=reranking_config,
    )

    if reranking_config["reward_source"] == "confidence_score" and confidence_rewards_path:
        save_csv(
            [record.to_dict() for record in reward_records],
            confidence_rewards_path,
        )

    reranked_records = rerank_generated_poses(
        generated_records=generated_records,
        reward_records=reward_records,
        descending=reranking_config["descending"],
        tie_breaker=reranking_config["tie_breaker"],
    )
    summary = {
        "stage": "confidence_reranking",
        "generated_manifest": str(generated_manifest_path),
        "reward_source": reranking_config["reward_source"],
        "descending": reranking_config["descending"],
        "tie_breaker": reranking_config["tie_breaker"],
        "aggregate": summarize_reranking(reranked_records),
    }

    save_reranked_manifest(reranked_records, reranked_manifest_path)
    save_json(summary, summary_json_path)
    save_text(
        (
            "# Confidence Reranking\n\n"
            f"- Generated poses: {summary['aggregate']['num_poses']}\n"
            f"- Complexes: {summary['aggregate']['num_complexes']}\n"
            f"- Reward source: {summary['reward_source']}\n"
            f"- Rank changes: {summary['aggregate']['num_rank_changed']}\n"
            f"- Mean absolute rank delta: "
            f"{summary['aggregate']['mean_absolute_rank_delta']}\n"
        ),
        summary_text_path,
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerank generated poses by confidence/reward records."
    )
    parser.add_argument(
        "--global-config",
        default="configs/global.yaml",
        help="Path to global config file.",
    )
    parser.add_argument(
        "--config",
        default="configs/diffdock/rerank_baseline.yaml",
        help="Path to reranking config file.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing generated_samples_manifest.json.",
    )
    parser.add_argument(
        "--generated-manifest",
        default=None,
        help="Optional override for generated_samples_manifest.json.",
    )
    parser.add_argument(
        "--rewards",
        default=None,
        help="Optional override for rewards.csv.",
    )

    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config = load_experiment_config(
        global_path=args.global_config,
        experiment_path=args.config,
    )

    run_reranking(
        generated_manifest_path=(
            Path(args.generated_manifest)
            if args.generated_manifest is not None
            else run_dir / "generated_samples_manifest.json"
        ),
        reward_path=Path(args.rewards) if args.rewards is not None else run_dir / "rewards.csv",
        reranked_manifest_path=run_dir / "reranked_generated_samples_manifest.json",
        summary_json_path=run_dir / "reranking_summary.json",
        summary_text_path=run_dir / "reranking_summary.md",
        confidence_rewards_path=run_dir / "confidence_rewards.csv",
        config=config,
    )

    print(f"Reranking complete: {run_dir}")


if __name__ == "__main__":
    main()
