from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.utils.config import load_yaml


SUPPORTED_ALGORITHMS = {
    "offline_reward_debug",
    "sft",
    "reinforce_surrogate",
    "surrogate_ppo",
}


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: str
    mode: str
    seed: int = 42


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str
    policy_mode: str = "surrogate"
    ppo_epochs: int = 1
    minibatch_size: int = 8


@dataclass(frozen=True)
class DataConfig:
    source_run_dir: str | None = None
    input_manifest: str | None = None
    generated_manifest: str | None = None
    train_manifest: str | None = None
    val_manifest: str | None = None


@dataclass(frozen=True)
class RewardConfig:
    type: str = "rmsd"
    weights: dict[str, float] = field(default_factory=lambda: {"rmsd": 1.0})
    sigma_angstrom: float = 2.0
    max_rmsd: float = 10.0
    invalid_reward: float = -1.0
    confidence_mode: str = "logit"
    confidence_temperature: float = 1.0


@dataclass(frozen=True)
class RolloutConfig:
    samples_per_complex: int = 10
    advantage_normalization: str = "zscore"
    min_valid_samples_per_complex: int = 1
    invalid_group_action: str = "zero"


@dataclass(frozen=True)
class ArtifactsConfig:
    run_root: str = "artifacts/runs"
    run_tag: str | None = None


@dataclass(frozen=True)
class RLConfig:
    experiment: ExperimentConfig
    algorithm: AlgorithmConfig
    data: DataConfig
    reward: RewardConfig
    rollout: RolloutConfig
    artifacts: ArtifactsConfig
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("raw", None)
        return data


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def parse_rl_config(data: dict[str, Any]) -> RLConfig:
    experiment = _required_mapping(data, "experiment")
    algorithm = _required_mapping(data, "algorithm")
    data_cfg = _required_mapping(data, "data")
    reward = _required_mapping(data, "reward")
    rollout = _required_mapping(data, "rollout")
    artifacts = _required_mapping(data, "artifacts")

    cfg = RLConfig(
        experiment=ExperimentConfig(
            name=experiment.get("name", "diffdock_rl"),
            model=experiment.get("model", "diffdock"),
            mode=experiment.get("mode", "posttraining_offline_reward_debug"),
            seed=int(experiment.get("seed", 42)),
        ),
        algorithm=AlgorithmConfig(
            name=algorithm.get("name", "offline_reward_debug"),
            policy_mode=algorithm.get("policy_mode", "surrogate"),
            ppo_epochs=int(algorithm.get("ppo_epochs", 1)),
            minibatch_size=int(algorithm.get("minibatch_size", 8)),
        ),
        data=DataConfig(
            source_run_dir=data_cfg.get("source_run_dir"),
            input_manifest=data_cfg.get("input_manifest"),
            generated_manifest=data_cfg.get("generated_manifest"),
            train_manifest=data_cfg.get("train_manifest"),
            val_manifest=data_cfg.get("val_manifest"),
        ),
        reward=RewardConfig(
            type=reward.get("type", "rmsd"),
            weights={
                str(key): float(value)
                for key, value in reward.get("weights", {"rmsd": 1.0}).items()
            },
            sigma_angstrom=float(reward.get("sigma_angstrom", 2.0)),
            max_rmsd=float(reward.get("max_rmsd", 10.0)),
            invalid_reward=float(reward.get("invalid_reward", -1.0)),
            confidence_mode=reward.get("confidence_mode", "logit"),
            confidence_temperature=float(reward.get("confidence_temperature", 1.0)),
        ),
        rollout=RolloutConfig(
            samples_per_complex=int(rollout.get("samples_per_complex", 10)),
            advantage_normalization=rollout.get("advantage_normalization", "zscore"),
            min_valid_samples_per_complex=int(
                rollout.get("min_valid_samples_per_complex", 1)
            ),
            invalid_group_action=rollout.get("invalid_group_action", "zero"),
        ),
        artifacts=ArtifactsConfig(
            run_root=artifacts.get("run_root", "artifacts/runs"),
            run_tag=artifacts.get("run_tag"),
        ),
        raw=data,
    )
    validate_rl_config(cfg)
    return cfg


def load_rl_config(path: str | Path) -> RLConfig:
    return parse_rl_config(load_yaml(path))


def _resolve_source_run_file(
    source_run_dir: str | None,
    explicit_path: str | None,
    default_name: str,
) -> str | None:
    if explicit_path:
        return explicit_path
    if source_run_dir:
        return str(Path(source_run_dir) / default_name)
    return None


def resolve_run_paths(cfg: RLConfig) -> RLConfig:
    data = DataConfig(
        source_run_dir=cfg.data.source_run_dir,
        input_manifest=_resolve_source_run_file(
            cfg.data.source_run_dir,
            cfg.data.input_manifest,
            "input_manifest.json",
        ),
        generated_manifest=_resolve_source_run_file(
            cfg.data.source_run_dir,
            cfg.data.generated_manifest,
            "generated_samples_manifest.json",
        ),
        train_manifest=cfg.data.train_manifest,
        val_manifest=cfg.data.val_manifest,
    )

    resolved = RLConfig(
        experiment=cfg.experiment,
        algorithm=cfg.algorithm,
        data=data,
        reward=cfg.reward,
        rollout=cfg.rollout,
        artifacts=cfg.artifacts,
        raw=cfg.raw,
    )
    validate_rl_config(resolved)
    return resolved


def validate_rl_config(cfg: RLConfig) -> None:
    if cfg.algorithm.name not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported RL algorithm: {cfg.algorithm.name}")

    if cfg.algorithm.name != "offline_reward_debug":
        raise NotImplementedError(
            f"{cfg.algorithm.name} is scaffolded but not implemented yet. "
            "Run offline_reward_debug first to validate rewards and rollouts."
        )

    if cfg.algorithm.policy_mode == "exact":
        raise NotImplementedError(
            "Exact PPO/log-prob training is not available until DiffDock sampler "
            "transition statistics are instrumented."
        )

    if cfg.rollout.samples_per_complex <= 0:
        raise ValueError("rollout.samples_per_complex must be positive")
    if cfg.rollout.min_valid_samples_per_complex <= 0:
        raise ValueError("rollout.min_valid_samples_per_complex must be positive")
    if cfg.rollout.advantage_normalization not in {"zscore", "center", "rank"}:
        raise ValueError("Unsupported rollout.advantage_normalization")
    if cfg.rollout.invalid_group_action not in {"zero", "drop"}:
        raise ValueError("Unsupported rollout.invalid_group_action")

    if cfg.reward.sigma_angstrom <= 0:
        raise ValueError("reward.sigma_angstrom must be positive")
    if cfg.reward.max_rmsd <= 0:
        raise ValueError("reward.max_rmsd must be positive")
    if not cfg.reward.weights:
        raise ValueError("reward.weights must not be empty")
    if all(value <= 0 for value in cfg.reward.weights.values()):
        raise ValueError("At least one reward weight must be positive")
    if cfg.reward.confidence_temperature <= 0:
        raise ValueError("reward.confidence_temperature must be positive")

    input_manifest = _resolve_source_run_file(
        cfg.data.source_run_dir,
        cfg.data.input_manifest,
        "input_manifest.json",
    )
    generated_manifest = _resolve_source_run_file(
        cfg.data.source_run_dir,
        cfg.data.generated_manifest,
        "generated_samples_manifest.json",
    )

    if cfg.algorithm.name == "offline_reward_debug":
        if not input_manifest:
            raise ValueError("data.input_manifest or data.source_run_dir is required")
        if not generated_manifest:
            raise ValueError("data.generated_manifest or data.source_run_dir is required")

        for path in [input_manifest, generated_manifest]:
            if not Path(path).is_file():
                raise FileNotFoundError(f"Required RL input file not found: {path}")
