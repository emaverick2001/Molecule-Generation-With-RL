from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.generation.generate_diffdock import generate_diffdock_poses
from src.rl.data import join_samples_with_complex_manifest
from src.rl.grpo import (
    LinearSurrogateState,
    linear_surrogate_score,
    save_linear_surrogate_checkpoint,
)
from src.rl.types import RLExample
from src.utils.artifact_logger import read_json
from src.utils.schemas import ComplexInput


class DiffDockRLAgent:
    """
    Boundary object for DiffDock RL/posttraining work.

    The current production-ready backend is `debug_linear`, which is a small
    trainable surrogate used to validate GRPO loss signs and checkpointing. The
    real DiffDock-loss backend is intentionally explicit and raises until the
    external DiffDock training graph/loss adapter is wired in.
    """

    def __init__(
        self,
        *,
        diffdock_repo_root: str | Path | None = None,
        model_dir: str | Path | None = None,
        ckpt: str | None = None,
        confidence_model_dir: str | Path | None = None,
        confidence_ckpt: str | None = None,
        device: str = "cuda",
        trainable_mode: str = "last_layers",
        surrogate_backend: str = "debug_linear",
        linear_state: LinearSurrogateState | None = None,
    ) -> None:
        if surrogate_backend not in {"debug_linear", "diffdock_loss"}:
            raise ValueError(f"Unsupported surrogate backend: {surrogate_backend}")
        if trainable_mode not in {"full", "last_layers", "lora", "none"}:
            raise ValueError(f"Unsupported trainable_mode: {trainable_mode}")

        self.diffdock_repo_root = Path(diffdock_repo_root) if diffdock_repo_root else None
        self.model_dir = Path(model_dir) if model_dir else None
        self.ckpt = ckpt
        self.confidence_model_dir = (
            Path(confidence_model_dir) if confidence_model_dir else None
        )
        self.confidence_ckpt = confidence_ckpt
        self.device = device
        self.trainable_mode = trainable_mode
        self.surrogate_backend = surrogate_backend
        self.linear_state = linear_state or LinearSurrogateState.initialized()
        self.trainable = trainable_mode != "none"

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        **kwargs,
    ) -> "DiffDockRLAgent":
        data = read_json(path)
        if not isinstance(data, dict) or "weights" not in data:
            raise ValueError(f"Unsupported DiffDockRLAgent checkpoint: {path}")

        return cls(
            linear_state=LinearSurrogateState(
                weights={str(key): float(value) for key, value in data["weights"].items()}
            ),
            **kwargs,
        )

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        metadata: dict | None = None,
    ) -> None:
        save_linear_surrogate_checkpoint(
            self.linear_state,
            path,
            metadata={
                "surrogate_backend": self.surrogate_backend,
                "trainable_mode": self.trainable_mode,
                **(metadata or {}),
            },
        )

    def freeze(self) -> None:
        self.trainable = False

    def unfreeze(self, trainable_mode: str | None = None) -> None:
        if trainable_mode is not None:
            if trainable_mode not in {"full", "last_layers", "lora"}:
                raise ValueError(f"Unsupported trainable_mode: {trainable_mode}")
            self.trainable_mode = trainable_mode
        self.trainable = True

    def frozen_old_policy(self) -> "DiffDockRLAgent":
        agent = DiffDockRLAgent(
            diffdock_repo_root=self.diffdock_repo_root,
            model_dir=self.model_dir,
            ckpt=self.ckpt,
            confidence_model_dir=self.confidence_model_dir,
            confidence_ckpt=self.confidence_ckpt,
            device=self.device,
            trainable_mode="none",
            surrogate_backend=self.surrogate_backend,
            linear_state=self.linear_state,
        )
        agent.freeze()
        return agent

    def frozen_reference_policy(self) -> "DiffDockRLAgent":
        return self.frozen_old_policy()

    def generate_samples(
        self,
        complexes: Sequence[ComplexInput],
        *,
        samples_per_complex: int,
        out_dir: str | Path,
        command_template: Sequence[str],
        config_path: str | Path,
        batch_size: int = 1,
        timeout_seconds: int | None = None,
        seed: int | None = None,
        inference_steps: int | None = None,
        actual_steps: int | None = None,
        save_visualisation: bool = False,
    ) -> list[RLExample]:
        del batch_size, seed, inference_steps, actual_steps, save_visualisation

        if self.diffdock_repo_root is None:
            raise ValueError("diffdock_repo_root is required for DiffDock generation")

        out_dir = Path(out_dir)
        generated = generate_diffdock_poses(
            records=list(complexes),
            output_dir=out_dir / "generated_samples",
            raw_output_dir=out_dir / "raw_diffdock_outputs",
            log_dir=out_dir / "logs",
            num_samples=samples_per_complex,
            command_template=command_template,
            repo_dir=self.diffdock_repo_root,
            config_path=config_path,
            timeout_seconds=timeout_seconds,
        )

        return join_samples_with_complex_manifest(
            generated,
            list(complexes),
            source_run_id=out_dir.name,
            source_checkpoint=self.ckpt,
        )

    def compute_surrogate_scores(
        self,
        examples: Sequence[RLExample],
        *,
        scoring_mode: str | None = None,
    ) -> list[float]:
        mode = scoring_mode or self.surrogate_backend

        if mode == "debug_linear":
            from src.rl.types import RewardBreakdown, RolloutRecord

            return [
                linear_surrogate_score(
                    RolloutRecord(
                        group_id=example.complex_id,
                        example=example,
                        reward=RewardBreakdown(total=0.0),
                    ),
                    self.linear_state,
                )
                for example in examples
            ]

        if mode == "diffdock_loss":
            raise NotImplementedError(
                "DiffDock-loss surrogate scoring requires an in-process adapter "
                "to external/DiffDock training graph construction and per-sample "
                "tr/rot/tor losses. The GRPO smoke path currently uses "
                "debug_linear."
            )

        raise ValueError(f"Unsupported scoring mode: {mode}")

    def compute_exact_logprobs(self, trajectories: Sequence[object]) -> list[float]:
        del trajectories
        raise NotImplementedError(
            "Exact PPO/log-prob computation is out of scope for this project. "
            "Use GRPO surrogate objectives instead."
        )
