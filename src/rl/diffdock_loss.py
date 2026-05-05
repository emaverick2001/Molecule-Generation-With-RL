from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log
from pathlib import Path
from statistics import mean
from typing import Sequence

from src.rl.types import RolloutRecord
from src.utils.artifact_logger import read_json, save_json


@dataclass(frozen=True)
class DiffDockLossSurrogateState:
    """State for the DiffDock-loss surrogate backend.

    This state intentionally keeps the first implementation lightweight: it
    learns a scalar affine calibration over a per-sample DiffDock-loss proxy.
    The production hook should replace `compute_diffdock_loss_proxy` with an
    in-process DiffDock loss call that returns one loss per generated pose.
    """

    loss_scale: float = 1.0
    loss_bias: float = 0.0

    @classmethod
    def initialized(cls) -> "DiffDockLossSurrogateState":
        return cls()

    @classmethod
    def from_file(cls, path: str | Path) -> "DiffDockLossSurrogateState":
        data = read_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid DiffDock-loss checkpoint: {path}")
        return cls(
            loss_scale=float(data.get("loss_scale", 1.0)),
            loss_bias=float(data.get("loss_bias", 0.0)),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _valid_grpo_records(records: Sequence[RolloutRecord]) -> list[RolloutRecord]:
    valid_records = [
        record
        for record in records
        if record.reward.valid and record.advantage is not None
    ]
    if not valid_records:
        raise ValueError("No valid rollout records with advantages for DiffDock-loss GRPO")
    return valid_records


def compute_diffdock_loss_proxy(
    record: RolloutRecord,
    *,
    sigma_angstrom: float = 2.0,
    eps: float = 1e-8,
) -> float:
    """Return a per-sample loss proxy for the DiffDock-loss backend.

    Target production score:
        s_theta(pose, complex) = -DiffDock_loss_theta(pose, complex)

    Until the in-process DiffDock loss call is wired, this proxy uses the RMSD
    reward component already computed by the offline reward path:
    - if raw RMSD is available, loss_proxy = RMSD / sigma
    - otherwise, invert reward ~= exp(-loss_proxy)

    This keeps sign, grouping, checkpoint, and logging behavior identical to
    the intended backend while avoiding a hard dependency on DiffDock internals
    in unit tests.
    """
    rmsd_component = record.reward.components.get("rmsd")
    if rmsd_component is not None and rmsd_component.raw_value is not None:
        return max(float(rmsd_component.raw_value), 0.0) / sigma_angstrom

    reward_total = max(float(record.reward.total), eps)
    return -log(reward_total)


def diffdock_loss_surrogate_score(
    record: RolloutRecord,
    state: DiffDockLossSurrogateState,
    *,
    sigma_angstrom: float = 2.0,
) -> float:
    loss_proxy = compute_diffdock_loss_proxy(
        record,
        sigma_angstrom=sigma_angstrom,
    )
    return -(state.loss_scale * loss_proxy + state.loss_bias)


def compute_grpo_diffdock_loss_surrogate_loss(
    records: Sequence[RolloutRecord],
    state: DiffDockLossSurrogateState,
    *,
    sigma_angstrom: float = 2.0,
) -> float:
    valid_records = _valid_grpo_records(records)
    terms = [
        float(record.advantage)
        * diffdock_loss_surrogate_score(
            record,
            state,
            sigma_angstrom=sigma_angstrom,
        )
        for record in valid_records
    ]
    return -mean(terms)


def train_diffdock_loss_grpo_step(
    records: Sequence[RolloutRecord],
    state: DiffDockLossSurrogateState,
    *,
    learning_rate: float,
    sigma_angstrom: float = 2.0,
) -> tuple[DiffDockLossSurrogateState, dict]:
    """Run one GRPO surrogate step over the DiffDock-loss score path.

    For the proxy implementation, gradients are analytic for the affine
    calibration parameters:
        loss = -mean(A_i * s_i)
        s_i = -(scale * loss_proxy_i + bias)

    When the real DiffDock backend is wired, this function is the seam where
    `loss_proxy_i` should become a differentiable per-pose DiffDock loss and
    the optimizer step should update DiffDock parameters instead of this small
    calibration state.
    """
    valid_records = _valid_grpo_records(records)
    loss_proxies = [
        compute_diffdock_loss_proxy(record, sigma_angstrom=sigma_angstrom)
        for record in valid_records
    ]
    advantages = [float(record.advantage) for record in valid_records]

    grad_loss_scale = mean(
        advantage * loss_proxy
        for advantage, loss_proxy in zip(advantages, loss_proxies)
    )
    grad_loss_bias = mean(advantages)

    updated_state = DiffDockLossSurrogateState(
        loss_scale=state.loss_scale - learning_rate * grad_loss_scale,
        loss_bias=state.loss_bias - learning_rate * grad_loss_bias,
    )

    metrics = {
        "num_records": len(valid_records),
        "learning_rate": learning_rate,
        "loss_before": compute_grpo_diffdock_loss_surrogate_loss(
            valid_records,
            state,
            sigma_angstrom=sigma_angstrom,
        ),
        "loss_after": compute_grpo_diffdock_loss_surrogate_loss(
            valid_records,
            updated_state,
            sigma_angstrom=sigma_angstrom,
        ),
        "gradients": {
            "loss_scale": grad_loss_scale,
            "loss_bias": grad_loss_bias,
        },
        "weights_before": state.to_dict(),
        "weights_after": updated_state.to_dict(),
        "loss_proxy_mean": mean(loss_proxies),
    }

    return updated_state, metrics


def build_diffdock_loss_score_rows(
    records: Sequence[RolloutRecord],
    state: DiffDockLossSurrogateState,
    *,
    sigma_angstrom: float = 2.0,
) -> list[dict]:
    rows = []
    for record in records:
        loss_proxy = compute_diffdock_loss_proxy(
            record,
            sigma_angstrom=sigma_angstrom,
        )
        rows.append(
            {
                "complex_id": record.example.complex_id,
                "sample_id": record.example.sample_id,
                "rank": record.example.sample_rank,
                "reward": record.reward.total,
                "advantage": record.advantage,
                "diffdock_loss_proxy": loss_proxy,
                "surrogate_score": diffdock_loss_surrogate_score(
                    record,
                    state,
                    sigma_angstrom=sigma_angstrom,
                ),
                "confidence": record.example.confidence_score,
            }
        )
    return rows


def save_diffdock_loss_surrogate_checkpoint(
    state: DiffDockLossSurrogateState,
    path: str | Path,
    *,
    metadata: dict | None = None,
) -> None:
    payload = state.to_dict()
    payload["metadata"] = metadata or {}
    save_json(payload, path)
