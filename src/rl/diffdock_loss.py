from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import inspect
from pathlib import Path
import sys
from typing import Any

from src.rl.types import RLExample


@dataclass(frozen=True)
class DiffDockLossWeights:
    tr: float = 1.0
    rot: float = 1.0
    tor: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class DiffDockLossComponents:
    tr_loss: list[float]
    rot_loss: list[float]
    tor_loss: list[float]
    total_loss: list[float]
    surrogate_scores: list[float]

    def to_rows(self, examples: Sequence[RLExample]) -> list[dict[str, Any]]:
        if len(examples) != len(self.total_loss):
            raise ValueError(
                "Number of examples does not match DiffDock loss component count"
            )

        return [
            {
                "complex_id": example.complex_id,
                "sample_id": example.sample_id,
                "rank": example.sample_rank,
                "tr_loss": self.tr_loss[index],
                "rot_loss": self.rot_loss[index],
                "tor_loss": self.tor_loss[index],
                "diffdock_loss": self.total_loss[index],
                "surrogate_score": self.surrogate_scores[index],
            }
            for index, example in enumerate(examples)
        ]


def _to_float_list(value: Any, *, name: str) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, (int, float)):
        return [float(value)]

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [float(item) for item in value]

    raise TypeError(f"{name} must be a scalar, sequence, or tensor-like value")


def _broadcast_component(values: list[float], *, target_length: int, name: str) -> list[float]:
    if len(values) == target_length:
        return values
    if len(values) == 1:
        return values * target_length
    raise ValueError(
        f"{name} length {len(values)} does not match expected length {target_length}"
    )


def _loss_component_from_mapping(
    raw_losses: Mapping[str, Any],
    key: str,
    *,
    default_length: int | None = None,
) -> list[float]:
    if key not in raw_losses:
        if default_length is None:
            raise KeyError(f"Missing DiffDock loss component: {key}")
        return [0.0 for _ in range(default_length)]

    return _to_float_list(raw_losses[key], name=key)


def combine_diffdock_loss_components(
    raw_losses: Mapping[str, Any] | Sequence[Any],
    *,
    weights: DiffDockLossWeights | None = None,
) -> DiffDockLossComponents:
    """
    Convert DiffDock per-sample loss outputs into GRPO surrogate scores.

    DiffDock's public `loss_function(..., apply_mean=False)` returns a tuple like:
    `(loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss)`.
    This helper also accepts a mapping with `tr_loss`, `rot_loss`, and `tor_loss`.
    The surrogate score is `s_theta = -weighted_loss`.
    """

    weights = weights or DiffDockLossWeights()

    if isinstance(raw_losses, Mapping):
        tr_loss = _loss_component_from_mapping(raw_losses, "tr_loss")
        target_length = len(tr_loss)
        rot_loss = _broadcast_component(
            _loss_component_from_mapping(raw_losses, "rot_loss"),
            target_length=target_length,
            name="rot_loss",
        )
        tor_loss = _broadcast_component(
            _loss_component_from_mapping(
                raw_losses,
                "tor_loss",
                default_length=target_length,
            ),
            target_length=target_length,
            name="tor_loss",
        )
    else:
        if len(raw_losses) < 4:
            raise ValueError(
                "DiffDock loss tuple must contain at least "
                "(loss, tr_loss, rot_loss, tor_loss)"
            )
        tr_loss = _to_float_list(raw_losses[1], name="tr_loss")
        target_length = len(tr_loss)
        rot_loss = _broadcast_component(
            _to_float_list(raw_losses[2], name="rot_loss"),
            target_length=target_length,
            name="rot_loss",
        )
        tor_loss = _broadcast_component(
            _to_float_list(raw_losses[3], name="tor_loss"),
            target_length=target_length,
            name="tor_loss",
        )

    total_loss = [
        weights.tr * tr_loss[index]
        + weights.rot * rot_loss[index]
        + weights.tor * tor_loss[index]
        for index in range(target_length)
    ]
    surrogate_scores = [-loss for loss in total_loss]

    return DiffDockLossComponents(
        tr_loss=tr_loss,
        rot_loss=rot_loss,
        tor_loss=tor_loss,
        total_loss=total_loss,
        surrogate_scores=surrogate_scores,
    )


class DiffDockLossBackend:
    """Callable backend that returns per-sample DiffDock loss components."""

    def __init__(
        self,
        scorer: Callable[[Sequence[RLExample]], Mapping[str, Any] | Sequence[Any]],
        *,
        weights: DiffDockLossWeights | None = None,
    ) -> None:
        self.scorer = scorer
        self.weights = weights or DiffDockLossWeights()
        self.last_components: DiffDockLossComponents | None = None

    def score_examples(self, examples: Sequence[RLExample]) -> list[float]:
        raw_losses = self.scorer(examples)
        components = combine_diffdock_loss_components(
            raw_losses,
            weights=self.weights,
        )

        if len(components.surrogate_scores) != len(examples):
            raise ValueError(
                "DiffDock loss backend returned "
                f"{len(components.surrogate_scores)} scores for "
                f"{len(examples)} examples"
            )

        self.last_components = components
        return components.surrogate_scores


@contextmanager
def _diffdock_import_path(repo_root: str | Path):
    repo_root = str(Path(repo_root))
    sys.path.insert(0, repo_root)
    try:
        yield
    finally:
        try:
            sys.path.remove(repo_root)
        except ValueError:
            pass


def import_diffdock_loss_function(repo_root: str | Path):
    repo_root = Path(repo_root)
    if not repo_root.is_dir():
        raise FileNotFoundError(f"DiffDock repo root not found: {repo_root}")

    with _diffdock_import_path(repo_root):
        from utils.training import loss_function

    return loss_function


class NativeDiffDockLossBackend(DiffDockLossBackend):
    """
    Adapter for DiffDock's native `loss_function(..., apply_mean=False)`.

    The caller must provide the hard project-specific piece: `batch_builder`,
    which converts final pose `RLExample`s into the DiffDock graph batch expected
    by the model and loss function. Keeping that boundary explicit avoids hiding
    a fragile dependency on a specific DiffDock checkout's dataset internals.
    """

    def __init__(
        self,
        *,
        repo_root: str | Path,
        model: Any,
        t_to_sigma: Callable,
        batch_builder: Callable[[Sequence[RLExample]], Any],
        device: Any,
        weights: DiffDockLossWeights | None = None,
        loss_function: Callable | None = None,
        no_torsion: bool = False,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.model = model
        self.t_to_sigma = t_to_sigma
        self.batch_builder = batch_builder
        self.device = device
        self.loss_function = loss_function or import_diffdock_loss_function(repo_root)
        self.no_torsion = no_torsion
        super().__init__(self._score_with_native_diffdock, weights=weights)

    def _score_with_native_diffdock(
        self,
        examples: Sequence[RLExample],
    ) -> Mapping[str, Any] | Sequence[Any]:
        if not examples:
            return {"tr_loss": [], "rot_loss": [], "tor_loss": []}

        batch = self.batch_builder(examples)
        predictions = self.model(batch)
        if not isinstance(predictions, Sequence):
            raise TypeError("DiffDock model must return prediction tuple/list")
        if len(predictions) < 3:
            raise ValueError("DiffDock model must return at least tr/rot/tor predictions")

        tr_pred, rot_pred, tor_pred = predictions[:3]
        sidechain_pred = predictions[3] if len(predictions) > 3 else None

        signature = inspect.signature(self.loss_function)
        kwargs = {
            "data": batch,
            "t_to_sigma": self.t_to_sigma,
            "device": self.device,
            "tr_weight": 1.0,
            "rot_weight": 1.0,
            "tor_weight": 1.0,
            "apply_mean": False,
            "no_torsion": self.no_torsion,
        }
        if "backbone_weight" in signature.parameters:
            kwargs["backbone_weight"] = 0.0
        if "sidechain_weight" in signature.parameters:
            kwargs["sidechain_weight"] = 0.0

        if "sidechain_pred" in signature.parameters:
            return self.loss_function(
                tr_pred,
                rot_pred,
                tor_pred,
                sidechain_pred,
                **kwargs,
            )

        return self.loss_function(tr_pred, rot_pred, tor_pred, **kwargs)
