from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Callable, Sequence

from src.rl.diffdock_loss import DiffDockLossWeights


@dataclass(frozen=True)
class NativeGRPOStepResult:
    loss: float
    grad_norm: float | None
    old_scores: list[float]
    scores_before: list[float]
    scores_after: list[float]
    tr_loss_before: list[float]
    rot_loss_before: list[float]
    tor_loss_before: list[float]
    total_loss_before: list[float]
    tr_loss_after: list[float]
    rot_loss_after: list[float]
    tor_loss_after: list[float]
    total_loss_after: list[float]
    ratios_before: list[float]
    clipped_ratios_before: list[float]
    objective_terms_before: list[float]


def _as_1d_tensor(value: Any, *, torch_module: Any, device: Any) -> Any:
    if not hasattr(value, "reshape"):
        value = torch_module.as_tensor(value, dtype=torch_module.float32, device=device)
    else:
        value = value.to(device)
    return value.reshape(-1)


def _broadcast_tensor(value: Any, *, target_length: int, torch_module: Any, device: Any) -> Any:
    value = _as_1d_tensor(value, torch_module=torch_module, device=device)
    if value.numel() == target_length:
        return value
    if value.numel() == 1:
        return value.repeat(target_length)
    raise ValueError(
        f"Loss component length {value.numel()} does not match expected {target_length}"
    )


def _call_diffdock_loss_function(
    *,
    model: Any,
    batch: Any,
    loss_function: Callable,
    t_to_sigma: Callable,
    device: Any,
    no_torsion: bool,
) -> Any:
    predictions = model(batch)
    if not isinstance(predictions, Sequence):
        raise TypeError("DiffDock model must return prediction tuple/list")
    if len(predictions) < 3:
        raise ValueError("DiffDock model must return at least tr/rot/tor predictions")

    tr_pred, rot_pred, tor_pred = predictions[:3]
    sidechain_pred = predictions[3] if len(predictions) > 3 else None

    signature = inspect.signature(loss_function)
    kwargs = {
        "data": batch,
        "t_to_sigma": t_to_sigma,
        "device": device,
        "tr_weight": 1.0,
        "rot_weight": 1.0,
        "tor_weight": 1.0,
        "apply_mean": False,
        "no_torsion": no_torsion,
    }
    if "backbone_weight" in signature.parameters:
        kwargs["backbone_weight"] = 0.0
    if "sidechain_weight" in signature.parameters:
        kwargs["sidechain_weight"] = 0.0

    if "sidechain_pred" in signature.parameters:
        return loss_function(
            tr_pred,
            rot_pred,
            tor_pred,
            sidechain_pred,
            **kwargs,
        )

    return loss_function(tr_pred, rot_pred, tor_pred, **kwargs)


def native_loss_components_from_raw(
    raw_losses: Sequence[Any],
    *,
    torch_module: Any,
    device: Any,
    weights: DiffDockLossWeights | None = None,
) -> dict[str, Any]:
    if len(raw_losses) < 4:
        raise ValueError(
            "DiffDock loss tuple must contain at least "
            "(loss, tr_loss, rot_loss, tor_loss)"
        )

    weights = weights or DiffDockLossWeights()
    tr_loss = _as_1d_tensor(raw_losses[1], torch_module=torch_module, device=device)
    target_length = tr_loss.numel()
    rot_loss = _broadcast_tensor(
        raw_losses[2],
        target_length=target_length,
        torch_module=torch_module,
        device=device,
    )
    tor_loss = _broadcast_tensor(
        raw_losses[3],
        target_length=target_length,
        torch_module=torch_module,
        device=device,
    )
    total_loss = weights.tr * tr_loss + weights.rot * rot_loss + weights.tor * tor_loss

    return {
        "tr_loss": tr_loss,
        "rot_loss": rot_loss,
        "tor_loss": tor_loss,
        "total_loss": total_loss,
        "scores": -total_loss,
    }


def compute_clipped_grpo_loss_from_scores(
    *,
    current_scores: Any,
    old_scores: Any,
    advantages: Any,
    torch_module: Any,
    clip_epsilon: float = 0.2,
    max_score_delta: float = 20.0,
) -> tuple[Any, dict[str, Any]]:
    score_delta = torch_module.clamp(
        current_scores - old_scores,
        min=-max_score_delta,
        max=max_score_delta,
    )
    ratios = torch_module.exp(score_delta)
    clipped_ratios = torch_module.clamp(
        ratios,
        min=1.0 - clip_epsilon,
        max=1.0 + clip_epsilon,
    )
    objective_terms = torch_module.minimum(
        ratios * advantages,
        clipped_ratios * advantages,
    )
    loss = -objective_terms.mean()

    return loss, {
        "ratios": ratios,
        "clipped_ratios": clipped_ratios,
        "objective_terms": objective_terms,
    }


def _to_float_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _trainable_parameters(model: Any) -> list[Any]:
    target = model.module if hasattr(model, "module") else model
    return [parameter for parameter in target.parameters() if parameter.requires_grad]


def _state_dict(model: Any) -> Any:
    target = model.module if hasattr(model, "module") else model
    return target.state_dict()


def run_native_diffdock_grpo_step(
    *,
    model: Any,
    batch: Any,
    advantages: Sequence[float],
    loss_function: Callable,
    t_to_sigma: Callable,
    device: Any,
    optimizer: Any,
    torch_module: Any,
    no_torsion: bool = False,
    weights: DiffDockLossWeights | None = None,
    clip_epsilon: float = 0.2,
    max_score_delta: float = 20.0,
    max_grad_norm: float | None = 1.0,
) -> tuple[NativeGRPOStepResult, Any]:
    trainable_parameters = _trainable_parameters(model)
    if not trainable_parameters:
        raise ValueError("No trainable DiffDock parameters found for GRPO step")

    model.train()
    with torch_module.no_grad():
        raw_old = _call_diffdock_loss_function(
            model=model,
            batch=batch,
            loss_function=loss_function,
            t_to_sigma=t_to_sigma,
            device=device,
            no_torsion=no_torsion,
        )
        old_components = native_loss_components_from_raw(
            raw_old,
            torch_module=torch_module,
            device=device,
            weights=weights,
        )
        old_scores = old_components["scores"].detach()

    optimizer.zero_grad(set_to_none=True)
    raw_before = _call_diffdock_loss_function(
        model=model,
        batch=batch,
        loss_function=loss_function,
        t_to_sigma=t_to_sigma,
        device=device,
        no_torsion=no_torsion,
    )
    before_components = native_loss_components_from_raw(
        raw_before,
        torch_module=torch_module,
        device=device,
        weights=weights,
    )
    advantages_tensor = torch_module.as_tensor(
        advantages,
        dtype=torch_module.float32,
        device=device,
    ).reshape(-1)
    if advantages_tensor.numel() != before_components["scores"].numel():
        raise ValueError(
            "Number of advantages does not match native score count: "
            f"{advantages_tensor.numel()} != {before_components['scores'].numel()}"
        )

    loss, objective = compute_clipped_grpo_loss_from_scores(
        current_scores=before_components["scores"],
        old_scores=old_scores,
        advantages=advantages_tensor,
        torch_module=torch_module,
        clip_epsilon=clip_epsilon,
        max_score_delta=max_score_delta,
    )
    loss.backward()

    grad_norm = None
    if max_grad_norm is not None:
        grad_norm_tensor = torch_module.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_grad_norm,
        )
        grad_norm = float(grad_norm_tensor.detach().cpu())

    optimizer.step()

    model.eval()
    with torch_module.no_grad():
        raw_after = _call_diffdock_loss_function(
            model=model,
            batch=batch,
            loss_function=loss_function,
            t_to_sigma=t_to_sigma,
            device=device,
            no_torsion=no_torsion,
        )
        after_components = native_loss_components_from_raw(
            raw_after,
            torch_module=torch_module,
            device=device,
            weights=weights,
        )

    result = NativeGRPOStepResult(
        loss=float(loss.detach().cpu()),
        grad_norm=grad_norm,
        old_scores=_to_float_list(old_scores),
        scores_before=_to_float_list(before_components["scores"]),
        scores_after=_to_float_list(after_components["scores"]),
        tr_loss_before=_to_float_list(before_components["tr_loss"]),
        rot_loss_before=_to_float_list(before_components["rot_loss"]),
        tor_loss_before=_to_float_list(before_components["tor_loss"]),
        total_loss_before=_to_float_list(before_components["total_loss"]),
        tr_loss_after=_to_float_list(after_components["tr_loss"]),
        rot_loss_after=_to_float_list(after_components["rot_loss"]),
        tor_loss_after=_to_float_list(after_components["tor_loss"]),
        total_loss_after=_to_float_list(after_components["total_loss"]),
        ratios_before=_to_float_list(objective["ratios"]),
        clipped_ratios_before=_to_float_list(objective["clipped_ratios"]),
        objective_terms_before=_to_float_list(objective["objective_terms"]),
    )

    return result, _state_dict(model)
