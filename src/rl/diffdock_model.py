from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

from src.rl.diffdock_loss import _diffdock_import_path
from src.utils.config import load_yaml


@dataclass(frozen=True)
class DiffDockScoreModelBundle:
    model: Any
    score_model_args: Namespace
    t_to_sigma: Callable
    checkpoint_path: Path


def load_score_model_args(
    *,
    model_dir: str | Path | None = None,
    model_args: str | Path | None = None,
) -> Namespace:
    if model_args is not None:
        args_path = Path(model_args)
    elif model_dir is not None:
        args_path = Path(model_dir) / "model_parameters.yml"
    else:
        raise ValueError("Either model_dir or model_args is required")

    if not args_path.is_file():
        raise FileNotFoundError(f"DiffDock model args file not found: {args_path}")

    return Namespace(**load_yaml(args_path))


def _extract_state_dict(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload

    for key in ["state_dict", "model_state_dict", "model"]:
        nested = payload.get(key)
        if isinstance(nested, dict):
            return nested

    return payload


def _load_state_dict(
    *,
    model: Any,
    state_dict: Any,
    strict: bool,
) -> Any:
    target = model.module if hasattr(model, "module") else model
    return target.load_state_dict(state_dict, strict=strict)


def load_diffdock_score_model(
    *,
    repo_root: str | Path,
    model_dir: str | Path,
    ckpt: str = "best_ema_inference_epoch_model.pt",
    device: Any,
    score_model_args: Namespace | None = None,
    old_score_model: bool = False,
    no_parallel: bool = False,
    strict: bool = True,
    get_model_fn: Callable | None = None,
    t_to_sigma_fn: Callable | None = None,
    torch_module: Any | None = None,
) -> DiffDockScoreModelBundle:
    model_dir = Path(model_dir)
    checkpoint_path = model_dir / ckpt
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DiffDock checkpoint not found: {checkpoint_path}")

    score_model_args = score_model_args or load_score_model_args(model_dir=model_dir)

    if get_model_fn is None or t_to_sigma_fn is None or torch_module is None:
        with _diffdock_import_path(repo_root):
            from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
            from utils.utils import get_model
            import torch

        get_model_fn = get_model_fn or get_model
        t_to_sigma_fn = t_to_sigma_fn or t_to_sigma_compl
        torch_module = torch_module or torch

    t_to_sigma = partial(t_to_sigma_fn, args=score_model_args)
    model = get_model_fn(
        score_model_args,
        device,
        t_to_sigma=t_to_sigma,
        no_parallel=no_parallel,
        old=old_score_model,
    )
    payload = torch_module.load(
        checkpoint_path,
        map_location=torch_module.device("cpu"),
    )
    _load_state_dict(
        model=model,
        state_dict=_extract_state_dict(payload),
        strict=strict,
    )
    model = model.to(device)
    model.eval()

    return DiffDockScoreModelBundle(
        model=model,
        score_model_args=score_model_args,
        t_to_sigma=t_to_sigma,
        checkpoint_path=checkpoint_path,
    )
