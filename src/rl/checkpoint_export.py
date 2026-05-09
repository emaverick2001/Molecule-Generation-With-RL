from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
from typing import Any

from src.rl.diffdock_model import _extract_state_dict


@dataclass(frozen=True)
class DiffDockInferenceCheckpointExport:
    source_checkpoint_path: str
    source_model_dir: str
    output_model_dir: str
    checkpoint_name: str
    checkpoint_path: str
    model_parameters_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def export_native_grpo_checkpoint_for_diffdock_inference(
    *,
    checkpoint_path: str | Path,
    source_model_dir: str | Path,
    output_model_dir: str | Path,
    checkpoint_name: str = "native_grpo_inference.pt",
    torch_module: Any | None = None,
) -> DiffDockInferenceCheckpointExport:
    """
    Convert a native GRPO checkpoint into the shape expected by DiffDock inference.

    Native GRPO checkpoints store training metadata and optimizer state. DiffDock's
    inference entry point loads ``args.model_dir/args.ckpt`` directly into
    ``model.load_state_dict(...)``, so this helper exports only the model state dict
    beside a copied ``model_parameters.yml``.
    """
    checkpoint_path = Path(checkpoint_path)
    source_model_dir = Path(source_model_dir)
    output_model_dir = Path(output_model_dir)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Native GRPO checkpoint not found: {checkpoint_path}")

    source_model_parameters = source_model_dir / "model_parameters.yml"
    if not source_model_parameters.is_file():
        raise FileNotFoundError(
            f"DiffDock model parameters not found: {source_model_parameters}"
        )

    if "/" in checkpoint_name or "\\" in checkpoint_name:
        raise ValueError("checkpoint_name must be a file name, not a path")

    if torch_module is None:
        import torch

        torch_module = torch

    output_model_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint_path = output_model_dir / checkpoint_name
    output_model_parameters = output_model_dir / "model_parameters.yml"

    payload = torch_module.load(
        checkpoint_path,
        map_location=torch_module.device("cpu"),
    )
    state_dict = _extract_state_dict(payload)

    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"Checkpoint does not contain a model state dict: {checkpoint_path}")

    torch_module.save(state_dict, output_checkpoint_path)
    shutil.copyfile(source_model_parameters, output_model_parameters)

    return DiffDockInferenceCheckpointExport(
        source_checkpoint_path=str(checkpoint_path),
        source_model_dir=str(source_model_dir),
        output_model_dir=str(output_model_dir),
        checkpoint_name=checkpoint_name,
        checkpoint_path=str(output_checkpoint_path),
        model_parameters_path=str(output_model_parameters),
    )
