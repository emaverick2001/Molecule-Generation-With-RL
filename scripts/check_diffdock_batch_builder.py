#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rl.data import load_offline_rl_examples
from src.rl.diffdock_batch_builder import DiffDockGeneratedPoseBatchBuilder


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{str(key): _to_namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _load_score_model_args(
    *,
    model_dir: str | Path | None,
    model_args: str | Path | None,
) -> Any:
    if model_args is not None:
        args_path = Path(model_args)
    elif model_dir is not None:
        args_path = Path(model_dir) / "model_parameters.yml"
    else:
        return SimpleNamespace()

    if not args_path.is_file():
        raise FileNotFoundError(f"DiffDock model args file not found: {args_path}")

    from src.utils.config import load_yaml

    return _to_namespace(load_yaml(args_path))


def _select_device(requested_device: str):
    import torch

    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    return device


def _graph_list(batch: Any) -> list[Any]:
    if isinstance(batch, list):
        return batch
    if hasattr(batch, "to_data_list"):
        return list(batch.to_data_list())
    return [batch]


def _graph_name(graph: Any) -> str:
    try:
        return str(graph["name"])
    except Exception:
        return str(getattr(graph, "name", "unknown"))


def _has_attr(graph: Any, name: str) -> bool:
    try:
        return hasattr(graph, name)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a DiffDock native-loss batch from a generated run manifest. "
            "Use this on ICRN after a successful DiffDock rollout."
        ),
    )
    parser.add_argument(
        "run_dir",
        help="Run directory containing input_manifest.json and generated_samples_manifest.json.",
    )
    parser.add_argument(
        "--repo-root",
        default="external/DiffDock",
        help="Path to the external DiffDock checkout.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="DiffDock score model directory containing model_parameters.yml.",
    )
    parser.add_argument(
        "--model-args",
        default=None,
        help="Explicit DiffDock model_parameters.yml path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Maximum generated poses to include in the test batch.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for the returned batch.",
    )
    parser.add_argument(
        "--work-dir",
        default="artifacts/tmp/diffdock_rl_batches",
        help="Temporary directory for DiffDock graph construction.",
    )
    parser.add_argument(
        "--lm-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable DiffDock ESM language-model embeddings during graph construction. "
            "Default is disabled for a fast backend smoke check."
        ),
    )
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    run_dir = Path(args.run_dir)
    input_manifest = run_dir / "input_manifest.json"
    generated_manifest = run_dir / "generated_samples_manifest.json"

    examples = load_offline_rl_examples(
        input_manifest,
        generated_manifest,
        source_run_id=run_dir.name,
        source_run_dir=run_dir,
    )
    examples = examples[: args.limit]
    if not examples:
        raise ValueError(f"No generated examples found in run: {run_dir}")

    score_model_args = _load_score_model_args(
        model_dir=args.model_dir,
        model_args=args.model_args,
    )
    device = _select_device(args.device)

    builder = DiffDockGeneratedPoseBatchBuilder.from_score_model_args(
        repo_root=args.repo_root,
        score_model_args=score_model_args,
        device=device,
        work_dir=args.work_dir,
        lm_embeddings=args.lm_embeddings,
    )
    batch = builder(examples)
    graphs = _graph_list(batch)

    print("DiffDock batch builder: ok")
    print(f"run_dir={run_dir}")
    print(f"repo_root={args.repo_root}")
    print(f"device={device}")
    print(f"lm_embeddings={args.lm_embeddings}")
    print(f"examples={len(examples)}")
    print(f"graphs={len(graphs)}")
    for graph in graphs:
        ligand_atom_count = graph["ligand"].pos.shape[0]
        print(
            "graph="
            f"{_graph_name(graph)} "
            f"ligand_atoms={ligand_atom_count} "
            f"has_tr_score={_has_attr(graph, 'tr_score')} "
            f"has_rot_score={_has_attr(graph, 'rot_score')} "
            f"has_tor_score={_has_attr(graph, 'tor_score')}"
        )


if __name__ == "__main__":
    main()
