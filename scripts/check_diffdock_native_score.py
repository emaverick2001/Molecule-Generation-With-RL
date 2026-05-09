#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rl.data import load_offline_rl_examples
from src.rl.diffdock_batch_builder import DiffDockGeneratedPoseBatchBuilder
from src.rl.diffdock_loss import NativeDiffDockLossBackend
from src.rl.diffdock_model import (
    load_diffdock_score_model,
    load_score_model_args,
    score_model_uses_lm_embeddings,
)
from src.utils.artifact_logger import save_csv


def _select_device(requested_device: str):
    import torch

    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    return device


def _load_examples(run_dir: Path, limit: int):
    examples = load_offline_rl_examples(
        run_dir / "input_manifest.json",
        run_dir / "generated_samples_manifest.json",
        source_run_id=run_dir.name,
        source_run_dir=run_dir,
    )
    examples = examples[:limit]
    if not examples:
        raise ValueError(f"No generated examples found in run: {run_dir}")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load a DiffDock score model, build a native-loss batch from generated "
            "poses, and print per-sample tr/rot/tor losses plus s_theta."
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
        default="external/DiffDock/workdir/v1.1/score_model",
        help="DiffDock score model directory containing model_parameters.yml and ckpt.",
    )
    parser.add_argument(
        "--ckpt",
        default="best_ema_inference_epoch_model.pt",
        help="Score model checkpoint filename inside --model-dir.",
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
        help="Maximum generated poses to score.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for model scoring.",
    )
    parser.add_argument(
        "--work-dir",
        default="artifacts/tmp/diffdock_rl_batches",
        help="Temporary directory for DiffDock graph construction.",
    )
    parser.add_argument(
        "--lm-embeddings",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable DiffDock ESM language-model embeddings during graph construction. "
            "Default is auto-detected from model_parameters.yml."
        ),
    )
    parser.add_argument(
        "--old-score-model",
        action="store_true",
        help="Use DiffDock's old score-model architecture flag.",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help=(
            "Disable DiffDock/PyG DataParallel. On CUDA, the default is parallel "
            "because DiffDock's loss_function expects list data."
        ),
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use strict checkpoint loading.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to save the per-sample score table.",
    )
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    run_dir = Path(args.run_dir)
    device = _select_device(args.device)
    examples = _load_examples(run_dir, args.limit)
    score_model_args = load_score_model_args(
        model_dir=args.model_dir,
        model_args=args.model_args,
    )
    model_needs_lm_embeddings = score_model_uses_lm_embeddings(score_model_args)
    if args.lm_embeddings is None:
        lm_embeddings = model_needs_lm_embeddings
    else:
        lm_embeddings = args.lm_embeddings

    if model_needs_lm_embeddings and not lm_embeddings:
        raise ValueError(
            "The score model appears to require ESM/LM receptor embeddings "
            "from model_parameters.yml, but --no-lm-embeddings was passed. "
            "Rerun with --lm-embeddings or omit the flag and let the checker "
            "auto-detect the setting."
        )

    # DiffDock's non-mean loss path expects list data on CUDA. Keep the default
    # model parallel wrapper on CUDA so model(batch) and loss_function(data=batch)
    # agree on the same list-style batch.
    no_parallel = args.no_parallel or device.type != "cuda"
    bundle = load_diffdock_score_model(
        repo_root=args.repo_root,
        model_dir=args.model_dir,
        ckpt=args.ckpt,
        device=device,
        score_model_args=score_model_args,
        old_score_model=args.old_score_model,
        no_parallel=no_parallel,
        strict=args.strict,
    )
    batch_builder = DiffDockGeneratedPoseBatchBuilder.from_score_model_args(
        repo_root=args.repo_root,
        score_model_args=bundle.score_model_args,
        device=device,
        work_dir=args.work_dir,
        lm_embeddings=lm_embeddings,
    )
    backend = NativeDiffDockLossBackend(
        repo_root=args.repo_root,
        model=bundle.model,
        t_to_sigma=bundle.t_to_sigma,
        batch_builder=batch_builder,
        device=device,
        no_torsion=getattr(bundle.score_model_args, "no_torsion", False),
    )

    import torch

    with torch.no_grad():
        backend.score_examples(examples)

    rows = backend.last_components.to_rows(examples)
    if args.output_csv:
        save_csv(rows, args.output_csv)

    print("DiffDock native score check: ok")
    print(f"run_dir={run_dir}")
    print(f"repo_root={args.repo_root}")
    print(f"model_dir={args.model_dir}")
    print(f"ckpt={args.ckpt}")
    print(f"checkpoint_path={bundle.checkpoint_path}")
    print(f"device={device}")
    print(f"model_parallel={not no_parallel}")
    print(f"lm_embeddings={lm_embeddings}")
    print(f"model_needs_lm_embeddings={model_needs_lm_embeddings}")
    print(f"examples={len(examples)}")
    print("complex_id,sample_id,rank,tr_loss,rot_loss,tor_loss,diffdock_loss,s_theta")
    for row in rows:
        print(
            ",".join(
                [
                    str(row["complex_id"]),
                    str(row["sample_id"]),
                    str(row["rank"]),
                    f"{row['tr_loss']:.6f}",
                    f"{row['rot_loss']:.6f}",
                    f"{row['tor_loss']:.6f}",
                    f"{row['diffdock_loss']:.6f}",
                    f"{row['surrogate_score']:.6f}",
                ]
            )
        )
    if args.output_csv:
        print(f"Wrote native score rows: {args.output_csv}")


if __name__ == "__main__":
    main()
