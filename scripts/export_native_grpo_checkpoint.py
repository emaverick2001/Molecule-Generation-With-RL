#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rl.checkpoint_export import export_native_grpo_checkpoint_for_diffdock_inference
from src.utils.artifact_logger import save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a native GRPO training checkpoint into a DiffDock inference "
            "model directory."
        ),
    )
    parser.add_argument("checkpoint_path")
    parser.add_argument(
        "--source-model-dir",
        default="external/DiffDock/workdir/v1.1/score_model",
        help="DiffDock score model directory containing model_parameters.yml.",
    )
    parser.add_argument(
        "--output-model-dir",
        required=True,
        help="Directory to write model_parameters.yml and the exported checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="native_grpo_inference.pt",
        help="File name for the exported DiffDock inference checkpoint.",
    )
    args = parser.parse_args()

    export = export_native_grpo_checkpoint_for_diffdock_inference(
        checkpoint_path=args.checkpoint_path,
        source_model_dir=args.source_model_dir,
        output_model_dir=args.output_model_dir,
        checkpoint_name=args.checkpoint_name,
    )
    save_json(export.to_dict(), Path(args.output_model_dir) / "export_metadata.json")

    print("Exported native GRPO checkpoint for DiffDock inference")
    print(f"output_model_dir={export.output_model_dir}")
    print(f"checkpoint={export.checkpoint_path}")
    print(f"ckpt_arg={export.checkpoint_name}")


if __name__ == "__main__":
    main()
