#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rl.diffdock_loss import import_diffdock_loss_function


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that the DiffDock native loss_function can be imported.",
    )
    parser.add_argument(
        "--repo-root",
        default="external/DiffDock",
        help="Path to the DiffDock checkout.",
    )
    args = parser.parse_args()

    loss_function = import_diffdock_loss_function(Path(args.repo_root))
    signature = inspect.signature(loss_function)
    source_file = inspect.getsourcefile(loss_function)

    print("DiffDock loss backend import: ok")
    print(f"repo_root={args.repo_root}")
    print(f"loss_function={loss_function.__module__}.{loss_function.__name__}")
    print(f"source_file={source_file}")
    print(f"signature={signature}")


if __name__ == "__main__":
    main()
