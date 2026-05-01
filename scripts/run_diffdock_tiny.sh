#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/create_tiny_pdbbind_mvp.py --exist-ok

uv run python -m src.pipeline.run_baseline \
  --config configs/diffdock/tiny.yaml \
  --exist-ok
