#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CONDA_PREFIX:-}" && -z "${DIFFDOCK_PYTHON:-}" ]]; then
  export DIFFDOCK_PYTHON="$CONDA_PREFIX/bin/python"
fi

uv run python scripts/create_tiny_pdbbind_mvp.py --exist-ok

uv run python -m src.pipeline.run_baseline \
  --config configs/diffdock/tiny.yaml \
  --exist-ok
