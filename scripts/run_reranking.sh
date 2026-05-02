#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_dir>" >&2
  exit 1
fi

uv run python -m src.pipeline.run_reranking \
  --run-dir "$1" \
  "${@:2}"
