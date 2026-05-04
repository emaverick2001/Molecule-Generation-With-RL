#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_frozen_eval_pipeline.sh [options]

Options:
  --config PATH             Baseline config. Default: configs/diffdock/main_eval_top10.yaml
  --seed N                  Random seed. Default: 42
  --run-tag TAG             Run ID tag. Default: main_eval
  --package-mode MODE       key or full. Default: key
  --package-output-dir DIR  Output dir for packaged archive. Default: packaged_runs
  --include-inputs          Include input protein/ligand/reference structures in package
  --skip-package            Run generation/evaluation without packaging artifacts
  -h, --help                Show this help message

This script assumes a frozen manifest already exists and runs:
  1. src.pipeline.run_baseline using the fixed config
  2. scripts/run_evaluation.sh <new_run_dir>
  3. scripts/diagnose_run_structures.py <new_run_dir>
  4. scripts/package_run_artifacts.sh <new_run_dir>
EOF
}

CONFIG="configs/diffdock/main_eval_top10.yaml"
SEED="42"
RUN_TAG="main_eval"
PACKAGE_MODE="key"
PACKAGE_OUTPUT_DIR="packaged_runs"
INCLUDE_INPUTS="false"
SKIP_PACKAGE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --package-mode)
      PACKAGE_MODE="$2"
      shift 2
      ;;
    --package-output-dir)
      PACKAGE_OUTPUT_DIR="$2"
      shift 2
      ;;
    --include-inputs)
      INCLUDE_INPUTS="true"
      shift
      ;;
    --skip-package)
      SKIP_PACKAGE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$PACKAGE_MODE" != "key" && "$PACKAGE_MODE" != "full" ]]; then
  echo "--package-mode must be 'key' or 'full'." >&2
  exit 1
fi

if [[ ! -d "scripts" || ! -d "src" ]]; then
  echo "Run this script from the repository root." >&2
  exit 1
fi

if [[ -n "${CONDA_PREFIX:-}" && -z "${DIFFDOCK_PYTHON:-}" ]]; then
  export DIFFDOCK_PYTHON="$CONDA_PREFIX/bin/python"
fi

RUNS_BEFORE="$(mktemp)"
RUNS_AFTER="$(mktemp)"
find artifacts/runs -mindepth 1 -maxdepth 1 -type d -print | sort > "$RUNS_BEFORE" 2>/dev/null || true

echo "==> Running frozen DiffDock baseline"
uv run python -m src.pipeline.run_baseline \
  --config "$CONFIG" \
  --seed "$SEED" \
  --run-tag "$RUN_TAG"

find artifacts/runs -mindepth 1 -maxdepth 1 -type d -print | sort > "$RUNS_AFTER"
NEW_RUN_DIR="$(comm -13 "$RUNS_BEFORE" "$RUNS_AFTER" | sort -r | head -1)"

if [[ -z "$NEW_RUN_DIR" ]]; then
  NEW_RUN_DIR="$(ls -td artifacts/runs/* | head -1)"
fi

rm -f "$RUNS_BEFORE" "$RUNS_AFTER"

if [[ -z "$NEW_RUN_DIR" || ! -d "$NEW_RUN_DIR" ]]; then
  echo "Could not detect run directory created by baseline." >&2
  exit 1
fi

echo "==> Detected run directory: $NEW_RUN_DIR"

echo "==> Running evaluation"
./scripts/run_evaluation.sh "$NEW_RUN_DIR"

echo "==> Running structure diagnostics"
./scripts/diagnose_run_structures.py "$NEW_RUN_DIR"

if [[ "$SKIP_PACKAGE" == "false" ]]; then
  echo "==> Packaging run artifacts"
  PACKAGE_ARGS=(
    "$NEW_RUN_DIR"
    "--$PACKAGE_MODE"
    --output-dir "$PACKAGE_OUTPUT_DIR"
  )

  if [[ "$INCLUDE_INPUTS" == "true" ]]; then
    PACKAGE_ARGS+=(--include-inputs)
  fi

  ./scripts/package_run_artifacts.sh "${PACKAGE_ARGS[@]}"
fi

echo "==> Frozen eval pipeline complete"
echo "Run directory:"
echo "  $NEW_RUN_DIR"
