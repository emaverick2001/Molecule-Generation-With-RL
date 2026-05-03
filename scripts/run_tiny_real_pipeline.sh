#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_tiny_real_pipeline.sh [options]

Options:
  --source DIR_OR_ARCHIVE   PDBBind source root/archive. Default: data/raw/pdbbind/P-L
  --num-complexes N         Number of complexes to sample. Default: 5
  --seed N                  Random seed. Default: 42
  --output-root DIR         Normalized real complex root. Default: data/raw/pdbbind_real
  --exclude-ids-file PATH   IDs to exclude from random sampling. Default: data/processed/diffdock/splits/exclude_ids.txt
  --baseline-config PATH    Baseline config. Default: configs/diffdock/tiny_real.yaml
  --package-mode MODE       key or full. Default: key
  --package-output-dir DIR  Output dir for packaged archive. Default: packaged_runs
  --include-inputs          Include input protein/ligand/reference structures in package
  --skip-package           Run generation/evaluation without packaging artifacts
  -h, --help               Show this help message

This script runs:
  1. scripts/create_tiny_real_pdbbind.py --random
  2. scripts/run_diffdock_tiny_real.sh
  3. scripts/run_evaluation.sh <new_run_dir>
  4. scripts/package_run_artifacts.sh <new_run_dir>
EOF
}

SOURCE="data/raw/pdbbind/P-L"
NUM_COMPLEXES="5"
SEED="42"
OUTPUT_ROOT="data/raw/pdbbind_real"
EXCLUDE_IDS_FILE="data/processed/diffdock/splits/exclude_ids.txt"
BASELINE_CONFIG="configs/diffdock/tiny_real.yaml"
PACKAGE_MODE="key"
PACKAGE_OUTPUT_DIR="packaged_runs"
SKIP_PACKAGE="false"
INCLUDE_INPUTS="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --num-complexes)
      NUM_COMPLEXES="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --exclude-ids-file)
      EXCLUDE_IDS_FILE="$2"
      shift 2
      ;;
    --baseline-config)
      BASELINE_CONFIG="$2"
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

echo "==> Building tiny real manifest"
uv run python scripts/create_tiny_real_pdbbind.py \
  --source "$SOURCE" \
  --random \
  --num-complexes "$NUM_COMPLEXES" \
  --seed "$SEED" \
  --output-root "$OUTPUT_ROOT" \
  --exclude-ids-file "$EXCLUDE_IDS_FILE"

RUNS_BEFORE="$(mktemp)"
RUNS_AFTER="$(mktemp)"
find artifacts/runs -mindepth 1 -maxdepth 1 -type d -print | sort > "$RUNS_BEFORE" 2>/dev/null || true

echo "==> Running DiffDock tiny real baseline"
uv run python -m src.pipeline.run_baseline \
  --config "$BASELINE_CONFIG" \
  --seed "$SEED"

find artifacts/runs -mindepth 1 -maxdepth 1 -type d -print | sort > "$RUNS_AFTER"

NEW_RUN_DIR="$(comm -13 "$RUNS_BEFORE" "$RUNS_AFTER" | sort -r | head -1)"

if [[ -z "$NEW_RUN_DIR" ]]; then
  NEW_RUN_DIR="$(ls -td artifacts/runs/* | head -1)"
fi

rm -f "$RUNS_BEFORE" "$RUNS_AFTER"

if [[ -z "$NEW_RUN_DIR" || ! -d "$NEW_RUN_DIR" ]]; then
  echo "Could not detect run directory created by DiffDock baseline." >&2
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

echo "==> Tiny real pipeline complete"
echo "Run directory:"
echo "  $NEW_RUN_DIR"
