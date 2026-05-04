#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  SMOKE_COMPLEX_ID=<pdb_id> scripts/run_rl_posttraining_smoke.sh [options]

Options:
  --smoke-complex-id ID     Real PDBBind-style complex ID. Defaults to SMOKE_COMPLEX_ID.
  --raw-root PATH           Root containing real complexes. Default: data/raw/pdbbind_real
  --baseline-config PATH    DiffDock rollout config. Default: configs/diffdock/smoke_top4.yaml
  --rl-config PATH          Reward-debug config. Default: configs/rl/offline_reward_smoke.yaml
  --grpo-config PATH        GRPO config. Default: configs/rl/grpo_surrogate_smoke.yaml
  --seed N                  Random seed. Default: 42
  --run-tag TAG             Run tag. Default: rl_smoke_<ID>_<HHMMSS>
  --package-output-dir DIR  Output dir for packaged archives. Default: packaged_runs
  --include-inputs          Include protein/ligand/reference structures in packages
  --skip-grpo               Skip the one-step GRPO surrogate smoke run
  --skip-package            Run smoke without packaging artifacts
  -h, --help                Show this help message

This script runs a one-complex RL smoke workflow:
  1. Build smoke split + manifest for one real complex
  2. Generate 4 DiffDock poses
  3. Run evaluation and structure diagnostics
  4. Run offline RL reward/advantage posttraining smoke
  5. Run one-step GRPO surrogate smoke
  6. Package baseline and posttraining artifacts

The GRPO smoke validates objective signs and checkpoint artifacts with a
debug-linear surrogate. It does not update DiffDock score-model weights yet.
EOF
}

SMOKE_COMPLEX_ID="${SMOKE_COMPLEX_ID:-}"
RAW_ROOT="${PDBBIND_RAW_ROOT:-data/raw/pdbbind_real}"
BASELINE_CONFIG="configs/diffdock/smoke_top4.yaml"
RL_CONFIG="configs/rl/offline_reward_smoke.yaml"
GRPO_CONFIG="configs/rl/grpo_surrogate_smoke.yaml"
SEED="42"
RUN_TAG=""
PACKAGE_OUTPUT_DIR="packaged_runs"
INCLUDE_INPUTS="false"
SKIP_GRPO="false"
SKIP_PACKAGE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke-complex-id)
      SMOKE_COMPLEX_ID="$2"
      shift 2
      ;;
    --raw-root)
      RAW_ROOT="$2"
      shift 2
      ;;
    --baseline-config)
      BASELINE_CONFIG="$2"
      shift 2
      ;;
    --rl-config)
      RL_CONFIG="$2"
      shift 2
      ;;
    --grpo-config)
      GRPO_CONFIG="$2"
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
    --package-output-dir)
      PACKAGE_OUTPUT_DIR="$2"
      shift 2
      ;;
    --include-inputs)
      INCLUDE_INPUTS="true"
      shift
      ;;
    --skip-grpo)
      SKIP_GRPO="true"
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

if [[ ! -d "scripts" || ! -d "src" ]]; then
  echo "Run this script from the repository root." >&2
  exit 1
fi

if [[ -z "$SMOKE_COMPLEX_ID" ]]; then
  echo "Set SMOKE_COMPLEX_ID or pass --smoke-complex-id." >&2
  usage >&2
  exit 1
fi

if [[ -z "$RUN_TAG" ]]; then
  RUN_TAG="rl_smoke_${SMOKE_COMPLEX_ID}_$(date +%H%M%S)"
fi

for required_file in \
  "$RAW_ROOT/$SMOKE_COMPLEX_ID/protein.pdb" \
  "$RAW_ROOT/$SMOKE_COMPLEX_ID/ligand.sdf" \
  "$RAW_ROOT/$SMOKE_COMPLEX_ID/ligand_gt.sdf"
do
  if [[ ! -s "$required_file" ]]; then
    echo "Missing required smoke input: $required_file" >&2
    exit 1
  fi
done

if [[ -n "${CONDA_PREFIX:-}" && -z "${DIFFDOCK_PYTHON:-}" ]]; then
  export DIFFDOCK_PYTHON="$CONDA_PREFIX/bin/python"
fi

export SMOKE_COMPLEX_ID
export PDBBIND_RAW_ROOT="$RAW_ROOT"

mkdir -p artifacts/runs

uv run python - <<'PY'
import os
from pathlib import Path

from src.data.manifests import build_and_save_manifest

complex_id = os.environ["SMOKE_COMPLEX_ID"]
raw_root = Path(os.environ.get("PDBBIND_RAW_ROOT", "data/raw/pdbbind_real"))

split_path = Path("data/processed/diffdock/splits/smoke.txt")
split_path.parent.mkdir(parents=True, exist_ok=True)
split_path.write_text(complex_id + "\n", encoding="utf-8")

build_and_save_manifest(
    ids_path=split_path,
    raw_root=raw_root,
    split="smoke",
    output_path=Path("data/processed/diffdock/manifests/smoke_manifest.json"),
)
PY

RUNS_BEFORE="$(mktemp)"
RUNS_AFTER="$(mktemp)"
cleanup() {
  rm -f "$RUNS_BEFORE" "$RUNS_AFTER"
}
trap cleanup EXIT

find artifacts/runs -mindepth 1 -maxdepth 1 -type d -print | sort > "$RUNS_BEFORE" 2>/dev/null || true

echo "==> Running one-complex DiffDock rollout"
uv run python -m src.pipeline.run_baseline \
  --config "$BASELINE_CONFIG" \
  --seed "$SEED" \
  --run-tag "$RUN_TAG"

find artifacts/runs -mindepth 1 -maxdepth 1 -type d -print | sort > "$RUNS_AFTER"
BASELINE_RUN_DIR="$(comm -13 "$RUNS_BEFORE" "$RUNS_AFTER" | sort -r | head -1)"

if [[ -z "$BASELINE_RUN_DIR" || ! -d "$BASELINE_RUN_DIR" ]]; then
  echo "Could not detect baseline run directory created by smoke rollout." >&2
  exit 1
fi

echo "==> Baseline rollout directory: $BASELINE_RUN_DIR"

echo "==> Running evaluation"
./scripts/run_evaluation.sh "$BASELINE_RUN_DIR"

echo "==> Running structure diagnostics"
./scripts/diagnose_run_structures.py "$BASELINE_RUN_DIR"

find artifacts/runs -mindepth 1 -maxdepth 1 -type d -print | sort > "$RUNS_BEFORE"

echo "==> Running RL posttraining smoke"
uv run python -m src.pipeline.run_posttraining \
  --config "$RL_CONFIG" \
  --source-run-dir "$BASELINE_RUN_DIR" \
  --run-tag "${RUN_TAG}_reward_debug" \
  --seed "$SEED"

find artifacts/runs -mindepth 1 -maxdepth 1 -type d -print | sort > "$RUNS_AFTER"
RL_RUN_DIR="$(comm -13 "$RUNS_BEFORE" "$RUNS_AFTER" | sort -r | head -1)"

if [[ -z "$RL_RUN_DIR" || ! -d "$RL_RUN_DIR" ]]; then
  echo "Could not detect RL posttraining run directory." >&2
  exit 1
fi

echo "==> RL smoke directory: $RL_RUN_DIR"

GRPO_RUN_DIR=""
if [[ "$SKIP_GRPO" == "false" ]]; then
  find artifacts/runs -mindepth 1 -maxdepth 1 -type d -print | sort > "$RUNS_BEFORE"

  echo "==> Running GRPO surrogate smoke"
  uv run python -m src.pipeline.run_posttraining \
    --config "$GRPO_CONFIG" \
    --source-run-dir "$BASELINE_RUN_DIR" \
    --run-tag "${RUN_TAG}_grpo" \
    --seed "$SEED"

  find artifacts/runs -mindepth 1 -maxdepth 1 -type d -print | sort > "$RUNS_AFTER"
  GRPO_RUN_DIR="$(comm -13 "$RUNS_BEFORE" "$RUNS_AFTER" | sort -r | head -1)"

  if [[ -z "$GRPO_RUN_DIR" || ! -d "$GRPO_RUN_DIR" ]]; then
    echo "Could not detect GRPO smoke run directory." >&2
    exit 1
  fi

  echo "==> GRPO smoke directory: $GRPO_RUN_DIR"
fi

if [[ "$SKIP_PACKAGE" == "false" ]]; then
  echo "==> Packaging baseline rollout artifacts"
  BASELINE_PACKAGE_ARGS=("$BASELINE_RUN_DIR" --key --output-dir "$PACKAGE_OUTPUT_DIR")
  RL_PACKAGE_ARGS=("$RL_RUN_DIR" --key --output-dir "$PACKAGE_OUTPUT_DIR")
  GRPO_PACKAGE_ARGS=("$GRPO_RUN_DIR" --key --output-dir "$PACKAGE_OUTPUT_DIR")

  if [[ "$INCLUDE_INPUTS" == "true" ]]; then
    BASELINE_PACKAGE_ARGS+=(--include-inputs)
    RL_PACKAGE_ARGS+=(--include-inputs)
    GRPO_PACKAGE_ARGS+=(--include-inputs)
  fi

  ./scripts/package_run_artifacts.sh "${BASELINE_PACKAGE_ARGS[@]}"

  echo "==> Packaging RL posttraining smoke artifacts"
  ./scripts/package_run_artifacts.sh "${RL_PACKAGE_ARGS[@]}"

  if [[ -n "$GRPO_RUN_DIR" ]]; then
    echo "==> Packaging GRPO smoke artifacts"
    ./scripts/package_run_artifacts.sh "${GRPO_PACKAGE_ARGS[@]}"
  fi
fi

echo "==> RL posttraining smoke complete"
echo "Baseline rollout directory:"
echo "  $BASELINE_RUN_DIR"
echo "RL smoke directory:"
echo "  $RL_RUN_DIR"
if [[ -n "$GRPO_RUN_DIR" ]]; then
  echo "GRPO smoke directory:"
  echo "  $GRPO_RUN_DIR"
fi
