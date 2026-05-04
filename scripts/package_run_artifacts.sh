#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

usage() {
  cat <<'EOF'
Usage:
  scripts/package_run_artifacts.sh [RUN_DIR] [--full|--key] [--include-inputs] [--output-dir DIR]

Examples:
  scripts/package_run_artifacts.sh
  scripts/package_run_artifacts.sh artifacts/runs/2026-05-02_diffdock_baseline_seed42
  scripts/package_run_artifacts.sh artifacts/runs/2026-05-02_diffdock_baseline_seed42 --full
  scripts/package_run_artifacts.sh artifacts/runs/2026-05-02_diffdock_baseline_seed42 --include-inputs
  scripts/package_run_artifacts.sh --output-dir ~/packaged_runs

Defaults:
  RUN_DIR      latest artifacts/runs/* directory
  mode         --key
  output-dir   ~/Molecule-Generation-With-RL/packaged_runs

--key includes configs, manifests, logs, generated samples, metrics, summaries,
and raw DiffDock outputs for the selected run. --full packages the entire run
directory. --include-inputs also includes protein.pdb, ligand.sdf, and
ligand_gt.sdf files referenced by input_manifest.json.
EOF
}

MODE="key"
RUN_DIR=""
OUTPUT_DIR=""
INCLUDE_INPUTS="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)
      MODE="full"
      shift
      ;;
    --key)
      MODE="key"
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --include-inputs)
      INCLUDE_INPUTS="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -n "$RUN_DIR" ]]; then
        echo "Only one RUN_DIR can be provided." >&2
        usage >&2
        exit 1
      fi
      RUN_DIR="$1"
      shift
      ;;
  esac
done

if [[ ! -d "artifacts/runs" ]]; then
  echo "Run this script from the repository root." >&2
  exit 1
fi

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(find artifacts/runs -mindepth 1 -maxdepth 1 -type d -print | sort -r | head -1)"
fi

if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "Run directory not found: ${RUN_DIR:-<none>}" >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="packaged_runs"
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd -P)"

RUN_NAME="$(basename "$RUN_DIR")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_PATH="$OUTPUT_DIR/${RUN_NAME}_${MODE}_${TIMESTAMP}.tar.gz"
STAGING_ROOT="$(mktemp -d)"
PATH_LIST_FILE=""
cleanup() {
  rm -rf "$STAGING_ROOT"
  rm -f "${INPUT_PATH_LIST_FILE:-}" "${ARCHIVE_PATH_LIST_FILE:-}" "${PATH_LIST_FILE:-}"
}
trap cleanup EXIT

STAGED_RUN_PARENT="$STAGING_ROOT/$(dirname "$RUN_DIR")"
STAGED_RUN_DIR="$STAGING_ROOT/$RUN_DIR"
INPUT_PATH_LIST_FILE="$(mktemp)"
ARCHIVE_PATH_LIST_FILE="$(mktemp)"
mkdir -p "$STAGED_RUN_PARENT"

collect_input_paths() {
  if [[ "$INCLUDE_INPUTS" == "true" ]]; then
    python "$SCRIPT_DIR/list_run_input_files.py" "$RUN_DIR" > "$INPUT_PATH_LIST_FILE"
  else
    : > "$INPUT_PATH_LIST_FILE"
  fi
}

stage_paths_from_file() {
  local path_list_file="$1"

  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    staged_parent="$STAGING_ROOT/$(dirname "$path")"
    mkdir -p "$staged_parent"
    cp -R "$path" "$staged_parent/"
  done < "$path_list_file"
}

write_archive_path_list() {
  printf '%s\n' "$RUN_DIR" > "$ARCHIVE_PATH_LIST_FILE"
  cat "$INPUT_PATH_LIST_FILE" >> "$ARCHIVE_PATH_LIST_FILE"
  sort -u "$ARCHIVE_PATH_LIST_FILE" -o "$ARCHIVE_PATH_LIST_FILE"
}

collect_input_paths

if [[ "$MODE" == "full" ]]; then
  cp -R "$RUN_DIR" "$STAGED_RUN_PARENT/"
  stage_paths_from_file "$INPUT_PATH_LIST_FILE"
  write_archive_path_list
  tar -C "$STAGING_ROOT" -czf "$ARCHIVE_PATH" -T "$ARCHIVE_PATH_LIST_FILE"
else
  PATH_LIST_FILE="$(mktemp)"
  find "$RUN_DIR" -maxdepth 1 \( \
    -name "config.yaml" -o \
    -name "config_snapshot.json" -o \
    -name "input_manifest.json" -o \
    -name "input_train_manifest.json" -o \
    -name "input_val_manifest.json" -o \
    -name "validation_report.json" -o \
    -name "dataset_summary.json" -o \
    -name "preflight_report.json" -o \
    -name "generated_samples_manifest.json" -o \
    -name "reranked_generated_samples_manifest.json" -o \
    -name "rewards.csv" -o \
    -name "confidence_rewards.csv" -o \
    -name "pose_metrics.csv" -o \
    -name "structure_diagnostics.csv" -o \
    -name "metrics.json" -o \
    -name "summary.md" -o \
    -name "evaluation_summary.md" -o \
    -name "reranking_summary.json" -o \
    -name "reranking_summary.md" -o \
    -name "reranking_comparison.csv" -o \
    -name "reranking_comparison.json" -o \
    -name "posttraining_summary.json" -o \
    -name "errors.log" -o \
    -name "logs" -o \
    -name "rollouts" -o \
    -name "checkpoints" -o \
    -name "eval" -o \
    -name "generated_samples" -o \
    -name "raw_diffdock_outputs" \
  \) -print | sort > "$PATH_LIST_FILE"

  if [[ ! -s "$PATH_LIST_FILE" ]]; then
    echo "No key artifacts found in: $RUN_DIR" >&2
    rm -f "$PATH_LIST_FILE"
    exit 1
  fi

  mkdir -p "$STAGED_RUN_DIR"
  stage_paths_from_file "$PATH_LIST_FILE"
  stage_paths_from_file "$INPUT_PATH_LIST_FILE"

  write_archive_path_list
  tar -C "$STAGING_ROOT" -czf "$ARCHIVE_PATH" -T "$ARCHIVE_PATH_LIST_FILE"
fi

echo "Packaged run artifacts:"
echo "  run:     $RUN_DIR"
echo "  mode:    $MODE"
echo "  inputs:  $INCLUDE_INPUTS"
echo "  archive: $ARCHIVE_PATH"
echo "  size:    $(du -h "$ARCHIVE_PATH" | cut -f1)"
echo
echo "Download this file from the Jupyter/ICRN file browser:"
echo "  $ARCHIVE_PATH"
