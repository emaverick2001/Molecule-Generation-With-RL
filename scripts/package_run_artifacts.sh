#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/package_run_artifacts.sh [RUN_DIR] [--full|--key] [--output-dir DIR]

Examples:
  scripts/package_run_artifacts.sh
  scripts/package_run_artifacts.sh artifacts/runs/2026-05-02_diffdock_baseline_seed42
  scripts/package_run_artifacts.sh artifacts/runs/2026-05-02_diffdock_baseline_seed42 --full
  scripts/package_run_artifacts.sh --output-dir ~/packaged_runs

Defaults:
  RUN_DIR      latest artifacts/runs/* directory
  mode         --key
  output-dir   ~/Molecule-Generation-With-RL/packaged_runs

--key includes configs, manifests, logs, generated samples, metrics, summaries,
and raw DiffDock outputs for the selected run. --full packages the entire run
directory.
EOF
}

MODE="key"
RUN_DIR=""
OUTPUT_DIR=""

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
cleanup() {
  rm -rf "$STAGING_ROOT"
}
trap cleanup EXIT

STAGED_RUN_PARENT="$STAGING_ROOT/$(dirname "$RUN_DIR")"
STAGED_RUN_DIR="$STAGING_ROOT/$RUN_DIR"
mkdir -p "$STAGED_RUN_PARENT"

if [[ "$MODE" == "full" ]]; then
  cp -a "$RUN_DIR" "$STAGED_RUN_PARENT/"
  tar -C "$STAGING_ROOT" -czf "$ARCHIVE_PATH" "$RUN_DIR"
else
  PATH_LIST_FILE="$(mktemp)"
  find "$RUN_DIR" -maxdepth 1 \( \
    -name "config.yaml" -o \
    -name "config_snapshot.json" -o \
    -name "input_manifest.json" -o \
    -name "validation_report.json" -o \
    -name "dataset_summary.json" -o \
    -name "generated_samples_manifest.json" -o \
    -name "reranked_generated_samples_manifest.json" -o \
    -name "rewards.csv" -o \
    -name "confidence_rewards.csv" -o \
    -name "pose_metrics.csv" -o \
    -name "metrics.json" -o \
    -name "summary.md" -o \
    -name "evaluation_summary.md" -o \
    -name "reranking_summary.json" -o \
    -name "reranking_summary.md" -o \
    -name "errors.log" -o \
    -name "logs" -o \
    -name "generated_samples" -o \
    -name "raw_diffdock_outputs" \
  \) -print | sort > "$PATH_LIST_FILE"

  if [[ ! -s "$PATH_LIST_FILE" ]]; then
    echo "No key artifacts found in: $RUN_DIR" >&2
    rm -f "$PATH_LIST_FILE"
    exit 1
  fi

  mkdir -p "$STAGED_RUN_DIR"
  while IFS= read -r path; do
    staged_parent="$STAGING_ROOT/$(dirname "$path")"
    mkdir -p "$staged_parent"
    cp -a "$path" "$staged_parent/"
  done < "$PATH_LIST_FILE"

  tar -C "$STAGING_ROOT" -czf "$ARCHIVE_PATH" "$RUN_DIR"
  rm -f "$PATH_LIST_FILE"
fi

echo "Packaged run artifacts:"
echo "  run:     $RUN_DIR"
echo "  mode:    $MODE"
echo "  archive: $ARCHIVE_PATH"
echo "  size:    $(du -h "$ARCHIVE_PATH" | cut -f1)"
echo
echo "Download this file from the Jupyter/ICRN file browser:"
echo "  $ARCHIVE_PATH"
