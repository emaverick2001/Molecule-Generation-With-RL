#!/usr/bin/env bash
set -euo pipefail

DIFFDOCK_REPO_URL="${DIFFDOCK_REPO_URL:-https://github.com/gcorso/DiffDock.git}"
DIFFDOCK_DIR="${DIFFDOCK_DIR:-external/DiffDock}"
DIFFDOCK_REF="${DIFFDOCK_REF:-}"

mkdir -p "$(dirname "$DIFFDOCK_DIR")"

if [[ -d "$DIFFDOCK_DIR/.git" ]]; then
  echo "DiffDock checkout already exists: $DIFFDOCK_DIR"
else
  git clone "$DIFFDOCK_REPO_URL" "$DIFFDOCK_DIR"
fi

if [[ -n "$DIFFDOCK_REF" ]]; then
  git -C "$DIFFDOCK_DIR" fetch --tags
  git -C "$DIFFDOCK_DIR" checkout "$DIFFDOCK_REF"
fi

if [[ ! -f "$DIFFDOCK_DIR/default_inference_args.yaml" ]]; then
  echo "Expected DiffDock config missing: $DIFFDOCK_DIR/default_inference_args.yaml" >&2
  exit 1
fi

cat <<EOF
DiffDock source is ready at: $DIFFDOCK_DIR

Next create the DiffDock environment:

  cd "$DIFFDOCK_DIR"
  conda env create --file environment.yml
  conda activate diffdock

Then return to this project and run:

  cd -
  ./scripts/run_diffdock_smoke.sh

If your compute environment already provides a DiffDock container or module,
you can skip the conda step and update configs/diffdock/*.yaml so the
command_template uses that environment's Python executable.
EOF
