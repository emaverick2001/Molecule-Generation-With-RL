#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CONDA_PREFIX:-}" && -z "${DIFFDOCK_PYTHON:-}" ]]; then
  export DIFFDOCK_PYTHON="$CONDA_PREFIX/bin/python"
fi

RAW_ROOT="${PDBBIND_RAW_ROOT:-data/raw/pdbbind_real}"

if [[ -z "${SMOKE_COMPLEX_ID:-}" ]]; then
  cat >&2 <<'EOF'
Set SMOKE_COMPLEX_ID to a real PDBBind-style complex before running DiffDock.

Expected files:
  data/raw/pdbbind_real/${SMOKE_COMPLEX_ID}/protein.pdb
  data/raw/pdbbind_real/${SMOKE_COMPLEX_ID}/ligand.sdf
  data/raw/pdbbind_real/${SMOKE_COMPLEX_ID}/ligand_gt.sdf

Example:
  SMOKE_COMPLEX_ID=1a30 ./scripts/run_diffdock_smoke.sh

The synthetic MVP dataset is useful for dry-run plumbing, but real DiffDock
inference should be tested on a real protein-ligand complex.
EOF
  exit 1
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

if grep -q "TINY MVP PROTEIN" "$RAW_ROOT/$SMOKE_COMPLEX_ID/protein.pdb" || \
   grep -q "TinyLigand" "$RAW_ROOT/$SMOKE_COMPLEX_ID/ligand.sdf"; then
  cat >&2 <<EOF
Refusing to run real DiffDock on the synthetic MVP complex: $SMOKE_COMPLEX_ID

Replace data/raw/pdbbind_real/$SMOKE_COMPLEX_ID with a real PDBBind-style complex,
or choose a different real complex ID:

  SMOKE_COMPLEX_ID=<real_pdbbind_id> ./scripts/run_diffdock_smoke.sh
EOF
  exit 1
fi

uv run python - <<'PY'
import os
from pathlib import Path

from src.data.manifests import build_and_save_manifest

split_path = Path("data/processed/diffdock/splits/smoke.txt")
split_path.parent.mkdir(parents=True, exist_ok=True)
split_path.write_text(os.environ["SMOKE_COMPLEX_ID"] + "\n", encoding="utf-8")

build_and_save_manifest(
    ids_path=split_path,
    raw_root=Path(os.environ.get("PDBBIND_RAW_ROOT", "data/raw/pdbbind_real")),
    split="smoke",
    output_path=Path("data/processed/diffdock/manifests/smoke_manifest.json"),
)
PY

uv run python -m src.pipeline.run_baseline \
  --config configs/diffdock/smoke.yaml \
  --exist-ok
