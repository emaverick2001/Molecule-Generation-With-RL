#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/create_tiny_pdbbind_mvp.py --exist-ok

uv run python - <<'PY'
from pathlib import Path

from src.data.manifests import build_and_save_manifest

split_path = Path("data/processed/diffdock/splits/smoke.txt")
split_path.parent.mkdir(parents=True, exist_ok=True)
split_path.write_text("1abc\n", encoding="utf-8")

build_and_save_manifest(
    ids_path=split_path,
    raw_root=Path("data/raw/pdbbind"),
    split="smoke",
    output_path=Path("data/processed/diffdock/manifests/smoke_manifest.json"),
)
PY

uv run python -m src.pipeline.run_baseline \
  --config configs/diffdock/smoke.yaml \
  --exist-ok
