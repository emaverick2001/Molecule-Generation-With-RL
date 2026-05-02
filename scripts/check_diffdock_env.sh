#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CONDA_PREFIX:-}" && -z "${DIFFDOCK_PYTHON:-}" ]]; then
  export DIFFDOCK_PYTHON="$CONDA_PREFIX/bin/python"
fi

DIFFDOCK_PYTHON="${DIFFDOCK_PYTHON:-python}"

"$DIFFDOCK_PYTHON" - <<'PY'
import importlib
import sys

print(f"python={sys.executable}")

try:
    import torch
except Exception as error:
    print(f"torch_import_error={error}")
    raise SystemExit(1)

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")

for module_name in [
    "torch_geometric",
    "torch_scatter",
    "torch_sparse",
    "torch_cluster",
    "torch_spline_conv",
]:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"{module_name}={version}")
    except Exception as error:
        print(f"{module_name}_import_error={error}")
        raise SystemExit(1)

from torch_cluster import radius, radius_graph

print("torch_cluster_radius_import=ok")
PY
