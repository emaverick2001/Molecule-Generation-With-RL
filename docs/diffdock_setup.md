# DiffDock Setup

This project treats DiffDock as a local external dependency, not as vendored
source code. Keep the checkout at:

```text
external/DiffDock
```

That directory is ignored by git.

## Local Setup

Clone DiffDock into the expected location:

```bash
./scripts/setup_diffdock.sh
```

The script clones:

```text
https://github.com/gcorso/DiffDock.git
```

and verifies that `default_inference_args.yaml` exists.

Create the DiffDock environment from inside the checkout:

```bash
cd external/DiffDock
conda env create --file environment.yml
conda activate diffdock
```

Run a one-complex smoke test from the project root:

```bash
conda activate diffdock
./scripts/run_diffdock_smoke.sh
```

Run the tiny multi-complex test:

```bash
conda activate diffdock
./scripts/run_diffdock_tiny.sh
```

The scripts use `uv run` for this project's lightweight pipeline code and
`DIFFDOCK_PYTHON` for the actual DiffDock subprocess. If a conda environment is
active, the scripts set:

```bash
DIFFDOCK_PYTHON="$CONDA_PREFIX/bin/python"
```

You can override it manually:

```bash
DIFFDOCK_PYTHON=/path/to/diffdock/python ./scripts/run_diffdock_smoke.sh
```

## ICRN / GPU Notes

Running DiffDock on Illinois Computes Research Notebooks is a good fit for
smoke tests, small baselines, and debugging because DiffDock inference benefits
from GPU acceleration. Start with the PyTorch image and a GPU server option.

Practical workflow:

1. Clone this repository into your persistent ICRN user directory.
2. Run `./scripts/setup_diffdock.sh`.
3. Create the DiffDock conda environment in `external/DiffDock`.
4. Run `nvidia-smi` and a one-complex smoke test before launching more samples.
5. Copy `artifacts/runs/...` back to local storage or a backed-up location.

ICRN notebooks are interactive and temporary, so avoid using them as the only
place where important outputs live. For longer sweeps, prefer a batch-oriented
resource such as Illinois Campus Cluster if your allocation permits it.

## Useful Config Knobs

The real inference backend is controlled by:

```yaml
generation:
  backend: diffdock
  num_samples: 1

diffdock:
  repo_dir: external/DiffDock
  config_path: external/DiffDock/default_inference_args.yaml
  timeout_seconds: 900
```

Use `num_samples: 1` until the smoke test is stable, then scale up.
