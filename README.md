# Molecule Generation With RL

Reward-based post-training experiments for target-conditioned molecular generation.

This repository is an MVP research codebase for testing whether reward-driven
post-training can improve molecular generation quality beyond pretrained
sampling and post-hoc reranking. The current focus is DiffDock-style
protein-ligand docking, with PepFlow peptide generation planned as the next
stage.

## Project Goal

The project studies a simple question:

> Can reward-based post-training improve the generated molecular structures
> themselves, rather than only improving how generated samples are ranked?

The intended workflow is:

1. Load a molecular design task from a dataset manifest.
2. Generate multiple candidate structures with a pretrained generator.
3. Score candidates with task rewards such as RMSD, docking score, energy, or a
   composite reward.
4. Compare pretrained baselines, reranking/filtering baselines, and
   reward-trained models.
5. Save configs, generated samples, rewards, metrics, and logs for reproducible
   analysis.

## Current Status

Implemented:

- Config loading and deep merging from `configs/global.yaml` plus experiment
  YAML files.
- Structured schemas for complexes, generated poses, reward records, and metric
  records.
- PDBBind-style manifest building, loading, and validation utilities.
- Dataset validation for missing files, empty files, wrong extensions,
  duplicate complex IDs, and invalid split names.
- A tiny 5-complex MVP PDBBind-style dataset generator for local smoke tests.
- Run directory creation, config snapshots, artifact logging, and error logs.
- A DiffDock baseline dry run that loads the mini manifest, validates complexes,
  and writes input manifests, validation reports, dataset summaries,
  generated-sample manifests, reward CSVs, metric JSON files, and summaries.
- Unit tests for config loading, path handling, run logging, artifact logging,
  error logging, manifest creation, dataset loading, and validation.

Planned / placeholder:

- Real DiffDock model adapter and generation code.
- Real RMSD, docking, energy, and composite reward implementations.
- DDPO, DPO, and reward-backpropagation post-training loops.
- PepFlow baseline and post-training pipelines.
- End-to-end evaluation and full experiment orchestration scripts.

## Repository Layout

```text
configs/
  global.yaml                       Shared paths, runtime, logging, evaluation defaults
  diffdock/
    baseline.yaml                   MVP mini-split DiffDock dry-run config
    rerank_baseline.yaml            Confidence reranking baseline config
    reward_filtering.yaml           Reward filtering baseline config
    posttraining.yaml               Reward-based post-training config
  pepflow/                          Planned PepFlow configs

src/
  data/                             Dataset manifests, loading, validation, preprocessing
  generation/                       Generation entry points and sampling utilities
  models/                           DiffDock and PepFlow adapters
  pipeline/                         Baseline, post-training, evaluation runners
  posttraining/                     DDPO, DPO, and reward-backprop trainers
  rewards/                          RMSD, docking, energy, composite rewards
  utils/                            Configs, schemas, paths, logging, seeds

docs/
  high_low_pipeline.md              System pipeline and artifact design
  experiments.md                    Research questions, metrics, baselines, ablations
  timeline.md                       Milestone plan and staged execution strategy

tests/                              Unit tests for the implemented MVP utilities
notebooks/                          Planned analysis and visualization notebooks
scripts/
  create_tiny_pdbbind_mvp.py        Creates the mini PDBBind-style dataset and manifest
  *.sh                              Planned shell wrappers for experiment runs
```

Generated outputs are written under `artifacts/`, which is intentionally ignored
by git. Local data under `data/` is also ignored by git.

Third-party model checkouts should live under `external/`, which is also ignored
by git. See `docs/diffdock_setup.md` for DiffDock setup and GPU notes.

## Setup

Requirements:

- Python `>=3.11,<3.13`
- `uv`

```bash
git clone https://github.com/emaverick2001/Molecule-Generation-With-RL.git
cd Molecule-Generation-With-RL

uv sync --dev
```

If you prefer to activate the environment manually:

```bash
source .venv/bin/activate
```

## Run Tests

```bash
uv run pytest
```

## Run the Mini MVP Experiment

The fastest end-to-end check is the mini MVP experiment. It creates a tiny
PDBBind-style dataset with 5 complexes, writes a mini split file, builds and
validates a manifest, then runs the current DiffDock baseline dry run.

```bash
uv run python scripts/create_tiny_pdbbind_mvp.py --run-baseline --exist-ok
```

This command writes dataset files under:

```text
data/raw/pdbbind_synthetic/
  1abc/
  2xyz/
  3def/
  4ghi/
  5jkl/

data/processed/diffdock/
  splits/mini.txt
  manifests/mini_manifest.json
  manifests/mini_validation_report.json
```

It also writes a run directory named with the current date, model, mode, and
seed, for example:

```text
artifacts/runs/2026-04-30_diffdock_baseline_seed42/
  config.yaml
  config_snapshot.json
  input_manifest.json
  validation_report.json
  dataset_summary.json
  generated_samples_manifest.json
  rewards.csv
  metrics.json
  errors.log
  summary.md
```

The mini dataset is synthetic and intended only to validate project plumbing.
It is not a scientifically meaningful docking benchmark.

## Run Only the Baseline Dry Run

After `data/processed/diffdock/manifests/mini_manifest.json` exists, you can run
the baseline dry run directly:

```bash
uv run python -m src.pipeline.run_baseline --exist-ok
```

The baseline runner does not invoke DiffDock yet. It loads the mini manifest,
validates the records, then simulates generated poses, reward records, and
aggregate metrics to validate the experiment artifact flow.

To run real DiffDock inference, first install the external DiffDock checkout:

```bash
./scripts/setup_diffdock.sh
```

Then run the one-complex smoke test:

```bash
SMOKE_COMPLEX_ID=<real_pdbbind_id> ./scripts/run_diffdock_smoke.sh
```

Package run artifacts for download from ICRN/Jupyter:

```bash
./scripts/package_run_artifacts.sh artifacts/runs/<run_id>
```

Real DiffDock smoke-test complexes should live under `data/raw/pdbbind_real/`.
Use the extractor after downloading a PDBBind package:

```bash
uv run python scripts/setup_pdbbind_smoke_complex.py \
  --complex-id <real_pdbbind_id>
```

Use `--config` to run another DiffDock config through the same entry point once
the corresponding pipeline behavior is implemented:

```bash
uv run python -m src.pipeline.run_baseline \
  --config configs/diffdock/reward_filtering.yaml \
  --exist-ok
```

## Experiment Design

The planned evaluation compares reward-based post-training against several
baselines:

- Pretrained DiffDock with no post-training.
- DiffDock with confidence reranking.
- DiffDock with reward filtering and no weight updates.
- DiffDock with reward-based post-training.
- Later stages repeat the structure for PepFlow peptide generation.

Primary metrics from `docs/experiments.md` include:

- Top-1 RMSD.
- Success@k for k = 1, 5, 10 using an RMSD threshold such as 2 Angstrom.
- Mean best-of-N reward.
- Average RMSD across generated samples.
- Reward distribution shift.
- Diversity and reward/RMSD correlation diagnostics.

The staged plan is:

1. Build and validate the DiffDock MVP pipeline.
2. Add DiffDock reward-based post-training.
3. Extend to PepFlow fixed-backbone peptide design.
4. Extend to full PepFlow sequence-structure co-design.
5. Add ablations, diagnostics, and exploratory inference-time guidance.

## Data and Checkpoints

Large datasets and model checkpoints are not stored in this repository. The
MVP scripts and configs assume the following repo-relative locations:

```text
data/raw/pdbbind_real/
data/raw/pdbbind_synthetic/
data/processed/diffdock/splits/
data/processed/diffdock/manifests/
data/raw/pepflow/
data/processed/pepflow/manifests/
artifacts/checkpoints/diffdock/
artifacts/checkpoints/pepflow/
external/DiffDock/
```

The mini experiment creates synthetic files in `data/raw/pdbbind_synthetic/` and
`data/processed/diffdock/`. These paths are ignored by git, along with
`artifacts/` and `external/`.

The canonical DiffDock manifest fields are represented by `ComplexInput` in
`src/utils/schemas.py`:

```json
{
  "complex_id": "1abc",
  "protein_path": "data/raw/pdbbind/1abc/protein.pdb",
  "ligand_path": "data/raw/pdbbind/1abc/ligand.sdf",
  "ground_truth_pose_path": "data/raw/pdbbind/1abc/ligand_gt.sdf",
  "split": "mini"
}
```

Manifest and dataset helpers live in:

- `src/data/manifests.py`: read split IDs, build manifest records, save/load
  manifest JSON.
- `src/data/validation.py`: validate paths, file sizes, extensions, duplicate
  complex IDs, and split names.
- `src/data/loaders.py`: load validated `ComplexInput` records from a manifest
  and filter records by split.

## Development Notes

- Keep experiment behavior config-driven through `configs/`.
- Save every run under `artifacts/runs/` with a config snapshot.
- Prefer adding small, testable pieces before connecting real model backends.
- Do not commit generated artifacts, checkpoints, or raw datasets.
- The default DiffDock baseline config currently targets the synthetic `mini`
  split. Update `dataset.split` and `dataset.manifest_path` when moving to a
  real train/val/test PDBBind manifest.
- `--exist-ok` allows rerunning the same dated run directory. Omit it when you
  want the runner to fail instead of overwriting files in an existing run.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Authors

- mae10@illinois.edu
- junkun3@illinois.edu
