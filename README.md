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
- Run directory creation, config snapshots, artifact logging, and error logs.
- A DiffDock baseline dry run that writes generated-sample manifests, reward
  CSVs, and metric JSON files.
- Unit tests for config loading, path handling, run logging, artifact logging,
  and error logging.

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
    baseline.yaml                   Pretrained DiffDock baseline config
    rerank_baseline.yaml            Confidence reranking baseline config
    reward_filtering.yaml           Reward filtering baseline config
    posttraining.yaml               Reward-based post-training config
  pepflow/                          Planned PepFlow configs

src/
  data/                             Dataset manifests, validation, preprocessing
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
scripts/                            Planned shell wrappers for experiment runs
```

Generated outputs are written under `artifacts/`, which is intentionally ignored
by git.

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

## Run the MVP DiffDock Baseline Dry Run

The current baseline runner is a dry run. It does not invoke DiffDock yet; it
simulates generated poses, reward records, and aggregate metrics to validate the
experiment plumbing.

```bash
uv run python -m src.pipeline.run_baseline --exist-ok
```

This writes a run directory similar to:

```text
artifacts/runs/diffdock_baseline_seed42/
  config_snapshot.json
  config.yaml
  generated_samples_manifest.json
  rewards.csv
  metrics.json
  errors.log
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
configs assume the following repo-relative locations:

```text
data/raw/pdbbind/
data/processed/diffdock/manifests/
data/raw/pepflow/
data/processed/pepflow/manifests/
artifacts/checkpoints/diffdock/
artifacts/checkpoints/pepflow/
```

The canonical DiffDock manifest fields are represented by `ComplexInput` in
`src/utils/schemas.py`:

```json
{
  "complex_id": "1abc",
  "protein_path": "data/raw/pdbbind/1abc/protein.pdb",
  "ligand_path": "data/raw/pdbbind/1abc/ligand.sdf",
  "ground_truth_pose_path": "data/raw/pdbbind/1abc/ligand_gt.sdf",
  "split": "test"
}
```

## Development Notes

- Keep experiment behavior config-driven through `configs/`.
- Save every run under `artifacts/runs/` with a config snapshot.
- Prefer adding small, testable pieces before connecting real model backends.
- Do not commit generated artifacts, checkpoints, or raw datasets.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Authors

- mae10@illinois.edu
- junkun3@illinois.edu
