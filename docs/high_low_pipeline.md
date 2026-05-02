#### **High-Level Pipeline + Low-level Pipeline**
- Describe the system as a sequence of stages
- Remember to first create most simplistic working system then build ontop of it
- 
1. First high-level then low-level
	1. for high-level mention the core data manipulations + actions + key system components + deliverables for respective section
	2. for low-level mention key datastructures + key artifact logging plan + implementation details to key components/deliverables for respective section so that you can implement the simplistic working system
	3. Once MVP is built add more details, optimizations, etc
	4. Optionally later on mention run artifacts produced + what datapoints to log
---
##### **Dataflow Example**
---
- Input: protein–ligand complex
- Output: generated docking poses + reward scores + RMSD metrics
```python
example_input = {  
    "complex_id": "1abc",  
    "protein_path": "data/raw/pdbbind/1abc/protein.pdb",  
    "ligand_path": "data/raw/pdbbind/1abc/ligand.sdf",  
    "ground_truth_pose_path": "data/raw/pdbbind/1abc/ligand_gt.sdf",  
    "split": "test"  
}
```
##### **0. Setup** 
###### **Key Components / Deliverables**
1. local code environment setup 
2. Setup Project Repository Structure
	1. Create top-level directories:
		- assets/
			- diagrams/
			- figures/
		- artifacts/
			- runs/
				- use this naming convention: {date}_{model}_{experiment}_{reward}_{seed}
				- each folder should contain: 
					- config.yaml  
					- metrics.json  
					- rewards.csv  
					- generated_samples_manifest.json  
					- training_log.csv  
					- errors.log  
					- summary.md
			- checkpoints/
				- diffdock/
				- pepflow/
			- generated_samples/
				- diffdock/
				- pepflow/
			- rewards/
				- diffdock/
				- pepflow/
			- metrics/
				- diffdock/
				- pepflow/
			- logs/
				- training/
				- evaluation/
				- errors/
		- configs/
			- diffdock/
				- baseline.yaml
				- rerank_baseline.yaml
				- reward_filtering.yaml
				- posttraining.yaml
			- pepflow/
				- baseline.yaml
				- posttraining.yaml
			- global.yaml
		- src/
			- \_\_init\_\_.py
			- data/
				- loaders.py
				- validation.py
				- preprocessing.py
				- manifests.py
			- models/
				- diffdock_adapter.py
				- pepflow_adapter.py
			- utils/
				- run_logger.py
				- artifact_logger.py
				- error_logger.py
				- config.py
				- paths.py
				- seeds.py
				- io.py
			- generation/
				- generate_diffdock.py
				- generate_pepflow.py
				- sampling.py
			- rewards/
				- rmsd_reward.py
				- docking_reward.py
				- energy_reward.py
				- composite_reward.py
			- posttraining/
				- ddpo_trainer.py
				- dpo_trainer.py
				- reward_backprop.py
				- training_utils.py
			- pipeline/
				- run_baseline.py
				- run_posttraining.py
				- run_evaluation.py
				- run_full_experiment.py
		- data/
			- raw/
				- pdbbind/
				- pepflow/
			- processed/
				- diffdock/
					- manifests/
					- splits/
				- pepflow/
		- tests/
			- test_data_loading.py  
			- test_reward_computation.py  
			- test_rmsd.py  
			- test_config_loading.py  
			- test_artifact_logging.py
		- docs/
			- experiment_plan.md
		- notebooks/
			- 01_data_exploration.ipynb  
			- 02_baseline_analysis.ipynb  
			- 03_reward_diagnostics.ipynb  
			- 04_results_visualization.ipynb
		- scripts/
			- setup_data.sh  
			- run_diffdock_baseline.sh  
			- run_diffdock_posttraining.sh  
			- evaluate_diffdock.sh  
			- run_all.sh
	2. Follow [[Software Implementation Guide]] until 2. Working on the Frontend or 3. Working on the Backend
##### **1. Dataset Loading + Processing
- Load protein–ligand complex
###### **Key Components / Deliverables**
0. Write a manifest builder
	- Create a reusable utility that scans a small PDBBind-style dataset folder and produces a standardized JSON manifest. The manifest becomes the single source of truth for downstream pipeline stages such as baseline generation, reward filtering, evaluation, and post-training.
	- Without a manifest, every stage would need to manually understand this folder structure:
		```
		data/raw/pdbbind/
		├── 1abc/│   
		├── protein.pdb│   
		├── ligand.sdf│   
		└── ligand_gt.sdf
		```
	1. Create src/data/manifests.py
		- This file should contain functions for: reading complex IDs, building manifest records, validating required files, saving manifest JSON, loading manifest JSON
		- Example output
			```
			[  
			{  
			"complex_id": "1abc",  
			"protein_path": "data/raw/pdbbind/1abc/protein.pdb",  
			"ligand_path": "data/raw/pdbbind/1abc/ligand.sdf",  
			"ground_truth_pose_path": "data/raw/pdbbind/1abc/ligand_gt.sdf",  
			"split": "mini"  
			}  
			]
			```
		1. read_complex_ids(path)
			- Read a `.txt` file containing one complex ID per line.
			1. Accept a path to a text file.
			2. Return a list of non-empty stripped IDs.
			3. Ignore blank lines.
			4. Raise `FileNotFoundError` if the file does not exist.
		2. build_manifest_records(complex_ids, raw_root, split)
			- Convert a list of complex IDs into standardized manifest records.
			1. For each `complex_id`, assume files are stored at
				```
				{raw_root}/{complex_id}/protein.pdb  
				{raw_root}/{complex_id}/ligand.sdf  
				{raw_root}/{complex_id}/ligand_gt.sdf
				```
			- Create a `ComplexInput` object for each complex.
		3. validate_manifest_records(records)
			- Ensure every manifest record points to actual files.
			1. For every record
				- Check that `protein_path` exists.
				- Check that `ligand_path` exists.
				- Check that `ground_truth_pose_path` exists.
				- Raise `FileNotFoundError` with a clear message if any required file is missing.
		4. save_manifest(records, path)
			- Save manifest records as JSON.
				1. Convert each `ComplexInput` dataclass to a dictionary.
				2. Save as indented JSON.
				3. Create parent directories if needed.
		5. load_manifest(path)
			- Load a manifest JSON file and convert it back into `ComplexInput` objects.
			1. Read the JSON file.
			2. Convert each dictionary into a `ComplexInput`.
			3. Return list of `ComplexInput`.
			4. Raise `FileNotFoundError` if manifest file does not exist.
		6. build_and_save_manifest(...)
			- One convenience function that handles the full flow
			1. Read IDs from `ids_path`.
			2. Build manifest records using `raw_root` and `split`.
			3. Validate that all required files exist.
			4. Save manifest to `output_path`.
			5. Return the records.
	2. Create ```tests/test_manifests.py```
		- Verifies:
			- read_complex_ids reads non-empty IDs  
			- build_manifest_records creates correct paths  
			- validate_manifest_records passes when files exist  
			- validate_manifest_records fails when files are missing  
			- save_manifest writes JSON  
			- load_manifest reads JSON into ComplexInput objects  
			- build_and_save_manifest runs the full workflow
1. Implement Path and File Validation 
	- Create a validation layer that checks whether a dataset manifest is safe to use before running baseline generation, reward filtering, evaluation, or post-training.
	- Catches 
		- missing files  
		- empty files  
		- wrong file extensions  
		- duplicate complex IDs  
		- invalid split names
	1. Create src/data/validation.py which should check:
		- `protein_path` exists
		- `ligand_path` exists
		- `ground_truth_pose_path` exists
		- files are not empty
		- file extensions are expected
		- `complex_id` is unique
		- split is valid
		1. validate_file_exists(path, field_name) 
		2. validate_file_not_empty(path, field_name)
		3. `find_duplicate_complex_ids(records)`
		4. validate_record(record, duplicate_ids, valid_splits)
		5. validate_manifest_records(records)
		6. validate_manifest_file(path)
		- Example output
			```
			validation_result = {  
			"num_complexes": 10,  
			"num_valid": 9,  
			"num_invalid": 1,  
			"invalid_complexes": [  
			{  
			"complex_id": "3bad",  
			"reason": "missing ligand_gt.sdf"  
			}  
			]  
			}
			```
	2. Create ```tests/test_validation.py```
2. Write the dataset loader 
	- Create a loader that reads a manifest JSON file and returns validated `ComplexInput` objects
	- The loader should **not** do molecular preprocessing, DiffDock conversion, reward computation, RMSD computation, or featurization.
	1. Create src/data/loaders.py
		- The loader should read a manifest and return `ComplexInput` objects.
		1. load_complex_manifest(manifest_path, validate=True)
			- Check that `manifest_path` exists.
			- Load JSON.
			- Verify the JSON root is a list.
			- Convert each item into a `ComplexInput`.
			- If `validate=True`, call `validate_manifest_records(records)`.
			- If invalid records exist, raise `ValueError` with a clear message.
			- Return `list[ComplexInput]`.
		2. `filter_records_by_split(records, split)`
			- Return only records matching the requested split.
		3. load_split_ids(path)
			1. 
	2. Add Dataset Loading Tests in tests/test_data_loading.py
		- manifest file exists  
		- manifest loads successfully  
		- each item has required fields  
		- all paths exist  
		- complex IDs are unique  
		- invalid manifest raises clear error
	3. Add Preprocessing only if necessary
		1. Start with lightweight preprocessing
			1. normalize file names  
			2. verify extensions  
			3. optionally copy/symlink files into a standardized location
		2. Avoid 
			1. coordinate transformations  
			2. molecular featurization  
			3. DiffDock tensor conversion  
			4. reward computation  
			5. RMSD computation
	4. Decide split structure
		1. Create data/processed/diffdock/splits/
			- mini.txt
			- train.txt
			- val.txt
			- test.txt
		2. Manifest Builder assigns splits based on these files
	5. Connect dataset loading to the dry-run pipeline
		1. Update src/pipeline/run_baseline.py
			- So instead of fake hardcoded inputs, it loads the real `mini_manifest.json`.
			- Your pipeline still should **not** call DiffDock yet.
			- it should
				1. create run folder  
				2. load mini manifest  
				3. validate complexes  
				4. save loaded manifest copy into run folder  
				5. save dataset summary
				```
				artifacts/runs/{run_id}/  
				├── config.yaml  
				├── input_manifest.json  
				├── dataset_summary.json  
				├── validation_report.json  
				└── summary.md
				```
3. Create a tiny real MVP dataset first and run it through the experiment pipeline
	- Create a small real PDBBind-style dataset with **5 to 10 protein-ligand complexes** and run it through the current MVP pipeline.
	- After this step, one command should be able to produce an experiment folder within
		```
		artifacts/runs/{run_id}/  
		├── config.yaml  
		├── config_snapshot.json  
		├── errors.log  
		├── generated_samples_manifest.json  
		├── rewards.csv  
		├── metrics.json  
		└── summary.md
		```
	1. Create Mini Split File
	2. Create Tiny Dataset Builder Script
	3. Run the Dataset Builder
	4. Build the Manifest
	5. Validate the Mini Manifest
	6. Update `configs/diffdock/baseline.yaml` for the Mini Dataset
	7. Modify `run_baseline.py` to Use the Manifest
##### **2. Baseline Generation**
- Run pretrained DiffDock and Generate N candidate poses
###### **Key Components / Deliverables**
0. Define the Baseline Generation Contract
	- Decide what baseline generation must produce for each complex.
	- [[DiffDock]]
1. **Create the Generation Interface**
	1. Create src/generation/dry_run_generator.py
		- generate fake pose records from real manifest-loaded complexes
		- later replace or extend it with `diffdock_generator.py`
	2. Implement MVP Dry-Run Generation 
		- For each `ComplexInput`:  
			1. Read `complex_id`  
			2. Loop over `num_samples`  
			3. Create output pose paths  
			4. Create `GeneratedPose` objects  
			5. Save them into `generated_samples_manifest.json`  
		- Do not run DiffDock yet.  
		- This validates the generation artifact shape before adding model complexity.
2. Define Generated Sample Artifact Layout
	- Use one run-local generated sample directory:
		```
		artifacts/runs/{run_id}/generated_samples/
		├── 1abc_sample_0.sdf
		├── 1abc_sample_1.sdf
		├── 2xyz_sample_0.sdf
		└── 2xyz_sample_1.sdf
		```
	- And one manifest: 
		```
		artifacts/runs/{run_id}/generated_samples_manifest.json
		```
3. Connect Generation to `run_baseline.py`
	- Update `run_baseline.py` so the flow becomes
		```
		load config
		→ initialize run folder
		→ load manifest
		→ validate dataset records
		→ run baseline generator
		→ save generated_samples_manifest.json
		→ save summary.md
		```
4. Create Baseline Generation Tests `tests/test_baseline_generation.py`
	- Create tests/test_baseline_generation.py
		- generator returns `num_complexes × num_samples` records  
		- each record has a valid `complex_id`  
		- each record has a unique `(complex_id, sample_id)`  
		- generated pose paths are inside the run directory  
		- generated manifest can be saved and reloaded
5. Add a Real DiffDock Wrapper Only After Dry-Run Works
	1. Create src/generation/generate_diffdock.py
		- Its job should be only to wrap DiffDock inference.
		- Do not mix evaluation, RMSD, reward scoring, or reranking into this file.
		- This file should only handle:
			- ComplexInput records→ DiffDock inference command→ generated pose files→ GeneratedPose records
		1. def generate_diffdock_poses()
			- This function should:
				- Accept manifest-loaded `ComplexInput` records.
				- Run DiffDock once per complex.
				- Store raw DiffDock outputs in a run-local folder.
				- Standardize output pose files into predictable names.
				- Return a list of `GeneratedPose` records.
				- Validate the output contract using `validate_generated_pose_records`.
6. Implement Actual DiffDock Inference 
	1. Add a DiffDock backend config
		```
		generation:
		  backend: diffdock
		  num_samples: 2
		  use_confidence_reranking: false
		diffdock:
		  repo_dir: external/DiffDock
		  config_path: external/DiffDock/default_inference_args.yaml
		  timeout_seconds: 900
		  command_template:
		    - python
		    - -m
		    - inference
		    - --config
		    - "{config_path}"
		    - --protein_path
		    - "{protein_path}"
		    - --ligand_description
		    - "{ligand_path}"
		    - --out_dir
		    - "{raw_output_dir}"
		    - --samples_per_complex
		    - "{num_samples}"
		    - --batch_size
		    - "1"
		```
		- Important correction: newer DiffDock `inference.py` uses `--ligand_description`, not just `--ligand`, based on the current parser.
	2. Add a preflight check before running DiffDock 
		1. Create a helper in `src/generation/generate_diffdock.py` or `src/generation/diffdock_preflight.py`.
			- Check 
				1. external/DiffDock exists  
				2. default_inference_args.yaml exists  
				3. manifest paths exist  
				4. protein files are .pdb  
				5. ligand files are .sdf or RDKit-readable  
				6. python command can run  
				7. generation.num_samples > 0
	3. Make the subprocess run inside the DiffDock repo
		- This matters because DiffDock imports its own local modules like `utils`, `datasets`, and its config paths are often repo-relative.
		```
		subprocess.run(  
		command,  
		cwd=repo_dir,  
		capture_output=True,  
		text=True,  
		timeout=timeout_seconds,  
		)
		```
		1. Update your command formatter so absolute paths are passed into DiffDock:
			- protein_path  
			- ligand_path  
			- raw_output_dir  
			- config_path
	4. Update `format_command_template` to support DiffDock placeholders
		```
		{complex_id}
		{protein_path}
		{ligand_path}
		{ground_truth_pose_path}
		{raw_output_dir}
		{num_samples}
		{config_path}
		{repo_dir}
		```
	5. Implement `run_diffdock_command(..., cwd=repo_dir)`
		- run command  
		- write stdout log  
		- write stderr log  
		- raise RuntimeError if return code != 0  
		- raise TimeoutError if timeout occurs
		- Each complex should get logs like:
			- artifacts/runs/{run_id}/logs/1abc.stdout.logartifacts/runs/{run_id}/logs/1abc.stderr.log
	6. Discover DiffDock-generated SDF outputs
		- After each DiffDock run, the wrapper searches:
			- artifacts/runs/{run_id}/raw_diffdock_outputs/{complex_id}/
		- for:
			- *.sdf
		- Parse DiffDock output names like:
			- `rank1.sdf`
			- `rank1_confidence-0.70.sdf`
			- `rank10_confidence-1.85.sdf`
		- Sort by numeric rank, not lexicographic filename order.
		- Prefer `rankN_confidenceX.sdf` over `rankN.sdf` when both exist.
		- Store parsed confidence values in `GeneratedPose.confidence_score`.
		- Then copy the first `num_samples` ranked poses into your standardized folder:
			- artifacts/runs/{run_id}/generated_samples/{complex_id}_sample_0.sdf
			- artifacts/runs/{run_id}/generated_samples/{complex_id}_sample_1.sdf
	7. Preserve your existing output contract
		- Even with real DiffDock, the pipeline output should still be:
			- generated_samples/
			- generated_samples_manifest.json
			- summary.md
			- logs/
			- raw_diffdock_outputs/
		- And `generated_samples_manifest.json` should still contain:
			```
			[
			  {
				"complex_id": "1abc",
				"sample_id": 0,
				"pose_path": "artifacts/runs/.../generated_samples/1abc_sample_0.sdf",
				"confidence_score": null
			  }
			]
			```
	8. Add the backend switch in `run_baseline.py`
		- add the ability to switch between diffdock and dry_run as configurations for the run_baseline.py 
	9. Start with one complex before running all 5–10
		1. Create a debug split data/processed/diffdock/splits/smoke.txt with one complex ID
		2. Create data/processed/diffdock/manifests/smoke_manifest.json
		3. Point a temporary config to that manifest:
			```
			dataset:
			  name: pdbbind_tiny
			  split: smoke
			  manifest_path: data/processed/diffdock/manifests/smoke_manifest.json
			generation:
			  backend: diffdock
			  num_samples: 1
			```
	10. Add a script for real DiffDock baseline scripts/run_diffdock_smoke.sh
	11. Add a script to test more complexes (5-10 complex mini manifest) scripts/run_diffdock_tiny.sh
	12. Completion Criteria 
		1. `run_baseline.py` loads the mini manifest  
		2. it creates generated sample records for every complex  
		3. it saves `generated_samples_manifest.json`  
		4. generated sample paths are run-specific  
		5. tests pass  
		6. the pipeline can be switched from dry-run generation to DiffDock generation without changing the dataset loader
		7. able to run ./scripts/run_diffdock_smoke.sh
		8. run folder contains 
			```
			config.yaml
			config_snapshot.json
			errors.log
			input_manifest.json
			dataset_summary.json
			validation_report.json
			raw_diffdock_outputs/
			logs/
			generated_samples/
			generated_samples_manifest.json
			summary.md
			```
		9. Next steps 
##### **3. Evaluation**
- This phase evaluates generated poses against the ground-truth bound ligand pose using an **offline** metric. For the MVP, the core metric should be symmetry-aware ligand RMSD, with top-k success thresholds matching common DiffDock reporting conventions. The original DiffDock paper reports success at RMSD < 2 Å, and the current repository evaluator computes `rmsds_below_2`, `rmsds_below_5`, `top5_rmsds_below_2`, and `top10_rmsds_below_2`, along with centroid-distance summaries. 
- RDKit’s documentation makes the intended RMSD choice clear. `SDMolSupplier` is the standard way to read SDF sets and should be checked for `None` values; `MolFromMolFile()` returns `None` on parse failure; and `rdMolAlign.CalcRMS()` is documented as useful for comparing docking poses and co-crystallized ligands because it computes RMSD in place without pre-aligning the probe to the reference. RDKit also warns that symmetry-aware matching can suffer combinatorial explosion when hydrogens are present. DiffDock’s own evaluator removes hydrogens before symmetry-aware RMSD and then reports RMSD, centroid-distance, and top-k summaries. That is the rationale for the default MVP design here: `CalcRMS`, symmetry-aware, `remove_hs=True`, and top-k aggregation at 2 Å and 5 Å.
###### **Key Components / Deliverables**
1. src/evaluation/__init__.py
2. src/evaluation/rmsd.py
	1. `load_single_sdf(...)`
		```python
		def load_single_sdf(path: Union[str, Path]) -> "Chem.Mol":
		```
		- use `Chem.SDMolSupplier(..., removeHs=False)`
		- return exactly one non-`None` molecule
		- raise:
		    - `FileNotFoundError` if missing,
		    - `ValueError` if no valid molecules found
	2. `compute_symmetry_corrected_rmsd(...)`
		```python
		def compute_symmetry_corrected_rmsd(
		    predicted_pose_path: Union[str, Path],
		    reference_pose_path: Union[str, Path],
		    remove_hs: bool = True,
		) -> float:
		```
		- load molecules
		- optional `Chem.RemoveHs(...)`
		- call `rdMolAlign.CalcRMS(pred, ref)`
		- return float RMSD
	3. `compute_centroid_distance(...)`
		```python
		def compute_centroid_distance(
		    predicted_pose_path: Union[str, Path],
		    reference_pose_path: Union[str, Path],
		    remove_hs: bool = True,
		) -> float:
		```
		- compute ligand-coordinate centroids
		- return Euclidean distance
3. src/evaluation/metrics.py
	1. `evaluate_generated_poses(...)`
		```python
		def evaluate_generated_poses(
		    input_records: list["ComplexInput"],
		    generated_records: list["GeneratedPose"],
		    rmsd_thresholds: tuple[float, float] = (2.0, 5.0),
		    remove_hs: bool = True,
		) -> list["PoseMetricRecord"]:
		```
		- build `complex_id -> ground_truth_pose_path` lookup from `input_records`
		- compute one `PoseMetricRecord` per generated pose
		- if a pose fails to load or compare:
			- emit invalid row, do not crash the whole run
	2. `aggregate_topk_metrics(...)`
		```python
		def aggregate_topk_metrics(
		    metric_records: list["PoseMetricRecord"],
		    top_k: list[int] = [1, 5, 10],
		) -> dict[str, object]:
		```
		- group valid rows by `complex_id`
		- sort within each complex by `rank`
		- compute:
		    - `num_complexes`
		    - `num_valid_poses`
		    - `num_invalid_poses`
		    - `mean_rmsd`
		    - `median_rmsd`
		    - `success_at_1`
		    - `success_at_5` if enough poses
		    - `success_at_10` if enough poses
		    - `best_of_n_mean_rmsd`
	3. `save_pose_metrics_csv(...)`
		```python
		def save_pose_metrics_csv(
		    records: list["PoseMetricRecord"],
		    path: Union[str, Path],
		) -> None:
		```
	4. `load_pose_metrics_csv(...)`
	```python
	def load_pose_metrics_csv(path: Union[str, Path]) -> list["PoseMetricRecord"]:
	```
4. src/pipeline/run_evaluation.py
5. tests/test_rmsd_evaluation.py
	- identical SDFs produce RMSD ≈ 0
	- moved conformer produces RMSD > 0
	- invalid SDF path raises `FileNotFoundError`
	- invalid SDF contents produce invalid metric row, not pipeline crash
6. tests/test_metrics_aggregation.py
	- aggregation computes correct `success_at_1` / `success_at_5`
	- top-k metrics are `None` when insufficient poses exist
	- reranked records are evaluated in reranked order
7. scripts/run_evaluation.sh
8. configs/diffdock/evaluation.yaml
##### **4. Confidence Reranking**
- This phase converts the per-pose DiffDock confidence signal into a run-local reward artifact that is downstream-friendly, reproducible, and separate from evaluation.
- Implementation status: offline reranking is implemented. It can rerank a completed run directory from either:
	- confidence scores stored in `generated_samples_manifest.json`
	- an existing `rewards.csv` artifact
- Run command:
	```bash
	./scripts/run_reranking.sh artifacts/runs/{run_id}
	```
- For dry-run artifacts that do not contain confidence scores, use a config without a `reranking.reward_source` override so the stage defaults to `rewards_csv`:
	```bash
	./scripts/run_reranking.sh artifacts/runs/{run_id} --config configs/diffdock/evaluation.yaml
	```
- Outputs:
	```text
	artifacts/runs/{run_id}/
	├── reranked_generated_samples_manifest.json
	├── reranking_summary.json
	└── reranking_summary.md
	```
- To evaluate the reranked order:
	```bash
	uv run python -m src.pipeline.run_evaluation \
	  --run-dir artifacts/runs/{run_id} \
	  --generated-manifest artifacts/runs/{run_id}/reranked_generated_samples_manifest.json
	```
###### **Key Components / Deliverables**
1. src/rewards/confidence_reward.py
	1. `require_confidence_scores(...)`
		- raise `ValueError` if any pose has `confidence_score is None`.
	2. `transform_confidence_score(...)`
		- `identity` → return raw score
		- `sigmoid` → `1 / (1 + exp(-score / temperature))`
	3. `build_confidence_reward_records(...)`
		- Convert `GeneratedPose.confidence_score` values into `RewardRecord` rows with `reward_type="confidence"`.
2. src/evaluation/reranking.py
	1. `build_reward_lookup(...)`
		```python
		def build_reward_lookup(
		    reward_records: list["RewardRecord"],
		) -> dict[tuple[str, int], "RewardRecord"]:
		```
	2. `rerank_generated_poses(...)`
		```python
		def rerank_generated_poses(
		    generated_records: list["GeneratedPose"],
		    reward_records: list["RewardRecord"],
		    descending: bool = True,
		    tie_breaker: str = "sample_id",
		) -> list["RerankedPose"]:
		```
		- join on `(complex_id, sample_id)`
		- group by `complex_id`
		- assign `original_rank` from original within-complex order
		- sort by reward descending
		- stable tie-break by `sample_id`
		- emit `RerankedPose` rows
		- **do not copy or rename SDF files**
	3. `summarize_reranking(...)`
		```python
		def summarize_reranking(
		    reranked_records: list["RerankedPose"],
		) -> dict[str, object]:
		```
		- count complexes
		- count poses
		- compute number of poses whose rank changed
		- compute mean absolute rank delta
	4. `save_reranked_manifest(...)`
		```python
		def save_reranked_manifest(
		    records: list["RerankedPose"],
		    path: Union[str, Path],
		) -> None:
		```
	5. `load_reranked_manifest(...)`
		```python
		def load_reranked_manifest(path: Union[str, Path]) -> list["RerankedPose"]:
		```
3. src/pipeline/run_reranking.py
	- Loads `generated_samples_manifest.json`.
	- Loads `rewards.csv` or derives confidence rewards from generated-pose confidence scores.
	- Writes `reranked_generated_samples_manifest.json`.
	- Writes `reranking_summary.json`.
	- Writes `reranking_summary.md`.
4. tests/test_reranking.py
	- reranking sorts descending reward within each complex
	- ties break by `sample_id`
	- missing reward row raises clear error
	- reranked manifest roundtrip works
	- reranking does not duplicate or rename pose files
	- confidence-only reranking is idempotent if original manifest is already sorted by confidence
	- evaluation uses manifest order for rank, so reranked manifests affect top-k metrics
5. scripts/run_reranking.sh
6. configs/diffdock/rerank_baseline.yaml
7. Completion Criteria
	1. reranking can consume a generated-samples manifest and reward records
	2. reranking preserves original SDF paths without copying or renaming files
	3. reranking writes a run-local reranked manifest
	4. missing confidence or reward rows fail with clear errors
	5. evaluation can consume the reranked manifest and respect reranked order
	6. tests pass
##### **5. Post-Training / RL Setup**
###### **Key Components / Deliverables**
8. [Key implementation steps / Key Data Manipulations / Key Components for this stage]
9. [key component name]
	1. [steps to implement key component/ tests to run for validation]
##### **7. Comparison + Reporting**
- Compare baseline vs post-trained model
###### **Key Components / Deliverables**
1. [Key implementation steps / Key Data Manipulations / Key Components for this stage]
2. [key component name]
	1. [steps to implement key component/ tests to run for validation]
##### **8. Extension 1: Better Reward Functions**
- docking score
- confidence score
- energy-based score
- composite reward
##### **9. Extension 2: Inference-Time Guidance Baseline**
- compare post-training vs sampling-time steering
##### **10. Extension 3: PepFlow**
- repeat structure for:
    - known-backbone peptide design
    - full peptide generation
##### **11. Extension 4: Diagnostics**
- reward vs RMSD correlation
- diversity preservation
- failure cases
- reward hacking detection
