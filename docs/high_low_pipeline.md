### ==**System Design (WHAT is built)**==
- Describe the **architecture of your system**
- Define:
    - components
    - dataflow
    - interactions
- **Output of this section should be:**
	- A **clear system architecture + pipeline**
---
#### **High-Level Pipeline + Low-level Pipeline**
- Describe the system as a sequence of stages
- Remember to first create most simplistic working system then build ontop of it
1. First high-level then low-level
	1. for high-level mention the core data manipulations + actions + key system components + deliverables for respective section
	2. for low-level mention key datastructures + key artifact logging plan + implementation details to key components/deliverables for respective section so that you can implement the simplistic working system
	3. Once MVP is built add more details, optimizations, etc
	4. Optionally later on mention run artifacts produced + what datapoints to log
---
##### **Dataflow Example + End-to-end pipeline**
- Step-by-step system execution:
- Example:
	1. Input is received
	2. Features are computed
	3. Decision is made
	4. Output is produced
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
- **Pipeline**
	1. Load protein–ligand complex
	2. Run pretrained DiffDock
	3. Generate N candidate poses
	4. Score poses using confidence model / reward function
	5. Compute RMSD against ground truth
	6. Save outputs + metrics
	7. Run reward-based post-training
	8. Re-generate poses with post-trained model
	9. Compare baseline vs post-trained model
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
##### **2. Baseline Generation**
###### **Key Components / Deliverables**
0. Define the Baseline Generation Contract
	- Decide what baseline generation must produce for each complex.
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
6. Run Baseline Generation on Tiny Dataset 
7. Define Acceptance Criteria for This Phase
	- Baseline Generation is complete when:  
		1. `run_baseline.py` loads the mini manifest  
		2. it creates generated sample records for every complex  
		3. it saves `generated_samples_manifest.json`  
		4. generated sample paths are run-specific  
		5. tests pass  
		6. the pipeline can be switched from dry-run generation to DiffDock generation without changing the dataset loader
##### **3. Reward Scoring**
###### **Key Components / Deliverables**
8. [Key implementation steps / Key Data Manipulations / Key Components for this stage]
9. [key component name]
	1. [steps to implement key component/ tests to run for validation]
##### **4. Evaluation**
###### **Key Components / Deliverables**
8. [Key implementation steps / Key Data Manipulations / Key Components for this stage]
9. [key component name]
	1. [steps to implement key component/ tests to run for validation]
##### **5. Post-Training**
###### **Key Components / Deliverables**
8. [Key implementation steps / Key Data Manipulations / Key Components for this stage]
9. [key component name]
	1. [steps to implement key component/ tests to run for validation]
##### **6. Post-Trained Generation**
###### **Key Components / Deliverables**
8. [Key implementation steps / Key Data Manipulations / Key Components for this stage]
9. [key component name]
	1. [steps to implement key component/ tests to run for validation]
##### **7. Comparison + Reporting**
###### **Key Components / Deliverables**
1. [Key implementation steps / Key Data Manipulations / Key Components for this stage]
2. [key component name]
	1. [steps to implement key component/ tests to run for validation]
##### **After MVP: Extensions**
###### **Extension 1: Better Reward Functions**
- docking score
- confidence score
- energy-based score
- composite reward
###### **Extension 2: Inference-Time Guidance Baseline**
- compare post-training vs sampling-time steering
###### **Extension 3: PepFlow**
- repeat structure for:
    - known-backbone peptide design
    - full peptide generation
###### **Extension 4: Diagnostics**
- reward vs RMSD correlation
- diversity preservation
- failure cases
- reward hacking detection