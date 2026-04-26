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
	- Follow [[Software Implementation Guide]]
2. Setup Project Repository Structure
	1. Create top-level directories:
		- assets/
			- diagrams/
			- figures/
		- artifacts/
			- runs/
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
##### **1. Dataset Loading + Processing
###### Datastructures
1. [Datastructure name]
	- [Datastructure description / purpose]
	```python
    # Datastructure example with fields and types
	```
###### **Key Components / Deliverables**
1. Add Error Handling & Fault Tolerance
	- Skip invalid molecule generations
	- Catch:
	    - invalid geometry
	    - failed reward computation
	- Log failed samples separately
2. Data Validation  
	- verify repo integrity  
	- ensure tests reproduce failure  
	- validate task schema
3. [Key implementation steps / Key Data Manipulations / Key Components for this stage]
4. [key component name]
	1. [steps to implement key component/ tests to run for validation]
##### **2. Baseline Generation**
##### **3. Reward Scoring**
##### **4. Evaluation**
##### **5. Post-Training**
##### **6. Post-Trained Generation**
##### **7. Comparison + Reporting**
###### Datastructures
1. [Datastructure name]
	- [Datastructure description / purpose]
	```python
    # Datastructure example with fields and types
	```
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
#### **Artifact Schema / Serialization Contract**
- Define how artifacts are structured and stored
- **Output of this section should be:**
	- A **reproducible artifact specification**
---
##### **Serialization Format**
- JSON / binary / database schema
- Key naming conventions
- Versioning format
---
- 
##### **Versioning Strategy**
- How are artifacts versioned?
- Examples:
	- dataset version
	- model checkpoint version
	- experiment ID
---
- 
#### **Threats to Validity (How could this be wrong?)**
- What could invalidate your conclusions?
- Types:
	- dataset bias
	- metric limitations
	- experimental setup bias
	- implementation artifacts
- **Output of this section should be:**
	- A **critical reflection on limitations of evaluation**
---
#### **Optimizations**
##### Orchestration / Pipeline Management
##### **Complexity / Scaling Behavior (Optional but Strong)**
- How does the method scale?
- Examples:
	- time complexity
	- memory usage
	- behavior with larger inputs
---
- 
##### Caching  
- 
##### Security / Isolation (Optional but Strong)