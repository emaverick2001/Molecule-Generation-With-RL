## ==**3. Experiments (How do we test and evaluate it?)**==
- **Purpose:**
    - Empirically evaluate whether the proposed method satisfies the **hypothesis and objectives** defined earlier
    - Provide **quantitative and qualitative evidence**
    - Research Questions → Metrics → Setup → Plan → Results → Analysis → Limitations
- **Key Question:**
    - _Does the proposed method work, under what conditions, and why?_
- **Constraints:**
    - No explanation of how the system works (belongs to Proposed Technique)
    - No restating problem motivation (belongs to Introduction)
    - Every result must map to:
        - a **research question**
        - a **metric**
    - Separate clearly:
        - **what is being tested**
        - **how it is tested**
        - **what the results are**
        - **why the results occur**
---
### ==**1. Research Questions (What are we testing?)**==
- Define the **key questions your evaluation answers**
	- What is the main question to be answered from the Experiment?
- Each should map to:
    - hypothesis
    - contribution
- **Output of this section should be:**
	- A **small set of testable evaluation questions**
---
#### **1. Effectiveness**
- Does the method improve task performance?
- Does it achieve the intended objective?
---
- 
#### **2. Efficiency**
- Does the method improve resource usage?
- Memory, latency, throughput?
---
- 
#### **3. Quality of Behavior / Decisions**
- Does the system make better intermediate decisions?
- Are outputs more stable, relevant, or consistent?
---
- 
#### _(Optional)_ Additional RQs:
- Robustness
- Generalization
- Scalability
---
- 
### ==**2. Experiment Evaluation**==
#### **1. Objective Function (What does “good” mean?)**
- What are we optimizing for?
- Is the objective **explicit (mathematical)** or **implicit (proxy-based)**?
- Define the main goal
---
#### **3. Efficiency Objectives**
---
#### **4. Correctness Objectives**
---
#### **5. Implicit Objectives**
---
#### **6. Tradeoffs**
- What competing objectives exist?
- Examples:
	- Quality vs memory
	- Latency vs accuracy
	- Throughput vs personalization
---
#### **7. Constraints**
- What **system-level or problem-level constraints** must be satisfied?
- Categories:
	- **Memory constraints:**
	    - e.g., KV cache budget ≤B\leq B≤B
	- **Latency constraints:**
	    - per-step or end-to-end latency
	- **Compute constraints:**
	    - FLOPs, GPU limits
	- **Quality constraints:**
	    - minimum accuracy / fidelity
	- **Causality / online constraints:**
	    - decisions must be made sequentially without future knowledge
- **Output of this section should be:**
	- A **set of formal constraints that define feasibility**
---
- 
#### **8. Assumptions**
- What assumptions are required to make the problem tractable?
- Examples:
	- Retrieval scores are available and meaningful
	- Attention reflects relevance
	- Data distribution is stationary
	- Errors are independent across steps
- **Output of this section should be:**
	- A **list of modeling assumptions**
---
- 
#### ==**9. Metrics**==
1. for each metric listed list what it measures, the formula, our hypothetical expected Direction of Change, why it matters.
2. rank the metrics based on highest impact and usability for this project
---
##### **1. Effectiveness (Primary Metrics)**
- These are your **most important metrics**.
---
1. Top-1 RMSD (Docking Accuracy)
	- **What it measures/ Why it matters**
	    - How close the _best predicted pose_ is to the ground-truth binding pose.
	    - This is the **gold standard metric for docking**
	    - Answers: Are we generating _better molecular poses_, not just ranking them better?
	- **Formula**
		- $\text{RMSD} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \| x_i^{pred} - x_i^{true} \|^2}$
	- **Expected Direction of Change**
	    - ⬇️ Lower is better (post-training should reduce RMSD)
2. Top-k Success Rate (k = 1, 5, 10)
	- **What it measures/ Why it matters**
		- Fraction of cases where at least one generated pose is “correct” (e.g., RMSD < 2Å)
		- Captures **practical usability**
		- Answers: Does post-training increase the probability of generating at least one good solution
	- **Formula**
		- $\text{Success@k} = \frac{1}{M} \sum_{i=1}^{M} \mathbf{1}(\min_{j \le k} \text{RMSD}_{i,j} < \tau)$
	- **Expected Direction of Change**
		- ⬆️ Higher is better
3. Mean Best-of-N Reward
	- **What it measures/ Why it matters**
		- Average reward score of the best sample per input
		- Directly measures **alignment with your optimization objective**
		- Answers: Did the model actually learn from the reward?
	- **Formula**
		- $\text{BestReward} = \frac{1}{M} \sum_{i=1}^{M} \max_{j \le N} R(x_{i,j})$
	- **Expected Direction of Change**
		- ⬆️ Higher is better
4. Average RMSD (Across All Samples)
	- **What it measures/ Why it matters**
		- Overall quality of the entire generated distribution
		- Tests whether improvement is global (distribution shift) vs localized (just best sample)
	- **Formula**
		- $\text{AvgRMSD} = \frac{1}{M \cdot N} \sum_{i=1}^{M} \sum_{j=1}^{N} \text{RMSD}_{i,j}$
	- **Expected Direction of Change**
		- ⬇️ Lower is better
5. Property Score (PepFlow Stage)
	- **What it measures/ Why it matters**
		- Stability / affinity / energy score of generated peptides
		- Ensures method generalizes beyond docking to **biochemical objectives**
	- **Formula**
		- $\text{PropertyScore} = \frac{1}{M} \sum_{i=1}^{M} f(x_i)$
	- **Expected Direction of Change**
		- ⬆️ Higher (or lower depending on energy definition)
6. [Metric n]
	- **What it measures/ Why it matters**
		- 
	- **Formula**
		- 
	- **Expected Direction of Change**
		- 
##### **2. Efficiency Metrics**
- These capture **cost vs benefit**.
---
1. Samples Required for Success (Sample Efficiency)
	- **What it measures/ Why it matters**
		- How many samples are needed to reach a correct pose
		- Answers: Does post-training reduce reliance on brute-force sampling?
	- **Formula**
		- $\mathbb{E}[\min \{ j : \text{RMSD}_{i,j} < \tau \}]$
	- **Expected Direction of Change**
		- ⬇️ Lower is better
2. Generation Time per Sample
	- **What it measures/ Why it matters**
		- Time required to generate one candidate
		- Ensures improvements aren’t too computationally expensive
	- **Formula**
		- $\text{Time/sample} = \frac{\text{Total generation time}}{\text{# samples}}$
	- **Expected Direction of Change**
		- Ideally ↔️ or slight ⬆️ (acceptable tradeoff)
3. Training Cost (Post-Training Overhead)
	- **What it measures/ Why it matters**
		- Compute required for reward-based fine-tuning
		- Evaluates practicality of your approach
	- **Formula**
		- $\text{Total GPU hours or FLOPs}$
	- **Expected Direction of Change**
		- ⬆️ (expected), but should be justified by gains
##### **3. Behavioral Diagnostics (VERY valuable for analysis)**
- These help explain _why_ your method works.
---
1. Reward Distribution Shift
	- **What it measures/ Why it matters**
		- How the distribution of rewards changes after training
		- Answers: The generator itself has improved, not just selection
	- **Formula**
		- $P_{\text{reward}}^{before}(x) \rightarrow P_{\text{reward}}^{after}(x)$
	- **Expected Direction of Change**
		- Shift toward higher rewards
2. Diversity Metrics (Structural Diversity)
	- **What it measures/ Why it matters**
		- Variety of generated molecules
		- Detects **over-optimization / collapse**
	- **Formula**
		- $\text{Diversity} = \mathbb{E}_{i,j}[\text{RMSD}(x_i, x_j)]$
	- **Expected Direction of Change**
		- Slight ⬇️ or stable (too much drop = mode collapse)
3. Reward vs RMSD Correlation
	- **What it measures/ Why it matters**
		- Whether reward aligns with actual correctness
		- Answers: Is the reward signal meaningful?
	- **Formula**
		- $\rho = \text{corr}(R(x), -\text{RMSD}(x))$
	- **Expected Direction of Change**
		- ⬆️ Higher correlation
4. Improvement per Iteration (Learning Curve)
	- **What it measures/ Why it matters**
		- How performance improves over training steps
		- Confirms training stability and convergence
	- **Formula**
		- $\text{Metric}(t) \text{ vs training step t}$
	- **Expected Direction of Change**
		- Monotonic improvement (or early plateau)
5. Failure Case Analysis
	- **What it measures/ Why it matters**
		- Cases where:
			- high reward but bad RMSD
			- low diversity outputs
			- invalid molecules
		- Identifies:
			- reward hacking
			- misalignment
			- edge cases
### ==**3. Experiment Structure**==
- **Purpose:**
    - Define all **artifacts** used, produced, and evaluated in the system
    - Ensure **reproducibility, traceability, and clarity of data flow**
    - Inputs → Transformations → Intermediate Artifacts → Outputs → Evaluation Artifacts
- **Key Question:**
	- What experiments are run?
	- What variables are controlled vs varied?
	- _What concrete objects (data, models, files, outputs) flow through the system, and how are they structured, versioned, and used?_
- **Constraints:**
    - No algorithmic explanations (belongs to Proposed Technique)
    - No performance claims (belongs to Evaluation)
    - Must clearly distinguish:
        - **Inputs**
        - **Intermediate artifacts**
        - **Outputs**
    - Every artifact should be:
        - identifiable
        - reproducible
        - versionable
---
#### **Benchmarks/Datasets**
- choose datasets / benchmarks and explain what it is + how it fits this project, how to setup
- What datasets are used?
- What is their structure and format?
- What preprocessing is applied?
- Include:
	- Dataset name + source
	- Size / scale
	- Format (JSON, CSV, text, etc.)
	- Access method (local, API, repo)
---
##### [[PDBBind(DiffDock Benchmark)]]
###### **Role in Project**
- Primary dataset for **Stage 1 (DiffDock)**
- Used to:
    - generate docking poses
    - evaluate pose accuracy (RMSD)
###### **Structure & Format**
- Protein–ligand complexes
- Includes:
    - protein structure (PDB files)
    - ligand structure (SDF/MOL2)
    - ground-truth binding pose
- Stored as structured directories:
    /complex_id/  
    	protein.pdb  
    	ligand.sdf  
    	label.json (optional metadata)
###### **Scale**
- ~4K–20K complexes depending on split (train/test)
- Standard splits used by DiffDock
###### **Preprocessing**
- Normalize coordinates
- Ensure ligand/protein alignment
- Convert to model-compatible tensors (DiffDock format)
---
##### [[PepFlow Peptide Dataset]]
###### **Role in Project**
- Used for **Stage 2 and Stage 3 (PepFlow)**
- Tasks:
    - fixed-backbone design
    - sequence-structure co-design
###### **Structure & Format**
- Peptide sequences + structures
- Includes:
    - amino acid sequence
    - backbone coordinates
    - full-atom structure (optional depending on task)
###### **Scale**
- Thousands to tens of thousands of peptide samples
###### **Preprocessing**
- Tokenize sequences
- Normalize structural representations
- Format into flow-matching input format
---
#### **Baselines**
- What is the structure of the baseline?
	1. Different modeling (e.g., simplex flow, CTMC for discrete diffusion, VAE)
	2. Different sampling (e.g., noise scheduler, temperature, self-condition)
	3. Reference Test Set Statistics to compare against actual molecules
- What models are used (if any)?
	- Model name / architecture
	- Training
	- Version / checkpoint
	- Role in system (e.g., embedding, generation, scoring)
---
##### **Novel Method: DiffDock + Reward-Based Post-Training**
- This is your **proposed system**
- Compared against ALL baselines above
##### **Baseline 1: DiffDock (Pretrained, No Post-Training)**
- **Model**
    - DiffDock diffusion model
- **Training**
    - Pretrained checkpoint (official release)
- **Role**
    - Baseline generator
- **Behavior**
	- Input: protein + ligand
	- Generate N poses via diffusion sampling
	- No modification to model weights
---
##### **Baseline 2: DiffDock + Confidence Reranking**
- **Model Components**
    - DiffDock (generator)
    - Confidence model (scorer)
- **Training**
    - Both pretrained (no additional tuning)
- **Role in System**
    - Generator + **post-hoc scoring module**
- **Behavior**
    1. Generate N candidate poses
    2. Score each using confidence model
    3. Select best candidate
---
##### **Baseline 3: DiffDock + Inference-Time Guidance**
- **Model**
    - DiffDock with modified sampling
- **Approach**
    - Inject guidance signal during diffusion sampling
        - classifier-free guidance
        - heuristic energy bias
        - test-time alignment
- **Training**
    - No additional training
- **Role in System**
    - **Generator with modified sampling dynamics**
- **Behavior**
    1. Modify noise or score function during sampling
    2. Bias generation toward higher-quality outputs
---
##### **Baseline 4: DiffDock + Naive Reward Filtering**
- **Model**
    - DiffDock (unchanged)
- **Approach**
    - Generate N samples
    - Compute reward (e.g., docking score)
    - Select best based on reward
- **Role**
    - **Reward-aware reranking (no training)**
- **Behavior**
    1. Generate candidates
    2. Evaluate with reward
    3. Pick best
##### **Baseline 1: PepFlow (raw)**
- **Model**
    - DiffDock diffusion model
- **Training**
    - Pretrained checkpoint (official release)
- **Role**
    - Baseline generator
- **Behavior**
	- Input: protein + ligand
	- Generate N poses via diffusion sampling
	- No modification to model weights
---
#### **Experiment Controls/Fairness Setup**
##### **Controlled Variables**
- Dataset splits (fixed across experiments)
- Number of generated samples per input (N)
- Model architecture (DiffDock / PepFlow unchanged)
- Evaluation metrics
- Random seed (where possible)
---
##### **Independent Variables (Varied)**
- Post-training method:
    - RL-style (DDPO)
    - preference-based (DPO)
    - reward backpropagation (AlignProp)
- Reward function definition:
    - docking score
    - energy / stability
- Training intensity:
    - number of fine-tuning steps
    - learning rate
- Sampling strategy (optional ablation)
---
#### **Ablation Studies**
- List each experiment + its ablation
---
- **Baseline vs Post-Training**
    - DiffDock (no training)
    - DiffDock + reward-based post-training
- **Post-Training Method Comparison**
    - RL-style vs preference-based vs differentiable reward
- **Reward Function Design**
    - simple reward vs composite reward
    - sparse vs dense reward
- **Training Strength**
    - few steps vs many steps
- **Inference vs Training Alignment**
    - inference guidance vs post-training
#### **Case Studies**
- List highest insight case studies
- Output for this section should be:
	- 
---

#### ==**Experiment Protocol (Explanation/Paragraphs)**==
- How is each experiment executed?
- Example:
	1. Load input
	2. Run system
	3. collect outputs
	4. compute metrics
---
1. **Load Input**
    - Select dataset split (DiffDock or PepFlow)
    - Load pretrained model checkpoint
2. **Run Generation**
    - Generate N samples per input
    - (Optional) apply inference guidance
3. **Apply Post-Training (if applicable)**
    - Fine-tune model using reward signal
    - Save updated checkpoint
4. **Generate Outputs**
    - Run generation again using updated model
    - Store generated molecules / poses
5. **Compute Evaluation Metrics**
    - Docking: RMSD, top-k success
    - Peptides: stability / affinity metrics
6. **Log Artifacts**
    - Save generated samples
    - Save metrics
    - Save config + seed
#### **(After Experiment Structure). Diagrams (System Visualization Layer)**
- Visual representations of the system
- **Output of this section should be:**
	- A **visual understanding of system + artifacts**
---
##### **System Overview Diagram**
- Bug Report → Context Retrieval → Neural Proposal → Impact Slicing → Constraint Extraction → Symbolic Execution → Counterexample Generation → Feedback Transformation → CEGF Loop
##### **Dataflow Diagram**
- task instance → repo context → patch diff → program slice → SMT constraints → solver result → critique → refined patch
##### **Repair Loop Diagram**
- shows the closed-loop interaction between the LLM and symbolic solver
##### **Artifact Flow Diagram**
- maps which files are consumed and produced at each stage
##### **Failure Mode Diagram**
- shows branching outcomes such as:
	- patch parse failure
	- unsupported symbolic construct
	- solver SAT / UNSAT / UNKNOWN / TIMEOUT
##### **Ablation Diagram**
- full Symbiotic-SWE vs. stripped variants:
	- neural only
	- neural + slicing
	- neural + solver without critique
	- neural + critique without slicing
	- full system

### **4. Threats to Validity (How could this be wrong?)**
- What could invalidate your conclusions?
- Types:
	- dataset bias
	- metric limitations
	- experimental setup bias
	- implementation artifacts
- **Output of this section should be:**
	- A **critical reflection on limitations of evaluation**
---
- **1. [Name of Threat]**
	- **Description**
		- 
	- **Potential Issues**
		- 
	- **Impact**
		- 
	- **Mitigation**
		- 
### **5. Failure Strategization**
- **Purpose:**
    - Identify where and why the system fails
    - Categorize failures by **component and root cause**
    - Provide insight into:
        - system limitations
        - future improvements
        - validity of assumptions
- **Output of this section should be:**
    - A structured breakdown of failure modes with explanations and diagnostic signals
---
- **1. [Name of Stage in Pipeline failure mode occurs]**
	- **Description**
		- 
	- **Failure Modes**
		- 
	- **Root Cause**
		- 
	- **Detection Signals**
		- 
	- **Mitigation** 
		- 