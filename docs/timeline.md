## **5. Timeline**

* **Purpose:**
  * Define a **structured execution plan** for the project
  * Translate the research idea into **phases, deliverables, and dependencies**
  * Phases → Dependencies → Execution Strategy → Parallelization → Risk Mitigation
* **Key Question:**
  * *What needs to be built, in what order, and how do we ensure progress and correctness?*
* **Constraints:**
  * Each phase must produce a **concrete deliverable**
  * Dependencies must be **explicitly defined**
  * Plan should support:
    * iteration
    * debugging
    * parallel work
  * Avoid vague tasks → prefer **actionable steps**

---

### **1. Phase-Based Plan (Milestone-Driven Execution)**

* Break the project into  **phases** , each with:
  * Define **ordering constraints between phases**
  * Define checkpoints to track progress
    * clear goal
    * concrete deliverable
    * actionable tasks
  * weekly milestones
  * demo checkpoints
  * evaluation checkpoints
* Example:
  * Week 1: Problem Formulation + Schemas
  * Week 2: Baseline System
  * Week 3: Core Mechanism (e.g., slicing)
  * Week 4: Feature / Signal Extraction
  * Week 5: Decision Module
  * Week 6: Feedback Loop
  * Week 7: Evaluation + Ablations
  * Week 8: Analysis + Writeup
* **Key Dependencies**
  * [Phase A] → required before [Phase B]
  * [Phase B] → required before [Phase C]
* **Output of this section should be:**
  * A **clear execution graph**
  * A **progress tracking system

---

#### **Week 1: Exploration and Setup**

##### **Goal**

* What is the objective of this phase?

---

* We will first spend one week reading code, reproducing a small baseline for DiffDock and PepFlow , and deciding which post-training objective is the cleanest starting point.
* We start with DiffDock first then move onto PepFlow since the DiffDock Pipeline is simpler
* Our initial candidates are policy-style fine-tuning as in DDPO, preference-style fine-tuning as in Diffusion-DPO  or D3PO, and direct reward backpropagation as in AlignProp when the reward is differentiable enough.

##### **Deliverables**

* What must be produced?

---

* (code, artifacts, diagrams, results)

##### **Tasks**

* What specific steps are required from  **High-Level Pipeline + Low-level Pipeline** ?

---

* 0. Setup + 1. Dataset Loading + Processing + 2. Baseline Generation + 3. Reward Scoring + 4. Evaluation

##### **Success Criteria (NEW — VERY IMPORTANT)**

* How do you know this phase is complete?
* What must be verified?

---

#### **Week 2: Stage 1 - DiffDock with RL / reward-based post-training**

##### **Goal**

* What is the objective of this phase?

---

* We will start with DiffDock, since it is the simplest and most self-contained setting, the reward signal is the easiest to define, and it is the best proof-of-concept for the overall idea.
* Our goal in this stage is to test whether direct optimization can improve the generated docking poses themselves, rather than only improving post-hoc reranking in DiffDock.
* We will begin with the simplest reward definitions and try one or two post-training objectives that are easiest to implement from Week 1.

##### **Deliverables**

* What must be produced?

---

* (code, artifacts, diagrams, results)

##### **Tasks**

* What specific steps are required from  **High-Level Pipeline + Low-level Pipeline** ?

---

* 5. Post-Training + 6. Post-Trained Generation

##### **Success Criteria (NEW — VERY IMPORTANT)**

* How do you know this phase is complete?
* What must be verified?

---

#### **Week 3: Stage 2 - PepFlow on known backbone / partial sampling**

##### **Goal**

* What is the objective of this phase?

---

* If Stage 1 is stable, we will move to the known-backbone setting in PepFlow, starting from fixed-backbone sequence design in PepFlow.
* This stage keeps more variables controlled, so training and evaluation should be more stable than full peptide co-design.
* Since PepFlow is based on multi-modal flow matching, this stage is also where flow-oriented post-training ideas such as Flow-GRPO become especially relevant.
* As a lighter baseline, we will also compare against inference-time control methods such as classifier-free guidance, Universal Guidance, or test-time alignment when they are easy to adapt.

##### **Deliverables**

* What must be produced?

---

* (code, artifacts, diagrams, results)

##### **Tasks**

* What specific steps are required from  **High-Level Pipeline + Low-level Pipeline** ?

---

##### **Success Criteria (NEW — VERY IMPORTANT)**

* How do you know this phase is complete?
* What must be verified?

#### **Week 4: Stage 3 - Full PepFlow**

##### **Goal**

* What is the objective of this phase?

---

* We will then move to the full PepFlow setting for sequence-structure co-design.
* This is a harder setting with a larger generation space, so the main question is whether reward-based post-training can still improve affinity- or stability-related objectives without clearly hurting overall sample quality. At this stage, we will use recent biomolecular alignment work such as AliDiff and direct energy-based preference optimization mainly as references for defining reasonable reward or preference signals.

##### **Deliverables**

* What must be produced?

---

* (code, artifacts, diagrams, results)

##### **Tasks**

* What specific steps are required from  **High-Level Pipeline + Low-level Pipeline** ?

---

* 7. Comparison + Reporting

##### **Success Criteria (NEW — VERY IMPORTANT)**

* How do you know this phase is complete?
* What must be verified?

---

#### **Week 5: Stage 4 - Extensions**

##### **Goal**

* What is the objective of this phase?

---

* If the earlier stages work, we will use the final week for extensions. One possible direction is inference steering, motivated by training-free guidance and test-time alignment methods in vision such as Universal Guidance and test-time alignment.
* Another possible direction is denoising-step or flow alignment across different modalities, motivated both by the multi-modal structure of PepFlow and by multimodal generation systems in vision such as Stable Diffusion 3 and Show-o. We see this stage as exploratory rather than required.

##### **Deliverables**

* What must be produced?

---

* (code, artifacts, diagrams, results)

##### **Tasks**

* What specific steps are required from  **High-Level Pipeline + Low-level Pipeline** ?

---

* After MVP: Extensions

##### **Success Criteria (NEW — VERY IMPORTANT)**

* How do you know this phase is complete?
* What must be verified?

---

### **2. Execution Strategy (How will work progress?)**

* Define how you will iterate through phases
* **Output of this section should be:**
  * A **repeatable development loop**

---

#### **Iteration Loop**

* Design → Implement → Test → Analyze → Refine

---

#### **Validation Strategy**

* How will each phase be validated before moving on?

---

### **3. Parallelization Plan (What can be done simultaneously?)**

* Identify independent workstreams
* Examples:
  * data processing vs model development
  * baseline vs proposed method
  * evaluation scripts vs system design
* Structure:
  * Task A and Task B can run in parallel because…
  * Task C depends on Task A
* **Output of this section should be:**
  * A **parallel execution strategy**

---
