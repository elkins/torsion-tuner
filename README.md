# Experimental Fine-Tuner: GNN-Based Protein Structure Refinement

The **Experimental Fine-Tuner** is a specialized machine learning tool designed to bridge the gap between "idealized" static protein structures (like those from AlphaFold) and "dynamic" solution-state experimental data (SAXS/NMR). 

By utilizing a JAX-based Graph Neural Network (GNN) and a differentiable kinematics layer, the model applies subtle adjustments to the backbone dihedral angles ($\phi/\psi$) to minimize the discrepancy between the predicted structure and real experimental scattering curves.

---

## 🔬 Theoretical Foundations

### 1. The Torsional Prediction Strategy
Instead of predicting 3D coordinates ($x, y, z$) directly—which can easily break chemical constraints like bond lengths—this model predicts **torsional deltas** ($\Delta\phi, \Delta\psi$). 
*   **Advantage:** By keeping bond lengths and bond angles fixed to their idealized values, we ensure the resulting structure is always chemically valid (no "stretched" bonds).
*   **Input:** Initial backbone angles from an AlphaFold PDB.
*   **Output:** Small angular adjustments that "nudge" the protein into a conformation that fits the data.

### 2. Differentiable Kinematics (NeRF)
To calculate a loss against experimental data, we must convert the predicted angles back into 3D coordinates. We use the **Natural Extension Reference Frame (NeRF)** algorithm, implemented in JAX.
*   **Forward Pass:** Angles $\rightarrow$ 3D Coordinates $\rightarrow$ Simulated SAXS.
*   **Backward Pass:** $\frac{\partial \text{Loss}}{\partial \text{SAXS}} \rightarrow \frac{\partial \text{SAXS}}{\partial \text{Coords}} \rightarrow \frac{\partial \text{Coords}}{\partial \text{Angles}} \rightarrow \frac{\partial \text{Angles}}{\partial \text{GNN Weights}}$.
*   This allows the experimental data to directly "teach" the GNN how to fold the protein.

### 3. The SAXS Loss Function (Debye Formula)
We use the **Debye formula** to simulate Small-Angle X-ray Scattering (SAXS) intensities $I(q)$ from the 3D coordinates:
$$I(q) = \sum_{i} \sum_{j} f_i(q) f_j(q) \frac{\sin(q r_{ij})}{q r_{ij}}$$
The model is trained to minimize the Mean Squared Error (MSE) between this simulated $I(q)$ and your experimental $I(q)$.

---

## 🛠 Software Architecture

*   **`src/data.py`**: Handles PDB loading using `Biotite`. It extracts the backbone (N, CA, C atoms) and constructs a graph where residues are nodes and edges represent sequence adjacency or spatial proximity.
*   **`src/model.py`**: An **Equinox** (JAX-native) GNN. It uses message-passing layers to allow residues to "communicate" with their neighbors before predicting their local angular shifts.
*   **`src/kinematics.py`**: The bridge between angles and 3D space. It ensures that any update to a single angle correctly propagates to all subsequent atoms in the chain.
*   **`src/train.py`**: The orchestration layer. It sets up the **Optax** optimizer (AdamW), applies gradient clipping for stability, and executes the training loop.

---

## 🚀 Getting Started

### 1. Installation
Ensure you have a Python 3.10+ environment.
```bash
# Install core dependencies and the local biophysics library
pip install -r requirements.txt
```

### 2. Preparing Your Data
1.  Place your "idealized" AlphaFold structure in the project root as `input.pdb`.
2.  (Optional) If you have a specific SAXS profile, update `src/train.py` to load your experimental $I(q)$ data instead of using the synthetic helix target.

### 3. Running a Refinement
You can test the system using the provided helix generator:
```bash
# 1. Generate a synthetic "target" helix
python generate_test_pdb.py

# 2. Run the fine-tuning optimization
PYTHONPATH=. python src/train.py
```

### 4. Interpreting Results
*   **`test_helix.pdb`**: Your starting "idealized" structure.
*   **`refined_helix.pdb`**: The output structure after the GNN has adjusted it to fit the experimental data.
*   Check the console output for the **Loss** values; a decreasing loss indicates the model is successfully "fitting" the structure to the data.

---

## 🔬 For Structural Biologists
...
*   **Data Types:** While currently set up for SAXS, the `diff-biophys` library supports NMR (RDCs, J-couplings) and can be easily swapped in `src/train.py`.

## 🏛 Integrating Research from the Montelione Group
The **Montelione Group** (Gaetano Montelione, RPI) has pioneered the integration of AI-predicted structures with NMR data. You can leverage their research findings in this project through the following workflows:

### 1. AlphaFold-NMR Workflow
Instead of refining a single structure, the Montelione group advocates for an **Ensemble Conformer Selection** approach.
*   **Step:** Use the Fine-Tuner to generate multiple refined structures by varying the `train_key` (random seed) or the `q_values` range.
*   **Step:** Score the resulting ensemble against experimental NOESY or chemical shift data using the **RPF-DP** (Recall, Precision, F-measure) metrics developed by the lab.

### 2. Chemical Shift Driven Refinement
The lab's research shows that AlphaFold models often rival the accuracy of experimental NMR structures for small proteins. You can further refine these models using chemical shift data.
*   **Implementation:** See `src/montelione_utils.py`. We have implemented a differentiable loss function based on **CA Chemical Shift Prediction**.
*   **Utility:** This allows you to "nudge" the AlphaFold structure to better match experimental $C_\alpha$ shifts, which are highly sensitive to secondary structure ($\alpha$-helices and $\beta$-sheets).

### 3. Structure Validation with PSVS
The **Protein Structure Validation Software (PSVS)** suite from the Montelione lab is the "gold standard" for validating how well a model fits NMR observables.
*   **Integration:** After refining your structure with the GNN, submit the `refined_structure.pdb` to the [PSVS server](http://psvs-1_5.nesg.org/).
*   **Metrics:** Focus on the **ANSURR** score (which measures structural rigidity vs. chemical shifts) and the **RPF-DP** scores to confirm that your refinement has improved the "NMR-quality" of the AlphaFold prediction.

## 💻 For Computer Scientists
...

*   **Framework:** JAX was chosen over PyTorch for its superior performance in "simulation-in-the-loop" training where the loss function itself involves heavy linear algebra (NeRF) and trigonometric operations.
*   **GNN Detail:** The default model is a 3-layer Graph Convolutional Network. You can increase the `hidden_dim` in `src/train.py` for more complex proteins.
*   **Stability:** We use `optax.clip_by_global_norm(1.0)` to prevent the "exploding gradient" problem often found in recurrent-like kinematic chains.
