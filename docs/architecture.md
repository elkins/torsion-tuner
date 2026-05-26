# Architecture & Theory

## 🔬 Theoretical Foundations

### 1. The Torsional Prediction Strategy
Instead of predicting 3D coordinates ($x, y, z$) directly—which can easily break chemical constraints like bond lengths—this model predicts **torsional deltas** ($\Delta\phi, \Delta\psi$). 
*   **Advantage:** By keeping bond lengths and bond angles fixed to their idealized values, we ensure the resulting structure is always chemically valid.
*   **Input:** Initial backbone angles from an AlphaFold PDB.
*   **Output:** Small angular adjustments that "nudge" the protein into a conformation that fits the data.

### 2. Differentiable Kinematics (NeRF)
To calculate a loss against experimental data, we convert the predicted angles back into 3D coordinates using the **Natural Extension Reference Frame (NeRF)** algorithm, implemented in JAX.
*   **Forward Pass:** Angles $\rightarrow$ 3D Coordinates $\rightarrow$ Simulated Observables (SAXS/NMR).
*   **Backward Pass:** Gradients flow from the experimental loss back through the kinematics layer to the GNN weights.

### 3. Multi-Objective Loss Function
The model optimizes a weighted combination of multiple experimental and structural constraints:
$$\mathcal{L}_{total} = w_{saxs}\mathcal{L}_{saxs} + w_{nmr}\mathcal{L}_{nmr} + w_{quality}\mathcal{L}_{quality} + w_{reg}\mathcal{L}_{reg}$$

*   **SAXS Loss:** Uses the Debye formula to fit Small-Angle X-ray Scattering profiles.
*   **NMR Loss:** Fits experimental $C_\alpha$ chemical shifts using a differentiable predictor.
*   **Ramachandran Geometry Regularization:** A soft potential penalizing phi/psi angles outside the favored regions of the Ramachandran plot (alpha-helix, beta-strand, left-handed alpha-helix). Analogous to the torsion-angle terms in CNS and Rosetta; similar in spirit to PROCHECK's G-factor.
*   **Regularization:** Prevents over-fitting by keeping the angular "nudges" small.

---

## 🛠 Software Architecture

The project is structured as a JAX-native pipeline, utilizing **Equinox** for the neural network architecture and **Optax** for optimization.

*   **`src/data.py`**: PDB loading and graph construction using `Biotite`. Handles the conversion of PDB coordinates into graph features (residue types, adjacency, and distances).
*   **`src/model.py`**: An Equinox GNN with message-passing layers. It utilizes both sequential and spatial adjacency to process the protein structure.
*   **`src/kinematics.py`**: The JAX-differentiable bridge between angles and 3D space. Implements the NeRF algorithm.
*   **`src/montelione_utils.py`**: Implementation of chemical shift losses and Ramachandran geometry regularization.
*   **`src/train.py`**: The orchestration layer. Defines the training loop, loss functions, and optimization step.

---

## 🏛 Research Integration
This project implements refinement strategies pioneered by the **Montelione Group** (RPI/Rutgers) for integrating AI-predicted structures with NMR data.

### 1. Chemical Shift Driven Refinement
Research shows that AlphaFold models often rival the accuracy of experimental NMR structures but can be further improved by fitting to backbone chemical shifts.
*   **Reference:** *Tejero R, et al. (2022). "AlphaFold models of small proteins rival the accuracy of solution NMR structures." Frontiers in Molecular Biosciences.*

### 2. Validation with RPF-DP and ANSURR
The Montelione lab advocates for rigorous validation of refined models using "goodness-of-fit" metrics like RPF-DP and ANSURR.

> **Note on ANSURR:** True ANSURR validation (Fowler, Sljoka & Williamson, 2020, *Nature Commun.* 11:6321) compares per-residue NMR flexibility (RCI, derived from backbone chemical shifts) against structural rigidity computed by the FIRST graph-theoretic algorithm. TorsionTuner currently uses a **Ramachandran geometry regularizer** as a training-time backbone quality term; for post-refinement ANSURR validation, structures should be submitted to the ANSURR web server (ansurr.shef.ac.uk).
*   **Reference:** *Huang YJ, et al. (2012). "RPF: a quality assessment tool for protein NMR structures." Nucleic Acids Research.*
*   **Reference:** *Fowler NJ, et al. (2020). "A method for validating the accuracy of NMR protein structures." Nature Communications.*
