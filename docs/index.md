# TorsionTuner

The **TorsionTuner** is a specialized machine learning tool designed to bridge the gap between "idealized" static protein structures (like those from AlphaFold) and "dynamic" solution-state experimental data (SAXS/NMR). 

By utilizing a JAX-based Graph Neural Network (GNN) and a differentiable kinematics layer, the model applies subtle adjustments to the backbone dihedral angles ($\phi/\psi$) to minimize the discrepancy between the predicted structure and real experimental observables.

---

## 🔬 Key Features

*   **Torsional Prediction Strategy**: Predicts $\Delta\phi/\Delta\psi$ to maintain chemical constraints.
*   **Differentiable Kinematics**: Natural Extension Reference Frame (NeRF) implemented in JAX.
*   **Multi-Objective Loss**: Integrates SAXS, NMR, and Ramachandran geometry regularization.
*   **GNN-Based**: Captures spatial and sequential relationships via an Equinox-based GNN.

---

## 📖 Key References
*   **RPF Scores:** Huang, Y. J., et al. (2005). *J. Am. Chem. Soc.*, 127(5), 1665–1674.
*   **Rosetta Refinement:** Mao, B., et al. (2014). *J. Am. Chem. Soc.*, 136(5), 1893–1906.
*   **AlphaFold-NMR Assessment:** Li, E. H., et al. (2023). *J. Magn. Reson.*, 352, 107481.
*   **Debye Formula for SAXS:** Debye, P. (1915). *Annalen der Physik*, 351(6), 809-876.
