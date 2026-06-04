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

## 🧪 Scientific Validation

TorsionTuner is validated at three levels:

1. **Internal parity** — physics engines (CS predictor, Debye SAXS) verified against independent implementations
2. **NESG benchmark** — Cα CSRMSD reduction on NESG targets using authentic BMRB experimental shifts
3. **External quality** — structural quality tracked via MolProbity, PSVS, and ANSURR (planned)

**Current benchmark results:**

| Target | BMRB | Residues | Initial CSRMSD | Final CSRMSD | Δ |
|--------|------|----------|----------------|--------------|---|
| 2KHD (α-helical) | 16238 | 17 | 0.430 ppm | 0.287 ppm | −33% |
| 2RN7 (mixed α/β) | 11017 | 91 | 1.819 ppm | — | see note¹ |

> ¹ 2RN7's CSRMSD improvement is limited by the residue-agnostic CS predictor floor (~1.8 ppm);
> see the roadmap for details.

For the full validation plan, benchmark methodology, and status of each item see
**[docs/VALIDATION_ROADMAP.md](VALIDATION_ROADMAP.md)**.

---

## 📖 Key References
*   **NeRF Kinematics:** Parsons, J., et al. (2005). *J. Comput. Chem.*, 26(10), 1063–1068.
*   **Chemical Shift Index (CSI):** Wishart, D. S., & Sykes, B. D. (1994). *J. Biomol. NMR*, 4(2), 171–180.
*   **Ramachandran Statistics:** Lovell, S. C., et al. (2003). *Proteins*, 50(3), 437–450.
*   **RPF Scores:** Huang, Y. J., et al. (2005). *J. Am. Chem. Soc.*, 127(5), 1665–1674.
*   **Rosetta Refinement:** Mao, B., et al. (2014). *J. Am. Chem. Soc.*, 136(5), 1893–1906.
*   **AlphaFold-NMR Assessment:** Li, E. H., et al. (2023). *J. Magn. Reson.*, 352, 107481.
*   **Debye Formula for SAXS:** Debye, P. (1915). *Annalen der Physik*, 351(6), 809-876.
