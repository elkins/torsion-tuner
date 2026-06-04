# Known Limitations

TorsionTuner is designed for targeted backbone refinement. To ensure scientifically sound results, users should be aware of the following physical and implementation-defined limitations.

---

## 🧪 Scientific Limitations

### 1. Chemical Shift Accuracy Floor
The current chemical shift predictor uses **residue-agnostic offsets** (fixed $\pm 3.1$ ppm for $\alpha$-helices and $-1.5$ ppm for $\beta$-sheets).
*   **Impact:** This results in an irreducible accuracy floor of approximately **1.0–1.8 ppm CSRMSD**.
*   **Note:** While this is sufficient for identifying secondary structure improvements, it cannot match the precision of residue-specific predictors like SPARTA+ or SHIFTX2.

### 2. SAXS Valid $q$-Range
The implementation uses the **Debye formula** with $q$-independent form factors (atomic numbers $Z$).
*   **Constraint:** The model is highly accurate for global shape and compaction at low scattering angles but diverges from reality at higher resolution.
*   **Recommendation:** Limit scattering data analysis to **$q < 0.15 \text{Å}^{-1}$**. Above this range, solvent scattering and atomic-level hydration shells (not modeled here) become dominant.

### 3. Backbone-Only Representation
TorsionTuner refines only the main-chain atoms ($N, C_\alpha, C, O$).
*   **Impact:** The model does not account for side-chain packing, rotamer favorability, or specific side-chain-to-side-chain interactions.
*   **Validation:** Users should use external tools like **MolProbity** post-refinement to verify that backbone changes haven't introduced non-physical side-chain clashes.

### 4. Peptide Bond Geometry ($\omega$)
The kinematics engine (NeRF) assumes a standard planar trans-peptide bond ($\omega = 180^\circ$).
*   **Constraint:** The model does not currently refine the $\omega$ angle. While this preserves physical planarity, it cannot model cis-proline bonds or significant peptide bond distortions unless they are present in the starting model and kept fixed.

---

## 💻 Computational Limitations

### 1. Single-Structure Refinement
The tool produces a single refined conformer rather than a structural ensemble.
*   **Context:** NMR and SAXS data are inherently ensemble-averaged measurements. A single structure fitting the data perfectly may represent an over-fitted "mean" rather than a physically populated state.

### 2. Gradient Vanishing in High-Quality Models
If a starting AlphaFold model is already near-perfect (CSRMSD $< 0.5$ ppm), the Gaussian soft-assignment for chemical shifts may produce very small gradients.
*   **Result:** In these cases, the "Torsional Nudge" may be negligible, as the model is already at a local minimum of the loss function.
