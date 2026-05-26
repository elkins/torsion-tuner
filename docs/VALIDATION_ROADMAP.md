# TorsionTuner — Validation & Documentation Roadmap

> Goal: Leave no scientific doubt. Establish TorsionTuner as a thoroughly
> peer-reviewable, reproducible structural biology tool.

---

## Part I — Additional Scientific Validation

### Tier 1: Internal Parity Tests (Highest Priority, ~1–2 weeks)

These tests confirm that TorsionTuner's internal physics engines
agree with established gold-standard software. They require no new
experimental data — just pre-computed reference outputs.

---

#### 1.1 Chemical Shift Predictor Parity vs. SPARTA+

**What to do:**  
For 3–5 proteins with known structures *and* deposited chemical shifts,
compute Cα shifts with both SPARTA+ and TorsionTuner's predictor.
Save SPARTA+ output as a reference file; write a pytest that asserts
TorsionTuner's predictions fall within a specified tolerance for the
same input angles.

**Why it matters:**  
The current predictor uses residue-agnostic +3.1/−1.5 ppm offsets.
A parity test quantifies *exactly* how large the approximation error is
(expected: ~0.5–1.5 ppm RMS residual vs. SPARTA+), so users can
calibrate their expectations.

**Success criterion:**  
Document the systematic bias. A tolerance of ±2 ppm is reasonable for
a simplified predictor; anything larger should motivate a residue-specific
lookup table upgrade.

**Tools needed:** SPARTA+ (free, standalone binary from Bax lab)

---

#### 1.2 SAXS Parity vs. CRYSOL or FoXS

**What to do:**  
Compute I(q) for a small test protein (e.g., ubiquitin, PDB 1UBQ) using
both CRYSOL/FoXS and `debye_saxs`. Compare normalized profiles over
q = 0.01–0.3 Å⁻¹. Write a pytest asserting χ² < 0.05 for the normalized
profiles at low q.

**Why it matters:**  
The Debye formula is correct, but the q-independent form factors (Z)
diverge from reality above ~0.15 Å⁻¹. A parity test quantifies the
q-range over which the approximation is valid, allowing documentation
of safe operating bounds.

**Success criterion:**  
Agreement within 5% at q < 0.15 Å⁻¹. Document the breakdown at higher q.

**Tools needed:** CRYSOL (ATSAS suite, free academic) or FoXS (web server)

---

#### 1.3 Omega Angle Integrity Check

**What to do:**  
After refinement, assert that all ω (omega) dihedral angles remain
within ±15° of 180° (trans) or ±15° of 0° (cis-Pro). Add this as a
post-refinement assertion in both `train.py` and the benchmark tests.

**Why it matters:**  
The index mapping in `kinematics.py` assumes ω is never modified, but
there is no enforcement. A distorted peptide bond would be unphysical
and invisible to the current loss function.

**Code change:** ~10 lines in `kinematics.py` or `train.py`.

---

### Tier 2: NESG Benchmark Expansion (~2–4 weeks)

#### 2.1 Complete All 9 Li et al. 2023 Targets

**Current status:** 2KHD and 2RN7 implemented.  
**Remaining:** 2KIW, 2KZV, 2L9R, 2LCI, 2M5O, 2M2L, 2M5Q
(verify exact list against Li et al. 2023 Table 1 first).

**What to do for each target:**
1. Download AlphaFold2 model from AF DB or generate with ColabFold
2. Download Cα shifts from corresponding BMRB entry
3. Run `test_2khd.py`-style benchmark
4. Record initial and final CSRMSD

**Why it matters:**  
Two proteins is not a statistically meaningful sample. Nine proteins
covering diverse folds (α, β, α/β) is the minimum for a credible claim.

**Success criterion:**  
Improvement in CSRMSD for ≥7/9 targets. Report mean ± SD reduction.

---

#### 2.2 Whole-Protein CSRMSD (Not Just Subset)

**Current state:**  
2KHD uses 17/108 residues; 2RN7 uses 91 residues (now corrected) but
only those assigned in BMRB 11017.

**What to do:**  
Use the full BMRB chemical shift assignment for each target, not just
a curated subset. Report CSRMSD separately for:
- Residues with experimental data (observed CSRMSD)
- Secondary structure elements only
- The full chain (predicted vs. expected using random-coil baseline)

---

#### 2.3 Statistical Significance Testing

**What to do:**  
Run each benchmark 5 times with different random seeds. Report:
- Mean CSRMSD improvement ± SD across seeds
- A one-sided Wilcoxon signed-rank test (or t-test) for improvement
- Whether results are consistent or seed-dependent

**Why it matters:**  
A single training run could get lucky. Showing consistent improvement
across seeds is essential for credibility.

---

### Tier 3: External Structural Quality Metrics (~2–4 weeks)

This is the strongest possible validation: show the refined structure
is better by *external*, *independent* tools — not TorsionTuner's own
loss function.

---

#### 3.1 Real PROCHECK / MolProbity Analysis

**What to do:**  
For each benchmark target:
1. Export initial AlphaFold model to PDB → run through MolProbity
2. Export TorsionTuner-refined model → run through MolProbity
3. Record Ramachandran favored %, clash score, rotamer outliers

**Why it matters:**  
The internal `ramachandran_penalty` is trained to minimize itself —
showing external MolProbity also improves proves the penalty is
physically meaningful, not just self-consistent.

**Tools needed:** MolProbity (web server or standalone)

---

#### 3.2 Real PSVS Submission for 2KHD and 2RN7

**What to do:**  
Actually submit both AlphaFold start models and TorsionTuner-refined
models to PSVS (http://psvs-1_5.nesg.org/). Save and commit the output
reports. Write a real (non-mock) version of `test_psvs_improvement.py`
that parses the real PSVS output files.

**Why it matters:**  
The current PSVS test uses entirely synthetic data. Real PSVS reports
would be the first genuine external validation of structural quality
improvement — directly comparable to what the Montelione group reports
in published refinement papers.

---

#### 3.3 ANSURR Validation (Post-Hoc, External)

**What to do:**  
Submit refined structures to the ANSURR web server (ansurr.shef.ac.uk)
along with the corresponding BMRB chemical shifts. Record the
correlation and RMSD scores before and after refinement.

**Why it matters:**  
ANSURR is the gold standard for NMR-specific structural accuracy as
advocated by the Montelione group. Showing improvement on ANSURR — even
as a post-hoc metric, not a training loss — would validate the entire
approach and correctly connect the project to the Fowler 2020 paper
already in the bibliography.

---

### Tier 4: Comparison vs. Established Tools (~4–8 weeks)

#### 4.1 Benchmark vs. CNS / XPLOR-NIH Refinement

**What to do:**  
On the same 2–3 targets, run a standard CS-Rosetta or CNS chemical
shift refinement protocol (or use the Montelione group's published
protocol from Li et al. 2023). Compare final CSRMSD.

**Why it matters:**  
Without a comparison baseline, it is impossible to know whether
TorsionTuner's improvements are competitive with existing methods.
A head-to-head comparison is required for any publication.

---

#### 4.2 Round-Trip Recovery Benchmark

**What to do:**  
Take a high-quality NMR structure (one of the NESG targets).
Perturb it with Gaussian noise on all ψ/φ angles (σ = 5°, 10°, 20°).
Refine with TorsionTuner using the experimental shifts.
Measure backbone RMSD to the true NMR structure after refinement.

**Why it matters:**  
Unlike the current test (which only measures CSRMSD improvement),
this directly tests whether the model recovers physically correct
3D geometry — the ultimate purpose of the tool.

---

## Part II — Documentation Upgrades

### D1. Known Limitations Section (High Priority)

Add a dedicated `docs/limitations.md` (and summary in README) covering:

| Limitation | Impact | Mitigation |
|---|---|---|
| Chemical shift predictor uses residue-agnostic offsets | CSRMSD numbers not directly comparable to SPARTA+ | Document expected systematic bias |
| SAXS form factors are q-independent (Z only) | Inaccurate above q ~ 0.15 Å⁻¹ | Document safe q-range |
| Backbone-only (N, CA, C) — no side chains | Cannot capture sidechain contacts or rotamer quality | State scope explicitly |
| No solvent/ensemble averaging | Single-structure refinement; may not represent NMR ensemble | Suggest multi-conformer extension |
| ANSURR is post-hoc only, not a training signal | Structure may improve CSRMSD but not ANSURR score | Direct users to ANSURR web server |

---

### D2. Scientific Methodology Section

Add `docs/scientific_methodology.md` explaining the physics behind each
component with explicit references:

- **Random coil shifts:** Wishart et al. 1995 — what values are used, where they come from
- **Secondary structure offsets:** Wishart & Sykes 1994 (CSI) — explain the ±3.1/−1.5 ppm as a one-coefficient approximation of SPARTA+; show the systematic error vs. per-residue coefficients
- **Debye formula:** Debye 1915 — the sinc approximation, why Z is used, the q-range where it is valid
- **NeRF kinematics:** Parsons et al. 2005 (*J. Comput. Chem.*) — the reference algorithm
- **Ramachandran penalty:** Lovell et al. 2003 — the source of the canonical region centers

---

### D3. Benchmark Results Dashboard

Add `docs/benchmarks.md` as a living table of results:

```markdown
| Target | Fold | Residues | BMRB | Initial CSRMSD | Final CSRMSD | Δ | Seeds |
|--------|------|----------|------|----------------|--------------|---|-------|
| 2KHD   | α/β  | 108      | 16238| 0.43 ppm       | 0.29 ppm     | −33% | n=1 |
| 2RN7   | TBD  | 108      | 11017| TBD            | TBD          | TBD | n=1 |
```

This forces honest tracking of which results are reproducible and
which remain preliminary (n=1).

---

### D4. Complete API Reference

Currently the MkDocs `docs/api/` directory exists but may be sparse.
Ensure every public function has:
- A one-line summary
- Full Args / Returns / Raises docstring
- A "Notes" section citing the scientific source for any formula
- A minimal `Example:` block

---

### D5. Tutorial Notebook

A `notebooks/quickstart.ipynb` walking through:
1. Generating a test PDB with `synth-pdb`
2. Running refinement
3. Visualizing the CSRMSD improvement
4. Exporting and submitting to PSVS

This is the most effective way to let new users understand what the
tool actually does.

---

### D6. Explicit PSVS Workflow Documentation

The `export.py` / `psvs_parser.py` infrastructure exists but is only
used in a test with mock data. Add `docs/psvs_workflow.md`:

1. How to export a refined structure with `save_for_psvs()`
2. How to submit to the PSVS server
3. How to parse results with `parse_psvs_summary()`
4. What scores to expect for a "good" AlphaFold refinement

---

### D7. Upgrade the SCIENTIFIC_VALIDATION.md Header

The document currently starts mid-task. Add a front-matter section:

```markdown
## Validation Philosophy

TorsionTuner is validated at three levels:
1. **Internal parity** — physics engines agree with established software
2. **NESG benchmark** — CSRMSD reduction on blind targets vs. BMRB data
3. **External quality** — independent validation via MolProbity, PSVS, ANSURR
```

This sets the scientific standard explicitly and makes the roadmap
easier to follow.

---

## Summary: Recommended Execution Order

| # | Task | Effort | Scientific Impact |
|---|------|--------|------------------|
| 1 | Omega angle integrity check | 1 day | Closes a known gap in the kinematics |
| 2 | SPARTA+ parity test | 2–3 days | Quantifies the #1 approximation |
| 3 | CRYSOL/FoXS SAXS parity | 2–3 days | Quantifies the #2 approximation |
| 4 | Known Limitations doc | 1 day | Essential for honest use |
| 5 | Expand NESG to all 9 targets | 1–2 weeks | Core credibility claim |
| 6 | Multi-seed statistical tests | 3 days | Required for any publication |
| 7 | Real MolProbity before/after | 3 days | Strongest external proof |
| 8 | Real PSVS submission | 1 day | Validates the entire PSVS pipeline |
| 9 | ANSURR web server post-hoc | 1 day | Closes the loop on the bibliography |
| 10 | Round-trip recovery benchmark | 1 week | Publication-grade proof of concept |
| 11 | Comparison vs. CNS/Rosetta | 2–4 weeks | Required for competitive claim |
| 12 | Tutorial notebook | 2 days | User adoption |
| 13 | API reference + methodology doc | 3 days | Long-term maintainability |
