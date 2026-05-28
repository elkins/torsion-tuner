# 🧬 TorsionTuner: Differentiable Structure Refinement

[![PyPI version](https://img.shields.io/pypi/v/torsiontuner.svg)](https://pypi.org/project/torsiontuner/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/torsiontuner.svg)](https://pypi.org/project/torsiontuner/)
[![Tests](https://github.com/elkins/TorsionTuner/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/TorsionTuner/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/Accelerated_by-JAX-blue.svg)](https://github.com/google/jax)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

TorsionTuner is a JAX-powered framework for refining protein structures using Graph Neural Networks (GNNs) and differentiable torsional kinematics.

---

### 🧪 For Structural Biologists
*   **Intelligent Refinement:** Improves the physical quality of protein models (e.g., Ramachandran scores) while staying true to experimental data.
*   **Kinematic Preservation:** Operates directly in torsional space, ensuring bond lengths and angles remain ideal during refinement.

### 🤖 For Machine Learning Geeks
*   **Differentiable NeRF:** Utilizes a JAX-based Natural Extension Reference Frame (NeRF) implementation to map GNN-predicted torsions back to 3D coordinates.
*   **End-to-End Optimization:** Flow gradients from complex biophysical loss functions directly back into the GNN weights.

---

## 🚀 Key Features

*   **Torsional GNN:** Message-passing architecture optimized for molecular dihedral prediction.
*   **Physics-Grounded Loss:** Built-in support for NMR and SAXS constraints from the Differentiable Biophysics suite.
*   **Ramachandran Regularization:** Incorporates statistical knowledge of protein backbone geometry.

## 📦 Installation

```bash
pip install torsiontuner
```

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
