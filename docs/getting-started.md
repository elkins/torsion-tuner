# Getting Started

## 1. Installation

### From Source
Clone the repository and install in editable mode:

```bash
git clone https://github.com/elkins/TorsionTuner.git
cd TorsionTuner

# Install core and dev dependencies
pip install -e .
```

### With Documentation Tools
If you want to build this documentation site locally:

```bash
pip install -e ".[docs]"
```

## 2. Running a Refinement

To see the model in action, you can run the synthetic helix refinement experiment.

### Step 1: Generate a Target Structure
First, generate a synthetic "target" helix using **synth-pdb**.

```bash
synth-pdb --length 20 --conformation alpha --output test_helix.pdb
```

### Step 2: Run Training
Run the multi-objective fine-tuning optimization. This script will load the helix, simulate SAXS/NMR data, and train the GNN to refine the structure.

```bash
PYTHONPATH=. python src/train.py
```

## 3. Configuration
The training process can be configured via the `Config` dataclass in `src/train.py`. You can adjust weights for different loss terms, learning rates, and more.
