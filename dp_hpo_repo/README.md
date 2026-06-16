# DP-HPO: A Dynamic Programming Framework for Neural Network Hyperparameter Optimisation

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://kartikjsonawane.github.io/dp-hpo/dp_hpo_terminal.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A Dynamic Programming Framework for Neural Network Hyperparameter Optimisation**  
> *Kartik Sonawane, Independent Researcher, 2026*

**[▶ Try the live interactive demo →](https://kartikjsonawane.github.io/dp-hpo/dp_hpo_terminal.html)**

---

## Overview

DP-HPO recasts neural network hyperparameter optimisation as a dynamic programming problem, exploiting the **optimal substructure** and **overlapping subproblems** properties of the configuration space. Compared to grid search (108 evaluations), DP-HPO finds a competitive configuration using only **10 evaluations** — a **90.7% reduction** — while achieving statistically equivalent accuracy across four benchmark datasets (Wilcoxon signed-rank, all p > 0.05).

---

## Results

All experiments use TensorFlow/Keras MLPs (Adam optimiser, early stopping patience=10, max 100 epochs, batch size 32). Results averaged over **5 random seeds (0–4)**. Statistical significance assessed via the Wilcoxon signed-rank test (n=5, two-sided).

| Dataset             | Method              | Accuracy (Mean ± Std %) | Evaluations |
|---------------------|---------------------|-------------------------|-------------|
| UCI Iris            | Grid Search         | 99.33 ± 1.33            | 108         |
| UCI Iris            | Random Search       | 99.33 ± 1.33            | 20          |
| UCI Iris            | Bayesian Opt        | 99.33 ± 1.33            | ~20         |
| UCI Iris            | Optuna (TPE)        | 98.67 ± 2.67            | ~16         |
| UCI Iris            | **DP-HPO (Proposed)** | **98.67 ± 2.67**      | **10**      |
| UCI Heart Disease   | Grid Search         | 89.67 ± 2.67            | 108         |
| UCI Heart Disease   | Random Search       | 87.33 ± 2.26            | 20          |
| UCI Heart Disease   | Bayesian Opt        | 88.33 ± 3.94            | ~20         |
| UCI Heart Disease   | Optuna (TPE)        | 89.00 ± 2.91            | ~16         |
| UCI Heart Disease   | **DP-HPO (Proposed)** | **87.00 ± 2.87**      | **10**      |
| MNIST (10K subset)  | Grid Search         | 94.49 ± 0.44            | 108         |
| MNIST (10K subset)  | Random Search       | 93.58 ± 0.55            | 20          |
| MNIST (10K subset)  | Bayesian Opt        | 94.68 ± 0.52            | ~20         |
| MNIST (10K subset)  | Optuna (TPE)        | 94.17 ± 0.33            | ~14         |
| MNIST (10K subset)  | **DP-HPO (Proposed)** | **93.10 ± 0.90**      | **10**      |
| UCI Breast Cancer   | Grid Search         | 98.95 ± 0.35            | 108         |
| UCI Breast Cancer   | Random Search       | 98.77 ± 0.43            | 20          |
| UCI Breast Cancer   | Bayesian Opt        | 98.77 ± 0.43            | ~20         |
| UCI Breast Cancer   | Optuna (TPE)        | 98.77 ± 0.43            | ~16         |
| UCI Breast Cancer   | **DP-HPO (Proposed)** | **98.60 ± 0.70**      | **10**      |

**All pairwise Wilcoxon tests are non-significant (p > 0.05).** DP-HPO achieves a **90.7% reduction in evaluations** (108 → 10) with no statistically significant loss in accuracy.

---

## Repository Structure

```
dp_hpo_repo/
├── dp_hpo.py              # Core: all 5 HPO methods (Grid, Random, Bayesian, Optuna, DP-HPO)
├── run_experiments.py     # Reproduce all paper results → results.json
├── results.json           # Full experimental results (4 datasets × 5 methods × 5 seeds)
├── plot_convergence.py    # Generate convergence curve figure
├── convergence_curve.png  # Figure 1 from the paper
└── README.md
```

---

## Installation

```bash
pip install tensorflow scikit-learn scikit-optimize optuna numpy pandas matplotlib scipy
```

Python 3.10+ recommended. No GPU required — all experiments run on CPU.

---

## Quick Start

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from dp_hpo import run_all_methods

X, y = load_iris(return_X_y=True)
Xr, Xv, yr, yv = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

sc = StandardScaler()
Xr = sc.fit_transform(Xr).astype(np.float32)
Xv = sc.transform(Xv).astype(np.float32)

results = run_all_methods(Xr, yr, Xv, yv, random_state=0)
for method, r in results.items():
    print(f"{method:25s}  acc={r['accuracy']:.4f}  evals={r['n_evaluations']}")
```

---

## Reproduce Paper Results

```bash
python run_experiments.py
```

Runtime: ~40–80 minutes on a standard laptop (5 seeds × 4 datasets). Results saved to `results.json`.

**UCI Heart Disease dataset:** download `processed.cleveland.data` from the [UCI archive](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data) and place it in the same directory as `run_experiments.py`.

---

## Search Space

| Hyperparameter | Values                 | Count |
|----------------|------------------------|-------|
| Hidden layers  | {1, 2, 3}              | 3     |
| Neurons/layer  | {32, 64, 128, 256}     | 4     |
| Learning rate  | {0.001, 0.01, 0.1}     | 3     |
| Activation     | {relu, tanh, logistic} | 3     |

**Total configurations:** 3 × 4 × 3 × 3 = **108**

---

## Algorithm

```
Initialise: committed ← DEFAULTS  (one default per dimension)
For each dimension i in {hidden_layers, neurons, lr, activation}:
    For each candidate value v in VALS[i]:
        config ← committed[:i] + [v] + DEFAULTS[i+1:]
        acc    ← evaluate(config)           # memoised
    committed[i] ← argmax_v acc
Return committed
```

Maximum evaluations: Σ|H_i| = 3 + 4 + 3 + 3 = **13**; in practice **10** (cache hits on shared defaults).

---

## Live Demo

An interactive terminal demo is hosted on GitHub Pages. Type commands like `run iris`, `results all`, `compare`, `algo step`, or `space` to explore the algorithm live.

**[https://kartikjsonawane.github.io/dp-hpo/dp_hpo_terminal.html](https://kartikjsonawane.github.io/dp-hpo/dp_hpo_terminal.html)**

Available commands: `run [iris|heart|mnist|breast]`, `results [dataset|all]`, `compare`, `algo`, `algo step`, `algo run`, `space`, `about`, `neofetch`, `views`

---

## Generate Figures

```bash
python plot_convergence.py   # outputs convergence_curve.png
```

---

## Citation

```bibtex
@article{sonawane2026dphpo,
  title   = {A Dynamic Programming Framework for Neural Network Hyperparameter Optimisation},
  author  = {Sonawane, Kartik},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
