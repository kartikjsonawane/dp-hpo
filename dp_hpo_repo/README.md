# DP-HPO: A Dynamic Programming Framework for Neural Network Hyperparameter Optimisation

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://kartikjsonawane.github.io/dp-hpo/dp_hpo_terminal.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A Dynamic Programming Framework for Neural Network Hyperparameter Optimisation**  
> *Kartik Sonawane, Independent Researcher, 2026*

**[▶ Try the live interactive demo →](https://kartikjsonawane.github.io/dp-hpo/dp_hpo_terminal.html)**

---

## Overview

DP-HPO recasts neural network hyperparameter optimisation as a dynamic programming problem, exploiting the **optimal substructure** and **overlapping subproblems** properties of layered configuration spaces. Compared to grid search (108 evaluations), DP-HPO finds a competitive configuration using only **10 evaluations** — a **90.7% reduction** — while achieving comparable accuracy on all three benchmark datasets.

---

## Results

All experiments use TensorFlow/Keras MLPs (Adam optimiser, early stopping patience=10, max 100 epochs). Results averaged over 3 random seeds.

| Dataset           | Method          | Accuracy (%)      | Evaluations |
|-------------------|-----------------|-------------------|-------------|
| UCI Iris          | Grid Search     | 100.00 ± 0.00     | 108         |
| UCI Iris          | Random Search   | 100.00 ± 0.00     | 20          |
| UCI Iris          | Bayesian Opt    | 100.00 ± 0.00     | 20          |
| UCI Iris          | **DP-HPO**      | **100.00 ± 0.00** | **10**      |
| UCI Heart Disease | Grid Search     | 91.11 ± 2.08      | 108         |
| UCI Heart Disease | Random Search   | 90.00 ± 1.36      | 20          |
| UCI Heart Disease | Bayesian Opt    | 88.89 ± 2.83      | 20          |
| UCI Heart Disease | **DP-HPO**      | **87.78 ± 1.57**  | **10**      |
| MNIST (10K)       | Grid Search     | 94.33 ± 0.20      | 108         |
| MNIST (10K)       | Random Search   | 93.52 ± 0.26      | 20          |
| MNIST (10K)       | Bayesian Opt    | 93.73 ± 0.45      | 20          |
| MNIST (10K)       | **DP-HPO**      | **92.87 ± 0.39**  | **10**      |

DP-HPO consistently reaches within 1–3% of grid search accuracy while evaluating **10.8× fewer configurations**.

---

## Repository Structure

```
dp_hpo_repo/
├── dp_hpo.py              # Core: all four HPO methods (Grid, Random, Bayesian, DP-HPO)
├── run_experiments.py     # Reproduce all paper results → results.json
├── plot_convergence.py    # Generate convergence curve figure
├── dp_hpo_terminal.html   # Interactive live demo (Linux terminal UI)
├── convergence_curve.png  # Figure 1 from the paper
├── DP_HPO_Draft_v4.docx   # Full paper draft
└── README.md
```

---

## Installation

```bash
pip install tensorflow scikit-learn scikit-optimize numpy pandas matplotlib
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

Runtime: ~20–40 minutes on a standard laptop. Results are saved to `results.json`.

**UCI Heart Disease dataset:** download `processed.cleveland.data` from the [UCI archive](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data) and place it in the same directory as `run_experiments.py`. The script checks for this file automatically before attempting a network download.

---

## Search Space

| Hyperparameter | Values                      | Count |
|----------------|-----------------------------|-------|
| Hidden layers  | {1, 2, 3}                   | 3     |
| Neurons/layer  | {32, 64, 128, 256}          | 4     |
| Learning rate  | {0.001, 0.01, 0.1}          | 3     |
| Activation     | {relu, tanh, logistic}      | 3     |

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

An interactive terminal demo is hosted on GitHub Pages. Type commands like `run iris`, `compare`, `algo step`, or `space` to explore the algorithm live.

**[https://kartikjsonawane.github.io/dp-hpo/dp_hpo_terminal.html](https://kartikjsonawane.github.io/dp-hpo/dp_hpo_terminal.html)**

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
