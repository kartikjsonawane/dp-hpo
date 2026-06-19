# DP-HPO: Approximate Dynamic Programming for Neural Network HPO

> **10 evaluations. Within 0.5% of exhaustive grid search accuracy. 90.7% fewer model evaluations.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-CC--BY--4.0-green)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20760182-blue)](https://doi.org/10.5281/zenodo.20760182)
[![GitHub Pages](https://img.shields.io/badge/demo-live-brightgreen)](https://kartikjsonawane.github.io/dp-hpo/)
[![Seeds](https://img.shields.io/badge/seeds-25-orange)](results_v2.json)

**[Live Demo & Results Dashboard →](https://kartikjsonawane.github.io/dp-hpo/)**

---

## What is DP-HPO?

DP-HPO formulates hyperparameter optimisation as a **finite-horizon Markov Decision Process** and solves it via Bellman's backward induction — the same principle as dynamic programming in operations research.

Instead of evaluating all combinations exhaustively (Grid Search: 108 configs) or sampling blindly (Random Search), DP-HPO commits each hyperparameter dimension **one at a time**, fixing the best value before moving to the next.

```
Stage 1: sweep hidden_layers ∈ {1, 2, 3}          → commit best (3 evals)
Stage 2: sweep neurons       ∈ {32, 64, 128, 256}  → commit best (4 evals)
Stage 3: sweep learning_rate ∈ {0.001, 0.01, 0.1} → commit best (3 evals)
Stage 4: sweep activation    ∈ {relu, tanh, logit} → commit best (3 evals)
                                                      TOTAL: 10 evals (+ 1 cached)
```

**Result:** 90.7% fewer evaluations. DP-HPO stays within 0.5% of grid search accuracy across all four benchmark datasets.

---

## Core Algorithm

```python
def dp_hpo(X_train, y_train, X_val, y_val, order=None):
    """
    Commits each hyperparameter dimension via Bellman backward induction.
    Exact under conditional independence.
    Gap bound: f* - f_DP-HPO <= (d-1) * epsilon  [Theorem 1]
    Evaluations: 1 + sum(|H_i| - 1) = 10         [Proposition 1]
    """
    committed = list(DEFAULTS)

    for i in (order or range(len(DIMS))):
        best_val, best_acc = None, -1
        for v in VALS[i]:
            config    = list(committed)
            config[i] = v
            acc       = _evaluate(config, X_train, y_train, X_val, y_val, memo)
            if acc > best_acc:
                best_val, best_acc = v, acc
        committed[i] = best_val              # Bellman argmax

    return best_acc, len(memo)
```

---

## Results — UCI Breast Cancer (25 Seeds, Sklearn MLP)

| Method | Mean Acc% | Std% | Evals | p-value |
|---|---|---|---|---|
| Grid Search | 98.70 | ±0.96 | 108 | ref |
| Hyperband | 98.53 | ±1.01 | ~30 | 0.0702 |
| Random (k=20) | 98.46 | ±1.20 | 20 | 0.1875 |
| Bayesian Opt | 98.42 | ±1.08 | 20 | 0.2278 |
| BOHB | 98.39 | ±1.04 | ~30 | 0.3008 |
| Optuna (TPE) | 98.35 | ±1.06 | 20 | 0.3805 |
| SMAC | 98.32 | ±1.02 | 20 | 0.6756 |
| **DP-HPO** | **98.28** | **±1.15** | **10** | proposed |
| Random (k=10) | 98.14 | ±1.32 | 10 | 0.6472 |

Wilcoxon signed-rank, two-sided, Bonferroni m=32, α_adj=0.00156.  
sklearn MLPClassifier, solver=adam, early_stopping=True, max_iter=150.

---

## Theory

### Bellman Equation

```
V*(sᵢ) = max_{a ∈ Hᵢ₊₁} V*(T(sᵢ, a))
```

### ADP Approximation (Completion Vector)

```
V̂(sᵢ, a) = f(h₁*, …, hᵢ*, a, DEFAULT_{i+2}, …, DEFAULT_d)
```

Reduces evaluations from O(Π|Hᵢ|) → O(Σ|Hᵢ|).

### Theorem 1 — Optimality Gap

```
f* − f_DP-HPO ≤ (d − 1) · ε
```

Exact (ε=0) when dimensions are conditionally independent.

### Proposition 1 — Evaluation Count

```
1 + (3−1) + (4−1) + (3−1) + (3−1) = 10
```

---

## Quick Start

```bash
git clone https://github.com/kartikjsonawane/dp-hpo.git
cd dp-hpo
pip install -r requirements.txt

# Run all 9 HPO methods, 25 seeds, 4 datasets
python run_experiments.py

# Auto-fill manuscript tables
python fill_results.py
```

For GPU runs, open a split Colab notebook from `notebooks/` and download the partial JSON. Then:

```bash
python merge_results.py   # merge 4 partial JSONs → results_v2.json
python fill_results.py    # fill manuscript tables
```

---

## Repository Structure

```
dp-hpo/
├── dp_hpo.py               # core: 9 HPO methods, MDP + ADP solver
├── run_experiments.py      # 25-seed runner, 4 datasets, stats
├── theory.py               # Theorem 1 verifier
├── fill_results.py         # auto-fill manuscript tables
├── merge_results.py        # merge split Colab results
├── results_v2.json         # 25-seed results (all 4 datasets)
├── dp-hpo.pdf              # published paper (DOI: 10.5281/zenodo.20760182)
├── requirements.txt
├── notebooks/              # Colab/Kaggle notebooks (one per dataset)
└── index.html              # GitHub Pages — live demo + dashboard
```

---

## 9 Methods Compared

| # | Method | Evals | Backend |
|---|---|---|---|
| 1 | Grid Search | 108 | exhaustive |
| 2 | Random (k=20) | 20 | numpy |
| 3 | Random (k=10) | 10 | numpy |
| 4 | Bayesian Opt | 20 | scikit-optimize (GP+EI) |
| 5 | Optuna (TPE) | 20 | optuna |
| 6 | Hyperband | ~30 | optuna (RandomSampler+HyperbandPruner) |
| 7 | BOHB | ~30 | optuna (TPESampler+HyperbandPruner) |
| 8 | SMAC | 20 | scikit-optimize (RF+EI) |
| 9 | **DP-HPO** | **10** | **custom MDP solver** |

---

## Citation

```bibtex
@misc{sonawane2026dphpo,
  title     = {DP-HPO: Approximate Dynamic Programming for Neural Network Hyperparameter Optimisation with Evaluation Caching},
  author    = {Sonawane, Kartik},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20760182},
  url       = {https://doi.org/10.5281/zenodo.20760182}
}
```

## License

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This work is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

© 2026 Kartik Sonawane · Published via [Zenodo](https://doi.org/10.5281/zenodo.20760182)
