# Changelog

## Version 2.0 — June 2026

### Method framing (key V2 fix)

- **Name kept:** DP-HPO — the method is legitimately grounded in dynamic programming
- **Framing corrected:** V1 claimed "this IS DP" without an MDP formulation. V2 correctly frames it as **approximate dynamic programming (ADP)** that is **exact under conditional independence** (Theorem 1)
- The one-sentence framing change: *"DP-HPO implements approximate DP for HPO: it is exact DP under conditional independence of hyperparameter dimensions (Theorem 1), and deviates from optimal by at most (d−1)·ε when independence is violated."*

### New files

- `theory.py` — formal MDP formulation, Theorem 1 (optimality gap bound), Lemma 2 (importance-first ordering), and Python utilities for empirical verification

### Theory (Contributions 1–3)

- **Contribution 1:** Complete MDP formulation — state, action, transition, reward, Bellman equation, and ADP approximation step
- **Theorem 1:** Optimality gap bound: `f* − f_DP-HPO ≤ (d−1)·ε` where ε = max pairwise interaction strength. Proof by induction on d.
- **Lemma 2:** Importance-first ordering (by decreasing marginal variance) minimises expected gap. Marginal variance is computed for free from the first-pass sweeps.
- **Proposition 1:** Corrected evaluation count: `1 + Σ(|Hᵢ|−1) = 10` (not Σ|Hᵢ| = 13). Proof accounts for the shared DEFAULTS config.

### Experiments

- Seeds: 5 → **25** (fixes fatal Wilcoxon flaw: at n=5, min achievable p = 0.0625; at n=25, power ≈ 0.86)
- Datasets: 4 → **6** (added Fashion-MNIST full 70K, Adult Income 48K; MNIST upgraded from 10K subset to full 70K)
- Statistics: Wilcoxon only → Wilcoxon + **Bonferroni correction** (m = 4 baselines × datasets) + Cohen's d effect size
- Ablation A1: 3 orderings → **all 4! = 24 orderings** across 5 seeds
- Ablation A2: Default sensitivity — 6 default configurations tested
- New output: `results_v2.json`

### Code

- `dp_hpo.py`: Updated docstrings to "approximate DP" framing; isolated memo caches per method (V1 shared cache confounded wall-clock comparisons); `dp_hpo()` now accepts `order` parameter for Lemma 2 validation
- `run_experiments.py`: 25 seeds, 6 datasets, Bonferroni, full ordering ablation, `FAST_MODE` flag

---

## Version 1.0 — April 2026

- Initial release as "DP-HPO"
- 5 HPO methods: Grid Search, Random Search, Bayesian Opt, Optuna, DP-HPO
- 4 datasets: UCI Iris, UCI Heart Disease, MNIST (10K subset), UCI Breast Cancer
- 5 seeds, Wilcoxon signed-rank tests
- Live terminal demo on GitHub Pages
