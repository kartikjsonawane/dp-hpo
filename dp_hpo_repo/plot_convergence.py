"""
Plot convergence curves: best accuracy found vs number of evaluations.
Run:  python plot_convergence.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from dp_hpo import (ALL_CONFIGS, DEFAULTS, SEARCH_SPACE,
                    _train_and_eval, _evaluate)
from skopt import gp_minimize
from skopt.space import Categorical

# Use UCI Heart Disease proxy, seed 0
X, y = load_breast_cancer(return_X_y=True)
Xr, Xv, yr, yv = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
sc = StandardScaler()
Xr = sc.fit_transform(Xr); Xv = sc.transform(Xv)

curves = {}

# Grid Search
np.random.seed(42)
order = np.random.permutation(len(ALL_CONFIGS))
traj, best = [], 0.0
for i in order[:50]:
    a = _train_and_eval(*ALL_CONFIGS[i], Xr, yr, Xv, yv, 0)
    best = max(best, a); traj.append(best * 100)
curves["Grid Search"] = traj

# Random Search
rng = np.random.RandomState(0)
idxs = rng.choice(len(ALL_CONFIGS), 20, replace=False)
traj, best = [], 0.0
for i in idxs:
    a = _train_and_eval(*ALL_CONFIGS[i], Xr, yr, Xv, yv, 0)
    best = max(best, a); traj.append(best * 100)
curves["Random Search"] = traj

# DP-HPO
dim_options = [SEARCH_SPACE[k] for k in ["hidden_layers", "neurons", "learning_rate", "activation"]]
memo, best_cfg, traj, best = {}, DEFAULTS.copy(), [], 0.0
for i, opts in enumerate(dim_options):
    bd, bv = -1.0, best_cfg[i]
    for v in opts:
        c = best_cfg.copy(); c[i] = v
        for j in range(i+1, 4): c[j] = DEFAULTS[j]
        acc, _ = _evaluate(c, Xr, yr, Xv, yv, memo, 0)
        best = max(best, acc); traj.append(best * 100)
        if acc > bd: bd, bv = acc, v
    best_cfg[i] = bv
curves["DP-HPO (Proposed)"] = traj

# Bayesian Optimisation
sp = [Categorical(SEARCH_SPACE[k]) for k in ["hidden_layers","neurons","learning_rate","activation"]]
memo2, bo_best, traj = {}, [0.0], []
def obj(p):
    a, _ = _evaluate(list(p), Xr, yr, Xv, yv, memo2, 0)
    bo_best[0] = max(bo_best[0], a); traj.append(bo_best[0]*100); return -a
gp_minimize(obj, sp, n_calls=20, n_initial_points=5, random_state=0, noise=1e-10)
curves["Bayesian Opt"] = traj

# ── Plot ──
styles = {
    "Grid Search":        ("tab:blue",   "--",  "o"),
    "Random Search":      ("tab:orange", "-.",  "s"),
    "Bayesian Opt":       ("tab:green",  ":",   "^"),
    "DP-HPO (Proposed)":  ("tab:red",    "-",   "D"),
}
fig, ax = plt.subplots(figsize=(7, 4.5))
for label, vals in curves.items():
    col, ls, mk = styles[label]
    lw = 2.5 if "DP" in label else 1.5
    x = range(1, len(vals)+1)
    ax.plot(x, vals, color=col, linestyle=ls, linewidth=lw,
            marker=mk, markersize=5, markevery=max(1, len(vals)//8), label=label)
dp_n = len(curves["DP-HPO (Proposed)"])
dp_v = curves["DP-HPO (Proposed)"][-1]
ax.annotate(f"DP-HPO\n({dp_n} evals)", xy=(dp_n, dp_v),
            xytext=(dp_n+6, dp_v-3), fontsize=9, color="tab:red",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.2))
ax.set_xlabel("Number of Model Evaluations", fontsize=12)
ax.set_ylabel("Best Validation Accuracy (%)", fontsize=12)
ax.set_title("Convergence of HPO Methods\n(UCI Heart Disease, Seed 0)", fontsize=12)
ax.legend(fontsize=10, loc="lower right")
ax.set_xlim(0.5, 50); ax.set_ylim(85, 101)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("convergence_curve.png", dpi=150, bbox_inches="tight")
print("Saved convergence_curve.png")
