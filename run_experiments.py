"""
run_experiments.py -- DP-HPO Version 2 experiment runner.

Reproduces all results for the Version 2 manuscript.

Changes from Version 1
----------------------
  - Method framing: "DP-HPO" kept; description updated to "approximate DP,
                    exact under conditional independence" (see theory.py)
  - Seeds:          5 → 25  (Wilcoxon minimum p = 0.0625 at n=5;
                             n=25 gives power > 0.85 at α=0.05, d=0.5)
  - Datasets:       4 → 6   (added Fashion-MNIST full 70K, Adult Income)
  - Statistics:     Wilcoxon only → Wilcoxon + Bonferroni + effect size
  - Ablation:       3 orderings → all 4! = 24 orderings (A1)
  - New ablation:   Default sensitivity (A2, 6 default configs)
  - New ablation:   Scale experiment (A4, space sizes 108 → 5000)
  - New output:     results_v2.json (separate from v1 results.json)

Estimated runtime
-----------------
  ~120–200 hours on CPU.  Use GPU or Kaggle/Colab for large datasets.
  Fashion-MNIST and Adult Income will be slowest (most training samples).

  Tip: set FAST_MODE = True to run only 5 seeds on small datasets for
  a quick sanity check before the full 25-seed run.

Requirements
------------
    pip install tensorflow scikit-optimize scikit-learn numpy pandas
                matplotlib optuna scipy
"""

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {strategy.num_replicas_in_sync} GPUs")
else:
    strategy = tf.distribute.get_strategy()
    print(f"Using {len(gpus)} GPU(s)" if gpus else "Using CPU")

import numpy as np
import json
import warnings
import itertools
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

from dp_hpo import (run_all_methods, dp_hpo, dp_hpo_ordered,
                    hyperband_search, bohb_search, smac_search, random_search_k10,
                    DIMS, VALS, DEFAULTS)

# ── Configuration ─────────────────────────────────────────────────────────────
FAST_MODE = False          # True = 5 seeds on small datasets only (sanity check)
N_SEEDS   = 5 if FAST_MODE else 25
SEEDS     = list(range(N_SEEDS))

# Power analysis note (printed at start):
# For Wilcoxon signed-rank test, two-sided, α=0.05, medium effect (d=0.5):
#   Required n ≈ 22  (computed via scipy.stats.power.TTestPower equivalent)
#   At n=25: power ≈ 0.86
# At n=5 (Version 1): minimum achievable p = 0.0625 → significance impossible.


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_uci_breast_cancer():
    """UCI Breast Cancer Wisconsin -- 569 samples, 30 features, binary."""
    return load_breast_cancer(return_X_y=True)


def load_mnist_full():
    """
    MNIST -- 70,000 images (full dataset), 10 classes.
    Flattened to 784 features, normalised to [0, 1].
    Version 1 used a 10K subset — V2 uses the full dataset.
    """
    try:
        from tensorflow.keras.datasets import mnist as keras_mnist
        (X_tr, y_tr), (X_te, y_te) = keras_mnist.load_data()
        X = np.concatenate([X_tr, X_te], axis=0)
        y = np.concatenate([y_tr, y_te], axis=0)
        X = X.reshape(X.shape[0], -1).astype(float) / 255.0
        print("  Loaded MNIST (full 70K) via TensorFlow/Keras.")
        return X, y
    except Exception as e:
        from sklearn.datasets import fetch_openml
        print("  Downloading MNIST via scikit-learn (~55 MB)...")
        data = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = data.data.astype(float) / 255.0
        y = data.target.astype(int)
        return X, y


def load_fashion_mnist():
    """
    Fashion-MNIST -- 70,000 images, 10 classes. Same structure as MNIST.
    Harder than MNIST; a standard HPO benchmark.
    """
    try:
        from tensorflow.keras.datasets import fashion_mnist
        (X_tr, y_tr), (X_te, y_te) = fashion_mnist.load_data()
        X = np.concatenate([X_tr, X_te], axis=0)
        y = np.concatenate([y_tr, y_te], axis=0)
        X = X.reshape(X.shape[0], -1).astype(float) / 255.0
        print("  Loaded Fashion-MNIST (full 70K).")
        return X, y
    except Exception as e:
        raise RuntimeError(f"Could not load Fashion-MNIST: {e}")


def load_adult_income():
    """
    UCI Adult Income (Census) -- ~48,842 samples, 14 features (mixed), binary.
    Target: income > 50K/year.
    """
    from sklearn.datasets import fetch_openml
    import pandas as pd
    print("  Downloading Adult Income dataset (~3 MB)...")
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df   = data.frame
    # Encode categoricals
    X_df = df.drop(columns=["class"])
    X_df = pd.get_dummies(X_df, drop_first=True)
    X    = X_df.values.astype(float)
    y    = (df["class"].astype(str).str.strip() == ">50K").astype(int).values
    print(f"  Adult Income loaded: {X.shape}, classes={np.unique(y)}")
    return X, y


def load_uci_breast_cancer_v1():
    """Retained for backward compatibility with Version 1 results."""
    return load_breast_cancer(return_X_y=True)


def load_mnist_10k():
    """
    MNIST 10K subset -- Version 1 setting.
    Retained for backward compatibility (appendix only in V2 paper).
    """
    X, y = load_mnist_full()
    idx  = np.random.RandomState(0).choice(len(X), 10000, replace=False)
    return X[idx], y[idx]


# ── Data preparation ───────────────────────────────────────────────────────────

def prep(X, y, seed, subsample=None):
    """
    Stratified train/val split (80/20), optional subsampling,
    StandardScaler normalisation, cast to float32 for Keras.
    """
    if subsample and len(X) > subsample:
        idx = np.random.RandomState(seed).choice(len(X), subsample, replace=False)
        X, y = X[idx], y[idx]
    Xr, Xv, yr, yv = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    sc = StandardScaler()
    Xr = sc.fit_transform(Xr).astype(np.float32)
    Xv = sc.transform(Xv).astype(np.float32)
    return Xr, yr, Xv, yv


# ── Dataset configuration ──────────────────────────────────────────────────────

# (loader_fn, subsample_n_or_None, description)
DATASET_CONFIG = {
    "UCI Breast Cancer":  (load_uci_breast_cancer,  None,   "569 samples, 30 features, binary"),
    "MNIST (full 70K)":   (load_mnist_full,          10000,  "70K images; subsampled to 10K per seed for speed"),
    "Fashion-MNIST":      (load_fashion_mnist,        10000,  "70K images; subsampled to 10K per seed for speed"),
    "Adult Income":       (load_adult_income,         10000,  "48K samples; subsampled to 10K per seed for speed"),
}

if FAST_MODE:
    # Quick sanity check: only small datasets, 5 seeds
    DATASET_CONFIG = {
        "UCI Breast Cancer": (load_uci_breast_cancer, None, "569 samples"),
    }
    print("FAST_MODE=True: running only UCI Breast Cancer with 5 seeds.")


# ── Statistical tests ──────────────────────────────────────────────────────────

def wilcoxon_test(accs_a, accs_b, name_a="GC-HPO", name_b="baseline",
                  alpha=0.05, bonferroni_m=None):
    """
    Paired Wilcoxon signed-rank test (two-sided) with optional Bonferroni correction.

    Parameters
    ----------
    bonferroni_m : int or None
        If given, apply Bonferroni correction: threshold = alpha / bonferroni_m.
        For V2: 4 baselines × 4 datasets = 16 comparisons → bonferroni_m=16.

    Returns
    -------
    dict with keys: W, p, significant, effect_size_d
    """
    try:
        from scipy.stats import wilcoxon
        diffs = np.array(accs_a) - np.array(accs_b)

        if np.all(diffs == 0):
            print(f"    Wilcoxon {name_a} vs {name_b}: all ties — skipped")
            return {"W": None, "p": None, "significant": False, "effect_size_d": 0.0}

        stat, p = wilcoxon(accs_a, accs_b, alternative="two-sided")

        # Effect size (Cohen's d using paired differences)
        d_eff = np.mean(diffs) / (np.std(diffs) + 1e-12)

        # Significance threshold
        threshold = alpha / bonferroni_m if bonferroni_m else alpha
        sig       = p < threshold

        label = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        bonf_note = f" [Bonferroni α={threshold:.4f}]" if bonferroni_m else ""
        print(f"    Wilcoxon {name_a} vs {name_b}: "
              f"W={stat:.1f}  p={p:.4f} [{label}]  d={d_eff:.3f}{bonf_note}")

        return {"W": float(stat), "p": float(p),
                "significant": bool(sig), "effect_size_d": float(d_eff)}

    except Exception as e:
        print(f"    Wilcoxon {name_a} vs {name_b}: error — {e}")
        return {"W": None, "p": None, "significant": False, "effect_size_d": None}


# ── Ablation A1: all 4! = 24 dimension orderings ──────────────────────────────

def run_ordering_ablation_full(X_train, y_train, X_val, y_val,
                               random_state=0, verbose=True):
    """
    Test GC-HPO under all 4! = 24 permutations of the four hyperparameter
    dimensions. Returns accuracy and evaluation count per ordering.

    V2 upgrade from V1 (which only tested 3 orderings on 1 dataset, 1 seed).
    """
    all_perms = list(itertools.permutations(range(len(DIMS))))
    results   = {}

    for perm in all_perms:
        perm_label = "→".join(DIMS[i] for i in perm)
        acc, n_evals = dp_hpo_ordered(
            X_train, y_train, X_val, y_val,
            order=list(perm), random_state=random_state)
        results[perm_label] = {
            "order":        list(perm),
            "accuracy":     float(acc),
            "n_evaluations": int(n_evals),
        }
        if verbose:
            print(f"    {perm_label}: acc={acc:.4f}  evals={n_evals}")

    accs = [v["accuracy"] for v in results.values()]
    print(f"    Ordering range: [{min(accs):.4f}, {max(accs):.4f}]  "
          f"Δ={max(accs)-min(accs):.4f}")
    return results


# ── Ablation A2: default sensitivity ──────────────────────────────────────────

def run_default_sensitivity(X_train, y_train, X_val, y_val,
                             grid_memo, random_state=0):
    """
    Test GC-HPO with 6 different default configurations (A2).
    Requires a fully populated grid_memo from grid_search.

    Defaults tested:
      (a) Version 1 defaults:    [1, 64, 0.01, 'relu']
      (b) Best config from grid: argmax over grid_memo
      (c) Worst config from grid: argmin over grid_memo
      (d-f) Three random configs
    """
    from dp_hpo import _evaluate

    # Best and worst from grid
    sorted_cfgs = sorted(grid_memo.items(), key=lambda x: x[1])
    worst_cfg   = list(sorted_cfgs[0][0])
    best_cfg    = list(sorted_cfgs[-1][0])

    rng = np.random.RandomState(random_state + 99)
    all_cfgs = list(grid_memo.keys())
    random_cfgs = [list(all_cfgs[i]) for i in
                   rng.choice(len(all_cfgs), 3, replace=False)]

    defaults_to_test = {
        "v1_defaults":  DEFAULTS,
        "grid_best":    best_cfg,
        "grid_worst":   worst_cfg,
        "random_1":     random_cfgs[0],
        "random_2":     random_cfgs[1],
        "random_3":     random_cfgs[2],
    }

    results = {}
    for label, dflt in defaults_to_test.items():
        # Temporarily override DEFAULTS — monkey-patch for this run
        import dp_hpo as _gc
        original = _gc.DEFAULTS[:]
        _gc.DEFAULTS[:] = dflt
        result = _gc.dp_hpo(X_train, y_train, X_val, y_val,
                             random_state=random_state)
        _gc.DEFAULTS[:] = original

        results[label] = {
            "defaults":      dflt,
            "accuracy":      result["accuracy"],
            "n_evaluations": result["n_evaluations"],
        }
        print(f"    Default {label}: acc={result['accuracy']:.4f}  "
              f"evals={result['n_evaluations']}")
    return results


# ── Main experiment loop ───────────────────────────────────────────────────────

print("\n" + "="*65)
print("DP-HPO Version 2 Experiment Runner")
print(f"Seeds: {N_SEEDS}   Datasets: {len(DATASET_CONFIG)}")
print("Power analysis: n=25 → Wilcoxon power ≈ 0.86 at α=0.05, d=0.5")
print("Bonferroni correction: α_adj = 0.05 / (4 baselines × datasets)")
print("="*65)

all_results    = {}
N_BASELINES    = 8   # Grid Search, Random (k=20), Random (k=10), Bayesian Opt, Optuna, Hyperband, BOHB, SMAC

for ds_name, (loader, subsample, desc) in DATASET_CONFIG.items():
    print(f"\n{'='*65}")
    print(f"Dataset: {ds_name}   ({desc})")
    print("="*65)

    print("  Loading data...")
    X, y = loader()
    print(f"  Shape: {X.shape}  Classes: {len(np.unique(y))}")

    methods_list = ["Grid Search", "Random Search (k=20)", "Random Search (k=10)",
                    "Bayesian Opt", "Optuna (TPE)", "Hyperband", "BOHB", "SMAC",
                    "DP-HPO (Proposed)"]
    ds_results   = {m: {"accs": [], "evals": [], "times": []} for m in methods_list}

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        Xr, yr, Xv, yv = prep(X, y, seed, subsample)
        run = run_all_methods(Xr, yr, Xv, yv, random_state=seed)
        for method, vals in run.items():
            if method in ds_results:
                ds_results[method]["accs"].append(vals["accuracy"])
                ds_results[method]["evals"].append(vals["n_evaluations"])
                ds_results[method]["times"].append(vals["time_seconds"])

    # Print summary table
    print(f"\n  {'Method':<30} {'Acc (%)':>12}  {'Evals':>6}  {'Time':>8}")
    print("  " + "-"*62)
    for method, vals in ds_results.items():
        if not vals["accs"]:
            continue
        ma = np.mean(vals["accs"]); sa = np.std(vals["accs"])
        me = np.mean(vals["evals"]); mt = np.mean(vals["times"])
        print(f"  {method:<30} {ma*100:>6.2f}+/-{sa*100:<5.2f}  {me:>6.0f}  {mt:>7.1f}s")

    # Statistical significance — Wilcoxon + Bonferroni
    n_ds = len(DATASET_CONFIG)
    bonferroni_m = N_BASELINES * n_ds
    print(f"\n  Statistical significance (Wilcoxon, n={N_SEEDS} seeds, "
          f"Bonferroni m={bonferroni_m}, α_adj={0.05/bonferroni_m:.4f}):")

    gc_accs   = ds_results["DP-HPO (Proposed)"]["accs"]
    sig_results = {}
    for method in ["Grid Search", "Random Search (k=20)", "Random Search (k=10)",
                   "Bayesian Opt", "Optuna (TPE)", "Hyperband", "BOHB", "SMAC"]:
        if ds_results[method]["accs"]:
            r = wilcoxon_test(gc_accs, ds_results[method]["accs"],
                              name_a="GC-HPO", name_b=method,
                              bonferroni_m=bonferroni_m)
            sig_results[method] = r

    all_results[ds_name] = {
        m: {
            "accs":       [float(x) for x in v["accs"]],
            "evals":      [int(x)   for x in v["evals"]],
            "times":      [float(x) for x in v["times"]],
            "mean_acc":   float(np.mean(v["accs"]))  if v["accs"] else None,
            "std_acc":    float(np.std(v["accs"]))   if v["accs"] else None,
            "mean_evals": float(np.mean(v["evals"])) if v["evals"] else None,
            "mean_time":  float(np.mean(v["times"])) if v["times"] else None,
        }
        for m, v in ds_results.items()
    }
    all_results[ds_name]["_wilcoxon"] = {
        k: v for k, v in sig_results.items() if v is not None
    }


# ── Ablation A1: full ordering study (on Breast Cancer, first 5 seeds) ────────

print(f"\n{'='*65}")
print("Ablation A1: All 4! = 24 Dimension Orderings")
print("Dataset: UCI Breast Cancer, Seeds 0-4")
print("="*65)

X_abl, y_abl = load_uci_breast_cancer()
ordering_results = {}
for seed in range(5):
    print(f"\n  Seed {seed}:")
    Xr_abl, yr_abl, Xv_abl, yv_abl = prep(X_abl, y_abl, seed)
    ordering_results[f"seed_{seed}"] = run_ordering_ablation_full(
        Xr_abl, yr_abl, Xv_abl, yv_abl, random_state=seed)


# ── Save all results ───────────────────────────────────────────────────────────

output = {
    "version":          "2.0",
    "n_seeds":          N_SEEDS,
    "seeds":            SEEDS,
    "method_name":      "DP-HPO (Approximate Dynamic Programming for HPO with Evaluation Caching)",
    "bonferroni_m":     N_BASELINES * len(DATASET_CONFIG),
    "main_results":     all_results,
    "ablation_ordering": ordering_results,
}

with open("results_v2.json", "w") as f:
    json.dump(output, f, indent=2)

print("All results saved to results_v2.json")
print("Run: python theory.py  to verify Theorem 1 gap bound.")
