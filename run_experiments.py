"""
Reproduce the DP-HPO paper experiments.

Run:  python run_experiments.py

Datasets:
  - UCI Iris           : built-in to scikit-learn (no download needed)
  - UCI Heart Disease  : local file or auto-download from UCI
  - MNIST              : auto-downloaded via tensorflow.keras (~55 MB, one-time)
  - UCI Breast Cancer  : built-in to scikit-learn (no download needed)

No GPU required. All models are TensorFlow/Keras MLPs trained on CPU.
Estimated runtime: ~40-80 minutes (5 seeds x 4 datasets).

Requirements:
    pip install tensorflow scikit-optimize scikit-learn numpy pandas matplotlib optuna scipy
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
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_breast_cancer

from dp_hpo import run_all_methods, dp_hpo_ordered, DIMS, VALS, DEFAULTS

SEEDS = [0, 1, 2, 3, 4]


# --- Dataset loaders ---------------------------------------------------------

def load_uci_iris():
    """UCI Iris -- 150 samples, 4 features, 3-class classification."""
    return load_iris(return_X_y=True)


def load_uci_breast_cancer():
    """
    UCI Breast Cancer Wisconsin -- 569 samples, 30 features, binary.
    Built-in to scikit-learn, no download required.
    """
    return load_breast_cancer(return_X_y=True)


def load_uci_heart_disease():
    """
    UCI Cleveland Heart Disease -- 303 samples, 13 features, binary.
    Checks for local file first, then tries UCI archive.
    """
    import pandas as pd
    import os

    col_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
    ]

    local_path = os.path.join(os.path.dirname(__file__), "processed.cleveland.data")
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "heart-disease/processed.cleveland.data")

    if os.path.exists(local_path):
        df = pd.read_csv(local_path, header=None, names=col_names, na_values="?")
        print("  Loaded UCI Heart Disease from local file.")
    else:
        try:
            df = pd.read_csv(url, header=None, names=col_names, na_values="?")
            print("  Downloaded UCI Heart Disease from UCI archive.")
        except Exception as e:
            raise RuntimeError(
                "Could not load UCI Heart Disease dataset: {}\n"
                "Download from: https://archive.ics.uci.edu/ml/machine-learning-databases/"
                "heart-disease/processed.cleveland.data\n"
                "Save as 'processed.cleveland.data' in the repo directory.".format(e)
            )

    df = df.dropna()
    X = df.drop("target", axis=1).values.astype(float)
    y = (df["target"].values > 0).astype(int)
    return X, y


def load_mnist():
    """
    MNIST -- 70,000 images, 10 classes. Flattened to 784 features, normalised.
    Downloaded via TensorFlow/Keras (~55 MB, cached after first run).
    """
    try:
        from tensorflow.keras.datasets import mnist as keras_mnist
        (X_tr, y_tr), (X_te, y_te) = keras_mnist.load_data()
        X = np.concatenate([X_tr, X_te], axis=0)
        y = np.concatenate([y_tr, y_te], axis=0)
        X = X.reshape(X.shape[0], -1).astype(float) / 255.0
        print("  Loaded MNIST via TensorFlow/Keras.")
        return X, y
    except ImportError:
        pass

    from sklearn.datasets import fetch_openml
    print("  Downloading MNIST via scikit-learn (~55 MB, please wait)...")
    data = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = data.data.astype(float) / 255.0
    y = data.target.astype(int)
    print("  MNIST downloaded.")
    return X, y


# --- Data preparation --------------------------------------------------------

def prep(X, y, seed, subsample=None):
    """Split, optionally subsample, standardise, cast to float32 for Keras."""
    if subsample and len(X) > subsample:
        idx = np.random.RandomState(seed).choice(len(X), subsample, replace=False)
        X, y = X[idx], y[idx]
    Xr, Xv, yr, yv = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    sc = StandardScaler()
    Xr = sc.fit_transform(Xr).astype(np.float32)
    Xv = sc.transform(Xv).astype(np.float32)
    return Xr, yr, Xv, yv


# --- Dataset configuration ---------------------------------------------------

DATASET_CONFIG = {
    "MNIST":             (load_mnist,             10000, SEEDS),
    "UCI Breast Cancer": (load_uci_breast_cancer, None,  SEEDS),
}


# --- Statistical significance testing ----------------------------------------

def wilcoxon_test(accs_a, accs_b, name_a, name_b):
    """Paired Wilcoxon signed-rank test. Returns p-value or None."""
    try:
        from scipy.stats import wilcoxon
        diffs = np.array(accs_a) - np.array(accs_b)
        if np.all(diffs == 0):
            print("    Wilcoxon {} vs {}: all differences zero, skipped".format(name_a, name_b))
            return None
        stat, p = wilcoxon(accs_a, accs_b)
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        print("    Wilcoxon {} vs {}: W={:.1f}, p={:.4f} [{}]".format(name_a, name_b, stat, p, sig))
        return float(p)
    except Exception as e:
        print("    Wilcoxon test skipped ({}): {}".format(name_b, e))
        return None


# --- Dimension-ordering ablation ---------------------------------------------

def run_ordering_ablation(X_train, y_train, X_val, y_val, random_state=0):
    """Test DP-HPO under 3 different dimension orderings."""
    orderings = {
        "layers->neurons->lr->activation (default)": [0, 1, 2, 3],
        "lr->activation->layers->neurons":           [2, 3, 0, 1],
        "activation->lr->neurons->layers (reverse)": [3, 2, 1, 0],
    }
    results = {}
    for label, order in orderings.items():
        acc, n_evals = dp_hpo_ordered(X_train, y_train, X_val, y_val,
                                      order=order, random_state=random_state)
        results[label] = {"accuracy": float(acc), "n_evaluations": int(n_evals)}
        print("    {}: acc={:.4f}  evals={}".format(label, acc, n_evals))
    return results


# --- Main experiment loop ----------------------------------------------------

all_results = {}

for ds_name, (loader, subsample, seeds) in DATASET_CONFIG.items():
    print("\n" + "="*65)
    print("Dataset: {}".format(ds_name))
    print("="*65)

    print("  Loading data...")
    X, y = loader()
    print("  Shape: {}  Classes: {}".format(X.shape, len(np.unique(y))))

    methods_list = ["Grid Search", "Random Search", "Bayesian Opt",
                    "Optuna", "DP-HPO (Proposed)"]
    ds_results = {m: {"accs": [], "evals": [], "times": []} for m in methods_list}

    for seed in seeds:
        print("\n  Seed {}:".format(seed))
        Xr, yr, Xv, yv = prep(X, y, seed, subsample)
        run = run_all_methods(Xr, yr, Xv, yv, random_state=seed)
        for method, vals in run.items():
            if method in ds_results:
                ds_results[method]["accs"].append(vals["accuracy"])
                ds_results[method]["evals"].append(vals["n_evaluations"])
                ds_results[method]["times"].append(vals["time_seconds"])

    print("\n  {:<28} {:>12}  {:>6}  {:>8}".format("Method", "Acc (%)", "Evals", "Time"))
    print("  " + "-"*60)
    for method, vals in ds_results.items():
        if not vals["accs"]:
            continue
        ma = np.mean(vals["accs"]); sa = np.std(vals["accs"])
        me = np.mean(vals["evals"]); mt = np.mean(vals["times"])
        print("  {:<28} {:>6.2f}+/-{:<5.2f}  {:>6.0f}  {:>7.1f}s".format(
              method, ma*100, sa*100, me, mt))

    print("\n  Statistical significance (Wilcoxon signed-rank, n={} seeds):".format(len(seeds)))
    dp_accs = ds_results["DP-HPO (Proposed)"]["accs"]
    sig_results = {}
    for method in ["Grid Search", "Random Search", "Bayesian Opt", "Optuna"]:
        if ds_results[method]["accs"]:
            p = wilcoxon_test(dp_accs, ds_results[method]["accs"], "DP-HPO", method)
            sig_results[method] = p

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
    all_results[ds_name]["_wilcoxon"] = sig_results


# --- Dimension-ordering ablation (UCI Iris, seed 0) --------------------------

print("\n" + "="*65)
print("Dimension-Ordering Ablation Study (UCI Iris, Seed 0)")
print("="*65)
X_abl, y_abl = load_uci_iris()
Xr_abl, yr_abl, Xv_abl, yv_abl = prep(X_abl, y_abl, seed=0)
ablation_results = run_ordering_ablation(Xr_abl, yr_abl, Xv_abl, yv_abl, random_state=0)


# --- Save results ------------------------------------------------------------

output = {
    "main_results": all_results,
    "ablation":     ablation_results,
}

with open("results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n\nAll results saved to results.json")
print("Share results.json to generate the final publication-ready paper.")
