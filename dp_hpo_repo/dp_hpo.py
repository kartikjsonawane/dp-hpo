"""
dp_hpo.py -- DP-HPO core implementation using TensorFlow/Keras.

Five HPO methods:
  1. Grid Search           -- exhaustive,  O(n^d) evaluations (108)
  2. Random Search         -- stochastic,  O(k)   evaluations (k=20)
  3. Bayesian Optimisation -- GP-based,    O(k)   evaluations (k=20)
  4. Optuna                -- TPE-based,   O(k)   evaluations (k=20)
  5. DP-HPO (Proposed)     -- greedy DP,   O(n*d) evaluations (~10-13)

Requirements:
    pip install tensorflow scikit-optimize scikit-learn numpy optuna
"""

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {strategy.num_replicas_in_sync} GPUs")
else:
    strategy = tf.distribute.get_strategy()
    print(f"Using {len(gpus)} GPU(s)" if gpus else "Using CPU")

import time
import warnings
import numpy as np
from itertools import product
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# --- Search space -----------------------------------------------------------
SEARCH_SPACE = {
    "hidden_layers": [1, 2, 3],
    "neurons":       [32, 64, 128, 256],
    "learning_rate": [0.001, 0.01, 0.1],
    "activation":    ["relu", "tanh", "logistic"],
}

DIMS        = list(SEARCH_SPACE.keys())
VALS        = list(SEARCH_SPACE.values())
ALL_CONFIGS = list(product(*VALS))   # 3 x 4 x 3 x 3 = 108

# Default value for each dimension (used by DP greedy sweep)
DEFAULTS = [1, 64, 0.01, "relu"]


# --- Keras model builder ----------------------------------------------------
def _build_model(hidden_layers, neurons, learning_rate, activation,
                 n_features, n_classes):
    """
    Construct a Keras Sequential MLP.
    Activation mapping: "logistic" -> "sigmoid" (sklearn name -> Keras name).
    """
    from tensorflow import keras
    keras_act = "sigmoid" if activation == "logistic" else activation

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(n_features,)))
    for _ in range(hidden_layers):
        model.add(keras.layers.Dense(neurons, activation=keras_act))

    if n_classes == 2:
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
    else:
        model.add(keras.layers.Dense(n_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


# --- Core evaluator (memoised) ----------------------------------------------
def _evaluate(config, X_train, y_train, X_val, y_val, memo, random_state):
    """
    Train a Keras MLP with config and return validation accuracy.
    Memoisation prevents redundant training runs across HPO methods.
    config = (hidden_layers, neurons, learning_rate, activation)
    """
    import tensorflow as tf

    key = tuple(config)
    if key in memo:
        return memo[key]

    hidden_layers, neurons, learning_rate, activation = config
    n_features = X_train.shape[1]
    n_classes  = len(np.unique(np.concatenate([y_train, y_val])))

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    model = _build_model(hidden_layers, neurons, learning_rate, activation,
                         n_features, n_classes)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=0,
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
    )

    if n_classes == 2:
        probs = model(X_val, training=False).numpy().flatten()
        preds = (probs > 0.5).astype(int)
    else:
        probs = model(X_val, training=False).numpy()
        preds = np.argmax(probs, axis=1)

    acc = float(np.mean(preds == y_val))
    memo[key] = acc
    return acc


# --- 1. Grid Search ---------------------------------------------------------
def grid_search(X_train, y_train, X_val, y_val, random_state=42):
    """Exhaustive evaluation of all 108 configurations. Complexity: O(n^d)."""
    memo = {}
    best_acc, best_cfg = -1, None
    t0 = time.time()

    for i, cfg in enumerate(ALL_CONFIGS, 1):
        acc = _evaluate(cfg, X_train, y_train, X_val, y_val, memo, random_state)
        if acc > best_acc:
            best_acc, best_cfg = acc, cfg
        print(f"\r      Grid Search: {i}/{len(ALL_CONFIGS)} configs  best={best_acc:.4f}",
              end="", flush=True)
    print()

    return {
        "config":        dict(zip(DIMS, best_cfg)),
        "accuracy":      best_acc,
        "n_evaluations": len(memo),
        "time_seconds":  time.time() - t0,
    }


# --- 2. Random Search -------------------------------------------------------
def random_search(X_train, y_train, X_val, y_val, k=20, random_state=42):
    """Uniformly sample k configurations. Complexity: O(k)."""
    rng  = np.random.RandomState(random_state)
    memo = {}
    best_acc, best_cfg = -1, None
    t0 = time.time()

    sampled = [ALL_CONFIGS[i]
               for i in rng.choice(len(ALL_CONFIGS), size=k, replace=False)]

    for cfg in sampled:
        acc = _evaluate(cfg, X_train, y_train, X_val, y_val, memo, random_state)
        if acc > best_acc:
            best_acc, best_cfg = acc, cfg

    return {
        "config":        dict(zip(DIMS, best_cfg)),
        "accuracy":      best_acc,
        "n_evaluations": len(memo),
        "time_seconds":  time.time() - t0,
    }


# --- 3. Bayesian Optimisation -----------------------------------------------
def bayesian_optimisation(X_train, y_train, X_val, y_val,
                          n_calls=20, random_state=42):
    """
    Gaussian-Process surrogate model (scikit-optimize gp_minimize).
    Complexity: O(n_calls).
    """
    from skopt import gp_minimize
    from skopt.space import Integer

    space = [
        Integer(0, len(VALS[0]) - 1, name="hidden_layers_idx"),
        Integer(0, len(VALS[1]) - 1, name="neurons_idx"),
        Integer(0, len(VALS[2]) - 1, name="learning_rate_idx"),
        Integer(0, len(VALS[3]) - 1, name="activation_idx"),
    ]

    memo = {}
    t0   = time.time()

    def objective(params):
        cfg = tuple(VALS[i][params[i]] for i in range(len(DIMS)))
        return -_evaluate(cfg, X_train, y_train, X_val, y_val, memo, random_state)

    result = gp_minimize(
        objective, space,
        n_calls=n_calls,
        random_state=random_state,
        verbose=False,
    )

    best_cfg = tuple(VALS[i][result.x[i]] for i in range(len(DIMS)))
    return {
        "config":        dict(zip(DIMS, best_cfg)),
        "accuracy":      -result.fun,
        "n_evaluations": len(memo),
        "time_seconds":  time.time() - t0,
    }


# --- 4. Optuna (TPE) --------------------------------------------------------
def optuna_search(X_train, y_train, X_val, y_val, n_trials=20, random_state=42):
    """
    Optuna Tree-structured Parzen Estimator (TPE) search.
    Complexity: O(n_trials).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    memo = {}
    t0   = time.time()

    def objective(trial):
        cfg = (
            trial.suggest_categorical("hidden_layers", VALS[0]),
            trial.suggest_categorical("neurons",       VALS[1]),
            trial.suggest_categorical("learning_rate", VALS[2]),
            trial.suggest_categorical("activation",    VALS[3]),
        )
        return -_evaluate(cfg, X_train, y_train, X_val, y_val, memo, random_state)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study   = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_cfg = (
        best_params["hidden_layers"],
        best_params["neurons"],
        best_params["learning_rate"],
        best_params["activation"],
    )
    return {
        "config":        dict(zip(DIMS, best_cfg)),
        "accuracy":      -study.best_value,
        "n_evaluations": len(memo),
        "time_seconds":  time.time() - t0,
    }


# --- 5. DP-HPO (Proposed) ---------------------------------------------------
def dp_hpo(X_train, y_train, X_val, y_val, random_state=42):
    """
    Greedy dimension-by-dimension dynamic programming search.

    Algorithm:
      committed = list(DEFAULTS)
      For each dimension i in {0, ..., d-1}:
        For each candidate v in VALS[i]:
          config = committed[:i] + [v] + DEFAULTS[i+1:]
          acc = _evaluate(config)   # memoised
        committed[i] = argmax_v acc

    Max evaluations = sum(|H_i|) = 3+4+3+3 = 13; ~10 in practice (cache hits).
    Complexity: O(n*d).
    """
    memo      = {}
    committed = list(DEFAULTS)
    t0        = time.time()
    best_acc  = -1

    for i, candidates in enumerate(VALS):
        best_val = committed[i]
        for v in candidates:
            cfg = tuple(committed[:i] + [v] + DEFAULTS[i + 1:])
            acc = _evaluate(cfg, X_train, y_train, X_val, y_val, memo, random_state)
            if acc > best_acc:
                best_acc, best_val = acc, v
        committed[i] = best_val

    return {
        "config":        dict(zip(DIMS, committed)),
        "accuracy":      best_acc,
        "n_evaluations": len(memo),
        "time_seconds":  time.time() - t0,
    }


def dp_hpo_ordered(X_train, y_train, X_val, y_val, order=None, random_state=42):
    """
    DP-HPO with a custom dimension processing order (for ablation study).

    Parameters
    ----------
    order : list of int, e.g. [2, 3, 0, 1]
        Indices into DIMS/VALS defining processing order.
        Default None uses [0, 1, 2, 3].

    Returns
    -------
    (best_accuracy, n_evaluations)
    """
    if order is None:
        order = list(range(len(DIMS)))

    memo      = {}
    committed = list(DEFAULTS)
    best_acc  = -1

    for i in order:
        best_val = committed[i]
        for v in VALS[i]:
            cfg = list(committed)
            cfg[i] = v
            acc = _evaluate(tuple(cfg), X_train, y_train, X_val, y_val, memo, random_state)
            if acc > best_acc:
                best_acc, best_val = acc, v
        committed[i] = best_val

    return best_acc, len(memo)


# --- Convenience wrapper ----------------------------------------------------
def run_all_methods(X_train, y_train, X_val, y_val, random_state=42):
    """
    Run all five HPO methods and return a dict of results.

    Returns
    -------
    dict mapping method name -> {config, accuracy, n_evaluations, time_seconds}
    """
    methods = {
        "Grid Search":        lambda: grid_search(
                                  X_train, y_train, X_val, y_val, random_state),
        "Random Search":      lambda: random_search(
                                  X_train, y_train, X_val, y_val,
                                  k=20, random_state=random_state),
        "Bayesian Opt":       lambda: bayesian_optimisation(
                                  X_train, y_train, X_val, y_val,
                                  n_calls=20, random_state=random_state),
        "Optuna":             lambda: optuna_search(
                                  X_train, y_train, X_val, y_val,
                                  n_trials=20, random_state=random_state),
        "DP-HPO (Proposed)":  lambda: dp_hpo(
                                  X_train, y_train, X_val, y_val, random_state),
    }

    results = {}
    for name, fn in methods.items():
        print(f"    Running {name} ...", flush=True)
        results[name] = fn()
        r = results[name]
        print(f"      acc={r['accuracy']:.4f}  evals={r['n_evaluations']}"
              f"  time={r['time_seconds']:.1f}s")
    return results
