import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# ============================================================
# ACTIVE BAYESIAN CLASSIFICATION (manuscript Section 2.4)
#
# Initial training set dominated by low-performance samples
# (45 Severe, 45 Poor, 5 Good, 5 Resistant). An RF classifier
# estimates the probability of the Resistant class; EI, PI, UCB
# and Random sampling select 10 candidates per iteration for
# 100 iterations (final training set: 1100 samples).
#
# One seed per run. The seed is read from the BO_SEED
# environment variable (default 0), e.g.
#   for s in 0 1 2 3 4; do BO_SEED=$s python scripts/active_classification.py; done
#
# Output (read by analyze_active_classification.py):
#   results/active_classification/seed{SEED}_active_history.csv
#   results/active_classification/seed{SEED}_selected_samples.csv
#   results/active_classification/seed{SEED}_active_final.csv
#
# Note: this file was reconstructed from the compiled bytecode
# (active_classification.cpython-39.pyc) of the script that
# produced the published results; only the input/output paths
# were changed from ~/corrosion_project to the repository root.
# ============================================================


# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]  # repository root
DATA_PATH = PROJECT_DIR / "data" / "Dataset No.1.csv"
RESULTS_DIR = PROJECT_DIR / "results" / "active_classification"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = int(os.environ.get("BO_SEED", "0"))

TARGET = "Behavior"

LABEL_MAP = {
    "Severe": 0,
    "Poor": 1,
    "Good": 2,
    "Resistant": 3,
}

CLASS_NAMES = {
    0: "Severe",
    1: "Poor",
    2: "Good",
    3: "Resistant",
}

# Initial training set dominated by low-performance samples
INITIAL_COUNTS = {
    0: 45,
    1: 45,
    2: 5,
    3: 5,
}

INITIAL_N = sum(INITIAL_COUNTS.values())

TEST_SIZE = 0.1
BATCH_SIZE = 10
MAX_ITER = 100

N_ESTIMATORS = 200

# Acquisition-function parameters
XI = 0.01
KAPPA = 2.0

STRATEGIES = ["EI", "PI", "UCB", "Random"]


# ============================================================
# 2. ACQUISITION FUNCTIONS
# ============================================================

def expected_improvement(mu, sigma, best, xi=0.01):
    improvement = mu - best - xi

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.divide(
            improvement,
            sigma,
            out=np.zeros_like(improvement),
            where=sigma > 0,
        )

    ei = (
        improvement * norm.cdf(z)
        + sigma * norm.pdf(z)
    )

    ei[sigma <= 0] = 0.0

    return ei


def probability_of_improvement(mu, sigma, best, xi=0.01):
    improvement = mu - best - xi

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.divide(
            improvement,
            sigma,
            out=np.zeros_like(improvement),
            where=sigma > 0,
        )

    pi = norm.cdf(z)

    # Without uncertainty, PI is 1 if the candidate improves, else 0
    pi = np.where(
        sigma <= 0,
        (improvement > 0).astype(float),
        pi,
    )

    return pi


def upper_confidence_bound(mu, sigma, kappa=2.0):
    return mu + kappa * sigma


def resistant_probability_and_uncertainty(model, X_candidate):
    """
    For each RF tree, obtain the predicted probability of the
    Resistant class (class 3).

    Mean across trees:
        estimated Resistant probability

    Standard deviation across trees:
        tree-level predictive disagreement
    """

    tree_probabilities = []

    for tree in model.estimators_:

        probabilities = tree.predict_proba(X_candidate)

        resistant_probability = np.zeros(len(X_candidate))

        if 3 in tree.classes_:
            class_position = np.where(tree.classes_ == 3)[0][0]
            resistant_probability = probabilities[:, class_position]

        tree_probabilities.append(resistant_probability)

    tree_probabilities = np.asarray(tree_probabilities)

    mu = tree_probabilities.mean(axis=0)
    sigma = tree_probabilities.std(axis=0)

    return mu, sigma


# ============================================================
# 3. LOAD AND PREPROCESS DATA
# ============================================================

print("=" * 75)
print("ACTIVE BAYESIAN CLASSIFICATION")
print("=" * 75)

print(f"Random seed: {SEED}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Maximum iterations: {MAX_ITER}")
print(f"RF trees: {N_ESTIMATORS}")

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

df[TARGET] = df[TARGET].map(LABEL_MAP)

if df[TARGET].isna().any():
    raise ValueError(
        "Unrecognized Behavior labels were found."
    )

df = df.drop(
    columns=["Pitting_label"],
    errors="ignore",
)

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

# The steel grade is described by its composition, so the
# categorical grade name is not used as a feature.
X = X.drop(
    columns=["ASTM Material"],
    errors="ignore",
)

X = pd.get_dummies(X, drop_first=False)

print(f"\nTotal dataset size: {len(X)}")
print(f"Number of features: {X.shape[1]}")

print("\nFull class distribution:")

for cls in sorted(CLASS_NAMES):
    count = int((y == cls).sum())
    print(
        f"{CLASS_NAMES[cls]:10s}: "
        f"{count:4d} ({count / len(y):.4f})"
    )


# ============================================================
# 4. HELD-OUT TEST SET
# ============================================================

X_pool, X_test, y_pool, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=SEED,
)

print(f"\nHeld-out test samples: {len(X_test)}")
print(f"Optimization pool: {len(X_pool)}")


# ============================================================
# 5. INITIAL TRAINING SET (45 / 45 / 5 / 5)
# ============================================================

rng_initial = np.random.default_rng(SEED)

initial_indices = []

for cls, required_n in INITIAL_COUNTS.items():

    class_indices = (
        y_pool[y_pool == cls]
    ).index.to_numpy()

    if len(class_indices) < required_n:
        raise ValueError(
            f"Not enough {CLASS_NAMES[cls]} samples for requested initialization."
        )

    selected = rng_initial.choice(
        class_indices,
        size=required_n,
        replace=False,
    )

    initial_indices.extend(selected.tolist())

initial_indices = np.asarray(initial_indices)

X_initial = X_pool.loc[initial_indices].copy()
y_initial = y_pool.loc[initial_indices].copy()

X_candidate_initial = X_pool.drop(
    index=initial_indices
).copy()

y_candidate_initial = y_pool.drop(
    index=initial_indices
).copy()

print("\nInitial training distribution:")

for cls in sorted(CLASS_NAMES):
    print(
        f"{CLASS_NAMES[cls]:10s}: "
        f"{int((y_initial == cls).sum())}"
    )

print(
    f"\nInitial training size: {len(X_initial)}"
)

print(
    f"Initial candidate pool: {len(X_candidate_initial)}"
)


# ============================================================
# 6. ACTIVE LEARNING LOOP
# ============================================================

all_histories = []
all_acquisitions = []
final_results = []

for strategy_index, strategy in enumerate(STRATEGIES):

    print("\n" + "=" * 75)
    print(f"STARTING SEED {SEED}: {strategy}")
    print("=" * 75)

    start_time = time.time()

    X_train = X_initial.copy()
    y_train = y_initial.copy()

    X_candidate = X_candidate_initial.copy()
    y_candidate = y_candidate_initial.copy()

    # Separate random generator for the Random baseline
    random_rng = np.random.default_rng(
        SEED * 1000 + 100 + strategy_index
    )

    initial_resistant = int(
        (y_initial == 3).sum()
    )

    # Iteration 0: evaluate the classifier trained on the initial set
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=SEED,
        n_jobs=-1,
        class_weight=None,
    )

    model.fit(X_train, y_train)

    test_prediction = model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        test_prediction,
    )

    test_macro_f1 = f1_score(
        y_test,
        test_prediction,
        average="macro",
    )

    resistant_total = int(
        (y_train == 3).sum()
    )

    resistant_discovered = (
        resistant_total - initial_resistant
    )

    acquired_total = 0

    resistant_ratio = (
        resistant_total / len(y_train)
    )

    discovery_rate = np.nan

    all_histories.append({
        "seed": SEED,
        "strategy": strategy,
        "iteration": 0,
        "train_size": len(X_train),
        "candidate_pool_size": len(X_candidate),
        "acquired_total": acquired_total,
        "resistant_total": resistant_total,
        "resistant_discovered": resistant_discovered,
        "resistant_class_ratio": resistant_ratio,
        "resistant_discovery_rate": discovery_rate,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
    })

    for iteration in range(1, MAX_ITER + 1):

        if len(X_candidate) == 0:
            print(
                f"{strategy}: candidate pool exhausted."
            )
            break

        # Surrogate used to rank the candidates
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=SEED + iteration,
            n_jobs=-1,
            class_weight=None,
        )

        model.fit(X_train, y_train)

        current_batch = min(
            BATCH_SIZE,
            len(X_candidate),
        )

        # ----------------------------------------------------
        # Candidate selection
        # ----------------------------------------------------

        if strategy == "Random":

            selected_positions = random_rng.choice(
                len(X_candidate),
                size=current_batch,
                replace=False,
            )

            selected_positions = np.asarray(
                selected_positions
            )

            selected_mu = np.full(
                current_batch,
                np.nan,
            )

            selected_sigma = np.full(
                current_batch,
                np.nan,
            )

            selected_acquisition = np.full(
                current_batch,
                np.nan,
            )

        else:

            # Resistant probability and tree uncertainty
            # for every candidate
            mu, sigma = (
                resistant_probability_and_uncertainty(
                    model,
                    X_candidate,
                )
            )

            # Best value so far: highest predicted Resistant
            # probability among the current training samples
            train_mu, _ = (
                resistant_probability_and_uncertainty(
                    model,
                    X_train,
                )
            )

            best_so_far = np.max(train_mu)

            if strategy == "EI":

                acquisition_values = (
                    expected_improvement(
                        mu,
                        sigma,
                        best_so_far,
                        xi=XI,
                    )
                )

            elif strategy == "PI":

                acquisition_values = (
                    probability_of_improvement(
                        mu,
                        sigma,
                        best_so_far,
                        xi=XI,
                    )
                )

            elif strategy == "UCB":

                acquisition_values = (
                    upper_confidence_bound(
                        mu,
                        sigma,
                        kappa=KAPPA,
                    )
                )

            else:
                raise ValueError(
                    f"Unknown strategy: {strategy}"
                )

            selected_positions = np.argsort(
                -acquisition_values,
                kind="stable",
            )[:current_batch]

            selected_mu = mu[
                selected_positions
            ]

            selected_sigma = sigma[
                selected_positions
            ]

            selected_acquisition = (
                acquisition_values[
                    selected_positions
                ]
            )

        # ----------------------------------------------------
        # Reveal labels and update the training set
        # ----------------------------------------------------

        X_selected = X_candidate.iloc[
            selected_positions
        ].copy()

        y_selected = y_candidate.iloc[
            selected_positions
        ].copy()

        for j, idx in enumerate(
            X_selected.index
        ):

            all_acquisitions.append({
                "seed": SEED,
                "strategy": strategy,
                "iteration": iteration,
                "sample_index": idx,
                "true_class": int(
                    y_selected.loc[idx]
                ),
                "true_behavior": CLASS_NAMES[
                    int(y_selected.loc[idx])
                ],
                "predicted_resistant_probability":
                    selected_mu[j],
                "predictive_uncertainty":
                    selected_sigma[j],
                "acquisition_value":
                    selected_acquisition[j],
            })

        X_train = pd.concat(
            [X_train, X_selected],
            axis=0,
        )

        y_train = pd.concat(
            [y_train, y_selected],
            axis=0,
        )

        selected_indices = X_selected.index

        X_candidate = X_candidate.drop(
            index=selected_indices
        )

        y_candidate = y_candidate.drop(
            index=selected_indices
        )

        # ----------------------------------------------------
        # Evaluate a classifier trained on the updated set
        # on the held-out test set
        # ----------------------------------------------------

        evaluation_model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=SEED + 10000 + iteration,
            n_jobs=-1,
            class_weight=None,
        )

        evaluation_model.fit(
            X_train,
            y_train,
        )

        test_prediction = (
            evaluation_model.predict(
                X_test
            )
        )

        test_accuracy = accuracy_score(
            y_test,
            test_prediction,
        )

        test_macro_f1 = f1_score(
            y_test,
            test_prediction,
            average="macro",
        )

        # ----------------------------------------------------
        # Discovery metrics
        # ----------------------------------------------------

        resistant_total = int(
            (y_train == 3).sum()
        )

        resistant_discovered = (
            resistant_total
            - initial_resistant
        )

        acquired_total = (
            len(y_train) - INITIAL_N
        )

        resistant_ratio = (
            resistant_total / len(y_train)
        )

        if acquired_total > 0:
            discovery_rate = (
                resistant_discovered
                / acquired_total
            )
        else:
            discovery_rate = np.nan

        all_histories.append({
            "seed": SEED,
            "strategy": strategy,
            "iteration": iteration,
            "train_size": len(X_train),
            "candidate_pool_size": len(X_candidate),
            "acquired_total": acquired_total,
            "resistant_total": resistant_total,
            "resistant_discovered": resistant_discovered,
            "resistant_class_ratio": resistant_ratio,
            "resistant_discovery_rate": discovery_rate,
            "test_accuracy": test_accuracy,
            "test_macro_f1": test_macro_f1,
        })

        if (
            iteration == 1
            or iteration % 10 == 0
            or iteration == MAX_ITER
        ):
            print(
                f"Iteration {iteration:3d} | Train="
                f"{len(X_train):4d} | Resistant discovered="
                f"{resistant_discovered:4d} | Discovery rate="
                f"{discovery_rate:.4f} | Macro-F1="
                f"{test_macro_f1:.4f}"
            )

    runtime = time.time() - start_time

    strategy_history = pd.DataFrame([
        row for row in all_histories
        if row["strategy"] == strategy
    ])

    final_row = strategy_history.iloc[-1]

    final_results.append({
        "seed": SEED,
        "strategy": strategy,
        "iterations_completed":
            int(final_row["iteration"]),
        "final_train_size":
            int(final_row["train_size"]),
        "resistant_discovered":
            int(final_row["resistant_discovered"]),
        "resistant_class_ratio":
            float(final_row["resistant_class_ratio"]),
        "resistant_discovery_rate":
            float(final_row["resistant_discovery_rate"]),
        "test_accuracy":
            float(final_row["test_accuracy"]),
        "test_macro_f1":
            float(final_row["test_macro_f1"]),
        "runtime_seconds": runtime,
    })

    print(
        f"\nFinished {strategy} in "
        f"{runtime:.2f} seconds."
    )


# ============================================================
# 7. SAVE RESULTS
# ============================================================

history_df = pd.DataFrame(
    all_histories
)

acquisition_df = pd.DataFrame(
    all_acquisitions
)

final_df = pd.DataFrame(
    final_results
)

history_file = (
    RESULTS_DIR /
    f"seed{SEED}_active_history.csv"
)

acquisition_file = (
    RESULTS_DIR /
    f"seed{SEED}_selected_samples.csv"
)

final_file = (
    RESULTS_DIR /
    f"seed{SEED}_active_final.csv"
)

history_df.to_csv(
    history_file,
    index=False,
)

acquisition_df.to_csv(
    acquisition_file,
    index=False,
)

final_df.to_csv(
    final_file,
    index=False,
)

print("\n" + "=" * 75)
print(f"SEED {SEED} COMPLETE")
print("=" * 75)

print("\nFinal results:")
print(
    final_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(history_file)
print(acquisition_file)
print(final_file)
