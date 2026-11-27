# ============================================================
# CLOSED-POOL BAYESIAN OPTIMIZATION
# Stage 1: Surrogate + Acquisition Function Comparison
#
# Combinations:
#   RF + EI
#   RF + PI
#   RF + UCB
#   GP + EI
#   GP + PI
#   GP + UCB
#
# Stage 1 uses random seed = 0 only.
# ============================================================

import os
import time
import warnings

import numpy as np
import pandas as pd

from scipy.stats import norm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("default")


# ============================================================
# 1. SETTINGS
# ============================================================

CSV_PATH = "data/Dataset No.1.csv"
RESULTS_DIR = "results"

TARGET = "Behavior"

RANDOM_STATE = int(os.environ.get("BO_SEED", "0"))

INIT_N = 100
BATCH_SIZE = 100
MAX_TRAIN = 3400
TEST_SIZE = 0.10

RF_TREES = 100

# IMPORTANT:
# Stage 1 deliberately tests GP without the previous 1000-sample cap.
GP_MAX_SAMPLES = None

# GP hyperparameter optimization
GP_RESTARTS = 0

# Acquisition-function parameters
XI = 0.01
KAPPA = 2.0

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print(f"RF/GP + EI/PI/UCB COMPARISON | SEED = {RANDOM_STATE}")
print("=" * 70)

print(f"\nLoading: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

print(f"Dataset shape: {df.shape}")

print("\nBehavior distribution:")
print(df[TARGET].value_counts())


# ============================================================
# 3. PREPROCESS DATA
# ============================================================

behavior_map = {
    "Severe": 0,
    "Poor": 1,
    "Good": 2,
    "Resistant": 3,
}

if not pd.api.types.is_numeric_dtype(df[TARGET]):
    df[TARGET] = df[TARGET].map(behavior_map)

if df[TARGET].isna().any():
    raise ValueError(
        "Some Behavior labels could not be mapped. "
        "Check the values in the Behavior column."
    )

# Pitting_label is not part of this optimization target.
df = df.drop(columns=["Pitting_label"], errors="ignore")

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(float)

# ASTM Material is categorical, so encode it.
X = pd.get_dummies(X, drop_first=False)

# Ensure all features are numerical.
X = X.astype(float)

print(f"\nSamples:  {len(X)}")
print(f"Features: {X.shape[1]}")


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_pool, X_test, y_pool, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Randomize the candidate pool reproducibly.
rng = np.random.default_rng(RANDOM_STATE)

order = rng.permutation(len(X_pool))

X_pool = X_pool.iloc[order].reset_index(drop=True)
y_pool = y_pool.iloc[order].reset_index(drop=True)

# Initial training set
X_initial = X_pool.iloc[:INIT_N].copy()
y_initial = y_pool.iloc[:INIT_N].copy()

# Remaining closed candidate pool
X_candidate_initial = X_pool.iloc[INIT_N:].copy()
y_candidate_initial = y_pool.iloc[INIT_N:].copy()

print(f"\nHeld-out test set: {len(X_test)}")
print(f"Initial training set: {len(X_initial)}")
print(f"Initial candidate pool: {len(X_candidate_initial)}")

print("\nInitial behavior distribution:")
print(y_initial.value_counts().sort_index())


# ============================================================
# 5. ACQUISITION FUNCTIONS
# ============================================================

def expected_improvement(mu, sigma, best, xi=0.01):

    sigma_safe = np.maximum(sigma, 1e-12)

    improvement = mu - best - xi
    z = improvement / sigma_safe

    ei = (
        improvement * norm.cdf(z)
        + sigma_safe * norm.pdf(z)
    )

    ei[sigma <= 1e-12] = 0.0

    return ei


def probability_of_improvement(mu, sigma, best, xi=0.01):

    sigma_safe = np.maximum(sigma, 1e-12)

    z = (
        mu - best - xi
    ) / sigma_safe

    pi = norm.cdf(z)

    pi[sigma <= 1e-12] = 0.0

    return pi


def upper_confidence_bound(mu, sigma, kappa=2.0):

    return mu + kappa * sigma


# ============================================================
# 6. SURROGATE MODELS
# ============================================================

def train_rf(X_train, y_train, seed):

    model = RandomForestRegressor(
        n_estimators=RF_TREES,
        bootstrap=True,
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def train_gp(X_train, y_train, seed):

    # No sample cap in Stage 1.
    if (
        GP_MAX_SAMPLES is not None
        and len(X_train) > GP_MAX_SAMPLES
    ):

        local_rng = np.random.default_rng(seed)

        selected = local_rng.choice(
            len(X_train),
            size=GP_MAX_SAMPLES,
            replace=False,
        )

        X_fit = X_train.iloc[selected]
        y_fit = y_train.iloc[selected]

    else:

        X_fit = X_train
        y_fit = y_train

    kernel = (
        ConstantKernel(
            1.0,
            constant_value_bounds=(1e-3, 1e3),
        )
        * Matern(
            length_scale=1.0,
            length_scale_bounds=(1e-3, 1e3),
            nu=1.5,
        )
        + WhiteKernel(
            noise_level=1e-6,
            noise_level_bounds=(1e-8, 1e-2),
        )
    )

    model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=GP_RESTARTS,
        normalize_y=True,
        random_state=seed,
    )

    model.fit(X_fit, y_fit)

    return model


# ============================================================
# 7. PREDICT MEAN + UNCERTAINTY
# ============================================================

def predict_rf(model, X_data):

    tree_predictions = np.vstack([
        tree.predict(X_data)
        for tree in model.estimators_
    ])

    mu = tree_predictions.mean(axis=0)
    sigma = tree_predictions.std(axis=0)

    return mu, sigma


def predict_gp(model, X_data):

    mu, sigma = model.predict(
        X_data,
        return_std=True,
    )

    return mu, sigma


def train_surrogate(model_type, X_train, y_train, seed):

    if model_type == "RF":
        return train_rf(
            X_train,
            y_train,
            seed,
        )

    if model_type == "GP":
        return train_gp(
            X_train,
            y_train,
            seed,
        )

    raise ValueError(
        f"Unknown surrogate: {model_type}"
    )


def predict_surrogate(model_type, model, X_data):

    if model_type == "RF":
        return predict_rf(model, X_data)

    if model_type == "GP":
        return predict_gp(model, X_data)

    raise ValueError(
        f"Unknown surrogate: {model_type}"
    )


# ============================================================
# 8. ACQUISITION SELECTION
# ============================================================

def calculate_acquisition(
    acquisition,
    mu,
    sigma,
    best,
):

    if acquisition == "EI":

        return expected_improvement(
            mu,
            sigma,
            best,
            xi=XI,
        )

    if acquisition == "PI":

        return probability_of_improvement(
            mu,
            sigma,
            best,
            xi=XI,
        )

    if acquisition == "UCB":

        return upper_confidence_bound(
            mu,
            sigma,
            kappa=KAPPA,
        )

    raise ValueError(
        f"Unknown acquisition function: {acquisition}"
    )


# ============================================================
# 9. RUN ONE SURROGATE + ACQUISITION COMBINATION
# ============================================================

def run_combination(model_type, acquisition):

    print("\n" + "=" * 70)
    print(f"STARTING: {model_type} + {acquisition}")
    print("=" * 70)

    combination_start = time.perf_counter()

    X_train = X_initial.copy()
    y_train = y_initial.copy()

    X_candidate = X_candidate_initial.copy()
    y_candidate = y_candidate_initial.copy()

    records = []

    iteration = 0

    while len(X_train) <= MAX_TRAIN:

        iteration_start = time.perf_counter()

        # ----------------------------------------------------
        # Train surrogate on CURRENT training set
        # ----------------------------------------------------

        fit_start = time.perf_counter()

        model = train_surrogate(
            model_type,
            X_train,
            y_train,
            RANDOM_STATE,
        )

        fit_seconds = (
            time.perf_counter()
            - fit_start
        )

        # ----------------------------------------------------
        # Evaluate on held-out test set
        # ----------------------------------------------------

        y_test_pred = model.predict(X_test)

        r2 = r2_score(
            y_test,
            y_test_pred,
        )

        mse = mean_squared_error(
            y_test,
            y_test_pred,
        )

        # ----------------------------------------------------
        # Predict candidate pool
        # ----------------------------------------------------

        if len(X_candidate) > 0:

            mu, sigma = predict_surrogate(
                model_type,
                model,
                X_candidate,
            )

            mean_uncertainty = float(
                np.mean(sigma)
            )

        else:

            mu = np.array([])
            sigma = np.array([])
            mean_uncertainty = np.nan

        iteration_seconds = (
            time.perf_counter()
            - iteration_start
        )

        records.append({
            "seed": RANDOM_STATE,
            "surrogate": model_type,
            "acquisition": acquisition,
            "iteration": iteration,
            "train_size": len(X_train),
            "candidate_pool_size": len(X_candidate),
            "R2": r2,
            "MSE": mse,
            "mean_uncertainty": mean_uncertainty,
            "fit_seconds": fit_seconds,
            "iteration_seconds": iteration_seconds,
        })

        print(
            f"Iteration {iteration:02d} | "
            f"Train={len(X_train):4d} | "
            f"Pool={len(X_candidate):4d} | "
            f"R²={r2:.4f} | "
            f"MSE={mse:.4f} | "
            f"Fit={fit_seconds:.1f}s"
        )

        # ----------------------------------------------------
        # Stopping conditions
        # ----------------------------------------------------

        if len(X_train) >= MAX_TRAIN:
            break

        if len(X_candidate) == 0:
            break

        # ----------------------------------------------------
        # Acquisition function
        # ----------------------------------------------------

        best_so_far = float(
            np.max(y_train)
        )

        acquisition_values = calculate_acquisition(
            acquisition,
            mu,
            sigma,
            best_so_far,
        )

        # Number that can actually be added.
        n_to_select = min(
            BATCH_SIZE,
            MAX_TRAIN - len(X_train),
            len(X_candidate),
        )

        # Highest acquisition values
        selected_positions = np.argsort(
            acquisition_values
        )[-n_to_select:][::-1]

        X_selected = X_candidate.iloc[
            selected_positions
        ].copy()

        y_selected = y_candidate.iloc[
            selected_positions
        ].copy()

        # ----------------------------------------------------
        # Add selected samples
        # ----------------------------------------------------

        X_train = pd.concat(
            [X_train, X_selected],
            ignore_index=True,
        )

        y_train = pd.concat(
            [y_train, y_selected],
            ignore_index=True,
        )

        # ----------------------------------------------------
        # Remove selected samples from candidate pool
        # ----------------------------------------------------

        keep_mask = np.ones(
            len(X_candidate),
            dtype=bool,
        )

        keep_mask[selected_positions] = False

        X_candidate = (
            X_candidate.iloc[keep_mask]
            .reset_index(drop=True)
        )

        y_candidate = (
            y_candidate.iloc[keep_mask]
            .reset_index(drop=True)
        )

        iteration += 1

    combination_seconds = (
        time.perf_counter()
        - combination_start
    )

    result = pd.DataFrame(records)

    result["total_combination_seconds"] = (
        combination_seconds
    )

    output_file = os.path.join(
        RESULTS_DIR,
        f"seed{RANDOM_STATE}_{model_type}_{acquisition}.csv",
    )

    result.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nFinished {model_type} + {acquisition}"
    )

    print(
        f"Runtime: {combination_seconds / 60:.2f} minutes"
    )

    print(
        f"Saved: {output_file}"
    )

    return result


# ============================================================
# 10. RUN ALL SIX COMBINATIONS
# ============================================================

all_results = []

surrogates = [
    "RF",
    "GP",
]

acquisitions = [
    "EI",
    "PI",
    "UCB",
]

overall_start = time.perf_counter()

for surrogate in surrogates:

    for acquisition in acquisitions:

        result = run_combination(
            surrogate,
            acquisition,
        )

        all_results.append(result)


# ============================================================
# 11. COMBINE RESULTS
# ============================================================

combined_results = pd.concat(
    all_results,
    ignore_index=True,
)

combined_file = os.path.join(
    RESULTS_DIR,
    f"seed{RANDOM_STATE}_all_combinations.csv",
)

combined_results.to_csv(
    combined_file,
    index=False,
)


# ============================================================
# 12. FINAL PERFORMANCE TABLE
# ============================================================

final_rows = (
    combined_results
    .sort_values("train_size")
    .groupby(
        ["surrogate", "acquisition"],
        as_index=False,
    )
    .tail(1)
)

final_table = final_rows[
    [
        "surrogate",
        "acquisition",
        "train_size",
        "R2",
        "MSE",
        "mean_uncertainty",
        "total_combination_seconds",
    ]
].copy()

final_file = os.path.join(
    RESULTS_DIR,
    f"seed{RANDOM_STATE}_final_performance.csv",
)

final_table.to_csv(
    final_file,
    index=False,
)


# ============================================================
# 13. PRINT FINAL SUMMARY
# ============================================================

overall_seconds = (
    time.perf_counter()
    - overall_start
)

print("\n" + "=" * 70)
print(f"SEED {RANDOM_STATE} COMPLETE")
print("=" * 70)

print("\nFinal performance:")

print(
    final_table.to_string(
        index=False
    )
)

print(
    f"\nTotal runtime: "
    f"{overall_seconds / 60:.2f} minutes"
)

print(f"\nCombined results: {combined_file}")
print(f"Final table:      {final_file}")
