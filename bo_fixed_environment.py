# ============================================================
# FIXED-ENVIRONMENT ALLOY SELECTION WITH BAYESIAN OPTIMIZATION
#
# Question: for a given corrosive environment, how many alloys
# must be tested before a corrosion-resistant one is found?
#
# Leave-one-environment-out design:
#   1. One environment is the "target"; its labels are hidden.
#   2. The surrogate is trained on all OTHER environments
#      (knowledge transfer from similar conditions).
#   3. One random alloy in the target environment is tested first
#      (the same alloy for every strategy, for a fair comparison).
#   4. Each strategy then selects ONE alloy at a time from the
#      remaining alloys of the target environment; its label is
#      revealed and the surrogate is retrained.
#   5. We record how many tests are needed to find the first
#      Resistant alloy, and how many are found within K tests.
#
# Strategies: EI, PI, UCB (as in Section 2.3), Greedy (mean only,
# no uncertainty) and Random. Only "informative" environments are
# used: those containing at least one Resistant AND at least one
# Severe or Poor alloy.
#
# Surrogate, acquisition functions, xi, kappa and the ordinal
# scale (Severe = 0 ... Resistant = 3) follow bo_multiseed.py.
#
# Run from ~/corrosion_project:
#   python scripts/bo_fixed_environment.py
#
# Environment variables (all optional):
#   FE_DATA      path to Dataset No. 1 CSV  (default data/Dataset No.1.csv)
#   FE_SEEDS     comma list of seeds        (default 0,1,2,3,4)
#   FE_MODE      transfer | cold            (default transfer)
#                cold = no data from other environments (ablation)
#   FE_KMAX      tests recorded per run     (default 10)
#   FE_OUT       results folder             (default results/fixed_environment)
#   FE_N_JOBS    cores for the random forest (default -1 = all)
#   FE_MAX_ENVS  only use the first N informative environments
#                (quick test; default: all)
# ============================================================

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 1. SETTINGS
# ============================================================

DATA = Path(os.environ.get("FE_DATA", "data/Dataset No.1.csv"))
SEEDS = [int(s) for s in os.environ.get("FE_SEEDS", "0,1,2,3,4").split(",")]
MODE = os.environ.get("FE_MODE", "transfer")
K_MAX = int(os.environ.get("FE_KMAX", "10"))
OUT = Path(os.environ.get("FE_OUT", "results/fixed_environment"))
N_JOBS = int(os.environ.get("FE_N_JOBS", "-1"))
MAX_ENVS = os.environ.get("FE_MAX_ENVS")

STRATEGIES = ["EI", "PI", "UCB", "Greedy", "Random"]

# Same settings as bo_multiseed.py
N_TREES = 100
XI = 0.01
KAPPA = 2.0
SIGMA_FLOOR = 1e-9

BEHAVIOR_MAP = {"Severe": 0, "Poor": 1, "Good": 2, "Resistant": 3}
RESISTANT = 3

ELEMENTS = {"C", "Mn", "Si", "P", "S", "Cr", "Mo", "Ni", "N", "Ti", "Nb",
            "Al"}
NON_FEATURES = {"ASTM Material", "Behavior", "Pitting_label"}

if MODE not in ("transfer", "cold"):
    raise ValueError("FE_MODE must be 'transfer' or 'cold'")

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DATA
# ============================================================

df = pd.read_csv(DATA)
df.columns = df.columns.str.strip()
df["Behavior"] = df["Behavior"].astype(str).str.strip().str.capitalize()
df = df[df["Behavior"].isin(BEHAVIOR_MAP)].reset_index(drop=True)
df["y"] = df["Behavior"].map(BEHAVIOR_MAP).astype(float)

comp_cols = [c for c in df.columns
             if c in ELEMENTS or c.lower().startswith("fe")]
env_cols = [c for c in df.columns
            if c not in comp_cols and c not in NON_FEATURES | {"y"}
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].nunique() > 1]
# ASTM Material is excluded, as in Section 2.2: alloys are described
# by their composition only.
feature_cols = comp_cols + env_cols

env_key = df[env_cols].round(6).astype(str).agg("|".join, axis=1)
df["env_id"] = pd.factorize(env_key)[0]

X_all = df[feature_cols].to_numpy(dtype=float)
y_all = df["y"].to_numpy()


def describe(env_rows):
    first = env_rows.iloc[0]
    return "; ".join(f"{c}={first[c]:g}" for c in env_cols if first[c] != 0)


envs = []
for env_id, g in df.groupby("env_id"):
    n_res = int((g["y"] == RESISTANT).sum())
    n_bad = int((g["y"] <= 1).sum())                 # Severe or Poor
    if n_res > 0 and n_bad > 0:
        envs.append({"env_id": env_id, "condition": describe(g),
                     "n_candidates": len(g), "n_resistant": n_res})

if MAX_ENVS:
    envs = envs[:int(MAX_ENVS)]

print("=" * 70)
print("FIXED-ENVIRONMENT ALLOY SELECTION")
print("=" * 70)
print(f"Data: {DATA} ({len(df)} rows)")
print(f"Features: {len(comp_cols)} composition + {len(env_cols)} environment")
print(f"Informative environments: {len(envs)} of {df['env_id'].nunique()}")
print(f"Mode: {MODE} | seeds: {SEEDS} | strategies: {STRATEGIES}")


# ============================================================
# 3. SURROGATE AND ACQUISITION (as in bo_multiseed.py)
# ============================================================

def fit_predict(X_train, y_train, X_cand, seed):
    model = RandomForestRegressor(n_estimators=N_TREES, bootstrap=True,
                                  random_state=seed, n_jobs=N_JOBS)
    model.fit(X_train, y_train)
    tree_preds = np.stack([t.predict(X_cand) for t in model.estimators_])
    mu = tree_preds.mean(axis=0)
    sigma = np.maximum(tree_preds.std(axis=0), SIGMA_FLOOR)
    return mu, sigma


def acquisition(name, mu, sigma, f_best):
    if name == "UCB":
        return mu + KAPPA * sigma
    if name == "Greedy":
        return mu
    z = (mu - f_best - XI) / sigma
    if name == "PI":
        return norm.cdf(z)
    if name == "EI":
        return (mu - f_best - XI) * norm.cdf(z) + sigma * norm.pdf(z)
    raise ValueError(name)


def random_expected_tests(n, r):
    """Expected number of random draws (without replacement) until the
    first of r successes among n items: (n + 1) / (r + 1)."""
    return (n + 1) / (r + 1)


# ============================================================
# 4. ONE RUN: one target environment, one seed, one strategy
# ============================================================

def run(env, seed, strategy):
    target = np.where(df["env_id"].to_numpy() == env["env_id"])[0]
    others = np.where(df["env_id"].to_numpy() != env["env_id"])[0]
    rng = np.random.default_rng([seed, env["env_id"]])

    # Same first test for every strategy (fair comparison)
    order = rng.permutation(target)
    tested = [order[0]]
    remaining = [i for i in target if i != order[0]]
    random_order = list(order[1:])            # used by "Random"

    while remaining and len(tested) < max(K_MAX, len(target)):
        found = (y_all[tested] == RESISTANT).any()
        if found and len(tested) >= K_MAX:
            break
        if strategy == "Random":
            pick = random_order.pop(0)
        else:
            if MODE == "transfer":
                train_idx = np.concatenate([others, tested])
            else:
                train_idx = np.array(tested)
            mu, sigma = fit_predict(X_all[train_idx], y_all[train_idx],
                                    X_all[remaining], seed)
            f_best = y_all[tested].max()
            scores = acquisition(strategy, mu, sigma, f_best)
            pick = remaining[int(np.argmax(scores))]
            if pick in random_order:
                random_order.remove(pick)
        tested.append(pick)
        remaining.remove(pick)

    labels = y_all[tested]
    hits = np.where(labels == RESISTANT)[0]
    first_hit = int(hits[0]) + 1 if len(hits) else np.nan
    cum_res = np.cumsum(labels == RESISTANT)
    row = {
        "env_id": env["env_id"], "condition": env["condition"],
        "n_candidates": env["n_candidates"],
        "n_resistant": env["n_resistant"],
        "seed": seed, "strategy": strategy, "mode": MODE,
        "tests_to_first_resistant": first_hit,
        "random_expected_tests": random_expected_tests(
            env["n_candidates"], env["n_resistant"]),
    }
    for k in range(1, K_MAX + 1):
        row[f"resistant_found_within_{k}"] = int(
            cum_res[min(k, len(cum_res)) - 1])
    return row


# ============================================================
# 5. MAIN LOOP
# ============================================================

rows = []
start = time.time()
for i, env in enumerate(envs, start=1):
    for seed in SEEDS:
        for strategy in STRATEGIES:
            rows.append(run(env, seed, strategy))
    print(f"  [{i:3d}/{len(envs)}] env {env['env_id']:3d} "
          f"({env['n_resistant']}/{env['n_candidates']} resistant) "
          f"done | {time.time() - start:6.0f} s", flush=True)

raw = pd.DataFrame(rows)
raw_path = OUT / f"fixed_env_runs_{MODE}.csv"
raw.to_csv(raw_path, index=False)


# ============================================================
# 6. SUMMARY
# ============================================================

def summarize(g):
    out = {
        "tests_to_first_mean": g["tests_to_first_resistant"].mean(),
        "tests_to_first_std": g["tests_to_first_resistant"].std(),
        "found_by_test_1": (g["tests_to_first_resistant"] <= 1).mean(),
        "found_within_3": (g["tests_to_first_resistant"] <= 3).mean(),
        "found_within_5": (g["tests_to_first_resistant"] <= 5).mean(),
        "resistant_in_first_5": g[f"resistant_found_within_{min(5, K_MAX)}"].mean(),
        f"resistant_in_first_{K_MAX}": g[f"resistant_found_within_{K_MAX}"].mean(),
    }
    return pd.Series(out)


summary = (raw.groupby("strategy")[raw.columns.drop("strategy")].apply(summarize)
           .reindex(STRATEGIES).round(3))
summary["random_expected_tests_analytic"] = round(
    raw.drop_duplicates("env_id")["random_expected_tests"].mean(), 3)
summary_path = OUT / f"fixed_env_summary_{MODE}.csv"
summary.to_csv(summary_path)

print("\n" + "=" * 70)
print(f"SUMMARY ({len(envs)} environments x {len(SEEDS)} seeds, mode = {MODE})")
print("=" * 70)
print(summary.to_string())
print("\ntests_to_first_mean : alloys tested until the first Resistant one "
      "(including the initial random test)")
print("found_within_3/5    : fraction of runs with a Resistant alloy "
      "among the first 3/5 tests")
print(f"\nSaved {raw_path}\nSaved {summary_path}")
print(f"Total time: {time.time() - start:.0f} s")


# ============================================================
# 7. FIGURE: fraction of environments solved vs number of tests
# ============================================================

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from corrosion_figstyle import (COLORS, MM, WIDTH_SINGLE,
                                        apply_style, save)
        apply_style()
        colors = {**{s: COLORS[s] for s in ["EI", "PI", "UCB", "Random"]},
                  "Greedy": "#9A4D8E"}
        size = (WIDTH_SINGLE * 1.25, 70 * MM)
    except ImportError:
        colors, size, save = {}, (5, 3.5), None

    ks = np.arange(1, K_MAX + 1)
    fig, ax = plt.subplots(figsize=size)
    for strategy in STRATEGIES:
        g = raw[raw["strategy"] == strategy]
        frac = [(g["tests_to_first_resistant"] <= k).mean() for k in ks]
        ax.plot(ks, frac, marker="o", ms=3, label=strategy,
                color=colors.get(strategy),
                ls="--" if strategy == "Random" else "-")
    ax.set_xlabel("Number of alloys tested")
    ax.set_ylabel("Fraction of runs with a\nresistant alloy found")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    if save is not None:
        save(fig, f"fixed_environment_success_{MODE}", out_dir=str(OUT))
    else:
        fig.savefig(OUT / f"fixed_environment_success_{MODE}.png",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
except Exception as exc:                      # figure is optional
    print(f"Figure skipped: {exc}")
