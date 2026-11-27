# ============================================================
# ANALYSIS OF MULTI-SEED BO EXPERIMENT
#
# Compares:
# RF-EI, RF-PI, RF-UCB
# GP-EI, GP-PI, GP-UCB
#
# Random seeds: 0, 1, 2, 3, 4
# ============================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
# ------------------------------------------------------------
# FIGURE FONT SIZES (revision: larger axis labels, ticks, legends)
# ------------------------------------------------------------
plt.rcParams.update({
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 18,
})
 
 
# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]  # repository root
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]

EXPECTED_COMBINATIONS = [
    ("RF", "EI"),
    ("RF", "PI"),
    ("RF", "UCB"),
    ("GP", "EI"),
    ("GP", "PI"),
    ("GP", "UCB"),
]


# ============================================================
# 2. CHECK THAT ALL FIVE SEEDS FINISHED
# ============================================================

print("=" * 75)
print("MULTI-SEED BAYESIAN OPTIMIZATION ANALYSIS")
print("=" * 75)

missing_files = []

for seed in SEEDS:

    file_path = RESULTS_DIR / f"seed{seed}_final_performance.csv"

    if not file_path.exists():
        missing_files.append(str(file_path))


if missing_files:

    print("\nERROR: The following result files are missing:\n")

    for f in missing_files:
        print(f"  {f}")

    raise SystemExit(
        "\nNot all seed calculations have finished. "
        "Run this script again after all five result files exist."
    )


print("\nAll five seed result files found.")


# ============================================================
# 3. LOAD FINAL PERFORMANCE FROM ALL SEEDS
# ============================================================

final_frames = []

for seed in SEEDS:

    file_path = RESULTS_DIR / f"seed{seed}_final_performance.csv"

    df = pd.read_csv(file_path)

    # Ensure correct seed value
    df["seed"] = seed

    final_frames.append(df)


all_final = pd.concat(
    final_frames,
    ignore_index=True
)


# ============================================================
# 4. VALIDATE COMBINATIONS
# ============================================================

print("\nChecking surrogate-acquisition combinations...")

for seed in SEEDS:

    seed_df = all_final[
        all_final["seed"] == seed
    ]

    observed = set(
        zip(
            seed_df["surrogate"],
            seed_df["acquisition"]
        )
    )

    expected = set(EXPECTED_COMBINATIONS)

    missing = expected - observed

    if missing:

        raise ValueError(
            f"Seed {seed} is missing combinations: {missing}"
        )

print("All six combinations are present for every seed.")


# ============================================================
# 5. SAVE ALL RAW FINAL RESULTS
# ============================================================

raw_output = RESULTS_DIR / "multiseed_all_final_results.csv"

all_final.to_csv(
    raw_output,
    index=False
)


# ============================================================
# 6. CALCULATE MEAN +/- STANDARD DEVIATION
# ============================================================

summary = (
    all_final
    .groupby(
        ["surrogate", "acquisition"]
    )
    .agg(
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),

        MSE_mean=("MSE", "mean"),
        MSE_std=("MSE", "std"),

        uncertainty_mean=("mean_uncertainty", "mean"),
        uncertainty_std=("mean_uncertainty", "std"),

        runtime_mean_seconds=(
            "total_combination_seconds",
            "mean"
        ),

        runtime_std_seconds=(
            "total_combination_seconds",
            "std"
        ),
    )
    .reset_index()
)


# ============================================================
# 7. RANK COMBINATIONS
# ============================================================

summary = summary.sort_values(
    by=["R2_mean", "MSE_mean"],
    ascending=[False, True]
).reset_index(drop=True)

summary.insert(
    0,
    "Rank",
    np.arange(1, len(summary) + 1)
)


# ============================================================
# 8. CREATE FORMATTED COLUMNS FOR PAPER
# ============================================================

summary["R2_mean±std"] = summary.apply(
    lambda row:
    f"{row['R2_mean']:.4f} ± {row['R2_std']:.4f}",
    axis=1
)

summary["MSE_mean±std"] = summary.apply(
    lambda row:
    f"{row['MSE_mean']:.4f} ± {row['MSE_std']:.4f}",
    axis=1
)

summary["Runtime_mean_s"] = summary[
    "runtime_mean_seconds"
].round(2)


# ============================================================
# 9. SAVE SUMMARY
# ============================================================

summary_output = (
    RESULTS_DIR /
    "multiseed_BO_summary.csv"
)

summary.to_csv(
    summary_output,
    index=False
)


publication_table = summary[
    [
        "Rank",
        "surrogate",
        "acquisition",
        "R2_mean±std",
        "MSE_mean±std",
        "Runtime_mean_s",
    ]
].copy()

publication_output = (
    RESULTS_DIR /
    "multiseed_publication_table.csv"
)

publication_table.to_csv(
    publication_output,
    index=False
)


# ============================================================
# 10. PRINT FINAL TABLE
# ============================================================

print("\n" + "=" * 75)
print("FINAL PERFORMANCE ACROSS FIVE RANDOM SEEDS")
print("=" * 75)

print(
    publication_table.to_string(
        index=False
    )
)


# ============================================================
# 11. R2 BAR PLOT
# ============================================================

plot_df = summary.copy()

plot_df["Combination"] = (
    plot_df["surrogate"]
    + "-"
    + plot_df["acquisition"]
)

plt.figure(figsize=(12, 8))

plt.bar(
    plot_df["Combination"],
    plot_df["R2_mean"],
    yerr=plot_df["R2_std"],
    capsize=5
)

plt.xlabel("Surrogate-Acquisition Combination")
plt.ylabel(r"Mean $R^2$")
plt.title(
    r"Final $R^2$ Across Five Random Seeds"
)

plt.xticks(rotation=45)

plt.tight_layout()

r2_figure = (
    FIGURES_DIR /
    "multiseed_R2_comparison.png"
)

plt.savefig(
    r2_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. MSE BAR PLOT
# ============================================================

plt.figure(figsize=(12, 8))

plt.bar(
    plot_df["Combination"],
    plot_df["MSE_mean"],
    yerr=plot_df["MSE_std"],
    capsize=5
)

plt.xlabel("Surrogate-Acquisition Combination")
plt.ylabel("Mean MSE")
plt.title(
    "Final MSE Across Five Random Seeds"
)

plt.xticks(rotation=45)

plt.tight_layout()

mse_figure = (
    FIGURES_DIR /
    "multiseed_MSE_comparison.png"
)

plt.savefig(
    mse_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. LOAD ITERATION HISTORIES
# ============================================================

history_frames = []

for seed in SEEDS:

    history_file = (
        RESULTS_DIR /
        f"seed{seed}_all_combinations.csv"
    )

    if not history_file.exists():

        print(
            f"\nWARNING: Missing history file: "
            f"{history_file}"
        )

        continue

    history = pd.read_csv(
        history_file
    )

    history["seed"] = seed

    history_frames.append(
        history
    )


# ============================================================
# 14. LEARNING-CURVE ANALYSIS
# ============================================================

if history_frames:

    histories = pd.concat(
        history_frames,
        ignore_index=True
    )

    history_output = (
        RESULTS_DIR /
        "multiseed_all_iteration_results.csv"
    )

    histories.to_csv(
        history_output,
        index=False
    )

    learning_summary = (
        histories
        .groupby(
            [
                "surrogate",
                "acquisition",
                "train_size",
            ]
        )
        .agg(
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
        )
        .reset_index()
    )

    learning_output = (
        RESULTS_DIR /
        "multiseed_learning_curve_summary.csv"
    )

    learning_summary.to_csv(
        learning_output,
        index=False
    )


    # ========================================================
    # 15. R2 LEARNING CURVES
    # ========================================================

    plt.figure(figsize=(12, 8))

    for surrogate, acquisition in EXPECTED_COMBINATIONS:

        subset = learning_summary[
            (
                learning_summary["surrogate"]
                == surrogate
            )
            &
            (
                learning_summary["acquisition"]
                == acquisition
            )
        ].sort_values("train_size")

        if subset.empty:
            continue

        label = f"{surrogate}-{acquisition}"

        plt.plot(
            subset["train_size"],
            subset["R2_mean"],
            marker="o",
            markersize=3,
            label=label
        )

        plt.fill_between(
            subset["train_size"],
            subset["R2_mean"] - subset["R2_std"],
            subset["R2_mean"] + subset["R2_std"],
            alpha=0.12
        )

    plt.xlabel("Training Set Size")
    plt.ylabel(r"Mean $R^2$")
    plt.title(
        r"BO Learning Curves Across Five Random Seeds"
    )

    plt.legend()

    plt.tight_layout()

    learning_figure = (
        FIGURES_DIR /
        "multiseed_R2_learning_curves.png"
    )

    plt.savefig(
        learning_figure,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # 16. MSE LEARNING CURVES
    # ========================================================

    plt.figure(figsize=(12, 8))

    for surrogate, acquisition in EXPECTED_COMBINATIONS:

        subset = learning_summary[
            (
                learning_summary["surrogate"]
                == surrogate
            )
            &
            (
                learning_summary["acquisition"]
                == acquisition
            )
        ].sort_values("train_size")

        if subset.empty:
            continue

        label = f"{surrogate}-{acquisition}"

        plt.plot(
            subset["train_size"],
            subset["MSE_mean"],
            marker="o",
            markersize=3,
            label=label
        )

        plt.fill_between(
            subset["train_size"],
            subset["MSE_mean"] - subset["MSE_std"],
            subset["MSE_mean"] + subset["MSE_std"],
            alpha=0.12
        )

    plt.xlabel("Training Set Size")
    plt.ylabel("Mean MSE")
    plt.title(
        "BO MSE Evolution Across Five Random Seeds"
    )

    plt.legend()

    plt.tight_layout()

    mse_learning_figure = (
        FIGURES_DIR /
        "multiseed_MSE_learning_curves.png"
    )

    plt.savefig(
        mse_learning_figure,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 17. PRINT BEST COMBINATION
# ============================================================

best = summary.iloc[0]

print("\n" + "=" * 75)
print("BEST COMBINATION BASED ON MEAN FINAL R2")
print("=" * 75)

print(
    f"\nSurrogate:   {best['surrogate']}"
)

print(
    f"Acquisition: {best['acquisition']}"
)

print(
    f"R²:          "
    f"{best['R2_mean']:.4f} ± "
    f"{best['R2_std']:.4f}"
)

print(
    f"MSE:         "
    f"{best['MSE_mean']:.4f} ± "
    f"{best['MSE_std']:.4f}"
)


# ============================================================
# 18. OUTPUT LOCATIONS
# ============================================================

print("\n" + "=" * 75)
print("OUTPUT FILES")
print("=" * 75)

print(f"\nRaw final results:")
print(raw_output)

print(f"\nSummary:")
print(summary_output)

print(f"\nPublication table:")
print(publication_output)

print(f"\nFigures:")
print(FIGURES_DIR)

print("\nAnalysis complete.")
