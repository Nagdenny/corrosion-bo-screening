import numpy as np
import pandas as pd
from pathlib import Path

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
# SETTINGS
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]  # repository root
RESULTS = PROJECT / "results" / "active_classification"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
STRATEGIES = ["EI", "PI", "UCB", "Random"]

CLASS_NAMES = {
    0: "Severe",
    1: "Poor",
    2: "Good",
    3: "Resistant"
}

# Actual initial training-set composition used in the experiment
INITIAL_COUNTS = {
    0: 45,
    1: 45,
    2: 5,
    3: 5
}


# ============================================================
# LOAD RESULTS
# ============================================================

final_list = []
history_list = []
selected_list = []

for seed in SEEDS:

    final_file = RESULTS / f"seed{seed}_active_final.csv"
    history_file = RESULTS / f"seed{seed}_active_history.csv"
    selected_file = RESULTS / f"seed{seed}_selected_samples.csv"

    for f in [final_file, history_file, selected_file]:
        if not f.exists():
            raise FileNotFoundError(f"Missing file: {f}")

    final_list.append(pd.read_csv(final_file))
    history_list.append(pd.read_csv(history_file))
    selected_list.append(pd.read_csv(selected_file))


final_df = pd.concat(final_list, ignore_index=True)
history_df = pd.concat(history_list, ignore_index=True)
selected_df = pd.concat(selected_list, ignore_index=True)

print("=" * 75)
print("FILES LOADED SUCCESSFULLY")
print("=" * 75)
print(f"Final-result rows: {len(final_df)}")
print(f"History rows: {len(history_df)}")
print(f"Selected-sample rows: {len(selected_df)}")


# ============================================================
# VALIDATION
# ============================================================

expected_combinations = len(SEEDS) * len(STRATEGIES)

if len(final_df) != expected_combinations:
    raise ValueError(
        f"Expected {expected_combinations} final rows, "
        f"but found {len(final_df)}."
    )

for seed in SEEDS:
    found = set(
        final_df.loc[final_df["seed"] == seed, "strategy"]
    )
    missing = set(STRATEGIES) - found

    if missing:
        raise ValueError(
            f"Seed {seed} missing strategies: {missing}"
        )

print("All seeds and strategies validated.")


# ============================================================
# SAVE COMBINED RAW DATA
# ============================================================

final_df.to_csv(
    RESULTS / "multiseed_active_all_final.csv",
    index=False
)

history_df.to_csv(
    RESULTS / "multiseed_active_all_history.csv",
    index=False
)

selected_df.to_csv(
    RESULTS / "multiseed_active_all_selected_samples.csv",
    index=False
)


# ============================================================
# FINAL MULTI-SEED SUMMARY
# ============================================================

summary = (
    final_df
    .groupby("strategy")
    .agg(
        resistant_mean=("resistant_discovered", "mean"),
        resistant_std=("resistant_discovered", "std"),

        rcr_mean=("resistant_class_ratio", "mean"),
        rcr_std=("resistant_class_ratio", "std"),

        discovery_rate_mean=("resistant_discovery_rate", "mean"),
        discovery_rate_std=("resistant_discovery_rate", "std"),

        accuracy_mean=("test_accuracy", "mean"),
        accuracy_std=("test_accuracy", "std"),

        macro_f1_mean=("test_macro_f1", "mean"),
        macro_f1_std=("test_macro_f1", "std"),

        runtime_mean=("runtime_seconds", "mean"),
        runtime_std=("runtime_seconds", "std")
    )
    .reset_index()
)


# Rank primarily by targeted Resistant discovery
summary = summary.sort_values(
    "resistant_mean",
    ascending=False
).reset_index(drop=True)

summary.insert(
    0,
    "Rank",
    np.arange(1, len(summary) + 1)
)


# Random baseline
random_resistant = float(
    summary.loc[
        summary["strategy"] == "Random",
        "resistant_mean"
    ].iloc[0]
)

random_rate = float(
    summary.loc[
        summary["strategy"] == "Random",
        "discovery_rate_mean"
    ].iloc[0]
)

summary["enrichment_vs_random"] = (
    summary["resistant_mean"] / random_resistant
)

summary["discovery_rate_gain_vs_random"] = (
    summary["discovery_rate_mean"] - random_rate
)


# ============================================================
# FORMATTED PUBLICATION TABLE
# ============================================================

publication = pd.DataFrame({
    "Rank": summary["Rank"],
    "Strategy": summary["strategy"],

    "Resistant_discovered_mean±std":
        summary.apply(
            lambda r:
            f"{r['resistant_mean']:.1f} ± {r['resistant_std']:.1f}",
            axis=1
        ),

    "Discovery_rate_mean±std":
        summary.apply(
            lambda r:
            f"{r['discovery_rate_mean']:.3f} ± "
            f"{r['discovery_rate_std']:.3f}",
            axis=1
        ),

    "RCR_mean±std":
        summary.apply(
            lambda r:
            f"{r['rcr_mean']:.3f} ± {r['rcr_std']:.3f}",
            axis=1
        ),

    "Accuracy_mean±std":
        summary.apply(
            lambda r:
            f"{r['accuracy_mean']:.3f} ± "
            f"{r['accuracy_std']:.3f}",
            axis=1
        ),

    "MacroF1_mean±std":
        summary.apply(
            lambda r:
            f"{r['macro_f1_mean']:.3f} ± "
            f"{r['macro_f1_std']:.3f}",
            axis=1
        ),

    "Enrichment_vs_Random":
        summary["enrichment_vs_random"].round(3),

    "Runtime_mean_s":
        summary["runtime_mean"].round(2)
})


summary.to_csv(
    RESULTS / "multiseed_active_summary.csv",
    index=False
)

publication.to_csv(
    RESULTS / "multiseed_active_publication_table.csv",
    index=False
)


print("\n" + "=" * 75)
print("PUBLICATION SUMMARY")
print("=" * 75)
print(publication.to_string(index=False))


# ============================================================
# ITERATION SUMMARY
# ============================================================

iteration_summary = (
    history_df
    .groupby(["strategy", "iteration"])
    .agg(
        train_size_mean=("train_size", "mean"),

        resistant_mean=("resistant_discovered", "mean"),
        resistant_std=("resistant_discovered", "std"),

        rcr_mean=("resistant_class_ratio", "mean"),
        rcr_std=("resistant_class_ratio", "std"),

        discovery_rate_mean=("resistant_discovery_rate", "mean"),
        discovery_rate_std=("resistant_discovery_rate", "std"),

        accuracy_mean=("test_accuracy", "mean"),
        accuracy_std=("test_accuracy", "std"),

        macro_f1_mean=("test_macro_f1", "mean"),
        macro_f1_std=("test_macro_f1", "std")
    )
    .reset_index()
)

iteration_summary.to_csv(
    RESULTS / "multiseed_active_iteration_summary.csv",
    index=False
)


# ============================================================
# GENERIC LEARNING-CURVE FUNCTION
# ============================================================

def plot_curve(
    mean_col,
    std_col,
    ylabel,
    title,
    filename
):

    plt.figure(figsize=(12, 8))

    for strategy in STRATEGIES:

        d = iteration_summary[
            iteration_summary["strategy"] == strategy
        ].sort_values("iteration")

        x = d["iteration"].to_numpy()
        mean = d[mean_col].to_numpy()
        std = d[std_col].fillna(0).to_numpy()

        plt.plot(
            x,
            mean,
            linewidth=2,
            label=strategy
        )

        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.12
        )

    plt.xlabel("Acquisition Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURES / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN EVOLUTION FIGURES
# ============================================================

plot_curve(
    "resistant_mean",
    "resistant_std",
    "Cumulative Resistant Samples Discovered",
    "Resistant Discovery During Active Acquisition",
    "resistant_discovery_vs_iteration.png"
)

plot_curve(
    "rcr_mean",
    "rcr_std",
    "Resistant Class Ratio",
    "Resistant Class Ratio During Active Acquisition",
    "resistant_class_ratio_vs_iteration.png"
)

plot_curve(
    "discovery_rate_mean",
    "discovery_rate_std",
    "Resistant Discovery Rate",
    "Resistant Discovery Rate During Active Acquisition",
    "resistant_discovery_rate_vs_iteration.png"
)

plot_curve(
    "accuracy_mean",
    "accuracy_std",
    "Test Accuracy",
    "Classification Accuracy During Active Acquisition",
    "test_accuracy_vs_iteration.png"
)

plot_curve(
    "macro_f1_mean",
    "macro_f1_std",
    "Macro-F1 Score",
    "Macro-F1 During Active Acquisition",
    "macro_f1_vs_iteration.png"
)


# ============================================================
# FINAL COMPARISON PLOTS
# ============================================================

plot_order = summary["strategy"].tolist()


def final_bar(
    mean_col,
    std_col,
    ylabel,
    title,
    filename
):

    d = summary.set_index("strategy").loc[plot_order]

    plt.figure(figsize=(12, 8))

    plt.bar(
        plot_order,
        d[mean_col],
        yerr=d[std_col],
        capsize=5
    )

    plt.xlabel("Acquisition Strategy")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIGURES / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


final_bar(
    "resistant_mean",
    "resistant_std",
    "Resistant Samples Discovered",
    "Final Resistant Discovery Across Five Random Seeds",
    "final_resistant_discovery_comparison.png"
)

final_bar(
    "rcr_mean",
    "rcr_std",
    "Resistant Class Ratio",
    "Final Resistant Class Ratio Across Five Random Seeds",
    "final_RCR_comparison.png"
)

final_bar(
    "accuracy_mean",
    "accuracy_std",
    "Test Accuracy",
    "Final Classification Accuracy Across Five Random Seeds",
    "final_accuracy_comparison.png"
)

final_bar(
    "macro_f1_mean",
    "macro_f1_std",
    "Macro-F1 Score",
    "Final Macro-F1 Across Five Random Seeds",
    "final_macroF1_comparison.png"
)


# ============================================================
# RECONSTRUCT TRAINING-SET CLASS DISTRIBUTION
# ============================================================

records = []

for seed in SEEDS:

    for strategy in STRATEGIES:

        selected = selected_df[
            (selected_df["seed"] == seed) &
            (selected_df["strategy"] == strategy)
        ].copy()

        counts = INITIAL_COUNTS.copy()

        # Initial training set, iteration 0
        for cls in range(4):

            records.append({
                "seed": seed,
                "strategy": strategy,
                "iteration": 0,
                "class_code": cls,
                "class_name": CLASS_NAMES[cls],
                "count": counts[cls]
            })

        max_iteration = int(
            selected["iteration"].max()
        )

        for iteration in range(1, max_iteration + 1):

            acquired = selected[
                selected["iteration"] == iteration
            ]

            for cls in range(4):

                counts[cls] += int(
                    (
                        acquired["true_class"] == cls
                    ).sum()
                )

            for cls in range(4):

                records.append({
                    "seed": seed,
                    "strategy": strategy,
                    "iteration": iteration,
                    "class_code": cls,
                    "class_name": CLASS_NAMES[cls],
                    "count": counts[cls]
                })


class_df = pd.DataFrame(records)

class_df.to_csv(
    RESULTS / "multiseed_class_distribution_raw.csv",
    index=False
)


class_summary = (
    class_df
    .groupby(
        [
            "strategy",
            "iteration",
            "class_code",
            "class_name"
        ]
    )
    .agg(
        count_mean=("count", "mean"),
        count_std=("count", "std")
    )
    .reset_index()
)

class_summary.to_csv(
    RESULTS / "multiseed_class_distribution_summary.csv",
    index=False
)


# ============================================================
# CLASS DISTRIBUTION IN TRAINING SET OVER ITERATIONS
# ============================================================

class_order = [
    "Severe",
    "Poor",
    "Good",
    "Resistant"
]

for strategy in STRATEGIES:

    d = class_summary[
        class_summary["strategy"] == strategy
    ]

    pivot = d.pivot(
        index="iteration",
        columns="class_name",
        values="count_mean"
    )

    pivot = pivot[class_order]

    plt.figure(figsize=(12, 8))

    plt.stackplot(
        pivot.index,
        *[
            pivot[c].to_numpy()
            for c in class_order
        ],
        labels=class_order,
        alpha=0.85
    )

    plt.xlabel("Acquisition Iteration")
    plt.ylabel("Mean Number of Samples")
    plt.title(
        f"Class Distribution in Training Set Over Iterations ({strategy})"
    )

    plt.legend(loc="upper left")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIGURES / f"class_distribution_{strategy}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# CLASS RATIO EVOLUTION
# ============================================================

for strategy in STRATEGIES:

    d = class_summary[
        class_summary["strategy"] == strategy
    ].copy()

    pivot = d.pivot(
        index="iteration",
        columns="class_name",
        values="count_mean"
    )

    pivot = pivot[class_order]

    ratio = pivot.div(
        pivot.sum(axis=1),
        axis=0
    )

    plt.figure(figsize=(12, 8))

    for class_name in class_order:

        plt.plot(
            ratio.index,
            ratio[class_name],
            linewidth=2,
            label=class_name
        )

    plt.xlabel("Acquisition Iteration")
    plt.ylabel("Mean Class Fraction")
    plt.title(
        f"Class Ratio in Training Set Over Iterations ({strategy})"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURES / f"class_ratio_{strategy}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SANITY CHECK: RECONSTRUCTED FINAL RESISTANT COUNTS
# ============================================================

print("\n" + "=" * 75)
print("CLASS-DISTRIBUTION RECONSTRUCTION CHECK")
print("=" * 75)

for seed in SEEDS:

    for strategy in STRATEGIES:

        reconstructed = class_df[
            (class_df["seed"] == seed) &
            (class_df["strategy"] == strategy) &
            (class_df["iteration"] == 100) &
            (class_df["class_code"] == 3)
        ]["count"].iloc[0]

        expected = final_df[
            (final_df["seed"] == seed) &
            (final_df["strategy"] == strategy)
        ]["resistant_discovered"].iloc[0] + INITIAL_COUNTS[3]

        if reconstructed != expected:

            raise ValueError(
                f"Mismatch for seed {seed}, {strategy}: "
                f"reconstructed={reconstructed}, "
                f"expected={expected}"
            )

print(
    "Reconstructed class distributions agree "
    "with the final Resistant counts."
)


# ============================================================
# PRINT ENRICHMENT
# ============================================================

print("\n" + "=" * 75)
print("TARGETED DISCOVERY RELATIVE TO RANDOM")
print("=" * 75)

for _, row in summary.iterrows():

    print(
        f"{row['strategy']:7s} | "
        f"Resistant = "
        f"{row['resistant_mean']:.1f} ± "
        f"{row['resistant_std']:.1f} | "
        f"RCR = "
        f"{row['rcr_mean']:.3f} ± "
        f"{row['rcr_std']:.3f} | "
        f"Enrichment = "
        f"{row['enrichment_vs_random']:.2f}x"
    )


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)

print(
    "\nPublication table:\n",
    RESULTS / "multiseed_active_publication_table.csv"
)

print(
    "\nFigures:\n",
    FIGURES
)