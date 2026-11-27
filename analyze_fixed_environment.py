# ============================================================
# ANALYSIS OF THE FIXED-ENVIRONMENT EXPERIMENT (Section 3.5)
#
# Reads  results/fixed_environment/fixed_env_runs_{transfer,cold}.csv
# Writes results/fixed_environment/analysis/
#   - table7_fixed_environment.csv     (Table 7)
#   - wilcoxon_vs_random.csv           (paired tests vs Random)
#   - breakdown_by_resistant_share.csv (where BO helps most)
#   - Figure10_fixed_environment.[svg|pdf|tiff|png]
# Run from ~/corrosion_project:  python scripts/analyze_fixed_environment.py
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RES = Path("results/fixed_environment")
OUT = RES / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
STRATEGIES = ["EI", "PI", "UCB", "Greedy", "Random"]
MODES = {"transfer": "With knowledge transfer", "cold": "Without knowledge transfer"}

runs = {m: pd.read_csv(RES / f"fixed_env_runs_{m}.csv") for m in MODES}

# ---------------- Table 7 ----------------
rows = []
for mode, df in runs.items():
    for s in STRATEGIES:
        g = df[df["strategy"] == s]
        t = g["tests_to_first_resistant"]
        rows.append({
            "mode": mode, "strategy": s,
            "tests_mean": t.mean(), "tests_std": t.std(),
            "found_within_3_pct": 100 * (t <= 3).mean(),
            "found_within_5_pct": 100 * (t <= 5).mean(),
            "resistant_in_first_10": g["resistant_found_within_10"].mean(),
        })
table7 = pd.DataFrame(rows).round(2)
table7.to_csv(OUT / "table7_fixed_environment.csv", index=False)
print("TABLE 7")
print(table7.to_string(index=False))

# ---------------- paired Wilcoxon tests vs Random ----------------
# One value per environment (mean over seeds), paired by environment.
tests = []
for mode, df in runs.items():
    per_env = (df.groupby(["env_id", "strategy"])["tests_to_first_resistant"]
                 .mean().unstack())
    for s in [x for x in STRATEGIES if x != "Random"]:
        diff = per_env[s] - per_env["Random"]
        stat, p = wilcoxon(per_env[s], per_env["Random"],
                           zero_method="wilcox", alternative="two-sided")
        tests.append({
            "mode": mode, "strategy": s, "n_environments": len(diff),
            "mean_tests": per_env[s].mean(),
            "mean_tests_random": per_env["Random"].mean(),
            "reduction_pct": 100 * (1 - per_env[s].mean() / per_env["Random"].mean()),
            "median_difference": diff.median(),
            "environments_better": int((diff < 0).sum()),
            "environments_worse": int((diff > 0).sum()),
            "wilcoxon_p": p,
        })
tests = pd.DataFrame(tests)
tests.to_csv(OUT / "wilcoxon_vs_random.csv", index=False)
print("\nPAIRED WILCOXON TESTS VS RANDOM (per environment, mean over seeds)")
print(tests.round({"mean_tests": 2, "mean_tests_random": 2, "reduction_pct": 1,
                   "median_difference": 2}).to_string(index=False))

# ---------------- breakdown by share of resistant alloys ----------------
df = runs["transfer"].copy()
df["resistant_share"] = df["n_resistant"] / df["n_candidates"]
df["share_group"] = pd.cut(df["resistant_share"], [0, 0.25, 0.5, 1.0],
                           labels=["<= 25%", "25-50%", "> 50%"])
brk = (df.groupby(["share_group", "strategy"], observed=True)
         ["tests_to_first_resistant"].mean().unstack()[STRATEGIES])
brk["n_environments"] = (df.groupby("share_group", observed=True)["env_id"]
                           .nunique())
brk = brk.round(2)
brk.to_csv(OUT / "breakdown_by_resistant_share.csv")
print("\nMEAN TESTS TO FIRST RESISTANT ALLOY BY SHARE OF RESISTANT ALLOYS (transfer)")
print(brk.to_string())

# ---------------- Figure 10 ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from corrosion_figstyle import COLORS, MM, WIDTH_DOUBLE, apply_style, panel_label, save
    apply_style()
    colors = {s: COLORS[s] for s in STRATEGIES}
    size = (WIDTH_DOUBLE, 66 * MM)
except ImportError:
    colors, size, save, panel_label = {}, (9, 3.5), None, None

ks = np.arange(1, 11)
fig, axes = plt.subplots(1, 2, figsize=size, sharey=True,
                         gridspec_kw={"wspace": 0.08})
for ax, (mode, title), letter in zip(axes, MODES.items(), "ab"):
    d = runs[mode]
    for s in STRATEGIES:
        t = d[d["strategy"] == s]["tests_to_first_resistant"]
        ax.plot(ks, [(t <= k).mean() for k in ks], marker="o", ms=3,
                label=s, color=colors.get(s), ls="--" if s == "Random" else "-")
    ax.set_title(title, pad=4)
    ax.set_xlabel("Number of alloys tested")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.02)
    if panel_label:
        panel_label(ax, letter, x=-0.12 if letter == "a" else -0.04)
axes[0].set_ylabel("Fraction of runs with a\nresistant alloy found")
axes[1].legend(loc="lower right")
if save is not None:
    save(fig, "Figure10_fixed_environment", out_dir=str(OUT), multipanel=True)
else:
    fig.savefig(OUT / "Figure10_fixed_environment.png", dpi=300, bbox_inches="tight")
print(f"\nSaved results in {OUT}")