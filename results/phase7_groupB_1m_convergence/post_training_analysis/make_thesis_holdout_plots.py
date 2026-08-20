from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(
    r"C:\Users\conor\OneDrive\Documents\Edinburgh master"
    r"\Dissertation\Python\mbt_gym\results"
    r"\phase7_groupB_1m_convergence\post_training_analysis"
)

GROUP_FILE = ROOT / "phase7_unseen_holdout_group_summary.csv"
CONTRAST_FILE = ROOT / "phase7_unseen_holdout_paired_contrasts.csv"
OUTPUT_DIR = ROOT / "thesis_plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not GROUP_FILE.exists():
    raise FileNotFoundError(f"Missing file: {GROUP_FILE}")

if not CONTRAST_FILE.exists():
    raise FileNotFoundError(f"Missing file: {CONTRAST_FILE}")

groups = pd.read_csv(GROUP_FILE)
contrasts = pd.read_csv(CONTRAST_FILE)

print("Group-summary columns:")
print(groups.columns.tolist())

print("\nContrast columns:")
print(contrasts.columns.tolist())


def find_column(frame, candidates, required=True):
    lookup = {str(column).lower(): column for column in frame.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    if required:
        raise KeyError(
            f"Could not find any of {candidates}. "
            f"Available columns: {frame.columns.tolist()}"
        )

    return None


def clean_policy_label(value):
    text = str(value).strip()
    key = text.lower().replace(" ", "_").replace("-", "_")

    replacements = {
        "hamilton_ppo_aggregate": "Hamilton PPO",
        "ppo_aggregate": "Hamilton PPO",
        "hamilton_ppo": "Hamilton PPO",
        "ppo": "Hamilton PPO",
        "analytical_oracle": "Analytical oracle",
        "oracle": "Analytical oracle",
        "belief_weighted": "Belief-weighted",
        "belief_weighted_policy": "Belief-weighted",
        "frozen_clone": "Frozen clone",
        "frozen_supervised_clone": "Frozen clone",
    }

    return replacements.get(key, text.replace("_", " ").title())


sns.set_theme(
    context="paper",
    style="whitegrid",
    font_scale=1.15,
)

# ============================================================
# Plot 1: mean objective on unseen holdout
# ============================================================

policy_col = find_column(
    groups,
    ["policy", "policy_name", "group", "agent", "method"],
)

mean_col = find_column(
    groups,
    [
        "mean_full_objective",
        "full_objective_mean",
        "mean_objective",
        "objective_mean",
        "mean",
    ],
)

se_col = find_column(
    groups,
    [
        "se",
        "full_objective_se",
        "objective_se",
        "mean_full_objective_se",
        "standard_error",
    ],
    required=False,
)

ci_lo_col = find_column(
    groups,
    [
        "ci_lo",
        "normal_ci_lo",
        "confidence_interval_lo",
        "full_objective_ci_lo",
    ],
    required=False,
)

ci_hi_col = find_column(
    groups,
    [
        "ci_hi",
        "normal_ci_hi",
        "confidence_interval_hi",
        "full_objective_ci_hi",
    ],
    required=False,
)

plot_groups = groups.copy()
plot_groups["display_policy"] = plot_groups[policy_col].map(
    clean_policy_label
)

plot_groups = (
    plot_groups
    .dropna(subset=[mean_col])
    .drop_duplicates(subset=["display_policy"], keep="first")
    .reset_index(drop=True)
)

preferred_order = [
    "Hamilton PPO",
    "Analytical oracle",
    "Belief-weighted",
    "Frozen clone",
]

observed = plot_groups["display_policy"].tolist()

order = [label for label in preferred_order if label in observed]
order += [label for label in observed if label not in order]

plot_groups["display_policy"] = pd.Categorical(
    plot_groups["display_policy"],
    categories=order,
    ordered=True,
)

plot_groups = plot_groups.sort_values("display_policy")

fig, ax = plt.subplots(figsize=(8.5, 5.2))

sns.barplot(
    data=plot_groups,
    x="display_policy",
    y=mean_col,
    order=order,
    errorbar=None,
    ax=ax,
)

means = plot_groups[mean_col].to_numpy(dtype=float)
x_positions = np.arange(len(plot_groups))

if ci_lo_col and ci_hi_col:
    lower = means - plot_groups[ci_lo_col].to_numpy(dtype=float)
    upper = plot_groups[ci_hi_col].to_numpy(dtype=float) - means
    yerr = np.vstack([lower, upper])

elif se_col:
    yerr = 1.96 * plot_groups[se_col].to_numpy(dtype=float)

else:
    yerr = None

if yerr is not None:
    ax.errorbar(
        x_positions,
        means,
        yerr=yerr,
        fmt="none",
        capsize=5,
        linewidth=1.4,
        zorder=10,
    )

for x_position, value in zip(x_positions, means):
    ax.text(
        x_position,
        value,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.set_title("Performance on 500 unseen paired market paths")
ax.set_xlabel("")
ax.set_ylabel("Mean objective")
ax.tick_params(axis="x", rotation=12)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "phase7_unseen_holdout_objective_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_DIR / "phase7_unseen_holdout_objective_comparison.pdf",
    bbox_inches="tight",
)

plt.close(fig)

# ============================================================
# Plot 2: paired-difference forest plot
# ============================================================

contrast_col = find_column(
    contrasts,
    [
        "contrast",
        "comparison",
        "benchmark",
        "comparator",
        "contrast_name",
    ],
)

diff_col = find_column(
    contrasts,
    [
        "mean_diff",
        "paired_mean_diff",
        "mean_difference",
        "difference_mean",
    ],
)

bootstrap_lo_col = find_column(
    contrasts,
    [
        "bootstrap_ci_lo",
        "bootstrap_95_ci_lo",
        "bootstrap_lower",
    ],
    required=False,
)

bootstrap_hi_col = find_column(
    contrasts,
    [
        "bootstrap_ci_hi",
        "bootstrap_95_ci_hi",
        "bootstrap_upper",
    ],
    required=False,
)

normal_lo_col = find_column(
    contrasts,
    [
        "ci_lo",
        "normal_ci_lo",
        "paired_ci_lo",
        "confidence_interval_lo",
    ],
    required=False,
)

normal_hi_col = find_column(
    contrasts,
    [
        "ci_hi",
        "normal_ci_hi",
        "paired_ci_hi",
        "confidence_interval_hi",
    ],
    required=False,
)

lo_col = bootstrap_lo_col or normal_lo_col
hi_col = bootstrap_hi_col or normal_hi_col

if lo_col is None or hi_col is None:
    raise KeyError(
        "Could not identify confidence-interval columns in the "
        "paired-contrast CSV."
    )

plot_contrasts = contrasts.copy()

seed_col = find_column(
    plot_contrasts,
    ["learner_seed", "seed", "ppo_seed"],
    required=False,
)

aggregation_col = find_column(
    plot_contrasts,
    ["aggregation", "scope", "level"],
    required=False,
)

if aggregation_col:
    aggregate_mask = (
        plot_contrasts[aggregation_col]
        .astype(str)
        .str.lower()
        .str.contains("aggregate|pooled|group|all")
    )

    if aggregate_mask.any():
        plot_contrasts = plot_contrasts[aggregate_mask].copy()

elif seed_col:
    aggregate_mask = (
        plot_contrasts[seed_col].isna()
        | plot_contrasts[seed_col]
        .astype(str)
        .str.lower()
        .isin(["aggregate", "pooled", "group", "all"])
    )

    if aggregate_mask.any():
        plot_contrasts = plot_contrasts[aggregate_mask].copy()

plot_contrasts = (
    plot_contrasts
    .dropna(subset=[diff_col, lo_col, hi_col])
    .drop_duplicates(subset=[contrast_col], keep="first")
    .reset_index(drop=True)
)

plot_contrasts["display_contrast"] = (
    plot_contrasts[contrast_col]
    .astype(str)
    .str.replace("_", " ", regex=False)
)

plot_contrasts = plot_contrasts.iloc[::-1].reset_index(drop=True)

differences = plot_contrasts[diff_col].to_numpy(dtype=float)
lower = plot_contrasts[lo_col].to_numpy(dtype=float)
upper = plot_contrasts[hi_col].to_numpy(dtype=float)

left_error = differences - lower
right_error = upper - differences
y_positions = np.arange(len(plot_contrasts))

fig, ax = plt.subplots(figsize=(8.5, 4.8))

sns.scatterplot(
    x=differences,
    y=y_positions,
    s=80,
    ax=ax,
    zorder=5,
)

ax.errorbar(
    differences,
    y_positions,
    xerr=np.vstack([left_error, right_error]),
    fmt="none",
    capsize=5,
    linewidth=1.5,
    zorder=4,
)

ax.axvline(0, linestyle="--", linewidth=1.2)

ax.set_yticks(y_positions)
ax.set_yticklabels(plot_contrasts["display_contrast"])
ax.set_xlabel("Paired objective difference")
ax.set_ylabel("")
ax.set_title("Paired differences on unseen market paths")

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "phase7_unseen_holdout_paired_differences.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_DIR / "phase7_unseen_holdout_paired_differences.pdf",
    bbox_inches="tight",
)

plt.close(fig)

# ============================================================
# LaTeX tables
# ============================================================

group_table_columns = [policy_col, mean_col]

if se_col:
    group_table_columns.append(se_col)

group_table = groups[group_table_columns].copy()
group_table[policy_col] = group_table[policy_col].map(clean_policy_label)

group_latex = group_table.to_latex(
    index=False,
    float_format=lambda value: f"{value:.2f}",
    caption=(
        "Performance on 500 previously unseen paired market paths."
    ),
    label="tab:phase7-unseen-holdout",
    escape=True,
    column_format="l" + "c" * (len(group_table_columns) - 1),
)

(
    OUTPUT_DIR / "phase7_unseen_holdout_table.tex"
).write_text(group_latex, encoding="utf-8")

contrast_table = plot_contrasts[
    [contrast_col, diff_col, lo_col, hi_col]
].copy()

contrast_latex = contrast_table.to_latex(
    index=False,
    float_format=lambda value: f"{value:.2f}",
    caption=(
        "Paired objective differences on the unseen holdout paths. "
        "Intervals are 95 percent confidence intervals."
    ),
    label="tab:phase7-unseen-holdout-contrasts",
    escape=True,
    column_format="lccc",
)

(
    OUTPUT_DIR / "phase7_unseen_holdout_contrasts.tex"
).write_text(contrast_latex, encoding="utf-8")

print("\nCreated files:")

for path in sorted(OUTPUT_DIR.iterdir()):
    print(path)
