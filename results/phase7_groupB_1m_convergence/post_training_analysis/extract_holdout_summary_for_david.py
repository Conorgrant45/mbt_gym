from pathlib import Path
import pandas as pd


ROOT = Path(
    r"C:\Users\conor\OneDrive\Documents\Edinburgh master"
    r"\Dissertation\Python\mbt_gym\results"
    r"\phase7_groupB_1m_convergence\post_training_analysis"
)

GROUP_FILE = ROOT / "phase7_unseen_holdout_group_summary.csv"
CONTRAST_FILE = ROOT / "phase7_unseen_holdout_paired_contrasts.csv"

groups = pd.read_csv(GROUP_FILE)
contrasts = pd.read_csv(CONTRAST_FILE)


def clean_group(value):
    names = {
        "analytical_oracle": "Analytical oracle",
        "belief_weighted": "Belief-weighted policy",
        "frozen_clone": "Frozen supervised clone",
        "hamilton_ppo_5_seed_aggregate": "Hamilton PPO (5-seed aggregate)",
        "hamilton_ppo": "Hamilton PPO (5-seed aggregate)",
    }

    text = str(value).strip()
    key = text.lower().replace(" ", "_").replace("-", "_")

    return names.get(key, text.replace("_", " ").title())


def clean_benchmark(value):
    names = {
        "analytical_oracle": "Analytical oracle",
        "oracle": "Analytical oracle",
        "belief_weighted": "Belief-weighted policy",
        "frozen_clone": "Frozen supervised clone",
    }

    text = str(value).strip()
    key = text.lower().replace(" ", "_").replace("-", "_")

    return names.get(key, text.replace("_", " ").title())


# ============================================================
# Overall performance summary
# ============================================================

policy_summary = groups[
    [
        "group",
        "n_seeds",
        "mean_full_objective",
        "cross_seed_sd_full_objective",
        "mean_fills",
        "mean_quoted_spread",
        "mean_abs_inventory",
    ]
].copy()

policy_summary["group"] = policy_summary["group"].map(clean_group)

policy_summary = policy_summary.rename(
    columns={
        "group": "Policy",
        "n_seeds": "Number of seeds",
        "mean_full_objective": "Mean objective",
        "cross_seed_sd_full_objective": "Cross-seed SD",
        "mean_fills": "Mean fills",
        "mean_quoted_spread": "Mean quoted spread",
        "mean_abs_inventory": "Mean absolute inventory",
    }
)

# ============================================================
# Paired statistical comparisons
# ============================================================

paired_summary = contrasts[
    [
        "ppo_group",
        "benchmark",
        "n_paths",
        "mean_paired_diff",
        "paired_sd",
        "se",
        "ci95_lo",
        "ci95_hi",
        "bootstrap_ci95_lo",
        "bootstrap_ci95_hi",
        "win_rate",
        "ci_includes_zero",
        "verdict",
    ]
].copy()

paired_summary["benchmark"] = paired_summary["benchmark"].map(
    clean_benchmark
)

paired_summary = paired_summary.rename(
    columns={
        "ppo_group": "PPO group",
        "benchmark": "Benchmark",
        "n_paths": "Number of paths",
        "mean_paired_diff": "Mean difference",
        "paired_sd": "Paired SD",
        "se": "Standard error",
        "ci95_lo": "95% CI lower",
        "ci95_hi": "95% CI upper",
        "bootstrap_ci95_lo": "Bootstrap CI lower",
        "bootstrap_ci95_hi": "Bootstrap CI upper",
        "win_rate": "PPO win rate",
        "ci_includes_zero": "CI includes zero",
        "verdict": "Verdict",
    }
)

# Convert proportion to percentage where applicable.
if paired_summary["PPO win rate"].max() <= 1:
    paired_summary["PPO win rate"] *= 100

# ============================================================
# Save CSVs
# ============================================================

policy_csv = ROOT / "phase7_david_policy_summary.csv"
paired_csv = ROOT / "phase7_david_paired_summary.csv"

policy_summary.to_csv(policy_csv, index=False)
paired_summary.to_csv(paired_csv, index=False)

# ============================================================
# Save LaTeX tables
# ============================================================

policy_latex = policy_summary.to_latex(
    index=False,
    float_format=lambda x: f"{x:.2f}",
    caption=(
        "Performance on 500 previously unseen paired market paths."
    ),
    label="tab:phase7-unseen-policy-summary",
    column_format="lcccccc",
    escape=True,
)

paired_latex = paired_summary[
    [
        "Benchmark",
        "Mean difference",
        "Standard error",
        "95% CI lower",
        "95% CI upper",
        "Bootstrap CI lower",
        "Bootstrap CI upper",
        "PPO win rate",
        "Verdict",
    ]
].to_latex(
    index=False,
    float_format=lambda x: f"{x:.2f}",
    caption=(
        "Paired differences between aggregate Hamilton PPO and each "
        "benchmark on 500 common unseen market paths. Differences are "
        "defined as Hamilton PPO minus the benchmark."
    ),
    label="tab:phase7-unseen-paired-comparisons",
    column_format="lccccccccl",
    escape=True,
)

policy_tex = ROOT / "phase7_david_policy_summary.tex"
paired_tex = ROOT / "phase7_david_paired_summary.tex"

policy_tex.write_text(policy_latex, encoding="utf-8")
paired_tex.write_text(paired_latex, encoding="utf-8")

# ============================================================
# Print concise meeting output
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")

print("\n" + "=" * 100)
print("OVERALL UNSEEN-HOLDOUT PERFORMANCE")
print("=" * 100)
print(policy_summary.to_string(index=False))

print("\n" + "=" * 100)
print("PAIRED HAMILTON PPO COMPARISONS")
print("Mean difference = Hamilton PPO minus benchmark")
print("=" * 100)
print(paired_summary.to_string(index=False))

print("\nCreated:")
print(policy_csv)
print(paired_csv)
print(policy_tex)
print(paired_tex)
