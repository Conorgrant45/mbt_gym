"""
phase3_analysis.py
----------------------
Phase 3 analysis: reads results/phase3_event_vs_fixed/phase3_holdout_episodes.csv
(frozen, already computed by phase3_holdout_evaluation.py) and
phase3_checkpoint_summary.csv, and produces:

    phase3_seed_summary.csv
    phase3_architecture_summary.csv
    phase3_benchmark_comparison.csv

Methodology (fixed before this script was ever run against the holdout
numbers -- this is the same script used to describe the required
statistics; nothing here was changed after inspecting results):

  - The learner-seed-specific policy MEAN (over its 200 holdout episodes) is
    the primary unit for architecture-level comparison -- individual
    episodes are NOT treated as independent trained-policy replicates.
  - Within each environment, all policies share the same 200 holdout seeds,
    so benchmark-relative gaps are computed as PAIRED per-episode
    differences (joined on evaluation_seed), not as a difference of two
    independent means -- tighter standard errors, same expectation.
  - Across the 5 learner seeds, event-vs-fixed differences are summarised
    with across-seed mean/std AND a bootstrap CI (resampling the 5
    seed-differences with replacement, 10,000 resamples) -- explicitly
    labelled weak, n=5.

Run from repo root:
    python phase3_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
PHASE3_RESULTS_DIR = REPO_ROOT / "results" / "phase3_event_vs_fixed"

RL_AGENT_TYPES = ("hamilton_ppo", "return_mlp_ppo", "return_lstm_ppo")
ANALYTIC_AGENT_TYPES = ("oracle", "belief_weighted", "randomised", "naive")
LEARNER_SEEDS = (0, 1, 2, 3, 4)
ENVIRONMENT_TYPES = ("fixed", "event")

METRICS = (
    "full_objective", "raw_pnl", "mean_quoted_spread", "fills",
    "mean_abs_inventory", "mean_signed_inventory", "terminal_abs_inventory",
    "terminal_signed_inventory", "running_penalty", "terminal_penalty",
    "adverse_selection_loss", "return_on_turnover_pct",
)

RNG_SEED_FOR_BOOTSTRAP = 20260727
N_BOOTSTRAP = 10_000


def load_holdout() -> pd.DataFrame:
    return pd.read_csv(PHASE3_RESULTS_DIR / "phase3_holdout_episodes.csv")


def load_checkpoint_summary() -> pd.DataFrame:
    return pd.read_csv(PHASE3_RESULTS_DIR / "phase3_checkpoint_summary.csv")


def se(x: np.ndarray) -> float:
    n = len(x)
    return float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0


# ======================================================================
# 1. Per-(environment, agent_type[, learner_seed]) seed-level summary
# ======================================================================
def build_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for env in ENVIRONMENT_TYPES:
        sub_env = df[df["environment_type"] == env]

        for agent_type in ANALYTIC_AGENT_TYPES:
            sub = sub_env[sub_env["agent_type"] == agent_type]
            rows.append(_summarise_group(env, agent_type, None, sub))

        for agent_type in RL_AGENT_TYPES:
            for seed in LEARNER_SEEDS:
                sub = sub_env[(sub_env["agent_type"] == agent_type) & (sub_env["training_seed"] == seed)]
                rows.append(_summarise_group(env, agent_type, seed, sub))
    return pd.DataFrame(rows)


def _summarise_group(env, agent_type, learner_seed, sub: pd.DataFrame) -> dict:
    x_obj = sub["full_objective"].to_numpy(dtype=float)
    row = dict(
        environment_type=env, agent_type=agent_type, learner_seed=learner_seed, n_episodes=len(sub),
        mean_full_objective=x_obj.mean(), std_full_objective=x_obj.std(ddof=1), se_full_objective=se(x_obj),
        loss_rate=float((x_obj < 0).mean()),
        mean_raw_pnl=sub["raw_pnl"].mean(),
        mean_quoted_spread=sub["mean_quoted_spread"].mean(), median_quoted_spread=sub["mean_quoted_spread"].median(),
        mean_fills=sub["fills"].mean(),
        mean_abs_inventory=sub["mean_abs_inventory"].mean(),
        mean_signed_inventory=sub["mean_signed_inventory"].mean(),
        mean_terminal_abs_inventory=sub["terminal_abs_inventory"].mean(),
        mean_terminal_signed_inventory=sub["terminal_signed_inventory"].mean(),
        mean_running_penalty=sub["running_penalty"].mean(),
        mean_terminal_penalty=sub["terminal_penalty"].mean(),
        mean_adverse_selection_loss=sub["adverse_selection_loss"].mean(),
        mean_return_on_turnover_pct=sub["return_on_turnover_pct"].replace([np.inf, -np.inf], np.nan).mean(),
        max_reward_reconciliation_error=sub["reward_reconciliation_error"].max(),
    )
    return row


# ======================================================================
# 2. Architecture-level (across-learner-seed) summary
# ======================================================================
def build_architecture_summary(seed_summary: pd.DataFrame, ckpt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for env in ENVIRONMENT_TYPES:
        for agent_type in RL_AGENT_TYPES:
            sub = seed_summary[(seed_summary["environment_type"] == env) & (seed_summary["agent_type"] == agent_type)]
            sub = sub.sort_values("learner_seed")
            seed_means = sub["mean_full_objective"].to_numpy(dtype=float)

            ckpt_sub = ckpt[(ckpt["environment_type"] == env) & (ckpt["agent_type"] == agent_type)].sort_values("learner_seed")
            validation_means = ckpt_sub["selected_validation_mean_objective"].to_numpy(dtype=float)
            selected_timesteps = ckpt_sub["selected_checkpoint_timestep"].tolist()

            row = dict(
                environment_type=env, agent_type=agent_type, n_seeds=len(seed_means),
                seed_mean_0=seed_means[0] if len(seed_means) > 0 else np.nan,
                seed_mean_1=seed_means[1] if len(seed_means) > 1 else np.nan,
                seed_mean_2=seed_means[2] if len(seed_means) > 2 else np.nan,
                seed_mean_3=seed_means[3] if len(seed_means) > 3 else np.nan,
                seed_mean_4=seed_means[4] if len(seed_means) > 4 else np.nan,
                across_seed_mean=seed_means.mean(), across_seed_std=seed_means.std(ddof=1),
                across_seed_se=se(seed_means),
                min_seed_result=seed_means.min(), max_seed_result=seed_means.max(),
                selected_checkpoint_timesteps=str(selected_timesteps),
                mean_selected_validation_objective=validation_means.mean(),
                mean_holdout_objective=seed_means.mean(),
                mean_validation_to_holdout_diff=(validation_means - seed_means).mean(),
                std_validation_to_holdout_diff=(validation_means - seed_means).std(ddof=1),
            )
            rows.append(row)
    return pd.DataFrame(rows)


# ======================================================================
# 3. Benchmark-relative gaps (paired within-environment, per learner seed)
# ======================================================================
def paired_gap(df_env: pd.DataFrame, agent_type: str, seed: int, benchmark: str, value_col: str = "full_objective") -> dict:
    learned = df_env[(df_env["agent_type"] == agent_type) & (df_env["training_seed"] == seed)].set_index("evaluation_seed")
    bench = df_env[df_env["agent_type"] == benchmark].set_index("evaluation_seed")
    common = learned.index.intersection(bench.index)
    diff = bench.loc[common, value_col] - learned.loc[common, value_col]  # J_benchmark - J_learned
    n = len(diff)
    mean_gap = float(diff.mean())
    se_gap = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    j_bench_mean = float(bench.loc[common, value_col].mean())
    j_learned_mean = float(learned.loc[common, value_col].mean())
    relative_gap = mean_gap / abs(j_bench_mean) if j_bench_mean != 0 else np.nan
    return dict(
        n_paired=n, absolute_gap=mean_gap, se_absolute_gap=se_gap, relative_gap=relative_gap,
        j_benchmark=j_bench_mean, j_learned=j_learned_mean,
    )


def build_benchmark_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for env in ENVIRONMENT_TYPES:
        df_env = df[df["environment_type"] == env]
        for agent_type in RL_AGENT_TYPES:
            for seed in LEARNER_SEEDS:
                for benchmark in ("belief_weighted", "oracle"):
                    g = paired_gap(df_env, agent_type, seed, benchmark)
                    row = dict(environment_type=env, agent_type=agent_type, learner_seed=seed, benchmark=benchmark)
                    row.update(g)
                    # behavioural deltas vs the belief-weighted benchmark specifically
                    if benchmark == "belief_weighted":
                        learned = df_env[(df_env["agent_type"] == agent_type) & (df_env["training_seed"] == seed)]
                        bench = df_env[df_env["agent_type"] == "belief_weighted"]
                        row["spread_gap_vs_belief"] = learned["mean_quoted_spread"].mean() - bench["mean_quoted_spread"].mean()
                        row["fills_gap_vs_belief"] = bench["fills"].mean() - learned["fills"].mean()
                        row["signed_inventory_gap_vs_belief"] = learned["mean_signed_inventory"].mean() - bench["mean_signed_inventory"].mean()
                        row["objective_std_gap_vs_belief"] = learned["full_objective"].std(ddof=1) - bench["full_objective"].std(ddof=1)
                    rows.append(row)
    return pd.DataFrame(rows)


# ======================================================================
# 4. Statistical analysis: event-vs-fixed across the 5 learner seeds
# ======================================================================
def bootstrap_ci(diffs: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED_FOR_BOOTSTRAP) -> dict:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)])
    return dict(
        mean=float(diffs.mean()), std=float(diffs.std(ddof=1)) if n > 1 else 0.0,
        ci_lo=float(np.percentile(boot_means, 2.5)), ci_hi=float(np.percentile(boot_means, 97.5)),
        n=n, n_bootstrap=n_boot,
    )


def build_event_vs_fixed_stats(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for agent_type in RL_AGENT_TYPES:
        fixed_sub = seed_summary[(seed_summary["environment_type"] == "fixed") & (seed_summary["agent_type"] == agent_type)].sort_values("learner_seed")
        event_sub = seed_summary[(seed_summary["environment_type"] == "event") & (seed_summary["agent_type"] == agent_type)].sort_values("learner_seed")
        fixed_means = fixed_sub.set_index("learner_seed")["mean_full_objective"]
        event_means = event_sub.set_index("learner_seed")["mean_full_objective"]
        common_seeds = fixed_means.index.intersection(event_means.index)
        diffs = (event_means.loc[common_seeds] - fixed_means.loc[common_seeds]).to_numpy(dtype=float)
        boot = bootstrap_ci(diffs)
        rows.append(dict(
            agent_type=agent_type, n_seed_pairs=len(diffs),
            per_seed_diffs=str(list(np.round(diffs, 4))),
            mean_diff_event_minus_fixed=boot["mean"], std_diff=boot["std"],
            bootstrap_ci_lo=boot["ci_lo"], bootstrap_ci_hi=boot["ci_hi"],
            n_bootstrap=boot["n_bootstrap"],
            frac_seeds_event_beats_fixed=float((diffs > 0).mean()),
        ))
    return pd.DataFrame(rows)


def main():
    df = load_holdout()
    ckpt = load_checkpoint_summary()

    print("Building seed-level summary...")
    seed_summary = build_seed_summary(df)
    seed_summary.to_csv(PHASE3_RESULTS_DIR / "phase3_seed_summary.csv", index=False)
    print(f"  {len(seed_summary)} rows -> phase3_seed_summary.csv")

    print("Building architecture-level summary...")
    arch_summary = build_architecture_summary(seed_summary, ckpt)
    arch_summary.to_csv(PHASE3_RESULTS_DIR / "phase3_architecture_summary.csv", index=False)
    print(f"  {len(arch_summary)} rows -> phase3_architecture_summary.csv")
    print(arch_summary[["environment_type", "agent_type", "across_seed_mean", "across_seed_std",
                         "min_seed_result", "max_seed_result", "mean_validation_to_holdout_diff"]].to_string(index=False))

    print("\nBuilding benchmark-relative comparison...")
    bench_comparison = build_benchmark_comparison(df)
    bench_comparison.to_csv(PHASE3_RESULTS_DIR / "phase3_benchmark_comparison.csv", index=False)
    print(f"  {len(bench_comparison)} rows -> phase3_benchmark_comparison.csv")

    print("\nEvent-vs-fixed across-seed statistics (weak, n=5):")
    stats_df = build_event_vs_fixed_stats(seed_summary)
    stats_df.to_csv(PHASE3_RESULTS_DIR / "phase3_event_vs_fixed_stats.csv", index=False)
    print(stats_df[["agent_type", "mean_diff_event_minus_fixed", "bootstrap_ci_lo", "bootstrap_ci_hi",
                     "frac_seeds_event_beats_fixed"]].to_string(index=False))

    print("\nAnalytical benchmarks (both environments):")
    print(seed_summary[seed_summary["agent_type"].isin(ANALYTIC_AGENT_TYPES)][
        ["environment_type", "agent_type", "mean_full_objective", "se_full_objective",
         "mean_quoted_spread", "mean_fills", "mean_abs_inventory"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
