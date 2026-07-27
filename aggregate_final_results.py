"""
aggregate_final_results.py
------------------------------
Stage 3 (final aggregation) for the five-seed dissertation experiment.
Pure aggregation from the already-produced
results/agent_comparison_final_all_seeds_episodes.csv -- does not re-run
any evaluation. Reuses evaluate_agents_common.py's aggregation functions
(summarise_group, aggregate_flat, hierarchical_rl_summary,
paired_or_seed_matched_diffs) unchanged.

Run from repo root:
    python aggregate_final_results.py
"""

from pathlib import Path

import pandas as pd

import evaluate_agents_common as EAC

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"

TRAINING_SEEDS = [0, 1, 2, 3, 4]
RL_AGENT_TYPES = EAC.RL_AGENT_TYPES
ANALYTIC_AGENT_TYPES = EAC.ANALYTIC_AGENT_TYPES

KEY_METRICS = ("full_objective", "raw_pnl", "mean_abs_inventory", "mean_signed_inventory",
               "terminal_abs_inventory", "terminal_signed_inventory", "fills",
               "mean_quoted_spread")


def per_seed_paired_summary(df: pd.DataFrame, agent_a: str, agent_b: str, value_col: str = "full_objective",
                             seeds_a=None, seeds_b=None) -> pd.DataFrame:
    """Per-training-seed paired diffs, THEN a summary across those 5 (or
    fewer) per-seed means -- the hierarchical treatment the task requires
    (never pooling 500 episodes as 500 independent replicates)."""
    seeds_a = seeds_a if seeds_a is not None else [None]
    seeds_b = seeds_b if seeds_b is not None else [None]
    rows = []
    for ts_a in seeds_a:
        for ts_b in seeds_b:
            d = EAC.paired_or_seed_matched_diffs(df, agent_a, agent_b, value_col,
                                                  training_seed_a=ts_a, training_seed_b=ts_b)
            d["training_seed_a"], d["training_seed_b"] = ts_a, ts_b
            rows.append(d)
    return pd.DataFrame(rows)


def summarise_across_seed_means(per_seed_df: pd.DataFrame) -> dict:
    import numpy as np
    means = per_seed_df["mean_diff"].to_numpy(dtype=float)
    n = len(means)
    mean_of_means = float(means.mean())
    std = float(means.std(ddof=1)) if n > 1 else 0.0
    se = std / (n ** 0.5) if n > 1 else 0.0
    return dict(n_seed_pairs=n, mean_of_seed_means=mean_of_means, std_across_seeds=std,
                se=se, ci_lo=mean_of_means - 1.96 * se, ci_hi=mean_of_means + 1.96 * se,
                positive_seed_fraction=float((means > 0).mean()))


def main():
    df = pd.read_csv(RESULTS_DIR / "agent_comparison_final_all_seeds_episodes.csv")
    print(f"Loaded {len(df)} rows: {df.groupby('agent_type').size().to_dict()}")

    # --- 1. Pooled summaries (across evaluation AND training seeds -- for context only) ---
    pooled = EAC.aggregate_flat(df, metrics=KEY_METRICS)
    pooled.to_csv(RESULTS_DIR / "final_pooled_summary.csv", index=False)
    print(f"\nPooled summary saved ({len(pooled)} rows)")

    # --- 2/3/4. Per-training-seed + between/within-training-seed hierarchy ---
    hier = EAC.hierarchical_rl_summary(df, metrics=KEY_METRICS)
    hier.to_csv(RESULTS_DIR / "final_hierarchical_summary.csv", index=False)
    print(f"Hierarchical (within/between-training-seed) summary saved ({len(hier)} rows)")

    print("\n" + "=" * 100)
    print("BETWEEN-TRAINING-SEED full_objective (n=5 training-seed means per agent)")
    print("=" * 100)
    bs = hier[(hier["level"] == "between_training_seeds") & (hier["metric"] == "full_objective")]
    print(bs[["agent_type", "n", "mean", "std", "ci_lo", "ci_hi", "median", "iqr"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    # --- 6. Seed-matched / paired episode-level differences ---
    print("\n" + "=" * 100)
    print("PAIRED COMPARISONS (per-training-seed diffs, then summarised across the 5 seed-means)")
    print("=" * 100)
    comparisons = [
        ("hamilton_ppo", "return_lstm_ppo", "RL", "RL"),
        ("return_lstm_ppo", "return_mlp_ppo", "RL", "RL"),
        ("hamilton_ppo", "return_mlp_ppo", "RL", "RL"),
    ]
    for a, b, kind_a, kind_b in comparisons:
        per_seed = per_seed_paired_summary(df, a, b, "full_objective", seeds_a=TRAINING_SEEDS, seeds_b=TRAINING_SEEDS)
        # Only same-seed pairs are meaningful as a single "agent a's seed i vs agent b's seed i" comparison;
        # additionally compute the full 5x5 cross-seed grid for robustness reporting.
        same_seed = per_seed[per_seed["training_seed_a"] == per_seed["training_seed_b"]]
        summ = summarise_across_seed_means(same_seed)
        summ["comparison"] = f"{a} - {b}"
        print(f"{a} - {b} (same-seed pairs, n={summ['n_seed_pairs']}): "
              f"mean={summ['mean_of_seed_means']:.4f} std={summ['std_across_seeds']:.4f} "
              f"CI=[{summ['ci_lo']:.4f},{summ['ci_hi']:.4f}] "
              f"frac_seeds_a>b={summ['positive_seed_fraction']:.2f}")
        same_seed.to_csv(RESULTS_DIR / f"final_paired_{a}_vs_{b}_per_seed.csv", index=False)

    print()
    for rl_agent in RL_AGENT_TYPES:
        for analytic in ANALYTIC_AGENT_TYPES:
            per_seed = per_seed_paired_summary(df, analytic, rl_agent, "full_objective", seeds_b=TRAINING_SEEDS)
            summ = summarise_across_seed_means(per_seed)
            print(f"{analytic} - {rl_agent} (n={summ['n_seed_pairs']} training seeds): "
                  f"mean={summ['mean_of_seed_means']:.4f} std={summ['std_across_seeds']:.4f} "
                  f"CI=[{summ['ci_lo']:.4f},{summ['ci_hi']:.4f}] "
                  f"frac_seeds_analytic_beats_rl={summ['positive_seed_fraction']:.2f}")
            per_seed.to_csv(RESULTS_DIR / f"final_paired_{analytic}_vs_{rl_agent}_per_seed.csv", index=False)

    # --- 7/8. Loss rates, median/IQR already in pooled + hierarchical CSVs (loss_rate, median, iqr columns) ---
    print("\nLoss rates (pooled, full_objective<0):")
    print(pooled[pooled["metric"] == "full_objective"][["agent_type", "loss_rate", "median", "iqr"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nDone.")


if __name__ == "__main__":
    main()
