"""
evaluate_hamilton_ppo_multiseed_holdout.py
----------------------------------------------
Common holdout evaluation (100 fresh seeds, 110000-110099, disjoint from
training seeds 0-4, dev eval seeds 90001-90005, and the previous
milestone's holdout 100000-100199) for all 5 independently-trained
Hamilton PPO models (seeds 0-4).

Fresh HamiltonPPOWrapper(seed=episode_seed) per episode -- never
reset(seed=...) on a reused wrapper. Deterministic policy predictions.

Produces:
    results/hamilton_ppo_multiseed_holdout_episodes.csv   (500 rows: 5 seeds x 100 episodes)
    results/hamilton_ppo_multiseed_summary.csv            (within-model AND between-training-seed
                                                             hierarchy, not a flat pool of 500)
    results/hamilton_ppo_multiseed_conditional_actions.csv (signed-policy + belief-quintile + tau-quartile
                                                              conditional action breakdown, per training seed)

Run from repo root:
    python evaluate_hamilton_ppo_multiseed_holdout.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

import hamilton_ppo_eval_lib as lib

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models" / "hamilton_ppo"
INVENTORY_SCALE = 10.0
TRAINING_SEEDS = [0, 1, 2, 3, 4]

PERFORMANCE_COLS = ["cumulative_objective", "raw_pnl", "running_inventory_penalty",
                    "terminal_inventory_penalty", "adverse_selection_loss", "spread_revenue", "total_fills"]
INVENTORY_COLS = ["mean_signed_inventory", "mean_abs_inventory", "max_abs_inventory",
                  "terminal_signed_inventory", "terminal_abs_inventory",
                  "frac_time_positive", "frac_time_negative"]
ACTION_COLS = ["bid_action_mean", "bid_action_std", "ask_action_mean", "ask_action_std",
              "frac_bid_near_low", "frac_bid_near_high", "frac_ask_near_low", "frac_ask_near_high"]
BELIEF_COLS = ["mean_belief", "belief_std", "belief_accuracy_offline"]
ALL_METRIC_COLS = PERFORMANCE_COLS + INVENTORY_COLS + ACTION_COLS + BELIEF_COLS


def evaluate_one_model(model, training_seed: int):
    bucket_acc = lib.BucketAccumulator()
    rows = []
    t0 = time.time()
    for i, seed in enumerate(lib.MULTISEED_HOLDOUT_SEEDS):
        r = lib.run_eval_episode_full(
            model, seed, INVENTORY_SCALE, deterministic=True, bucket_accumulator=bucket_acc,
        )
        r["training_seed"] = training_seed
        rows.append(r)
        assert r["reconciliation_abs_diff"] < 1e-6, f"seed {training_seed}/{seed}: reconciliation failed"
        assert r["episode_length"] == 4000
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{len(lib.MULTISEED_HOLDOUT_SEEDS)} episodes "
                  f"({elapsed:.1f}s, {elapsed/(i+1):.2f}s/ep)")
    return pd.DataFrame(rows), bucket_acc


def conditional_breakdown_for_seed(bdf: pd.DataFrame, training_seed: int) -> pd.DataFrame:
    bdf = bdf.copy()
    bdf["near_ask_high"] = (np.abs(bdf["ask"] - 1.0) < 0.01).astype(int)
    bdf["near_ask_low"] = (np.abs(bdf["ask"] - (-1.0)) < 0.01).astype(int)
    bdf["near_bid_low"] = (np.abs(bdf["bid"] - (-1.0)) < 0.01).astype(int)
    bdf["near_bid_high"] = (np.abs(bdf["bid"] - 1.0) < 0.01).astype(int)
    bdf["belief_quintile"] = pd.qcut(bdf["belief"], 5, labels=[f"Q{i+1}" for i in range(5)], duplicates="drop")
    bdf["tau_bucket"] = pd.cut(
        bdf["tau"], bins=[-0.001, 0.25, 0.5, 0.75, 1.001],
        labels=["[0.00,0.25]", "(0.25,0.50]", "(0.50,0.75]", "(0.75,1.00]"],
    )

    rows = []
    agg_cols = dict(n=("bid", "size"), mean_bid=("bid", "mean"), mean_ask=("ask", "mean"),
                     frac_ask_high=("near_ask_high", "mean"), frac_ask_low=("near_ask_low", "mean"),
                     frac_bid_low=("near_bid_low", "mean"), frac_bid_high=("near_bid_high", "mean"))

    for group_col, group_name in [("inv_sign", "inventory_sign"),
                                    ("belief_quintile", "belief_quintile"),
                                    ("tau_bucket", "tau_bucket")]:
        g = bdf.groupby(group_col, observed=True).agg(**agg_cols).reset_index()
        g = g.rename(columns={group_col: "bucket_value"})
        g["breakdown"] = group_name
        rows.append(g)

    out = pd.concat(rows, ignore_index=True)
    out["training_seed"] = training_seed
    return out[["training_seed", "breakdown", "bucket_value", "n", "mean_bid", "mean_ask",
                "frac_bid_low", "frac_bid_high", "frac_ask_low", "frac_ask_high"]]


def hierarchical_summary(episodes_df: pd.DataFrame) -> pd.DataFrame:
    """Within-model (per training_seed, across its 100 holdout episodes) AND
    between-training-seed (across the 5 per-seed means) summaries -- NOT a
    flat pool of 500 episodes treated as iid."""
    rows = []
    for metric in ALL_METRIC_COLS:
        per_seed_means = []
        for ts in TRAINING_SEEDS:
            sub = episodes_df[episodes_df["training_seed"] == ts][metric].to_numpy(dtype=float)
            n = len(sub)
            mean = float(sub.mean())
            std = float(sub.std(ddof=1))
            se = std / np.sqrt(n)
            per_seed_means.append(mean)
            rows.append(dict(
                level="within_seed", training_seed=ts, metric=metric, n=n,
                mean=mean, std=std, median=float(np.median(sub)),
                ci_lo=mean - 1.96 * se, ci_hi=mean + 1.96 * se, min=float(sub.min()), max=float(sub.max()),
            ))
        per_seed_means = np.array(per_seed_means)
        n_seeds = len(per_seed_means)
        mean_of_means = float(per_seed_means.mean())
        std_across_seeds = float(per_seed_means.std(ddof=1))
        se_across_seeds = std_across_seeds / np.sqrt(n_seeds)
        rows.append(dict(
            level="between_training_seeds", training_seed="ALL", metric=metric, n=n_seeds,
            mean=mean_of_means, std=std_across_seeds, median=float(np.median(per_seed_means)),
            ci_lo=mean_of_means - 1.96 * se_across_seeds, ci_hi=mean_of_means + 1.96 * se_across_seeds,
            min=float(per_seed_means.min()), max=float(per_seed_means.max()),
        ))
    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Multiseed holdout evaluation on {len(lib.MULTISEED_HOLDOUT_SEEDS)} seeds: "
          f"{lib.MULTISEED_HOLDOUT_SEEDS[0]}..{lib.MULTISEED_HOLDOUT_SEEDS[-1]}")

    all_episodes = []
    all_conditional = []
    for ts in TRAINING_SEEDS:
        model_path = MODELS_DIR / f"ppo_hamilton_seed_{ts}.zip"
        print(f"\n--- training_seed={ts} ({model_path}) ---")
        model = PPO.load(str(model_path))
        df, bucket_acc = evaluate_one_model(model, ts)
        all_episodes.append(df)

        bdf = pd.DataFrame(bucket_acc.rows)
        cond = conditional_breakdown_for_seed(bdf, ts)
        all_conditional.append(cond)

        print(f"  seed {ts}: mean_objective={df['cumulative_objective'].mean():.4f}  "
              f"mean_terminal_signed_inv={df['terminal_signed_inventory'].mean():.3f}  "
              f"frac_positive_terminal={(df['terminal_signed_inventory']>0).mean():.3f}")

    episodes_df = pd.concat(all_episodes, ignore_index=True)
    episodes_path = RESULTS_DIR / "hamilton_ppo_multiseed_holdout_episodes.csv"
    episodes_df.to_csv(episodes_path, index=False)
    print(f"\nEpisode-level results (500 rows) saved to {episodes_path}")

    summary_df = hierarchical_summary(episodes_df)
    summary_path = RESULTS_DIR / "hamilton_ppo_multiseed_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Hierarchical (within-model + between-seed) summary saved to {summary_path}")

    conditional_df = pd.concat(all_conditional, ignore_index=True)
    conditional_path = RESULTS_DIR / "hamilton_ppo_multiseed_conditional_actions.csv"
    conditional_df.to_csv(conditional_path, index=False)
    print(f"Conditional-action breakdown saved to {conditional_path}")

    # --- Console report: between-seed summary for the headline metrics ---
    pd.set_option("display.width", 160)
    print("\n" + "=" * 110)
    print("BETWEEN-TRAINING-SEED SUMMARY (n=5 training seeds; mean/std/min/max of each seed's own mean)")
    print("=" * 110)
    headline = ["cumulative_objective", "terminal_signed_inventory", "mean_signed_inventory",
                "ask_action_mean", "bid_action_mean", "frac_ask_near_high", "frac_bid_near_low",
                "frac_time_positive"]
    bs = summary_df[(summary_df["level"] == "between_training_seeds") & (summary_df["metric"].isin(headline))]
    print(bs[["metric", "mean", "std", "min", "max", "ci_lo", "ci_hi"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 110)
    print("PER-SEED headline metrics")
    print("=" * 110)
    ws = summary_df[(summary_df["level"] == "within_seed") & (summary_df["metric"].isin(headline))]
    piv = ws.pivot(index="training_seed", columns="metric", values="mean")
    print(piv[headline].to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nDone.")


if __name__ == "__main__":
    main()
