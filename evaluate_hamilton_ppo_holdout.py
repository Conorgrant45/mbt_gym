"""
evaluate_hamilton_ppo_holdout.py
-----------------------------------
Unseen-holdout evaluation of the saved Hamilton PPO model
(models/hamilton_ppo/ppo_hamilton_v1.zip) on 200 fresh episode seeds
(100000..100199) never used in training or periodic evaluation (which
used seeds 90001-90005). A fresh HamiltonPPOWrapper(seed=...) is built
per episode -- never reset(seed=...) on a reused wrapper.

Also accumulates per-step (inventory sign, belief quintile, tau quartile,
true_regime) conditional action statistics as a byproduct of the same
200-episode rollout, for the ask-side-saturation investigation
(true_regime used for OFFLINE analysis only, never supplied to the
policy -- see hamilton_ppo_eval_lib.BucketAccumulator).

Does not modify the market dynamics, Hamilton filter, reward, or the
trained model.

Run from repo root:
    python evaluate_hamilton_ppo_holdout.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

import hamilton_ppo_eval_lib as lib

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
MODEL_PATH = REPO_ROOT / "models" / "hamilton_ppo" / "ppo_hamilton_v1.zip"
INVENTORY_SCALE = 10.0

METRIC_COLS = [
    "cumulative_objective", "raw_pnl", "running_inventory_penalty", "terminal_inventory_penalty",
    "mean_signed_inventory", "mean_abs_inventory", "terminal_signed_inventory", "terminal_abs_inventory",
    "total_fills", "bid_action_mean", "bid_action_std", "ask_action_mean", "ask_action_std",
    "frac_bid_near_low", "frac_bid_near_high", "frac_ask_near_low", "frac_ask_near_high",
    "mean_belief", "episode_length",
]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = PPO.load(str(MODEL_PATH))
    print(f"Loaded model from {MODEL_PATH}")
    print(f"Evaluating on {len(lib.HOLDOUT_SEEDS)} holdout seeds: "
          f"{lib.HOLDOUT_SEEDS[0]}..{lib.HOLDOUT_SEEDS[-1]}")

    bucket_acc = lib.BucketAccumulator()
    rows = []
    t0 = time.time()
    for i, seed in enumerate(lib.HOLDOUT_SEEDS):
        r = lib.run_eval_episode_detailed(
            model, seed, INVENTORY_SCALE, belief_override=None,
            deterministic=True, bucket_accumulator=bucket_acc,
        )
        rows.append(r)
        assert r["reconciliation_abs_diff"] < 1e-6, f"reconciliation failed for seed {seed}"
        assert r["episode_length"] == 4000
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(lib.HOLDOUT_SEEDS)} episodes done ({elapsed:.1f}s elapsed, "
                  f"{elapsed/(i+1):.2f}s/episode)")

    elapsed = time.time() - t0
    print(f"\nCompleted {len(lib.HOLDOUT_SEEDS)} holdout episodes in {elapsed:.1f}s")

    episodes_df = pd.DataFrame(rows)
    episodes_path = RESULTS_DIR / "hamilton_ppo_holdout_episodes_v1.csv"
    episodes_df.to_csv(episodes_path, index=False)
    print(f"Per-episode holdout results saved to {episodes_path}")

    summary_rows = [lib.summarise_column(episodes_df, c) for c in METRIC_COLS]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "hamilton_ppo_holdout_summary_v1.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Holdout summary (mean/std/median/95% CI) saved to {summary_path}")

    print("\n" + "=" * 100)
    print("HOLDOUT SUMMARY (n=%d episodes)" % len(lib.HOLDOUT_SEEDS))
    print("=" * 100)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Conditional saturation breakdown (item 6) ---
    bdf = pd.DataFrame(bucket_acc.rows)
    bdf["near_ask_high"] = (np.abs(bdf["ask"] - 1.0) < 0.01).astype(int)
    bdf["near_bid_low"] = (np.abs(bdf["bid"] - (-1.0)) < 0.01).astype(int)

    bdf["belief_quintile"] = pd.qcut(bdf["belief"], 5, labels=[f"Q{i+1}" for i in range(5)], duplicates="drop")
    bdf["tau_quartile"] = pd.cut(
        bdf["tau"], bins=[-0.001, 0.25, 0.5, 0.75, 1.001],
        labels=["T1 (0-0.25)", "T2 (0.25-0.5)", "T3 (0.5-0.75)", "T4 (0.75-1.0)"],
    )

    print("\n" + "=" * 100)
    print("CONDITIONAL ASK-SATURATION BREAKDOWN (item 6) -- fraction of steps with ask near +1 bound")
    print("=" * 100)

    print("\nBy inventory sign:")
    g = bdf.groupby("inv_sign", observed=True).agg(
        n=("near_ask_high", "size"), frac_ask_high=("near_ask_high", "mean"),
        frac_bid_low=("near_bid_low", "mean"), mean_ask=("ask", "mean"), mean_bid=("bid", "mean"),
    )
    print(g.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nBy belief quintile:")
    g = bdf.groupby("belief_quintile", observed=True).agg(
        n=("near_ask_high", "size"), frac_ask_high=("near_ask_high", "mean"),
        frac_bid_low=("near_bid_low", "mean"), mean_belief=("belief", "mean"),
        mean_ask=("ask", "mean"), mean_bid=("bid", "mean"),
    )
    print(g.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nBy time-to-horizon quartile:")
    g = bdf.groupby("tau_quartile", observed=True).agg(
        n=("near_ask_high", "size"), frac_ask_high=("near_ask_high", "mean"),
        frac_bid_low=("near_bid_low", "mean"), mean_ask=("ask", "mean"), mean_bid=("bid", "mean"),
    )
    print(g.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nBy true regime (OFFLINE diagnostic only -- never supplied to the policy):")
    g = bdf.groupby("true_regime", observed=True).agg(
        n=("near_ask_high", "size"), frac_ask_high=("near_ask_high", "mean"),
        frac_bid_low=("near_bid_low", "mean"), mean_ask=("ask", "mean"), mean_bid=("bid", "mean"),
    )
    print(g.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nBy inventory sign x belief quintile (top breakdown):")
    g = bdf.groupby(["inv_sign", "belief_quintile"], observed=True).agg(
        n=("near_ask_high", "size"), frac_ask_high=("near_ask_high", "mean"), mean_ask=("ask", "mean"),
    )
    print(g.to_string(float_format=lambda x: f"{x:.4f}"))

    # Terminal signed inventory distribution
    print("\nTerminal signed inventory distribution (holdout episodes):")
    ti = episodes_df["terminal_signed_inventory"]
    print(f"  mean={ti.mean():.3f}  std={ti.std():.3f}  median={ti.median():.3f}  "
          f"frac positive={float((ti > 0).mean()):.3f}  frac negative={float((ti < 0).mean()):.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
