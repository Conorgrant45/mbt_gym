"""
final_aggregate_validation.py
------------------------------------
Reads the raw validation_path_level.csv (written incrementally by
final_run_training.py's per-checkpoint evaluation -- deterministic
PRIMARY metrics + separately-retained stochastic action diagnostics, on
IDENTICAL fixed validation paths across every architecture, learner seed,
and checkpoint) and produces the two aggregated deliverables:

    validation_seed_summary.csv        -- one row per
                                           (architecture, learner_seed,
                                            checkpoint_transition), averaged
                                           over the 50 validation paths.
    validation_architecture_summary.csv -- one row per
                                           (architecture, checkpoint_transition),
                                           averaged over the 5 learner seeds'
                                           own seed-level means (cross-seed
                                           mean/SD/SE -- NOT a flat pool of
                                           250 path-level observations, to
                                           keep learner-seed uncertainty and
                                           path-level uncertainty separate).

Never evaluates a model or reads a checkpoint file -- purely a read-only
aggregation of already-produced CSV rows, reproducible at any time.

Run from repo root:
    python final_aggregate_validation.py
"""
import numpy as np
import pandas as pd

import final_common as FC

MEAN_FIELDS = [
    "full_objective", "raw_pnl", "fills", "buy_side_fills", "sell_side_fills",
    "total_arrivals", "buy_side_arrivals", "sell_side_arrivals", "fill_to_arrival_ratio",
    "mean_quoted_spread", "mean_bid_depth", "mean_ask_depth",
    "spread_revenue", "spread_revenue_per_fill", "adverse_selection_loss",
    "running_penalty", "terminal_penalty",
    "mean_abs_inventory", "terminal_abs_inventory", "mean_signed_inventory",
    "terminal_signed_inventory", "max_abs_inventory",
    "mean_bid_action", "mean_ask_action", "action_std_bid", "action_std_ask",
    "near_bound_rate_deterministic", "near_bound_rate_stochastic",
    "stochastic_action_std_bid", "stochastic_action_std_ask", "stochastic_full_objective",
    "episode_duration", "n_decision_events", "reward_reconciliation_error",
]


def main():
    path = FC.RESULTS_DIR / "validation_path_level.csv"
    assert path.exists(), f"validation_path_level.csv not found at {path} -- run final_run_training.py first"
    df = pd.read_csv(path)
    assert df["deterministic"].all(), "validation_path_level.csv rows must all be the deterministic-primary row"

    seed_rows = []
    for (arch, seed, t), sub in df.groupby(["architecture", "learner_seed", "checkpoint_transition"]):
        row = dict(architecture=arch, learner_seed=seed, checkpoint_transition=t, n_paths=len(sub))
        for field in MEAN_FIELDS:
            row[f"mean_{field}"] = float(sub[field].mean())
        row["sd_full_objective"] = float(sub["full_objective"].std(ddof=1))
        row["se_full_objective"] = row["sd_full_objective"] / np.sqrt(len(sub))
        row["mean_wall_clock_elapsed_seconds"] = float(sub["wall_clock_elapsed_seconds"].mean())
        seed_rows.append(row)
    seed_df = pd.DataFrame(seed_rows).sort_values(["architecture", "learner_seed", "checkpoint_transition"])
    seed_path = FC.RESULTS_DIR / "validation_seed_summary.csv"
    seed_df.to_csv(seed_path, index=False)
    print(f"Saved {seed_path} ({len(seed_df)} rows)")

    arch_rows = []
    for (arch, t), sub in seed_df.groupby(["architecture", "checkpoint_transition"]):
        row = dict(architecture=arch, checkpoint_transition=t, n_seeds=len(sub))
        row["cross_seed_mean_full_objective"] = float(sub["mean_full_objective"].mean())
        row["cross_seed_sd_full_objective"] = float(sub["mean_full_objective"].std(ddof=1)) if len(sub) > 1 else 0.0
        row["cross_seed_se_full_objective"] = row["cross_seed_sd_full_objective"] / np.sqrt(len(sub))
        for field in MEAN_FIELDS:
            row[f"cross_seed_mean_{field}"] = float(sub[f"mean_{field}"].mean())
        arch_rows.append(row)
    arch_df = pd.DataFrame(arch_rows).sort_values(["architecture", "checkpoint_transition"])
    arch_path = FC.RESULTS_DIR / "validation_architecture_summary.csv"
    arch_df.to_csv(arch_path, index=False)
    print(f"Saved {arch_path} ({len(arch_df)} rows)")


if __name__ == "__main__":
    main()
