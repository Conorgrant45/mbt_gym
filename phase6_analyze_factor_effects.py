"""
phase6_analyze_factor_effects.py
--------------------------------------
Phase 6, Sections 6-7: aggregate the holdout episode results
(phase6_holdout_episodes.csv, produced by phase6_evaluate_holdout.py) into
per-seed and per-group summaries, then estimate the four primary factor
effects (reduced exploration: B-A, D-C; clone initialisation: C-A, D-B) plus
the initialisation x exploration interaction, using the LEARNER-SEED-
SPECIFIC HOLDOUT MEAN as the primary unit (n=5 per group, never treating
the 200 holdout episodes as 200 independent trained policies).

Run from repo root (after phase6_evaluate_holdout.py):
    python phase6_analyze_factor_effects.py
"""
import json

import numpy as np
import pandas as pd

import phase6_common as P6

N_BOOTSTRAP = 10_000
RNG_SEED_FOR_BOOTSTRAP = 999_001  # analysis-only RNG, not an environment seed


def seed_summary_row(group: str, learner_seed: int, episodes: pd.DataFrame,
                      checkpoint_df: pd.DataFrame) -> dict:
    obj = episodes["full_objective"].to_numpy()
    n = len(obj)
    mean_obj = float(obj.mean())
    sd_obj = float(obj.std(ddof=1))
    se_obj = sd_obj / np.sqrt(n)

    selection = json.loads(P6.offline_selection_path(group, learner_seed).read_text())
    selected_t = selection["selected_timestep"]
    validation_obj = selection["selected_det_mean_objective"]

    cp_row = checkpoint_df[(checkpoint_df["group"] == group) & (checkpoint_df["learner_seed"] == learner_seed) &
                            (checkpoint_df["timestep"] == selected_t)]
    action_saturation_rate = float(cp_row["det_frac_near_bound"].iloc[0]) if len(cp_row) else float("nan")

    return dict(
        group=group, learner_seed=learner_seed, n_holdout_episodes=n,
        mean_full_objective=mean_obj, episode_sd=sd_obj, se=se_obj,
        mean_raw_pnl=float(episodes["raw_pnl"].mean()),
        loss_rate=float((obj < 0).mean()),
        mean_quoted_spread=float(episodes["mean_quoted_spread"].mean()),
        median_quoted_spread=float(episodes["mean_quoted_spread"].median()),
        mean_fills=float(episodes["fills"].mean()),
        mean_abs_inventory=float(episodes["mean_abs_inventory"].mean()),
        mean_signed_inventory=float(episodes["mean_signed_inventory"].mean()),
        mean_terminal_abs_inventory=float(episodes["terminal_abs_inventory"].mean()),
        mean_terminal_signed_inventory=float(episodes["terminal_signed_inventory"].mean()),
        mean_running_penalty=float(episodes["running_penalty"].mean()),
        mean_terminal_penalty=float(episodes["terminal_penalty"].mean()),
        mean_adverse_selection_loss=float(episodes["adverse_selection_loss"].mean()),
        mean_return_on_turnover_pct=float(episodes["return_on_turnover_pct"].mean()),
        selected_checkpoint_timestep=int(selected_t),
        validation_objective=float(validation_obj),
        validation_to_holdout_diff=float(validation_obj - mean_obj),
        final_log_std_bid=float(episodes["log_std_bid"].iloc[0]),
        final_log_std_ask=float(episodes["log_std_ask"].iloc[0]),
        action_saturation_rate=action_saturation_rate,
    )


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED_FOR_BOOTSTRAP) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def contrast_row(name: str, group_hi: str, group_lo: str, seed_means: dict) -> dict:
    diffs = np.array([seed_means[group_hi][s] - seed_means[group_lo][s] for s in P6.LEARNER_SEEDS])
    ci_lo, ci_hi = bootstrap_ci(diffs)
    return dict(
        contrast=name, group_hi=group_hi, group_lo=group_lo,
        n_seeds=len(diffs), mean_diff=float(diffs.mean()), sd_diff=float(diffs.std(ddof=1)),
        min_diff=float(diffs.min()), max_diff=float(diffs.max()),
        seed0_diff=diffs[0], seed1_diff=diffs[1], seed2_diff=diffs[2], seed3_diff=diffs[3], seed4_diff=diffs[4],
        bootstrap_ci_lo=ci_lo, bootstrap_ci_hi=ci_hi,
        bootstrap_note="WEAK: bootstrap resamples only n=5 seed-level differences -- CI width is not a "
                       "substitute for a larger-n design, reported per Section 7's explicit instruction.",
    )


def main():
    ep = pd.read_csv(P6.RESULTS_DIR / "phase6_holdout_episodes.csv")
    checkpoint_df = pd.read_csv(P6.RESULTS_DIR / "phase6_checkpoint_summary.csv")

    seed_rows = []
    for group in ("A", "B", "C", "D"):
        for learner_seed in P6.LEARNER_SEEDS:
            sub = ep[(ep["group"] == group) & (ep["learner_seed"] == learner_seed)]
            assert len(sub) == len(P6.HOLDOUT_SEEDS), f"{group}/{learner_seed}: expected {len(P6.HOLDOUT_SEEDS)} episodes, got {len(sub)}"
            seed_rows.append(seed_summary_row(group, learner_seed, sub, checkpoint_df))
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(P6.RESULTS_DIR / "phase6_seed_summary.csv", index=False)
    print(f"Saved phase6_seed_summary.csv ({len(seed_df)} rows)")

    group_rows = []
    for group in ("A", "B", "C", "D"):
        sub = seed_df[seed_df["group"] == group]
        metrics = ["mean_full_objective", "mean_raw_pnl", "loss_rate", "mean_fills", "mean_abs_inventory",
                   "mean_signed_inventory", "mean_terminal_abs_inventory", "action_saturation_rate",
                   "final_log_std_bid", "validation_to_holdout_diff"]
        row = dict(group=group, n_seeds=len(sub))
        for m in metrics:
            row[f"{m}_mean"] = float(sub[m].mean())
            row[f"{m}_sd"] = float(sub[m].std(ddof=1))
            row[f"{m}_min"] = float(sub[m].min())
            row[f"{m}_max"] = float(sub[m].max())
        group_rows.append(row)
    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(P6.RESULTS_DIR / "phase6_group_summary.csv", index=False)
    print(f"Saved phase6_group_summary.csv ({len(group_df)} rows)")
    print(group_df[["group", "mean_full_objective_mean", "mean_full_objective_sd",
                     "mean_full_objective_min", "mean_full_objective_max"]].to_string(index=False))

    seed_means = {
        group: dict(zip(seed_df[seed_df["group"] == group]["learner_seed"],
                         seed_df[seed_df["group"] == group]["mean_full_objective"]))
        for group in ("A", "B", "C", "D")
    }

    contrasts = [
        contrast_row("reduced_exploration_random_init", "B", "A", seed_means),
        contrast_row("reduced_exploration_clone_init", "D", "C", seed_means),
        contrast_row("clone_init_default_exploration", "C", "A", seed_means),
        contrast_row("clone_init_reduced_exploration", "D", "B", seed_means),
    ]
    d_minus_c = np.array([seed_means["D"][s] - seed_means["C"][s] for s in P6.LEARNER_SEEDS])
    b_minus_a = np.array([seed_means["B"][s] - seed_means["A"][s] for s in P6.LEARNER_SEEDS])
    interaction = d_minus_c - b_minus_a
    ci_lo, ci_hi = bootstrap_ci(interaction)
    contrasts.append(dict(
        contrast="interaction_(D-C)-(B-A)", group_hi="D-C", group_lo="B-A",
        n_seeds=len(interaction), mean_diff=float(interaction.mean()), sd_diff=float(interaction.std(ddof=1)),
        min_diff=float(interaction.min()), max_diff=float(interaction.max()),
        seed0_diff=interaction[0], seed1_diff=interaction[1], seed2_diff=interaction[2],
        seed3_diff=interaction[3], seed4_diff=interaction[4],
        bootstrap_ci_lo=ci_lo, bootstrap_ci_hi=ci_hi,
        bootstrap_note="WEAK: bootstrap resamples only n=5 seed-level interaction values.",
    ))

    effects_df = pd.DataFrame(contrasts)
    effects_df.to_csv(P6.RESULTS_DIR / "phase6_factor_effects.csv", index=False)
    print(f"\nSaved phase6_factor_effects.csv ({len(effects_df)} rows)")
    print(effects_df[["contrast", "mean_diff", "sd_diff", "min_diff", "max_diff",
                       "bootstrap_ci_lo", "bootstrap_ci_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
