"""
phase7_analyze_convergence.py
------------------------------------
Phase 7: convergence analysis from phase7_checkpoint_summary.csv and
phase7_training_diagnostics.csv. Computes, per seed: performance at
200,000 transitions, best performance before/after 200,000, final
performance at 1,000,000, and the slope of deterministic validation
objective over the final 200,000 transitions (least-squares fit over the
800k/900k/1,000k checkpoints). Aggregates to seed-level and group-level
summaries and applies the five decision rules (A-E) from the task brief.

Run from repo root (after phase7_run_training.py has completed):
    python phase7_analyze_convergence.py
"""
import json

import numpy as np
import pandas as pd

import phase7_common as P7

PRE_200K_MAX = 200_000
POST_200K_MIN = 300_000
SLOPE_WINDOW = (800_000, 1_000_000)


def seed_convergence_row(seed: int, cp: pd.DataFrame, diag: pd.DataFrame) -> dict:
    cp = cp.sort_values("timestep")
    at_200k = cp[cp["timestep"] == 200_000]["det_mean_objective"].iloc[0]

    pre = cp[cp["timestep"] < PRE_200K_MAX]
    best_before_200k = float(pre["det_mean_objective"].max())
    best_before_200k_t = int(pre.loc[pre["det_mean_objective"].idxmax(), "timestep"])

    post = cp[cp["timestep"] >= POST_200K_MIN]
    best_after_200k = float(post["det_mean_objective"].max())
    best_after_200k_t = int(post.loc[post["det_mean_objective"].idxmax(), "timestep"])

    final = cp[cp["timestep"] == P7.TOTAL_TRANSITIONS]["det_mean_objective"].iloc[0]

    slope_df = cp[(cp["timestep"] >= SLOPE_WINDOW[0]) & (cp["timestep"] <= SLOPE_WINDOW[1])]
    if len(slope_df) >= 2:
        slope, intercept = np.polyfit(slope_df["timestep"], slope_df["det_mean_objective"], 1)
    else:
        slope = float("nan")

    seed_diag = diag[diag["learner_seed"] == seed] if "learner_seed" in diag.columns else diag
    late_diag = seed_diag[seed_diag["num_timesteps"] >= SLOPE_WINDOW[0]]
    mean_explained_variance_late = float(late_diag["explained_variance"].mean()) if len(late_diag) else float("nan")
    mean_clip_fraction_late = float(late_diag["clip_fraction"].mean()) if len(late_diag) else float("nan")
    mean_approx_kl_late = float(late_diag["approx_kl"].mean()) if len(late_diag) else float("nan")
    std_value_loss_late = float(late_diag["value_loss"].std()) if len(late_diag) else float("nan")
    mean_value_loss_late = float(late_diag["value_loss"].mean()) if len(late_diag) else float("nan")

    final_row = cp[cp["timestep"] == P7.TOTAL_TRANSITIONS].iloc[0]

    return dict(
        learner_seed=seed,
        det_objective_at_200k=float(at_200k),
        best_det_objective_before_200k=best_before_200k,
        best_det_objective_before_200k_timestep=best_before_200k_t,
        best_det_objective_after_200k=best_after_200k,
        best_det_objective_after_200k_timestep=best_after_200k_t,
        final_det_objective_at_1m=float(final),
        improvement_1m_minus_200k=float(final - at_200k),
        improvement_best_after_minus_best_before=float(best_after_200k - best_before_200k),
        slope_final_200k=float(slope),
        mean_explained_variance_final_200k=mean_explained_variance_late,
        mean_clip_fraction_final_200k=mean_clip_fraction_late,
        mean_approx_kl_final_200k=mean_approx_kl_late,
        mean_value_loss_final_200k=mean_value_loss_late,
        std_value_loss_final_200k=std_value_loss_late,
        final_log_std_bid=float(final_row["log_std_bid"]),
        final_log_std_ask=float(final_row["log_std_ask"]),
        final_action_saturation_rate_det=float(final_row["det_frac_near_bound"]),
        final_stoch_frac_near_bound=float(final_row["stoch_frac_near_bound"]),
        final_mean_fills=float(final_row["det_mean_fills"]),
        final_mean_abs_inventory=float(final_row["det_mean_abs_inventory"]),
        final_mean_signed_inventory=float(final_row["det_mean_signed_inventory"]),
    )


def classify_seed(row: dict) -> str:
    """Per-seed classification against decision rules A-D (E is reserved
    for the GROUP-level cross-seed disagreement check in main())."""
    rising = row["improvement_best_after_minus_best_before"] > 1.0 and row["improvement_1m_minus_200k"] > 0.5
    if rising:
        return "A_BUDGET_LIMITED"

    poor_ev = row["mean_explained_variance_final_200k"] < 0.3
    unstable_value = row["std_value_loss_final_200k"] > 0.5 * abs(row["mean_value_loss_final_200k"])
    if poor_ev or unstable_value:
        return "B_CRITIC_LIMITED"

    peaked_then_fell = (row["best_det_objective_after_200k"] - row["final_det_objective_at_1m"]) > 2.0
    unstable_updates = row["mean_clip_fraction_final_200k"] > 0.05 or row["mean_approx_kl_final_200k"] > 0.02
    if peaked_then_fell and unstable_updates:
        return "C_PPO_UPDATE_INSTABILITY"

    return "D_EXPLORATION_OR_LOCAL_OPTIMUM"


def main():
    cp_all = pd.read_csv(P7.RESULTS_DIR / "phase7_checkpoint_summary.csv")
    diag_all = pd.read_csv(P7.RESULTS_DIR / "phase7_training_diagnostics.csv")

    seed_rows = []
    for seed in P7.LEARNER_SEEDS:
        cp = cp_all[cp_all["learner_seed"] == seed]
        assert len(cp) == len(P7.CHECKPOINT_TIMESTEPS), f"seed {seed}: expected {len(P7.CHECKPOINT_TIMESTEPS)} checkpoints, got {len(cp)}"
        row = seed_convergence_row(seed, cp, diag_all)
        row["classification"] = classify_seed(row)
        seed_rows.append(row)

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(P7.RESULTS_DIR / "phase7_seed_summary.csv", index=False)
    print(f"Saved phase7_seed_summary.csv ({len(seed_df)} rows)")
    print(seed_df[["learner_seed", "det_objective_at_200k", "best_det_objective_before_200k",
                   "best_det_objective_after_200k", "final_det_objective_at_1m", "slope_final_200k",
                   "classification"]].to_string(index=False))

    classifications = set(seed_df["classification"])
    group_row = dict(
        group="B",
        n_seeds=len(seed_df),
        mean_det_objective_at_200k=float(seed_df["det_objective_at_200k"].mean()),
        sd_det_objective_at_200k=float(seed_df["det_objective_at_200k"].std(ddof=1)),
        mean_best_before_200k=float(seed_df["best_det_objective_before_200k"].mean()),
        mean_best_after_200k=float(seed_df["best_det_objective_after_200k"].mean()),
        mean_final_at_1m=float(seed_df["final_det_objective_at_1m"].mean()),
        sd_final_at_1m=float(seed_df["final_det_objective_at_1m"].std(ddof=1)),
        mean_slope_final_200k=float(seed_df["slope_final_200k"].mean()),
        mean_explained_variance_final_200k=float(seed_df["mean_explained_variance_final_200k"].mean()),
        mean_clip_fraction_final_200k=float(seed_df["mean_clip_fraction_final_200k"].mean()),
        n_distinct_seed_classifications=len(classifications),
        seed_classifications=sorted(classifications),
    )
    if len(classifications) > 1:
        group_row["group_classification"] = "E_INCONCLUSIVE"
    else:
        group_row["group_classification"] = classifications.pop()

    group_df = pd.DataFrame([group_row])
    group_df.to_csv(P7.RESULTS_DIR / "phase7_group_summary.csv", index=False)
    print(f"\nSaved phase7_group_summary.csv")
    print(f"Group-level classification: {group_row['group_classification']} "
          f"(seed-level classifications: {group_row['seed_classifications']})")


if __name__ == "__main__":
    main()
