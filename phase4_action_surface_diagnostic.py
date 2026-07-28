"""
phase4_action_surface_diagnostic.py
---------------------------------------
Phase 4, Sections 2-3: dense (q, tau, belief) grid, analytical belief-
weighted vs. all 10 Hamilton PPO offline-best models (5 fixed-step, 5
event-driven), action-surface comparison metrics, finite-difference
sensitivities, monotonicity/saturation/symmetry checks.

Run from repo root:
    python phase4_action_surface_diagnostic.py
"""
import time

import numpy as np
import pandas as pd

import phase4_common as P4

SKEW_DEADZONE_Q = 2.0  # |q| below this is not checked for "wrong skew" (near-zero-inventory noise)
BOUND_TOL = 0.95  # |action| > this counts as "near bound"


def compute_grid_dataframe(models: dict, controls: dict) -> pd.DataFrame:
    q_grid, tau_grid, b_grid = P4.default_grid()
    rows = []
    t0 = time.time()
    for tau in tau_grid:
        for b in b_grid:
            for q in q_grid:
                bid_a, ask_a = P4.analytical_belief_weighted_depths(controls, tau, q, b)
                rows.append(dict(q=q, tau=tau, belief=b, policy="analytical",
                                  bid_depth=bid_a, ask_depth=ask_a,
                                  bid_action=P4.SBW.normalise_depth(bid_a), ask_action=P4.SBW.normalise_depth(ask_a)))
                for (env_type, seed), entry in models.items():
                    bid_p, ask_p = P4.hamilton_ppo_depths(entry["model"], q, tau, b)
                    rows.append(dict(q=q, tau=tau, belief=b, policy=f"{env_type}_seed{seed}",
                                      bid_depth=bid_p, ask_depth=ask_p,
                                      bid_action=(bid_p / P4.MAX_DEPTH) * 2.0 - 1.0,
                                      ask_action=(ask_p / P4.MAX_DEPTH) * 2.0 - 1.0))
    df = pd.DataFrame(rows)
    print(f"  grid computed: {len(q_grid)}x{len(tau_grid)}x{len(b_grid)} states x "
          f"{1 + len(models)} policies = {len(df)} rows in {time.time()-t0:.1f}s")
    return df


def finite_diff_sensitivity(df_policy: pd.DataFrame, q_grid, tau_grid, b_grid) -> dict:
    """Central finite-difference sensitivity of bid/ask depth w.r.t. q, tau,
    belief, holding the other two fixed at their grid median."""
    piv_bid = df_policy.pivot_table(index="q", columns=["tau", "belief"], values="bid_depth")
    piv_ask = df_policy.pivot_table(index="q", columns=["tau", "belief"], values="ask_depth")

    def mean_abs_central_diff(series_by_axis, axis_vals):
        vals = np.array(series_by_axis, dtype=float)
        diffs = np.diff(vals) / np.diff(axis_vals)
        return float(np.mean(np.abs(diffs)))

    # d/dq at median tau, median belief
    tau_med, b_med = tau_grid[len(tau_grid) // 2], b_grid[len(b_grid) // 2]
    bid_vs_q = df_policy[(df_policy.tau == tau_med) & (df_policy.belief == b_med)].sort_values("q")
    dq_bid = mean_abs_central_diff(bid_vs_q["bid_depth"].to_numpy(), bid_vs_q["q"].to_numpy())
    dq_ask = mean_abs_central_diff(bid_vs_q["ask_depth"].to_numpy(), bid_vs_q["q"].to_numpy())

    q_med = q_grid[len(q_grid) // 2]
    bid_vs_tau = df_policy[(df_policy.q == q_med) & (df_policy.belief == b_med)].sort_values("tau")
    dtau_bid = mean_abs_central_diff(bid_vs_tau["bid_depth"].to_numpy(), bid_vs_tau["tau"].to_numpy())
    dtau_ask = mean_abs_central_diff(bid_vs_tau["ask_depth"].to_numpy(), bid_vs_tau["tau"].to_numpy())

    bid_vs_b = df_policy[(df_policy.q == q_med) & (df_policy.tau == tau_med)].sort_values("belief")
    db_bid = mean_abs_central_diff(bid_vs_b["bid_depth"].to_numpy(), bid_vs_b["belief"].to_numpy())
    db_ask = mean_abs_central_diff(bid_vs_b["ask_depth"].to_numpy(), bid_vs_b["belief"].to_numpy())

    return dict(
        sensitivity_dq_bid=dq_bid, sensitivity_dq_ask=dq_ask,
        sensitivity_dtau_bid=dtau_bid, sensitivity_dtau_ask=dtau_ask,
        sensitivity_dbelief_bid=db_bid, sensitivity_dbelief_ask=db_ask,
    )


def monotonicity_violations(df_policy: pd.DataFrame) -> dict:
    """CJ theory: as q increases, ask_depth should be non-increasing and
    bid_depth non-decreasing (inventory skew becomes more sell-encouraging),
    holding tau/belief fixed. Fraction of adjacent-q pairs violating this,
    averaged over all (tau, belief) slices."""
    violations_ask, violations_bid, total = 0, 0, 0
    for (tau, b), sub in df_policy.groupby(["tau", "belief"]):
        sub = sub.sort_values("q")
        ask_vals = sub["ask_depth"].to_numpy()
        bid_vals = sub["bid_depth"].to_numpy()
        violations_ask += int(np.sum(np.diff(ask_vals) > 1e-9))  # should be <= 0 (non-increasing)
        violations_bid += int(np.sum(np.diff(bid_vals) < -1e-9))  # should be >= 0 (non-decreasing)
        total += len(ask_vals) - 1
    return dict(
        frac_ask_monotonicity_violations=violations_ask / total,
        frac_bid_monotonicity_violations=violations_bid / total,
    )


def zero_inventory_symmetry(df_policy: pd.DataFrame) -> float:
    at_zero = df_policy[df_policy.q == 0.0]
    asym = (at_zero["ask_depth"] - at_zero["bid_depth"]).abs()
    return float(asym.mean())


def wrong_skew_fraction(df_policy: pd.DataFrame) -> float:
    skew = df_policy["ask_depth"] - df_policy["bid_depth"]
    q = df_policy["q"]
    pos = q >= SKEW_DEADZONE_Q
    neg = q <= -SKEW_DEADZONE_Q
    wrong_pos = (skew[pos] > 0).sum()  # positive inventory should give skew <= 0
    wrong_neg = (skew[neg] < 0).sum()  # negative inventory should give skew >= 0
    n = pos.sum() + neg.sum()
    return float((wrong_pos + wrong_neg) / n) if n > 0 else float("nan")


def compute_metrics(df: pd.DataFrame, policy_name: str, q_grid, tau_grid, b_grid) -> dict:
    analytical = df[df.policy == "analytical"].set_index(["q", "tau", "belief"])
    learned = df[df.policy == policy_name].set_index(["q", "tau", "belief"])
    joined = analytical.join(learned, lsuffix="_analytical", rsuffix="_learned")

    bid_err = joined["bid_action_learned"] - joined["bid_action_analytical"]
    ask_err = joined["ask_action_learned"] - joined["ask_action_analytical"]
    bid_depth_err = joined["bid_depth_learned"] - joined["bid_depth_analytical"]
    ask_depth_err = joined["ask_depth_learned"] - joined["ask_depth_analytical"]

    spread_learned = joined["bid_depth_learned"] + joined["ask_depth_learned"]
    spread_analytical = joined["bid_depth_analytical"] + joined["ask_depth_analytical"]
    skew_learned = joined["ask_depth_learned"] - joined["bid_depth_learned"]
    skew_analytical = joined["ask_depth_analytical"] - joined["bid_depth_analytical"]

    df_learned = df[df.policy == policy_name]
    near_bound = (df_learned["bid_action"].abs() > BOUND_TOL) | (df_learned["ask_action"].abs() > BOUND_TOL)

    metrics = dict(
        policy=policy_name,
        bid_action_mse=float((bid_err ** 2).mean()),
        ask_action_mse=float((ask_err ** 2).mean()),
        mean_abs_action_error=float(pd.concat([bid_err.abs(), ask_err.abs()]).mean()),
        max_abs_action_error=float(pd.concat([bid_err.abs(), ask_err.abs()]).max()),
        mean_quoted_spread_error_signed=float((spread_learned - spread_analytical).mean()),
        mean_quoted_spread_error_abs=float((spread_learned - spread_analytical).abs().mean()),
        mean_inventory_skew_error_signed=float((skew_learned - skew_analytical).mean()),
        mean_inventory_skew_error_abs=float((skew_learned - skew_analytical).abs().mean()),
        corr_bid=float(np.corrcoef(joined["bid_depth_learned"], joined["bid_depth_analytical"])[0, 1]),
        corr_ask=float(np.corrcoef(joined["ask_depth_learned"], joined["ask_depth_analytical"])[0, 1]),
        frac_near_action_bound=float(near_bound.mean()),
        frac_wider_than_analytical=float((spread_learned > spread_analytical).mean()),
        frac_wrong_inventory_skew=wrong_skew_fraction(df_learned),
        zero_inventory_asymmetry=zero_inventory_symmetry(df_learned),
    )
    metrics.update(finite_diff_sensitivity(df_learned, q_grid, tau_grid, b_grid))
    metrics.update(monotonicity_violations(df_learned))
    return metrics


def main():
    P4.PHASE4_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading and verifying 10 Hamilton PPO offline-best models...")
    models = P4.verify_and_load_hamilton_models()
    print(f"  {len(models)} models verified and loaded.")

    print("Building analytical optimal-control tables (unchanged existing solver)...")
    controls = P4.build_analytical_controls()

    print("Computing action-surface grid...")
    df_grid = compute_grid_dataframe(models, controls)
    grid_path = P4.PHASE4_RESULTS_DIR / "phase4_action_surface_grid.csv"
    df_grid.to_csv(grid_path, index=False)
    print(f"  grid saved to {grid_path}")

    q_grid, tau_grid, b_grid = P4.default_grid()
    print("\nComputing per-policy action-surface metrics vs analytical belief-weighted policy...")
    policy_names = [f"{env}_seed{seed}" for (env, seed) in models.keys()]
    rows = [compute_metrics(df_grid, name, q_grid, tau_grid, b_grid) for name in policy_names]
    metrics_df = pd.DataFrame(rows)
    metrics_path = P4.PHASE4_RESULTS_DIR / "phase4_action_surface_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  metrics saved to {metrics_path}")
    print(metrics_df[["policy", "bid_action_mse", "ask_action_mse", "corr_bid", "corr_ask",
                       "frac_near_action_bound", "frac_wrong_inventory_skew",
                       "zero_inventory_asymmetry", "mean_quoted_spread_error_signed"]].to_string(index=False))


if __name__ == "__main__":
    main()
