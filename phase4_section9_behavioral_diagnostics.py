"""
phase4_section9_behavioral_diagnostics.py
------------------------------------------------
Phase 4, Section 9: architecture-independent behavioural diagnostics
(spread, fills, inventory bias, action saturation, state visitation,
performance decomposition) applied to ALL THREE RL agent types
(hamilton_ppo, return_mlp_ppo, return_lstm_ppo).

Per the task brief, the analytical-policy imitation/representability test
(Sections 6-7) is NOT applied to return_mlp_ppo/return_lstm_ppo -- those
agents never receive the belief-state observation, so there is no shared
action surface to imitate. This script instead reuses two ALREADY-COMPUTED,
already-validated data sources rather than running new expensive rollouts:

  1. results/phase3_event_vs_fixed/phase3_holdout_episodes.csv -- 1000
     held-out episodes per RL agent type per environment (5 training seeds
     x 200 evaluation seeds), giving per-episode action mean/std (an
     episode-level saturation proxy) and the full performance decomposition
     (spread_revenue, adverse_selection_loss, running/terminal penalty).
  2. results/phase4_policy_diagnostic/phase4_state_visitation.csv -- fresh,
     never-before-used seeds (210000-210049), per-step instrumented, giving
     realised fill rate, mean quoted spread, mean expected fill probability,
     and inventory mean/std at ACTUALLY VISITED states for all 3
     architectures.

Run from repo root:
    python phase4_section9_behavioral_diagnostics.py
"""
import numpy as np
import pandas as pd

import phase4_common as P4

RL_AGENT_TYPES = ("hamilton_ppo", "return_mlp_ppo", "return_lstm_ppo")
SATURATION_TOL = 0.95


def summarise_phase3_holdout() -> pd.DataFrame:
    df = pd.read_csv("results/phase3_event_vs_fixed/phase3_holdout_episodes.csv")
    df = df[df["agent_type"].isin(RL_AGENT_TYPES)].copy()
    df["bid_saturated"] = df["bid_action_mean"].abs() > SATURATION_TOL
    df["ask_saturated"] = df["ask_action_mean"].abs() > SATURATION_TOL
    df["episode_saturated"] = df["bid_saturated"] | df["ask_saturated"]

    rows = []
    for (agent_type, env_type), sub in df.groupby(["agent_type", "environment_type"]):
        n = len(sub)
        se = lambda col: float(sub[col].std(ddof=1) / np.sqrt(n))
        rows.append(dict(
            agent_type=agent_type, environment_type=env_type, n_episodes=n,
            mean_full_objective=float(sub["full_objective"].mean()), se_full_objective=se("full_objective"),
            mean_raw_pnl=float(sub["raw_pnl"].mean()),
            mean_fills=float(sub["fills"].mean()),
            mean_quoted_spread=float(sub["mean_quoted_spread"].mean()),
            mean_abs_inventory=float(sub["mean_abs_inventory"].mean()),
            mean_signed_inventory=float(sub["mean_signed_inventory"].mean()),
            mean_terminal_abs_inventory=float(sub["terminal_abs_inventory"].mean()),
            mean_running_penalty=float(sub["running_penalty"].mean()),
            mean_terminal_penalty=float(sub["terminal_penalty"].mean()),
            mean_adverse_selection_loss=float(sub["adverse_selection_loss"].mean()),
            mean_bid_action=float(sub["bid_action_mean"].mean()),
            mean_ask_action=float(sub["ask_action_mean"].mean()),
            frac_episodes_action_saturated=float(sub["episode_saturated"].mean()),
            max_abs_signed_inventory_seen=float(sub["mean_signed_inventory"].abs().max()),
        ))
    return pd.DataFrame(rows)


def summarise_state_visitation() -> pd.DataFrame:
    df = pd.read_csv(P4.PHASE4_RESULTS_DIR / "phase4_state_visitation.csv")
    df = df[df["agent_type"].isin(RL_AGENT_TYPES)].copy()
    rows = []
    for (agent_type, env_type), sub in df.groupby(["agent_type", "environment_type"]):
        rows.append(dict(
            agent_type=agent_type, environment_type=env_type, n_seeds=len(sub),
            mean_realised_fill_rate_per_step=float(sub["realised_fill_rate_per_step"].mean()),
            mean_quoted_spread=float(sub["mean_quoted_spread"].mean()),
            mean_p_fill_total=float(sub["mean_p_fill_total"].mean()),
            mean_inventory_mean=float(sub["inventory_mean"].mean()),
            mean_inventory_std=float(sub["inventory_std"].mean()),
            worst_seed_abs_inventory_mean=float(sub["inventory_mean"].abs().max()),
        ))
    return pd.DataFrame(rows)


def main():
    P4.PHASE4_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    holdout_summary = summarise_phase3_holdout()
    visitation_summary = summarise_state_visitation()

    out_path = P4.PHASE4_RESULTS_DIR / "phase4_section9_holdout_behavioral_summary.csv"
    holdout_summary.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(holdout_summary.to_string(index=False))

    out_path2 = P4.PHASE4_RESULTS_DIR / "phase4_section9_visitation_behavioral_summary.csv"
    visitation_summary.to_csv(out_path2, index=False)
    print(f"\nSaved {out_path2}")
    print(visitation_summary.to_string(index=False))

    # Analytical (belief_weighted) reference rows, both sources, for direct comparison.
    df3 = pd.read_csv("results/phase3_event_vs_fixed/phase3_holdout_episodes.csv")
    analytic = df3[df3["agent_type"] == "belief_weighted"]
    print("\nAnalytical belief_weighted reference (Phase 3 holdout, 200 episodes/env):")
    print(analytic.groupby("environment_type")[["full_objective", "fills", "mean_quoted_spread",
                                                  "mean_abs_inventory", "mean_signed_inventory"]].mean().to_string())


if __name__ == "__main__":
    main()
