"""
phase4_policy_evaluation.py
--------------------------------
Phase 4, Section 7: evaluates the frozen supervised clone (selected by
VALIDATION MSE only, in phase4_supervised_clone.py -- never re-tuned after
inspecting these results) as an actual POLICY, alongside the analytical
belief-weighted policy and all 5 fixed-step / 5 event-driven Hamilton PPO
offline-best models, on FRESH unseen seeds in both environments.

Reuses evaluate_agents_common.run_hamilton_agent_episode /
evaluate_agents_event_driven.run_event_hamilton_agent_episode UNMODIFIED --
the clone is passed in via a duck-typed `.predict()` shim
(SupervisedCloneAgent), so it goes through the EXACT SAME
HamiltonPPOWrapper / EventDrivenHamiltonPPOWrapper observation, belief-
filter timing and action transformation as real Hamilton PPO.

Fresh evaluation seeds (225001-225050): disjoint from Phase 3's
195001-195050/200000-200199, Phase 4's diagnostic seeds (210000-210049) and
the supervised clone's own train/val/test seeds (220000-222999 range).

Run from repo root:
    python phase4_policy_evaluation.py
"""
import time

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

import phase4_common as P4
import evaluate_agents_common as EAC
import evaluate_agents_event_driven as EAED
import simulate_belief_weighted as SBW
from phase4_supervised_clone import build_mlp, SupervisedCloneAgent

EVAL_SEEDS = list(range(225_001, 225_051))  # 50 fresh seeds


def load_frozen_clone() -> SupervisedCloneAgent:
    net = build_mlp(0)
    net.load_state_dict(torch.load(P4.PHASE4_LOGS_DIR / "supervised_clone_best.pt"))
    net.eval()
    return SupervisedCloneAgent(net)


def main():
    controls = P4.build_analytical_controls()
    clone = load_frozen_clone()

    rows = []
    t0 = time.time()

    print("Analytical belief-weighted (fixed-step)...")
    for seed in EVAL_SEEDS:
        m = EAC.run_analytic_agent_episode("belief_weighted", controls, seed)
        rows.append(dict(policy="analytical_belief_weighted", environment_type="fixed", **m))

    print("Analytical belief-weighted (event-driven)...")
    for seed in EVAL_SEEDS:
        m = EAED.run_event_analytic_agent_episode("belief_weighted", controls, seed)
        rows.append(dict(policy="analytical_belief_weighted", environment_type="event", **m))

    print("Supervised clone (fixed-step)...")
    for seed in EVAL_SEEDS:
        m = EAC.run_hamilton_agent_episode(clone, seed)
        rows.append(dict(policy="supervised_clone", environment_type="fixed", **m))

    print("Supervised clone (event-driven)...")
    for seed in EVAL_SEEDS:
        m = EAED.run_event_hamilton_agent_episode(clone, seed)
        rows.append(dict(policy="supervised_clone", environment_type="event", **m))

    for seed_idx in P4.LEARNER_SEEDS:
        print(f"Hamilton PPO fixed-step seed {seed_idx}...")
        model = PPO.load(str(P4.fixed_model_path(seed_idx)))
        for seed in EVAL_SEEDS:
            m = EAC.run_hamilton_agent_episode(model, seed)
            rows.append(dict(policy=f"hamilton_ppo_fixed_seed{seed_idx}", environment_type="fixed", **m))

        print(f"Hamilton PPO event-driven seed {seed_idx}...")
        model = PPO.load(str(P4.event_model_path(seed_idx)))
        for seed in EVAL_SEEDS:
            m = EAED.run_event_hamilton_agent_episode(model, seed)
            rows.append(dict(policy=f"hamilton_ppo_event_seed{seed_idx}", environment_type="event", **m))

    df = pd.DataFrame(rows)
    out_path = P4.PHASE4_RESULTS_DIR / "phase4_policy_evaluation.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{len(df)} rows written to {out_path} ({time.time()-t0:.1f}s)")

    summary = df.groupby(["policy", "environment_type"]).agg(
        n=("full_objective", "size"),
        mean_full_objective=("full_objective", "mean"),
        se_full_objective=("full_objective", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        mean_raw_pnl=("raw_pnl", "mean"),
        mean_fills=("fills", "mean"),
        mean_quoted_spread=("mean_quoted_spread", "mean"),
        mean_abs_inventory=("mean_abs_inventory", "mean"),
        mean_signed_inventory=("mean_signed_inventory", "mean"),
        mean_terminal_abs_inventory=("terminal_abs_inventory", "mean"),
        mean_running_penalty=("running_penalty", "mean"),
        mean_terminal_penalty=("terminal_penalty", "mean"),
        mean_adverse_selection_loss=("adverse_selection_loss", "mean"),
    ).reset_index()
    summary_path = P4.PHASE4_RESULTS_DIR / "phase4_policy_evaluation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))

    max_recon = df["reward_reconciliation_error"].max()
    print(f"\nMax reward-reconciliation error across all {len(df)} episodes: {max_recon:.3e}")


if __name__ == "__main__":
    main()
