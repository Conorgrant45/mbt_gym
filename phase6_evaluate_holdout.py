"""
phase6_evaluate_holdout.py
--------------------------------
Phase 6, Sections 6-7: evaluate every selected offline-best model (4 groups
x 5 learner seeds = 20 models) on the SAME fresh 200-episode holdout set
(phase6_common.HOLDOUT_SEEDS), alongside the analytical oracle/
belief-weighted benchmarks and the frozen Phase 4 supervised clone --
matched exogenous paths (same seed -> same environment path) across every
policy, via evaluate_agents_event_driven's already-validated per-episode
runners (reused unmodified, not reimplemented).

MUST NOT be run until all 20 training runs are complete, all 20 checkpoints
are selected (offline_selection.json present + hash-verified for every
(group, seed)), and this script itself is finalised -- see
phase6_common.run_is_complete and main()'s pre-flight check below.

Run from repo root:
    python phase6_evaluate_holdout.py
"""
import time

import pandas as pd
from stable_baselines3 import PPO

import phase4_common as P4
import phase5_common as P5
import phase6_common as P6
import evaluate_agents_event_driven as EAED
from phase4_supervised_clone import SupervisedCloneAgent
from envs.make_envs import INITIAL_PRICE

ANALYTIC_POLICIES = ("oracle", "belief_weighted")


def preflight_check_all_runs_complete():
    missing = []
    for group in ("A", "B", "C", "D"):
        for seed in P6.LEARNER_SEEDS:
            if not P6.run_is_complete(group, seed):
                missing.append(f"{group}/seed{seed}")
    if missing:
        raise RuntimeError(
            f"Holdout evaluation refused: {len(missing)} run(s) not yet complete/verified: {missing}. "
            f"Run phase6_run_training.py to completion first."
        )
    print("Pre-flight check passed: all 20 (group, seed) runs are complete and hash-verified.")


def evaluate_analytic(policy_name: str, controls: dict, seeds: list) -> list:
    rows = []
    for seed in seeds:
        m = EAED.run_event_analytic_agent_episode(policy_name, controls, seed)
        rows.append(dict(group=policy_name, learner_seed=None, **m))
    return rows


def main():
    preflight_check_all_runs_complete()

    P6.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    controls = P4.build_analytical_controls()
    seeds = P6.HOLDOUT_SEEDS

    all_rows = []
    t0 = time.time()

    for policy_name in ANALYTIC_POLICIES:
        print(f"Evaluating analytical policy: {policy_name} ({len(seeds)} holdout episodes)...")
        all_rows += evaluate_analytic(policy_name, controls, seeds)

    print(f"Evaluating frozen supervised clone ({len(seeds)} holdout episodes)...")
    clone_net = P5.load_supervised_clone_net()
    clone_agent = SupervisedCloneAgent(clone_net)
    for seed in seeds:
        m = EAED.run_event_hamilton_agent_episode(clone_agent, seed)
        m["evaluation_seed"] = seed  # _run_event_rl_episode hardcodes this to None; fill in the real seed
        all_rows.append(dict(group="frozen_clone", learner_seed=None, **m))

    for group in ("A", "B", "C", "D"):
        for learner_seed in P6.LEARNER_SEEDS:
            print(f"Evaluating group {group} seed {learner_seed} offline-best model "
                  f"({len(seeds)} holdout episodes)...")
            model = PPO.load(str(P6.offline_best_path(group, learner_seed)))
            log_std = model.policy.log_std.detach().numpy().copy()
            for seed in seeds:
                m = EAED.run_event_hamilton_agent_episode(model, seed)
                m["evaluation_seed"] = seed  # _run_event_rl_episode hardcodes this to None; fill in the real seed
                turnover = m["fills"] * INITIAL_PRICE
                return_on_turnover_pct = (m["raw_pnl"] / turnover * 100.0) if turnover > 0 else float("nan")
                all_rows.append(dict(
                    group=group, learner_seed=learner_seed,
                    log_std_bid=float(log_std[0]), log_std_ask=float(log_std[1]),
                    return_on_turnover_pct=return_on_turnover_pct,
                    **m,
                ))
            del model

    elapsed = time.time() - t0
    df = pd.DataFrame(all_rows)
    out_path = P6.RESULTS_DIR / "phase6_holdout_episodes.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{len(df)} holdout episode rows saved to {out_path} ({elapsed:.1f}s)")

    max_recon = df["reward_reconciliation_error"].max()
    print(f"Max reward-reconciliation error across all {len(df)} episodes: {max_recon:.3e}")


if __name__ == "__main__":
    main()
