"""
phase3_holdout_evaluation.py
--------------------------------
Phase 3 holdout evaluation. Uses the 200 fresh holdout seeds frozen in
results/phase3_event_vs_fixed/phase3_experiment_manifest.json (200000-200199
-- never used for training, monitoring, validation or checkpoint selection
anywhere in this project), evaluating:

  - all 15 fixed-step offline-best RL models, in the fixed-step environment;
  - all 15 event-driven offline-best RL models, in the event-driven
    environment;
  - the 4 analytical benchmarks, in the fixed-step environment;
  - the 4 analytical benchmarks, in the event-driven environment.

Within each environment, every policy uses the SAME 200 seed VALUES (same
exogenous path within that environment for a given seed). The two
environments are NOT claimed to be pathwise identical for the same seed
value -- they are statistically equivalent (Phase 1), not identical
simulators.

Reuses, READ-ONLY: evaluate_agents_common.py's run_hamilton_agent_episode/
run_return_agent_episode/run_analytic_agent_episode/build_row (fixed-step),
evaluate_agents_event_driven.py's event-driven counterparts. Does not
reimplement any accounting.

Resumable: writes one partial CSV per (environment, agent_type, learner_seed
or "analytic") group; skips a group if its partial file already has exactly
len(holdout_seeds) rows.

Run from repo root:
    python phase3_holdout_evaluation.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

import evaluate_agents_common as EAC
import evaluate_agents_event_driven as EAED
import simulate_belief_weighted as SBW

REPO_ROOT = Path(__file__).resolve().parent
PHASE3_RESULTS_DIR = REPO_ROOT / "results" / "phase3_event_vs_fixed"
PARTIAL_DIR = PHASE3_RESULTS_DIR / "holdout_partial"

RECURRENT_AGENT_TYPES = ("return_lstm_ppo",)


def load_manifest() -> dict:
    return json.load(open(PHASE3_RESULTS_DIR / "phase3_experiment_manifest.json"))


def fixed_run_tag(seed: int) -> str:
    return f"offlinecv_env70000_learner_{seed}"


def event_run_tag(seed: int, training_env_seed: int) -> str:
    return f"phase3_event_env{training_env_seed}_learner_{seed}"


def build_controls():
    controls = {}
    for regime, params in SBW.REGIME_PARAMS.items():
        da, db, qag, qbg = SBW.build_optimal_control(**params)
        controls[regime] = {"delta_ask": da, "delta_bid": db, "q_ask": qag, "q_bid": qbg}
    return controls


def partial_path(env_type: str, agent_type: str, group: str) -> Path:
    return PARTIAL_DIR / f"{env_type}_{agent_type}_{group}.csv"


def group_already_done(path: Path, n_expected: int) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    return len(df) == n_expected


def evaluate_analytics(env_type: str, holdout_seeds: list, controls: dict) -> pd.DataFrame:
    rows_all = []
    for agent_type in EAC.ANALYTIC_AGENT_TYPES:
        p = partial_path(env_type, agent_type, "analytic")
        if group_already_done(p, len(holdout_seeds)):
            print(f"  [SKIP] {env_type}/{agent_type}/analytic already done ({p})")
            rows_all.append(pd.read_csv(p))
            continue
        rows = []
        t0 = time.time()
        for seed in holdout_seeds:
            if env_type == "fixed":
                m = EAC.run_analytic_agent_episode(agent_type, controls, seed)
            else:
                m = EAED.run_event_analytic_agent_episode(agent_type, controls, seed)
            row = EAC.build_row(agent_type, None, None, None, f"analytic_{env_type}", m)
            row["environment_type"] = env_type
            rows.append(row)
        df = pd.DataFrame(rows)
        PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        print(f"  [DONE] {env_type}/{agent_type}/analytic: {len(holdout_seeds)} episodes in {time.time()-t0:.1f}s")
        rows_all.append(df)
    return pd.concat(rows_all, ignore_index=True)


def evaluate_rl_fixed(agent_type: str, seed: int, holdout_seeds: list, training_env_seed: int) -> pd.DataFrame:
    tag = fixed_run_tag(seed)
    p = partial_path("fixed", agent_type, f"learner_{seed}")
    if group_already_done(p, len(holdout_seeds)):
        print(f"  [SKIP] fixed/{agent_type}/learner_{seed} already done ({p})")
        return pd.read_csv(p)

    path = REPO_ROOT / "models" / agent_type / f"ppo_{agent_type}_{tag}_offline_best.zip"
    is_recurrent = agent_type in RECURRENT_AGENT_TYPES
    model_cls = RecurrentPPO if is_recurrent else PPO
    model = model_cls.load(str(path))

    rows = []
    t0 = time.time()
    for hseed in holdout_seeds:
        if agent_type == "hamilton_ppo":
            m = EAC.run_hamilton_agent_episode(model, hseed)
        else:
            m = EAC.run_return_agent_episode(model, hseed, is_recurrent=is_recurrent)
        row = EAC.build_row(agent_type, seed, str(path), 200_000, "fixed_holdout", m,
                             checkpoint_selection="offline_best", loaded_model_path=str(path),
                             training_env_seed=training_env_seed)
        row["environment_type"] = "fixed"
        rows.append(row)
    df = pd.DataFrame(rows)
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"  [DONE] fixed/{agent_type}/learner_{seed}: {len(holdout_seeds)} episodes in {time.time()-t0:.1f}s")
    return df


def evaluate_rl_event(agent_type: str, seed: int, holdout_seeds: list, training_env_seed: int) -> pd.DataFrame:
    tag = event_run_tag(seed, training_env_seed)
    p = partial_path("event", agent_type, f"learner_{seed}")
    if group_already_done(p, len(holdout_seeds)):
        print(f"  [SKIP] event/{agent_type}/learner_{seed} already done ({p})")
        return pd.read_csv(p)

    path = REPO_ROOT / "models" / f"{agent_type}_event" / f"ppo_{agent_type}_{tag}_offline_best.zip"
    is_recurrent = agent_type in RECURRENT_AGENT_TYPES
    model_cls = RecurrentPPO if is_recurrent else PPO
    model = model_cls.load(str(path))

    rows = []
    t0 = time.time()
    for hseed in holdout_seeds:
        if agent_type == "hamilton_ppo":
            m = EAED.run_event_hamilton_agent_episode(model, hseed)
        else:
            m = EAED.run_event_return_agent_episode(model, hseed, is_recurrent=is_recurrent)
        m["evaluation_seed"] = hseed
        row = EAC.build_row(agent_type, seed, str(path), 200_000, "event_holdout", m,
                             checkpoint_selection="offline_best", loaded_model_path=str(path),
                             training_env_seed=training_env_seed)
        row["environment_type"] = "event"
        rows.append(row)
    df = pd.DataFrame(rows)
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"  [DONE] event/{agent_type}/learner_{seed}: {len(holdout_seeds)} episodes in {time.time()-t0:.1f}s")
    return df


def main():
    manifest = load_manifest()
    holdout_seeds = list(range(manifest["holdout_seeds_start"],
                                manifest["holdout_seeds_start"] + manifest["holdout_seeds_count"]))
    learner_seeds = manifest["learner_seeds"]
    agent_types = manifest["agent_types"]
    training_env_seed = manifest["training_env_seed"]

    print(f"Holdout seeds: {holdout_seeds[0]}..{holdout_seeds[-1]} ({len(holdout_seeds)} total)")
    print(f"Learner seeds: {learner_seeds}")
    t_start = time.time()

    controls = build_controls()

    print("\n=== Analytical benchmarks: fixed-step environment ===")
    df_analytic_fixed = evaluate_analytics("fixed", holdout_seeds, controls)

    print("\n=== Analytical benchmarks: event-driven environment ===")
    df_analytic_event = evaluate_analytics("event", holdout_seeds, controls)

    print("\n=== RL agents: fixed-step environment (15 offline-best models) ===")
    fixed_rl_frames = []
    for agent_type in agent_types:
        for seed in learner_seeds:
            fixed_rl_frames.append(evaluate_rl_fixed(agent_type, seed, holdout_seeds, training_env_seed))

    print("\n=== RL agents: event-driven environment (15 offline-best models) ===")
    event_rl_frames = []
    for agent_type in agent_types:
        for seed in learner_seeds:
            event_rl_frames.append(evaluate_rl_event(agent_type, seed, holdout_seeds, training_env_seed))

    all_df = pd.concat(
        [df_analytic_fixed, df_analytic_event] + fixed_rl_frames + event_rl_frames,
        ignore_index=True,
    )
    out_path = PHASE3_RESULTS_DIR / "phase3_holdout_episodes.csv"
    all_df.to_csv(out_path, index=False)
    print(f"\n{len(all_df)} total holdout episode rows written to {out_path}")
    print(f"Total elapsed: {time.time()-t_start:.1f}s")

    max_recon = all_df["reward_reconciliation_error"].max()
    print(f"Maximum reward-reconciliation discrepancy across all {len(all_df)} episodes: {max_recon:.3e}")


if __name__ == "__main__":
    main()
