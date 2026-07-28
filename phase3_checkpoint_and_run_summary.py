"""
phase3_checkpoint_and_run_summary.py
----------------------------------------
Reconstructs phase3_checkpoint_summary.csv (Section 5/10) and augments it
with the per-run training-cost bookkeeping Section 7 requires (total
transitions, completed episodes, total simulated market time, number of PPO
updates, total observed arrivals, wall-clock duration), for all 15 fixed-step
and 15 event-driven runs, by reading the JSON artefacts already on disk --
does not re-run or re-instrument any training.

IMPORTANT, stated explicitly (not concealed): train_agents.py's own
run_config/run_summary do not instrument a live per-training-run counter of
completed episodes / total simulated market time / total observed arrivals
(Monitor's ep_info_buffer only retains a bounded recent window, not a
cumulative total across 200,000 transitions). Where the EXACT realised count
is not available from the artefacts on disk, this script reports the
THEORETICALLY EXPECTED value instead (labelled `_expected` in the column
name) -- justified because both environments' transition-generating
processes are exactly known and Monte-Carlo-verified (Phase 1's equivalence
diagnostic): fixed-step training always completes EXACTLY 200000/4000=50
episodes (n_steps=4000 exactly matches the episode length, so this one is
exact, not an expectation); event-driven training completes an EXPECTED
200000/(2*LAMBDA)=714.3 episodes (Poisson-distributed actual count, not
re-measured from the specific realised training run).

Run from repo root:
    python phase3_checkpoint_and_run_summary.py
"""
import json
from pathlib import Path

import pandas as pd

from envs.make_envs import LAMBDA, N_STEPS, TERMINAL_TIME

REPO_ROOT = Path(__file__).resolve().parent
PHASE3_RESULTS_DIR = REPO_ROOT / "results" / "phase3_event_vs_fixed"

AGENT_TYPES = ("hamilton_ppo", "return_mlp_ppo", "return_lstm_ppo")
LEARNER_SEEDS = (0, 1, 2, 3, 4)
TRAINING_ENV_SEED = 70000
TOTAL_TIMESTEPS = 200_000

MEAN_TRANSITIONS_PER_EPISODE_EVENT = 2 * LAMBDA  # 280, see envs/event_driven_regime_env.py


def fixed_run_tag(seed: int) -> str:
    return f"offlinecv_env70000_learner_{seed}"


def event_run_tag(seed: int) -> str:
    return f"phase3_event_env{TRAINING_ENV_SEED}_learner_{seed}"


def load_json(path: Path):
    if not path.exists():
        return None
    return json.load(open(path))


def summarise_run(agent_type: str, seed: int, environment_type: str) -> dict:
    tag = event_run_tag(seed) if environment_type == "event" else fixed_run_tag(seed)
    agent_dir = f"{agent_type}_event" if environment_type == "event" else agent_type
    log_dir = REPO_ROOT / "logs" / agent_dir

    cfg = load_json(log_dir / f"run_config_{tag}.json")
    summary = load_json(log_dir / f"run_summary_{tag}.json")
    manifest = load_json(log_dir / f"checkpoint_manifest_{tag}.json")
    selection = load_json(log_dir / f"offline_selection_{tag}.json")

    row = dict(agent_type=agent_type, environment_type=environment_type, learner_seed=seed, run_tag=tag)
    if cfg is None or summary is None or manifest is None or selection is None:
        row["status"] = "MISSING_ARTEFACTS"
        return row

    rg = cfg["rollout_geometry"]
    n_rollouts = rg["n_rollouts_total"]
    n_updates = rg["n_gradient_updates_per_rollout"] * n_rollouts

    if environment_type == "fixed":
        completed_episodes = TOTAL_TIMESTEPS // N_STEPS  # exact: 50
        completed_episodes_is_exact = True
        total_market_time = completed_episodes * TERMINAL_TIME
        total_arrivals_expected = 2 * LAMBDA * total_market_time  # theoretical expectation
    else:
        completed_episodes = TOTAL_TIMESTEPS / MEAN_TRANSITIONS_PER_EPISODE_EVENT  # expectation, not exact
        completed_episodes_is_exact = False
        total_market_time = completed_episodes * TERMINAL_TIME  # expectation
        total_arrivals_expected = TOTAL_TIMESTEPS - completed_episodes  # expectation (one terminal per episode)

    row.update(dict(
        status="OK",
        total_transitions=TOTAL_TIMESTEPS,
        completed_episodes_expected=completed_episodes,
        completed_episodes_is_exact=completed_episodes_is_exact,
        total_simulated_market_time_expected=total_market_time,
        n_ppo_updates=n_updates,
        total_observed_arrivals_expected=total_arrivals_expected,
        train_elapsed_seconds=summary["train_elapsed_seconds"],
        total_elapsed_seconds=summary["total_elapsed_seconds"],
        n_candidate_checkpoints=len(manifest["checkpoints"]),
        candidate_timesteps=str([c["timestep"] for c in sorted(manifest["checkpoints"], key=lambda c: c["timestep"])]),
        selected_checkpoint_timestep=selection["selected_timestep"],
        selected_validation_mean_objective=selection["selected_mean_cumulative_reward"],
        selected_validation_std_objective=selection["selected_std_cumulative_reward"],
        validation_seed_start=selection["validation_seed_start"],
        validation_seed_count=selection["validation_seed_count"],
        hash_verified=selection["hash_verified"],
        save_reload_equivalence_ok=summary.get("save_reload_equivalence_ok"),
        improvement_over_baseline=summary.get("improvement_over_baseline_mean_reward",
                                                summary.get("improvement_over_baseline_mean_objective")),
    ))
    return row


def main():
    rows = []
    for agent_type in AGENT_TYPES:
        for seed in LEARNER_SEEDS:
            rows.append(summarise_run(agent_type, seed, "fixed"))
            rows.append(summarise_run(agent_type, seed, "event"))

    df = pd.DataFrame(rows)
    missing = df[df["status"] == "MISSING_ARTEFACTS"]
    if len(missing):
        print("WARNING: missing artefacts for:")
        print(missing[["agent_type", "environment_type", "learner_seed"]].to_string(index=False))

    PHASE3_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PHASE3_RESULTS_DIR / "phase3_checkpoint_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{len(df)} rows written to {out_path}")
    print(df[["agent_type", "environment_type", "learner_seed", "status", "selected_checkpoint_timestep",
              "selected_validation_mean_objective", "total_elapsed_seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
