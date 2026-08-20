"""
final_evaluate_holdout.py
--------------------------------
Final unseen-holdout evaluation. Evaluates, on the SAME 500 fresh holdout
paths (seeds disjoint from every previously-used range in this project,
verified programmatically):

  - the FIXED 1,000,000-transition checkpoint (primary result) AND the
    FIXED 200,000-transition checkpoint (for the 200k-vs-1m analysis) --
    for all three architectures x 5 learner seeds;
  - the analytical oracle, belief-weighted policy, and frozen supervised
    clone (checkpoint_transition = NaN for these -- they are not
    checkpointed models).

Written as ONE unified episode-level file, indexed exactly as specified:
(policy, learner_seed, checkpoint_transition, path_seed).

Also captures full step-level trajectories for a small, documented subset
of holdout paths (5 paths x {hamilton_ppo, return_mlp_ppo, return_lstm_ppo}
seed 0 at the 1,000,000-transition checkpoint, plus belief_weighted as an
analytical reference) -- NOT for all 500 paths, per the brief's own storage
guidance. Episode-level data remains complete for all 500 paths regardless.

Run from repo root:
    python final_evaluate_holdout.py
"""
import json
import time

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

import final_common as FC
import final_episode_runner as ER
import phase4_common as P4
import phase5_common as P5
from phase4_supervised_clone import SupervisedCloneAgent
from evaluate_agents_event_driven import make_event_time_filter, continuous_time_control
from envs.event_driven_regime_env import EventDrivenRegimeSwitchingEnv
from envs.event_driven_hamilton_ppo_wrapper import EventDrivenHamiltonPPOWrapper
from envs.event_driven_return_ppo_wrapper import EventDrivenReturnPPOWrapper
from mbt_gym.gym.index_names import TIME_INDEX
import simulate_belief_weighted as SBW

STOCHASTIC_TORCH_SEED_BASE = 950_000
TRAJECTORY_PATH_SEEDS = None  # set in main(): first 5 of FC.HOLDOUT_SEEDS, documented explicitly
N_TRAJECTORY_PATHS = 5
TRAJECTORY_POLICIES = [
    ("hamilton_ppo", 0, FC.TOTAL_TRANSITIONS),
    ("return_mlp_ppo", 0, FC.TOTAL_TRANSITIONS),
    ("return_lstm_ppo", 0, FC.TOTAL_TRANSITIONS),
    ("belief_weighted", None, None),
]


def load_model_for_checkpoint(architecture: str, learner_seed: int, timestep: int):
    if architecture == "hamilton_ppo":
        path = FC.hamilton_reused_checkpoint_path(learner_seed, timestep)
        return PPO.load(str(path))
    path = FC.checkpoint_path(architecture, learner_seed, timestep)
    cls = RecurrentPPO if FC.is_recurrent(architecture) else PPO
    return cls.load(str(path))


def evaluate_all_checkpoints_on_holdout(checkpoint_transitions: list) -> pd.DataFrame:
    rows = []
    controls = P4.build_analytical_controls()
    clone_net = P5.load_supervised_clone_net()
    clone_agent = SupervisedCloneAgent(clone_net)

    for policy_name in ("oracle", "belief_weighted"):
        print(f"Evaluating analytical policy: {policy_name} ({len(FC.HOLDOUT_SEEDS)} holdout paths)...")
        for seed in FC.HOLDOUT_SEEDS:
            row = ER.evaluate_benchmark_row(policy_name, controls, seed)
            row.update(policy=policy_name, learner_seed=None, checkpoint_transition=None,
                       path_seed=row.pop("evaluation_seed"))
            rows.append(row)

    print(f"Evaluating frozen supervised clone ({len(FC.HOLDOUT_SEEDS)} holdout paths)...")
    for seed in FC.HOLDOUT_SEEDS:
        row = ER.evaluate_clone_row(clone_agent, seed)
        row.update(policy="frozen_clone", learner_seed=None, checkpoint_transition=None,
                    path_seed=row.pop("evaluation_seed"))
        rows.append(row)

    for architecture in FC.ARCHITECTURES:
        for seed in FC.LEARNER_SEEDS:
            for timestep in checkpoint_transitions:
                print(f"Evaluating {architecture} seed {seed} @ t={timestep:,} "
                      f"({len(FC.HOLDOUT_SEEDS)} holdout paths)...")
                model = load_model_for_checkpoint(architecture, seed, timestep)
                for path_seed in FC.HOLDOUT_SEEDS:
                    row = ER.evaluate_episode_pair(
                        model, architecture, path_seed,
                        stochastic_torch_seed=STOCHASTIC_TORCH_SEED_BASE + timestep + path_seed,
                    )
                    row.update(policy=architecture, learner_seed=seed, checkpoint_transition=timestep,
                               path_seed=row.pop("evaluation_seed"))
                    rows.append(row)
                del model

    return pd.DataFrame(rows)


# ======================================================================
# Step-level trajectory capture (small documented subset only)
# ======================================================================
def capture_trajectory(policy_name: str, learner_seed, timestep, path_seed: int) -> list:
    """Returns a list of per-step dicts. `policy_name` in ARCHITECTURES uses
    the appropriate PPO wrapper + loaded checkpoint; 'belief_weighted' uses
    the raw environment + analytical control (no wrapper, no model)."""
    base_env = EventDrivenRegimeSwitchingEnv(seed=path_seed)
    steps = []

    if policy_name in FC.ARCHITECTURES:
        model = load_model_for_checkpoint(policy_name, learner_seed, timestep)
        recurrent = FC.is_recurrent(policy_name)
        if policy_name == "hamilton_ppo":
            wrapper = EventDrivenHamiltonPPOWrapper(base_env=base_env, inventory_scale=FC.INVENTORY_SCALE)
        else:
            wrapper = EventDrivenReturnPPOWrapper(base_env=base_env, inventory_scale=FC.INVENTORY_SCALE,
                                                    return_scale=FC.RETURN_SCALE)
        obs, _ = wrapper.reset()
        lstm_states = None
        episode_start = np.array([True], dtype=bool)
        terminated = truncated = False
        info = None
        while not (terminated or truncated):
            if recurrent:
                action, new_lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start,
                                                          deterministic=True)
                hidden_norm = float(np.linalg.norm(new_lstm_states[0][0])) if new_lstm_states is not None else float("nan")
                lstm_states = new_lstm_states
            else:
                action, _ = model.predict(obs, deterministic=True)
                hidden_norm = float("nan")
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            obs, reward, terminated, truncated, info = wrapper.step(action)
            episode_start = np.array([False], dtype=bool)
            belief = wrapper.belief if policy_name == "hamilton_ppo" else None
            steps.append(_trajectory_step_row(info, action, reward, hidden_norm, belief=belief))
        del model
    elif policy_name == "belief_weighted":
        controls = P4.build_analytical_controls()
        filt = make_event_time_filter()
        raw_state = base_env.reset()
        filt.reset(initial_price=base_env.raw_midprice)
        q, t_elapsed = base_env.raw_inventory, 0.0
        done = False
        info = None
        while not done:
            belief = filt.belief
            c0, c1 = controls[0], controls[1]
            a0, b0 = continuous_time_control(c0["delta_ask"], c0["delta_bid"], c0["q_ask"], c0["q_bid"], t_elapsed, q)
            a1, b1 = continuous_time_control(c1["delta_ask"], c1["delta_bid"], c1["q_ask"], c1["q_bid"], t_elapsed, q)
            ask_depth = (1.0 - belief) * a0 + belief * a1
            bid_depth = (1.0 - belief) * b0 + belief * b1
            action = np.array([SBW.normalise_depth(bid_depth), SBW.normalise_depth(ask_depth)])
            raw_state, reward, done, info = base_env.step(action)
            filt.update(info["price_after"], info["elapsed_time"], info["event_type"] == "arrival")
            steps.append(_trajectory_step_row(info, action, reward, float("nan"), belief=belief))
            q = info["inventory_after"]
            t_elapsed = float(info["raw_state"][TIME_INDEX])
    else:
        raise ValueError(f"Unsupported trajectory policy: {policy_name!r}")

    return steps


def _trajectory_step_row(info: dict, action: np.ndarray, reward: float, lstm_hidden_norm: float,
                          belief: float = None) -> dict:
    bid_action, ask_action = float(action[0]), float(action[1])
    bid_depth, ask_depth = float(info["bid_depth"]), float(info["ask_depth"])
    price_before = float(info["price_before"])
    price_after = float(info["price_after"])
    return dict(
        time=float(info["raw_state"][TIME_INDEX]),
        event_type=info["event_type"], arrival_side=info["arrival_side"],
        hidden_regime=info.get("regime_at_event"),
        hamilton_belief=belief if belief is not None else float("nan"),
        # Matches EventDrivenReturnPPOWrapper's own raw_return formula
        # exactly: (price_after - price_before) / price_before, computed
        # every step (arrival or terminal) -- not gated on event_type,
        # since the wrapper itself does not gate it either.
        observed_return=(price_after - price_before) / price_before,
        inventory_after=float(info["inventory_after"]),
        bid_depth=bid_depth, ask_depth=ask_depth,
        # Quote-price convention (documented, not asserted as literal env
        # state): bid fills execute at (price_before_jump - bid_depth), ask
        # fills at (price_before_jump + ask_depth) -- see
        # envs/event_driven_regime_env.py's own module docstring/cash
        # update. price_before is the closest exposed reference price (the
        # exact pre-jump price at fill time is not separately surfaced in
        # `info`); bid_price/ask_price below are therefore an approximation
        # for trajectory-inspection purposes only, not used in any reward
        # or objective calculation.
        bid_price_approx=price_before - bid_depth, ask_price_approx=price_before + ask_depth,
        fill_bid=int(info["fill_indicator"][0]), fill_ask=int(info["fill_indicator"][1]),
        adverse_jump=float(info.get("jump_increment", 0.0)),
        cash_after=float(info["cash_after"]),
        mark_to_market_wealth=float(info["cash_after"] + info["inventory_after"] * info["price_after"]),
        reward=float(reward),
        running_penalty_increment=float(info["running_penalty_increment"]),
        terminal_penalty_increment=float(info["terminal_penalty_increment"]),
        recurrent_hidden_state_l2_norm=lstm_hidden_norm,
        bid_action=bid_action, ask_action=ask_action,
    )


def main():
    t0 = time.time()
    FC.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    disjoint_check = FC.verify_seed_range_disjoint(FC.HOLDOUT_SEEDS, "final_holdout")
    assert disjoint_check["disjoint"], f"Holdout seeds NOT disjoint: {disjoint_check['overlaps']}"
    print(f"Holdout range {FC.HOLDOUT_SEEDS[0]}-{FC.HOLDOUT_SEEDS[-1]} "
          f"({len(FC.HOLDOUT_SEEDS)} paths) verified disjoint from all prior ranges.")

    episodes_df = evaluate_all_checkpoints_on_holdout(FC.FIXED_200K_1M_TIMESTEPS)
    episodes_path = FC.RESULTS_DIR / "final_holdout_episode_level.csv"
    episodes_df.to_csv(episodes_path, index=False)
    max_recon = episodes_df["reward_reconciliation_error"].max()
    print(f"\n{len(episodes_df)} episode rows saved to {episodes_path}")
    print(f"Max reward-reconciliation error: {max_recon:.3e}")

    trajectory_seeds = FC.HOLDOUT_SEEDS[:N_TRAJECTORY_PATHS]
    print(f"\nCapturing step-level trajectories for {len(TRAJECTORY_POLICIES)} policies x "
          f"{len(trajectory_seeds)} paths (seeds {trajectory_seeds})...")
    traj_rows = []
    for policy_name, learner_seed, timestep in TRAJECTORY_POLICIES:
        for path_seed in trajectory_seeds:
            steps = capture_trajectory(policy_name, learner_seed, timestep, path_seed)
            for step_idx, s in enumerate(steps):
                s.update(policy=policy_name, learner_seed=learner_seed, checkpoint_transition=timestep,
                          path_seed=path_seed, step_index=step_idx)
                traj_rows.append(s)
    traj_df = pd.DataFrame(traj_rows)
    traj_path = FC.RESULTS_DIR / "trajectory_subset.csv"
    traj_df.to_csv(traj_path, index=False)
    print(f"Saved {traj_path} ({len(traj_df)} rows, {len(TRAJECTORY_POLICIES)} policies x "
          f"{len(trajectory_seeds)} paths)")

    elapsed = time.time() - t0
    manifest_note = dict(
        holdout_seed_start=FC.HOLDOUT_SEEDS[0], holdout_seed_count=len(FC.HOLDOUT_SEEDS),
        disjoint_check=disjoint_check,
        checkpoint_transitions_evaluated=FC.FIXED_200K_1M_TIMESTEPS,
        trajectory_subset_path_seeds=trajectory_seeds,
        trajectory_subset_policies=TRAJECTORY_POLICIES,
        n_episode_rows=len(episodes_df), max_reward_reconciliation_error=float(max_recon),
        elapsed_seconds=elapsed,
    )
    (FC.RESULTS_DIR / "final_holdout_evaluation_manifest.json").write_text(json.dumps(manifest_note, indent=2, default=str))
    print(f"\nHoldout evaluation complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
