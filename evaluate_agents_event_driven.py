"""
evaluate_agents_event_driven.py
----------------------------------
Event-driven counterpart of evaluate_agents_common.py. Does NOT modify that
file -- reuses its schema (SCHEMA_COLUMNS, build_row), constants
(ANALYTIC_AGENT_TYPES, RL_AGENT_TYPES, ANALYTIC_POLICY_NAME_MAP) and
aggregation helpers (summarise_group, aggregate_flat, hierarchical_rl_summary)
READ-ONLY, exactly as evaluate_agents_common.py itself reuses
compare_four_policies_paired.py and simulate_belief_weighted.py read-only.

Adapts the fixed-step evaluator to the event-driven environment:
  - variable episode length (no assumption of exactly N_STEPS transitions);
  - incremental (RunningStats-based) action-statistic accumulation, so
    memory does not grow with the number of evaluated timesteps -- the
    concern this project has previously hit a genuine MemoryError over
    (logs/offline_selection_batch/seed3_holdout_error.txt);
  - the analytical oracle/belief-weighted/randomised policies adapted to
    the continuous decision time (see continuous_time_control) and the
    event-time Hamilton filter (beliefs/event_time_hamilton_filter.py);
  - an exogenous-path hash proving all policies see identical arrival
    times/sides, hidden regime paths, Brownian innovations and jump
    innovations for a shared seed.

Run from repo root:
    python evaluate_agents_event_driven.py --run-tag event_diag --training-seed 0
"""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

import simulate_belief_weighted as SBW  # READ ONLY
import evaluate_agents_common as EAC  # READ ONLY (schema, build_row, aggregation)
from train_agents import RECURRENT_AGENT_TYPES, RunningStats  # READ ONLY
from envs.make_envs import STEP_SIZE, N_STEPS, TERMINAL_TIME
from envs.event_driven_regime_env import EventDrivenRegimeSwitchingEnv
from envs.event_driven_hamilton_ppo_wrapper import EventDrivenHamiltonPPOWrapper, make_event_time_filter
from envs.event_driven_return_ppo_wrapper import EventDrivenReturnPPOWrapper, DEFAULT_INVENTORY_SCALE, DEFAULT_RETURN_SCALE
from mbt_gym.gym.index_names import TIME_INDEX

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"

ENVIRONMENT_TYPE = "event"

# Fresh holdout range, disjoint from evaluate_agents_common.DEFAULT_HOLDOUT_SEEDS
# (120000-120099) and every fixed-step range used elsewhere in this project.
DEFAULT_HOLDOUT_SEEDS = list(range(130_000, 130_100))

RECONCILIATION_TOL = 1e-6


# ======================================================================
# Continuous-time analytic-policy control lookup
# ======================================================================
def continuous_time_control(d_ask, d_bid, q_ask, q_bid, elapsed_time: float, inv_sc: float,
                             dt: float = STEP_SIZE, n_t: int = N_STEPS):
    """
    Maps a CONTINUOUS decision time to the nearest row of the existing
    discrete optimal-control table (built once, unchanged, by
    simulate_belief_weighted.build_optimal_control on its own fixed
    n_t=N_STEPS-row time grid t_grid[n] = n*dt) and reuses
    simulate_belief_weighted.get_control (imported read-only) to look up
    the control at that row -- the SAME solver/table as every fixed-step
    analytic policy, not a re-derived one.

    Nearest-grid-point lookup introduces a timing error of at most dt/2 =
    1.25e-4 (STEP_SIZE=0.00025 in production), negligible against the
    ~0.0036 mean inter-arrival gap this control is actually evaluated at.
    This is NOT claimed to be the globally optimal continuous-time control
    -- see module docstring / Section 7 of the Phase 2 report for why this
    stays labelled a "regime-conditioned benchmark", not an oracle.
    """
    t_idx = int(np.clip(round(elapsed_time / dt), 0, n_t - 1))
    return SBW.get_control(d_ask, d_bid, q_ask, q_bid, t_idx, inv_sc)


# ======================================================================
# Exogenous path hash (Section 8): independent of actions.
# ======================================================================
def compute_exogenous_path_hash(seed: int, action_fn) -> str:
    """
    Roll out EventDrivenRegimeSwitchingEnv directly (no wrapper) for one
    full episode using actions from action_fn() (called with no arguments --
    these checks intentionally use action-sequences that do not depend on
    the (policy-independent) observation, since the point is to vary the
    ACTIONS themselves and confirm the exogenous path is unaffected), hashing
    only the fields that must NEVER depend on the policy's actions:
    event_type, arrival_side, regime_at_event, elapsed_time,
    brownian_increment, jump_increment. Fill/cash/inventory are deliberately
    excluded (those DO depend on actions).
    """
    env = EventDrivenRegimeSwitchingEnv(seed=seed)
    env.reset()
    h = hashlib.sha256()
    done = False
    side_code = {"buy": 0, "sell": 1, None: 2}
    while not done:
        action = action_fn()
        _, _, done, info = env.step(action)
        h.update(np.array([
            0 if info["event_type"] == "arrival" else 1,
            side_code[info["arrival_side"]],
            info["regime_at_event"],
        ], dtype=np.int64).tobytes())
        h.update(np.array([
            info["elapsed_time"], info["brownian_increment"], info["jump_increment"],
        ], dtype=np.float64).tobytes())
    return h.hexdigest()


def verify_exogenous_path_pairing(seed: int = 777_001) -> dict:
    """Confirms compute_exogenous_path_hash is IDENTICAL across four
    structurally different action sequences (all-zero, all-max, naive
    constant depth, uniform-random) for one shared seed -- the event-driven
    counterpart of evaluate_agents_common.verify_path_pairing."""
    rng = np.random.default_rng(seed + 555_000)
    naive_action_value = SBW.normalise_depth(SBW.NAIVE_DEPTH)
    action_fns = {
        "all_zero": lambda: np.array([0.0, 0.0]),
        "all_max": lambda: np.array([1.0, 1.0]),
        "naive_constant": lambda: np.array([naive_action_value, naive_action_value]),
        "uniform_random": lambda: rng.uniform(-1.0, 1.0, size=2),
    }
    hashes = {name: compute_exogenous_path_hash(seed, fn) for name, fn in action_fns.items()}
    all_matched = len(set(hashes.values())) == 1
    return dict(seed=seed, hashes=hashes, all_matched=all_matched)


# ======================================================================
# Analytic-policy episode runner (event-driven)
# ======================================================================
def run_event_analytic_agent_episode(agent_type: str, controls: dict, seed: int) -> dict:
    policy = EAC.ANALYTIC_POLICY_NAME_MAP[agent_type]

    env = EventDrivenRegimeSwitchingEnv(seed=seed)
    filt = make_event_time_filter() if policy in ("belief", "randomised") else None
    rng_random_policy = np.random.default_rng(seed + 1_000_000) if policy == "randomised" else None

    raw_state = env.reset()
    if filt is not None:
        filt.reset(initial_price=env.raw_midprice)
    cash_0, inv_0, mid_0 = env.raw_cash, env.raw_inventory, env.raw_midprice

    inv_sum = inv_abs_sum = inv_max_abs = 0.0
    running_penalty = spread_revenue = adverse_selection_loss = total_fills = 0.0
    cumulative_objective = 0.0
    action_stats = RunningStats(2)
    depth_stats = RunningStats(2)

    q = inv_0
    t_elapsed = 0.0
    n_steps = 0
    done = False
    info = None
    while not done:
        inv_sc = q / SBW.INV_UNIT

        if policy == "oracle":
            regime = env.current_regime  # privileged -- the regime-conditioned benchmark's whole point
            ctrl = controls[regime]
            ask_depth, bid_depth = continuous_time_control(
                ctrl["delta_ask"], ctrl["delta_bid"], ctrl["q_ask"], ctrl["q_bid"], t_elapsed, inv_sc,
            )
        elif policy == "belief":
            belief = filt.belief  # belief already updated through the PREVIOUS event -- available at decision time
            c0, c1 = controls[0], controls[1]
            a0, b0 = continuous_time_control(c0["delta_ask"], c0["delta_bid"], c0["q_ask"], c0["q_bid"], t_elapsed, inv_sc)
            a1, b1 = continuous_time_control(c1["delta_ask"], c1["delta_bid"], c1["q_ask"], c1["q_bid"], t_elapsed, inv_sc)
            ask_depth = (1.0 - belief) * a0 + belief * a1
            bid_depth = (1.0 - belief) * b0 + belief * b1
        elif policy == "randomised":
            belief = filt.belief
            sampled_regime = int(rng_random_policy.choice(2, p=[1.0 - belief, belief]))
            ctrl = controls[sampled_regime]
            ask_depth, bid_depth = continuous_time_control(
                ctrl["delta_ask"], ctrl["delta_bid"], ctrl["q_ask"], ctrl["q_bid"], t_elapsed, inv_sc,
            )
        else:  # naive
            ask_depth = bid_depth = SBW.NAIVE_DEPTH

        bid_action = SBW.normalise_depth(bid_depth)
        ask_action = SBW.normalise_depth(ask_depth)
        action_stats.add(np.array([bid_action, ask_action]))
        depth_stats.add(np.array([bid_depth, ask_depth]))

        raw_state, reward, done, info = env.step(np.array([bid_action, ask_action]))

        if filt is not None:
            filt.update(info["price_after"], info["elapsed_time"], info["event_type"] == "arrival")

        fill_bid, fill_ask = info["fill_indicator"]
        total_fills += int(fill_bid) + int(fill_ask)
        spread_revenue += ask_depth * fill_ask + bid_depth * fill_bid
        if info["regime_at_event"] == 1:
            adverse_selection_loss += abs(info["jump_increment"]) * (fill_ask + fill_bid)

        inv_after = info["inventory_after"]
        inv_sum += inv_after
        inv_abs_sum += abs(inv_after)
        inv_max_abs = max(inv_max_abs, abs(inv_after))
        running_penalty += info["running_penalty_increment"]
        cumulative_objective += float(reward)

        q = inv_after
        t_elapsed = float(info["raw_state"][TIME_INDEX])
        n_steps += 1

    cash_T, inv_T, mid_T = info["cash_after"], info["inventory_after"], info["price_after"]
    raw_pnl = (cash_T + inv_T * mid_T) - (cash_0 + inv_0 * mid_0)
    terminal_penalty = info["terminal_penalty_increment"]
    full_objective = raw_pnl - running_penalty - terminal_penalty
    reconciliation_abs_diff = abs(full_objective - cumulative_objective)

    return dict(
        evaluation_seed=seed,
        raw_pnl=float(raw_pnl), full_objective=float(full_objective),
        spread_revenue=spread_revenue, adverse_selection_loss=adverse_selection_loss,
        running_penalty=running_penalty, terminal_penalty=terminal_penalty,
        mean_abs_inventory=inv_abs_sum / n_steps, max_abs_inventory=inv_max_abs,
        mean_signed_inventory=inv_sum / n_steps, terminal_signed_inventory=float(inv_T),
        terminal_abs_inventory=abs(float(inv_T)), fills=total_fills,
        bid_action_mean=float(action_stats.mean[0]), ask_action_mean=float(action_stats.mean[1]),
        bid_action_std=float(action_stats.std[0]), ask_action_std=float(action_stats.std[1]),
        mean_quoted_spread=float(depth_stats.mean[0] + depth_stats.mean[1]), episode_length=n_steps,
        reward_reconciliation_error=reconciliation_abs_diff,
        recurrent_state_reset_verified=None,
    )


# ======================================================================
# RL-agent episode runners (event-driven)
# ======================================================================
def run_event_hamilton_agent_episode(model, seed: int, inventory_scale: float = DEFAULT_INVENTORY_SCALE) -> dict:
    base_env = EventDrivenRegimeSwitchingEnv(seed=seed)
    wrapper = EventDrivenHamiltonPPOWrapper(base_env=base_env, inventory_scale=inventory_scale)
    return _run_event_rl_episode(model, wrapper, is_recurrent=False)


def run_event_return_agent_episode(model, seed: int, is_recurrent: bool,
                                    inventory_scale: float = DEFAULT_INVENTORY_SCALE,
                                    return_scale: float = DEFAULT_RETURN_SCALE) -> dict:
    base_env = EventDrivenRegimeSwitchingEnv(seed=seed)
    wrapper = EventDrivenReturnPPOWrapper(base_env=base_env, inventory_scale=inventory_scale, return_scale=return_scale)
    return _run_event_rl_episode(model, wrapper, is_recurrent=is_recurrent)


def _run_event_rl_episode(model, wrapper, is_recurrent: bool) -> dict:
    obs, _ = wrapper.reset()
    cash_0, inv_0, mid_0 = wrapper.base_env.raw_cash, wrapper.base_env.raw_inventory, wrapper.base_env.raw_midprice

    lstm_states = None
    episode_start = np.array([True], dtype=bool)
    state_changed_within_episode = False
    prev_state_hidden = None

    running_penalty = spread_revenue = adverse_selection_loss = total_fills = 0.0
    inv_sum = inv_abs_sum = inv_max_abs = 0.0
    cumulative_objective = 0.0
    action_stats = RunningStats(2)
    depth_stats = RunningStats(2)

    n_steps = 0
    terminated = truncated = False
    info = None
    while not (terminated or truncated):
        if is_recurrent:
            action, new_lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True,
            )
            if lstm_states is not None and prev_state_hidden is not None:
                if not np.allclose(new_lstm_states[0], prev_state_hidden):
                    state_changed_within_episode = True
            prev_state_hidden = new_lstm_states[0].copy()
            lstm_states = new_lstm_states
        else:
            action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float64)
        action_stats.add(action)

        obs, reward, terminated, truncated, info = wrapper.step(action)
        episode_start = np.array([False], dtype=bool)

        bid_action, ask_action = float(action[0]), float(action[1])
        ask_depth = (ask_action + 1.0) / 2.0 * SBW.MAX_DEPTH
        bid_depth = (bid_action + 1.0) / 2.0 * SBW.MAX_DEPTH
        depth_stats.add(np.array([bid_depth, ask_depth]))

        fill_bid, fill_ask = info["fill_indicator"]
        total_fills += int(fill_bid) + int(fill_ask)
        spread_revenue += ask_depth * fill_ask + bid_depth * fill_bid
        if info["regime_at_event"] == 1:
            adverse_selection_loss += abs(info["jump_increment"]) * (fill_ask + fill_bid)

        inv_after = info["inventory_after"]
        inv_sum += inv_after
        inv_abs_sum += abs(inv_after)
        inv_max_abs = max(inv_max_abs, abs(inv_after))
        running_penalty += info["running_penalty_increment"]
        cumulative_objective += float(reward)
        n_steps += 1

    cash_T, inv_T, mid_T = info["cash_after"], info["inventory_after"], info["price_after"]
    raw_pnl = (cash_T + inv_T * mid_T) - (cash_0 + inv_0 * mid_0)
    terminal_penalty = info["terminal_penalty_increment"]
    full_objective = raw_pnl - running_penalty - terminal_penalty
    reconciliation_abs_diff = abs(full_objective - cumulative_objective)

    return dict(
        evaluation_seed=None,
        raw_pnl=raw_pnl, full_objective=full_objective,
        spread_revenue=spread_revenue, adverse_selection_loss=adverse_selection_loss,
        running_penalty=running_penalty, terminal_penalty=terminal_penalty,
        mean_abs_inventory=inv_abs_sum / n_steps, max_abs_inventory=inv_max_abs,
        mean_signed_inventory=inv_sum / n_steps, terminal_signed_inventory=inv_T,
        terminal_abs_inventory=abs(inv_T), fills=total_fills,
        bid_action_mean=float(action_stats.mean[0]), ask_action_mean=float(action_stats.mean[1]),
        bid_action_std=float(action_stats.std[0]), ask_action_std=float(action_stats.std[1]),
        mean_quoted_spread=float(depth_stats.mean[0] + depth_stats.mean[1]), episode_length=n_steps,
        reward_reconciliation_error=reconciliation_abs_diff,
        recurrent_state_reset_verified=(state_changed_within_episode) if is_recurrent else None,
    )


# ======================================================================
# Orchestration
# ======================================================================
def evaluate_all_agents_event_driven(holdout_seeds, model_paths: dict, training_seeds: dict,
                                      total_timesteps: dict, training_env_seeds: dict = None) -> pd.DataFrame:
    training_env_seeds = training_env_seeds or {}
    print("Verifying exogenous path pairing (event-driven)...")
    pairing = verify_exogenous_path_pairing()
    print(f"  hashes match across 4 structurally different action sequences: {pairing['all_matched']}")
    assert pairing["all_matched"], "Event-driven exogenous path hash differs across action sequences -- see pairing['hashes']."
    status = "exogenous_paired_fills_not_claimed_paired" if pairing["all_matched"] else "EXOGENOUS_PAIRING_FAILED"

    print("\nBuilding oracle optimal control tables (unchanged existing solver)...")
    controls = {}
    for regime, params in SBW.REGIME_PARAMS.items():
        da, db, qag, qbg = SBW.build_optimal_control(**params)
        controls[regime] = {"delta_ask": da, "delta_bid": db, "q_ask": qag, "q_bid": qbg}

    rows = []
    for agent_type in EAC.ANALYTIC_AGENT_TYPES:
        print(f"\n--- {agent_type} (event-driven) ---")
        for seed in holdout_seeds:
            m = run_event_analytic_agent_episode(agent_type, controls, seed)
            rows.append(EAC.build_row(agent_type, None, None, None, status, m))

    for agent_type in EAC.RL_AGENT_TYPES:
        path = model_paths.get(agent_type)
        if path is None:
            print(f"\n--- {agent_type}: SKIPPED (no model path given) ---")
            continue
        print(f"\n--- {agent_type} ({path}) [event-driven] ---")
        is_recurrent = agent_type in RECURRENT_AGENT_TYPES
        model_cls = RecurrentPPO if is_recurrent else PPO
        model = model_cls.load(str(path))
        _assert_model_is_event_driven(model, agent_type)
        ts = training_seeds.get(agent_type)
        tes = training_env_seeds.get(agent_type)
        tt = total_timesteps.get(agent_type)
        for seed in holdout_seeds:
            if agent_type == "hamilton_ppo":
                m = run_event_hamilton_agent_episode(model, seed)
            else:
                m = run_event_return_agent_episode(model, seed, is_recurrent=is_recurrent)
            m["evaluation_seed"] = seed
            rows.append(EAC.build_row(
                agent_type, ts, path, tt, status, m,
                checkpoint_selection="final", loaded_model_path=path, training_env_seed=tes,
            ))

    df = pd.DataFrame(rows, columns=EAC.SCHEMA_COLUMNS)
    df["environment_type"] = ENVIRONMENT_TYPE
    return df


def _assert_model_is_event_driven(model, agent_type: str):
    """Item 2's 'fail clearly if an event-driven model is evaluated using a
    fixed-step wrapper, or vice versa' -- checked here via the observation
    space width: event-driven return agents have a 4-wide observation
    ([q, tau, r, delta_tau]) vs the fixed-step return wrapper's 3-wide one
    ([q, tau, r]); the Hamilton wrapper is 3-wide in BOTH environment types
    (belief replaces the extra slot instead of adding one), so hamilton_ppo
    is instead checked via select_checkpoint_offline.py's/train_agents.py's
    run_config environment_type metadata (see tests/test_event_driven_agent_
    integration.py's mismatch tests) rather than observation width alone."""
    expected_shape = model.observation_space.shape
    if agent_type in ("return_mlp_ppo", "return_lstm_ppo") and expected_shape != (4,):
        raise ValueError(
            f"Model at this path has observation_space shape {expected_shape}, expected (4,) for an "
            f"event-driven {agent_type} model -- this looks like a FIXED-STEP model being evaluated "
            f"with the event-driven evaluator (or vice versa). Refusing to proceed."
        )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-tag", type=str, default="event_diag")
    p.add_argument("--training-seed", type=int, default=0)
    p.add_argument("--training-env-seed", type=int, default=None)
    p.add_argument("--holdout-seeds-start", type=int, default=DEFAULT_HOLDOUT_SEEDS[0])
    p.add_argument("--holdout-seeds-count", type=int, default=len(DEFAULT_HOLDOUT_SEEDS))
    p.add_argument("--output", type=str, default=str(RESULTS_DIR / "agent_comparison_event_episodes.csv"))
    p.add_argument("--total-training-timesteps", type=int, default=None)
    p.add_argument("--model-dir-suffix", type=str, default="_event",
                    help="Event-driven models live in models/<agent_type><suffix>/ (default '_event', "
                         "matching train_agents.py's own directory-separation convention).")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    holdout_seeds = list(range(args.holdout_seeds_start, args.holdout_seeds_start + args.holdout_seeds_count))

    model_paths = {}
    for at in EAC.RL_AGENT_TYPES:
        p = REPO_ROOT / "models" / f"{at}{args.model_dir_suffix}" / f"ppo_{at}_{args.run_tag}.zip"
        model_paths[at] = p if p.exists() else None
        if not p.exists():
            print(f"  NOTE: {p} not found -- {at} will be SKIPPED (analytic policies still evaluated).")

    training_seeds = {k: args.training_seed for k in EAC.RL_AGENT_TYPES}
    training_env_seed = args.training_env_seed if args.training_env_seed is not None else args.training_seed
    training_env_seeds = {k: training_env_seed for k in EAC.RL_AGENT_TYPES}
    total_timesteps = {k: args.total_training_timesteps for k in EAC.RL_AGENT_TYPES}

    t0 = time.time()
    df = evaluate_all_agents_event_driven(holdout_seeds, model_paths, training_seeds, total_timesteps, training_env_seeds)
    elapsed = time.time() - t0

    df.to_csv(args.output, index=False)
    print(f"\n{len(df)} rows written to {args.output} ({elapsed:.1f}s total)")

    flat = EAC.aggregate_flat(df, metrics=("full_objective", "raw_pnl", "mean_abs_inventory", "terminal_abs_inventory"))
    print("\n" + "=" * 100)
    print("POOLED SUMMARY (event-driven, all evaluation seeds, per agent_type) -- full_objective")
    print("=" * 100)
    print(flat[flat["metric"] == "full_objective"][
        ["agent_type", "n", "mean", "std", "ci_lo", "ci_hi", "median", "loss_rate"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    max_recon = df["reward_reconciliation_error"].max()
    print(f"\nMaximum reconciliation discrepancy across all {len(df)} episodes: {max_recon:.3e}")


if __name__ == "__main__":
    main()
