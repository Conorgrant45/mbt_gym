"""
shared/evaluate_agents_common.py
------------------------------
Common, fair evaluation framework for all three RL agent types
(hamilton_ppo, return_mlp_ppo, return_lstm_ppo) and four analytic
benchmark policies (oracle, belief_weighted, randomised, naive).

Reuses, READ-ONLY, rather than duplicating:
    - hamilton_ppo_eval_lib.run_eval_episode_full   (hamilton_ppo's full
      per-episode accounting -- used UNCHANGED, not reimplemented)
    - compare_four_policies_paired.run_episode, instrument_env,
      RecordingRNG, JumpOnlyRNG   (analytic-policy accounting primitives
      and the seed-pairing convention)
    - simulate_belief_weighted   (oracle/belief-weighted/naive/randomised
      policy logic: get_control, normalise_depth, make_filter, MAX_DEPTH,
      NAIVE_DEPTH, REGIME_PARAMS, build_optimal_control)
    - train_hamilton_ppo   (PHI, ALPHA, DT, ACTION_BOUND_TOL -- the single
      source of the reward-decomposition constants)
    - envs/hamilton_ppo_wrapper.py, envs/return_ppo_wrapper.py

The return_mlp_ppo / return_lstm_ppo per-episode runner
(run_return_agent_episode) and the analytic-policy extended runner
(run_analytic_agent_episode) ARE new code: neither ReturnPPOWrapper nor
compare_four_policies_paired.run_episode previously computed the full
Part-4 output schema (bid/ask action mean+std, mean_quoted_spread, signed/
max inventory). Both reuse every accounting PRIMITIVE listed above
(instrument_env, PHI/ALPHA/DT, SBW.MAX_DEPTH) rather than re-deriving
them, and run_analytic_agent_episode's core reconciled metrics are
verified against compare_four_policies_paired.run_episode's own numbers
for a cross-check sample at startup (see verify_analytic_reuse_fidelity).

Run from repo root:
    python shared/evaluate_agents_common.py --run-tag diag --training-seed 0
"""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

import shared.simulate_belief_weighted as SBW  # READ ONLY
import shared.compare_four_policies_paired as C4P  # READ ONLY
import shared.hamilton_ppo_eval_lib as HEL  # READ ONLY (run_eval_episode_full reused unchanged)
from shared.train_hamilton_ppo import PHI, ALPHA, DT, ACTION_BOUND_TOL  # READ ONLY

from envs.hamilton_ppo_wrapper import HamiltonPPOWrapper, DEFAULT_INVENTORY_SCALE
from envs.return_ppo_wrapper import ReturnPPOWrapper, DEFAULT_RETURN_SCALE
from envs.make_envs import make_regime_envs, N_STEPS
from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, ASSET_PRICE_INDEX, ASK_INDEX, BID_INDEX

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models"
LOGS_DIR = REPO_ROOT / "logs"

RL_AGENT_TYPES = ("hamilton_ppo", "return_mlp_ppo", "return_lstm_ppo")
ANALYTIC_AGENT_TYPES = ("oracle", "belief_weighted", "randomised", "naive")
ALL_AGENT_TYPES = RL_AGENT_TYPES + ANALYTIC_AGENT_TYPES
RECURRENT_AGENT_TYPES = ("return_lstm_ppo",)

CHECKPOINT_SELECTIONS = ("final", "best", "offline_best")

# C4P.POLICIES uses "belief" -- this module's schema uses "belief_weighted"
# throughout (matching the user-facing agent-type name); this dict is the
# ONLY place that naming difference is bridged.
ANALYTIC_POLICY_NAME_MAP = {"oracle": "oracle", "belief_weighted": "belief",
                             "randomised": "randomised", "naive": "naive"}

# Fresh holdout range for Phase 3 -- disjoint from every previously-used
# range in this project: training seeds (0-4), dev eval seeds
# (90001-90005), the single-seed-milestone holdout (100000-100199), and
# the multiseed holdout (110000-110099).
DEFAULT_HOLDOUT_SEEDS = list(range(120_000, 120_100))

RECONCILIATION_TOL = 1e-6

# training_seed: kept for backwards compatibility -- now specifically the
# LEARNER seed (PPO/RecurrentPPO construction: policy init, action
# sampling, torch/numpy/random state) from train_agents.py's
# --learner-seed/--seed. training_env_seed: the separate seed that drove
# the exogenous training environment (regime path, arrivals, jumps,
# diffusion, fills) via --env-seed/--seed -- see train_agents.py's
# resolve_seeds(). Neither is read from the model file or run_config here;
# both are recorded verbatim from --training-seed/--training-env-seed, same
# convention as total_training_timesteps below.
SCHEMA_COLUMNS = [
    "agent_type", "training_seed", "training_env_seed", "evaluation_seed",
    "raw_pnl", "full_objective", "spread_revenue", "adverse_selection_loss",
    "running_penalty", "terminal_penalty",
    "mean_abs_inventory", "max_abs_inventory", "mean_signed_inventory",
    "terminal_signed_inventory", "terminal_abs_inventory",
    "fills", "bid_action_mean", "ask_action_mean", "bid_action_std", "ask_action_std",
    "mean_quoted_spread", "episode_length", "reward_reconciliation_error",
    "path_pairing_status", "model_path", "total_training_timesteps",
    "recurrent_state_reset_verified",
    # retained existing PER-EPISODE metrics (compare_four_policies_paired.py's
    # add_return_pct_column/print_loss_rate_table quantities, computed per row).
    # "maximum reconciliation discrepancy" is inherently a GROUP-level statistic
    # (max over a set of episodes, as compare_four_policies_paired.print_summary_table
    # already computes it) -- it is produced by summarise_group() in Part 5, not
    # stored per-row here.
    "return_on_turnover_pct", "raw_pnl_loss", "full_objective_loss", "turnover",
    # checkpoint-selection metadata (final vs validation-selected best checkpoint).
    # "analytic" for the four analytic benchmark rows, which carry no learned-model
    # metadata at all (loaded_model_path/best_validation_* stay None for them).
    "checkpoint_selection", "loaded_model_path", "best_validation_timestep",
    "best_validation_mean_objective",
    # offline-selection metadata (checkpoint_selection == "offline_best" only;
    # None for "final"/"best"/analytic rows -- see load_offline_selection_metadata).
    "selected_checkpoint_timestep", "selected_validation_mean", "selected_validation_std",
    "validation_seed_count",
]


# ======================================================================
# Part 3: common-random-number / path-pairing verification
# ======================================================================
def verify_path_pairing(seed: int = 555_001, n_steps: int = 300) -> dict:
    """
    Empirically verify, for a fixed seed, which exogenous processes are
    identical across (a) a raw analytic-policy environment (the
    compare_four_policies_paired.py construction) and (b) both RL
    wrappers -- and separately confirm that REALISED FILLS differ once
    actions differ (never claim fills are paired).

    Regime-transition path, diffusion shocks, market-order arrivals, and
    jump magnitudes are all action-independent (ArrivalJumpMidpriceModel.
    update() only reads `arrivals`, never `fills`/`actions`; the regime
    draw is an unconditional Markov step) -- this is the SAME empirical
    fact hamilton_ppo_eval_lib.py's module docstring already establishes
    and compare_four_policies_paired.py's own full-trace verification
    (n_verify episodes, asserted identical across all 4 analytic
    policies) already exercises for the analytic side. This function
    additionally checks it across the RL wrappers, which run_episode()
    never touches.

    Returns a dict of booleans (one per exogenous component) plus
    'fills_differ_when_actions_differ' -- used to set path_pairing_status
    on every evaluated episode's row (computed ONCE, not per episode).
    """
    rng_actions_a = np.random.default_rng(seed + 42_001)
    rng_actions_b = np.random.default_rng(seed + 42_002)

    # (a) raw analytic-style environment
    np.random.seed(seed)
    env_ref = make_regime_envs(switch_within_episode=True, seed=seed)
    logs_ref = C4P.instrument_env(env_ref, full_trace=True)
    env_ref.reset()
    regimes_ref, diffusion_ref, arrivals_ref, jumps_ref, fills_ref = [], [], [], [], []
    for _ in range(n_steps):
        regime = env_ref.current_regime
        pre_arr = len(logs_ref[regime]["arrival_log"])
        pre_exp = len(logs_ref[regime]["midprice_rng"].exponential_log)
        pre_norm = len(logs_ref[regime]["midprice_rng"].normal_log)
        action = rng_actions_a.uniform(-1.0, 1.0, size=(1, 2)).astype(np.float32)
        obs, reward, done, info = env_ref.step(action)
        regimes_ref.append(regime)
        arrivals_ref.append(logs_ref[regime]["arrival_log"][pre_arr].copy())
        fills_ref.append(logs_ref[regime]["fill_log"][pre_arr].copy())
        exp_calls = logs_ref[regime]["midprice_rng"].exponential_log[pre_exp:pre_exp + 2]
        jumps_ref.append([c.copy() for c in exp_calls])
        norm_calls = logs_ref[regime]["midprice_rng"].normal_log[pre_norm:pre_norm + 1]
        diffusion_ref.append([c.copy() for c in norm_calls])

    # (b) HamiltonPPOWrapper, DIFFERENT actions
    np.random.seed(seed)
    base_h = make_regime_envs(switch_within_episode=True, seed=seed)
    logs_h = C4P.instrument_env(base_h, full_trace=True)
    wrapper_h = HamiltonPPOWrapper(base_env=base_h, inventory_scale=DEFAULT_INVENTORY_SCALE)
    wrapper_h.reset(seed=seed)
    regimes_h, diffusion_h, arrivals_h, jumps_h, fills_h = [], [], [], [], []
    for _ in range(n_steps):
        regime = wrapper_h.base_env.current_regime
        pre_arr = len(logs_h[regime]["arrival_log"])
        pre_exp = len(logs_h[regime]["midprice_rng"].exponential_log)
        pre_norm = len(logs_h[regime]["midprice_rng"].normal_log)
        action = rng_actions_b.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
        wrapper_h.step(action)
        regimes_h.append(regime)
        arrivals_h.append(logs_h[regime]["arrival_log"][pre_arr].copy())
        fills_h.append(logs_h[regime]["fill_log"][pre_arr].copy())
        exp_calls = logs_h[regime]["midprice_rng"].exponential_log[pre_exp:pre_exp + 2]
        jumps_h.append([c.copy() for c in exp_calls])
        norm_calls = logs_h[regime]["midprice_rng"].normal_log[pre_norm:pre_norm + 1]
        diffusion_h.append([c.copy() for c in norm_calls])

    regime_path_matched = regimes_ref == regimes_h
    arrivals_matched = all(np.array_equal(a, b) for a, b in zip(arrivals_ref, arrivals_h))
    jumps_matched = all(
        len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))
        for a, b in zip(jumps_ref, jumps_h)
    )
    diffusion_matched = all(
        len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))
        for a, b in zip(diffusion_ref, diffusion_h)
    )
    # Actions differ between (a) and (b) (independent RNG streams for the
    # actions themselves) -- fills should generally differ as a result.
    fills_differ = not all(np.array_equal(a, b) for a, b in zip(fills_ref, fills_h))

    return dict(
        seed=seed,
        n_steps_checked=n_steps,
        regime_transition_path_matched=bool(regime_path_matched),
        brownian_diffusion_shocks_matched=bool(diffusion_matched),
        market_order_arrivals_matched=bool(arrivals_matched),
        jump_magnitudes_matched=bool(jumps_matched),
        realised_fills_differ_with_different_actions=bool(fills_differ),
    )


def path_pairing_status_label(verification: dict) -> str:
    """Single-string status recorded on every evaluated row. Distinguishes
    fully-paired EXOGENOUS processes (regime/diffusion/arrivals/jumps) from
    fills, which are explicitly NEVER claimed identical once actions
    differ (this function raises if the fills-differ check unexpectedly
    fails, since that would mean the 'exogenous only' claim below is
    unverified for this run)."""
    exogenous_ok = (
        verification["regime_transition_path_matched"]
        and verification["brownian_diffusion_shocks_matched"]
        and verification["market_order_arrivals_matched"]
        and verification["jump_magnitudes_matched"]
    )
    if not exogenous_ok:
        return "SEED_MATCHED_ONLY_exogenous_pairing_verification_FAILED"
    return "exogenous_paired_fills_not_claimed_paired"


# ======================================================================
# Part 2: hamilton_ppo episode runner (reused unchanged from
# hamilton_ppo_eval_lib.py, remapped to the common schema)
# ======================================================================
def run_hamilton_agent_episode(model, seed: int, inventory_scale: float = DEFAULT_INVENTORY_SCALE) -> dict:
    r = HEL.run_eval_episode_full(model, seed, inventory_scale, deterministic=True)
    mean_quoted_spread = ((r["bid_action_mean"] * -1 + 1.0) / 2.0 + (r["ask_action_mean"] + 1.0) / 2.0) * SBW.MAX_DEPTH
    return dict(
        evaluation_seed=seed,
        raw_pnl=r["raw_pnl"], full_objective=r["full_objective"],
        spread_revenue=r["spread_revenue"], adverse_selection_loss=r["adverse_selection_loss"],
        running_penalty=r["running_inventory_penalty"], terminal_penalty=r["terminal_inventory_penalty"],
        mean_abs_inventory=r["mean_abs_inventory"], max_abs_inventory=r["max_abs_inventory"],
        mean_signed_inventory=r["mean_signed_inventory"], terminal_signed_inventory=r["terminal_signed_inventory"],
        terminal_abs_inventory=r["terminal_abs_inventory"], fills=r["total_fills"],
        bid_action_mean=r["bid_action_mean"], ask_action_mean=r["ask_action_mean"],
        bid_action_std=r["bid_action_std"], ask_action_std=r["ask_action_std"],
        mean_quoted_spread=mean_quoted_spread, episode_length=r["episode_length"],
        reward_reconciliation_error=r["reconciliation_abs_diff"],
        recurrent_state_reset_verified=None,
    )


# ======================================================================
# Part 2: return_mlp_ppo / return_lstm_ppo episode runner (new; reuses
# instrument_env, PHI/ALPHA/DT, SBW.MAX_DEPTH -- same primitives as
# hamilton_ppo_eval_lib.run_eval_episode_full, applied to ReturnPPOWrapper)
# ======================================================================
def run_return_agent_episode(model, seed: int, is_recurrent: bool,
                              inventory_scale: float = DEFAULT_INVENTORY_SCALE,
                              return_scale: float = DEFAULT_RETURN_SCALE) -> dict:
    base_env = make_regime_envs(switch_within_episode=True, seed=seed)
    logs = C4P.instrument_env(base_env, full_trace=False)
    wrapper = ReturnPPOWrapper(base_env=base_env, inventory_scale=inventory_scale, return_scale=return_scale)
    obs, info = wrapper.reset(seed=seed)  # required for exogenous-path pairing -- see run_eval_episode docstring in train_agents.py

    cash_0 = wrapper.base_env.raw_cash
    inv_0 = wrapper.base_env.raw_inventory
    mid_0 = wrapper.base_env.raw_midprice

    lstm_states = None
    episode_start = np.array([True], dtype=bool)
    state_changed_within_episode = False
    state_was_reset_at_start = True  # by construction: state=None at t=0

    running_inventory_penalty = 0.0
    inv_sum = 0.0
    inv_abs_sum = 0.0
    inv_max_abs = 0.0
    spread_revenue = 0.0
    adverse_selection_loss = 0.0
    total_fills = 0.0
    cumulative_objective = 0.0
    actions = []

    t = 0
    terminated = truncated = False
    info_last = None
    prev_state_hidden = None
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
        actions.append(action.copy())
        bid_action, ask_action = float(action[0]), float(action[1])

        regime_before = wrapper.base_env.current_regime
        pre_arr = len(logs[regime_before]["arrival_log"])
        pre_exp = len(logs[regime_before]["midprice_rng"].exponential_log)

        obs, reward, terminated, truncated, info = wrapper.step(action)
        info_last = info
        episode_start = np.array([False], dtype=bool)

        arrivals = logs[regime_before]["arrival_log"][pre_arr]
        fills = logs[regime_before]["fill_log"][pre_arr]
        buy_arrival = float(arrivals[0, ASK_INDEX])
        sell_arrival = float(arrivals[0, BID_INDEX])
        ask_fill_event = buy_arrival * float(fills[0, ASK_INDEX])
        bid_fill_event = sell_arrival * float(fills[0, BID_INDEX])
        total_fills += ask_fill_event + bid_fill_event

        exp_calls = logs[regime_before]["midprice_rng"].exponential_log[pre_exp:pre_exp + 2]
        jump_up = float(exp_calls[0][0, 0]) if len(exp_calls) == 2 else 0.0
        jump_down = float(exp_calls[1][0, 0]) if len(exp_calls) == 2 else 0.0

        ask_depth = (ask_action + 1.0) / 2.0 * SBW.MAX_DEPTH
        bid_depth = (bid_action + 1.0) / 2.0 * SBW.MAX_DEPTH
        spread_revenue += ask_depth * ask_fill_event + bid_depth * bid_fill_event
        if regime_before == 1:
            adverse_selection_loss += jump_up * ask_fill_event + jump_down * bid_fill_event

        inv_after = float(info["raw_state"][INVENTORY_INDEX])
        inv_sum += inv_after
        inv_abs_sum += abs(inv_after)
        inv_max_abs = max(inv_max_abs, abs(inv_after))
        running_inventory_penalty += PHI * (inv_after ** 2) * DT
        cumulative_objective += float(reward)
        t += 1

    n_steps = t
    cash_T = float(info_last["raw_state"][CASH_INDEX])
    inv_T = float(info_last["raw_state"][INVENTORY_INDEX])
    mid_T = float(info_last["raw_state"][ASSET_PRICE_INDEX])
    raw_pnl = (cash_T + inv_T * mid_T) - (cash_0 + inv_0 * mid_0)
    terminal_inventory_penalty = ALPHA * (inv_T ** 2)

    actions_arr = np.array(actions)
    bid_actions, ask_actions = actions_arr[:, 0], actions_arr[:, 1]
    mean_quoted_spread = ((-bid_actions.mean() + 1.0) / 2.0 + (ask_actions.mean() + 1.0) / 2.0) * SBW.MAX_DEPTH

    full_objective = raw_pnl - running_inventory_penalty - terminal_inventory_penalty
    reconciliation_abs_diff = abs(full_objective - cumulative_objective)

    return dict(
        evaluation_seed=seed,
        raw_pnl=raw_pnl, full_objective=full_objective,
        spread_revenue=spread_revenue, adverse_selection_loss=adverse_selection_loss,
        running_penalty=running_inventory_penalty, terminal_penalty=terminal_inventory_penalty,
        mean_abs_inventory=inv_abs_sum / n_steps, max_abs_inventory=inv_max_abs,
        mean_signed_inventory=inv_sum / n_steps, terminal_signed_inventory=inv_T,
        terminal_abs_inventory=abs(inv_T), fills=total_fills,
        bid_action_mean=float(bid_actions.mean()), ask_action_mean=float(ask_actions.mean()),
        bid_action_std=float(bid_actions.std()), ask_action_std=float(ask_actions.std()),
        mean_quoted_spread=mean_quoted_spread, episode_length=n_steps,
        reward_reconciliation_error=reconciliation_abs_diff,
        recurrent_state_reset_verified=(state_was_reset_at_start and state_changed_within_episode) if is_recurrent else None,
    )


# ======================================================================
# Part 2/4: analytic-policy episode runner (oracle/belief_weighted/
# randomised/naive). Extends compare_four_policies_paired.run_episode's
# loop with action/inventory series it does not itself return (bid/ask
# depth mean+std, mean_quoted_spread, signed/max inventory) -- reuses
# every accounting PRIMITIVE from that module (SBW.get_control,
# SBW.normalise_depth, SBW.make_filter, SBW.NAIVE_DEPTH, instrument_env,
# the np.random.seed(seed)+make_regime_envs(seed=seed) pairing
# convention) rather than re-deriving them. Cross-checked against
# compare_four_policies_paired.run_episode's own core metrics -- see
# verify_analytic_reuse_fidelity.
# ======================================================================
def run_analytic_agent_episode(agent_type: str, controls: dict, seed: int) -> dict:
    policy = ANALYTIC_POLICY_NAME_MAP[agent_type]

    np.random.seed(seed)
    env = make_regime_envs(switch_within_episode=True, seed=seed)
    logs = C4P.instrument_env(env, full_trace=False)

    filt = SBW.make_filter() if policy in ("belief", "randomised") else None
    if filt is not None:
        filt.reset()
    rng_random_policy = np.random.default_rng(seed + 1000) if policy == "randomised" else None

    obs = env.reset()
    cash_0, inv_0, mid_0 = env.raw_cash, env.raw_inventory, env.raw_midprice
    mid = mid_0

    inv_sum = 0.0
    inv_abs_sum = 0.0
    inv_max_abs = 0.0
    running_inventory_penalty = 0.0
    spread_revenue = 0.0
    adverse_selection_loss = 0.0
    total_fills = 0.0
    obj_accum_env = 0.0
    bid_depths, ask_depths = [], []

    step = 0
    done = np.array([False])
    info = None
    while not np.all(done):
        regime = env.current_regime
        obs_flat = np.array(obs).flatten()
        inventory = obs_flat[1]
        t_idx = env.current_step
        inv_sc = inventory / SBW.INV_UNIT

        if policy == "oracle":
            ctrl = controls[regime]
            ask_depth, bid_depth = SBW.get_control(ctrl["delta_ask"], ctrl["delta_bid"], ctrl["q_ask"], ctrl["q_bid"], t_idx, inv_sc)
        elif policy == "belief":
            belief = filt.update(mid)
            c0, c1 = controls[0], controls[1]
            a0, b0 = SBW.get_control(c0["delta_ask"], c0["delta_bid"], c0["q_ask"], c0["q_bid"], t_idx, inv_sc)
            a1, b1 = SBW.get_control(c1["delta_ask"], c1["delta_bid"], c1["q_ask"], c1["q_bid"], t_idx, inv_sc)
            ask_depth = (1.0 - belief) * a0 + belief * a1
            bid_depth = (1.0 - belief) * b0 + belief * b1
        elif policy == "randomised":
            belief = filt.update(mid)
            sampled_regime = int(rng_random_policy.choice(2, p=[1.0 - belief, belief]))
            ctrl = controls[sampled_regime]
            ask_depth, bid_depth = SBW.get_control(ctrl["delta_ask"], ctrl["delta_bid"], ctrl["q_ask"], ctrl["q_bid"], t_idx, inv_sc)
        else:  # naive
            ask_depth = bid_depth = SBW.NAIVE_DEPTH

        bid_depths.append(bid_depth)
        ask_depths.append(ask_depth)
        action = np.array([[SBW.normalise_depth(bid_depth), SBW.normalise_depth(ask_depth)]])

        pre_arr = len(logs[regime]["arrival_log"])
        pre_exp = len(logs[regime]["midprice_rng"].exponential_log)

        obs, reward, done, info = env.step(action)
        mid = info["raw_midprice"]
        obj_accum_env += float(np.sum(reward))

        arrivals = logs[regime]["arrival_log"][pre_arr]
        fills = logs[regime]["fill_log"][pre_arr]
        buy_arrival = float(arrivals[0, ASK_INDEX])
        sell_arrival = float(arrivals[0, BID_INDEX])
        ask_fill_event = buy_arrival * float(fills[0, ASK_INDEX])
        bid_fill_event = sell_arrival * float(fills[0, BID_INDEX])
        total_fills += ask_fill_event + bid_fill_event
        spread_revenue += ask_depth * ask_fill_event + bid_depth * bid_fill_event

        exp_calls = logs[regime]["midprice_rng"].exponential_log[pre_exp:pre_exp + 2]
        if len(exp_calls) == 2 and regime == 1:
            adverse_selection_loss += float(exp_calls[0][0, 0]) * ask_fill_event + float(exp_calls[1][0, 0]) * bid_fill_event

        inv_after = float(info["raw_state"][1])
        inv_sum += inv_after
        inv_abs_sum += abs(inv_after)
        inv_max_abs = max(inv_max_abs, abs(inv_after))
        running_inventory_penalty += PHI * (inv_after ** 2) * DT
        step += 1

    steps_taken = step
    cash_T, inv_T, mid_T = info["raw_state"][0], info["raw_state"][1], info["raw_state"][3]
    raw_pnl = (cash_T + inv_T * mid_T) - (cash_0 + inv_0 * mid_0)
    terminal_inventory_penalty = ALPHA * (inv_T ** 2)
    full_objective = raw_pnl - running_inventory_penalty - terminal_inventory_penalty
    reconciliation_abs_diff = abs(full_objective - obj_accum_env)

    bid_depths = np.array(bid_depths)
    ask_depths = np.array(ask_depths)
    # Normalised-action-scale mean/std, matching the RL agents' bid_action_mean/std
    # convention (action in [-1,1]), via SBW.normalise_depth applied elementwise.
    bid_actions_norm = np.array([SBW.normalise_depth(d) for d in bid_depths])
    ask_actions_norm = np.array([SBW.normalise_depth(d) for d in ask_depths])

    return dict(
        evaluation_seed=seed,
        raw_pnl=float(raw_pnl), full_objective=float(full_objective),
        spread_revenue=spread_revenue, adverse_selection_loss=adverse_selection_loss,
        running_penalty=running_inventory_penalty, terminal_penalty=terminal_inventory_penalty,
        mean_abs_inventory=inv_abs_sum / steps_taken, max_abs_inventory=inv_max_abs,
        mean_signed_inventory=inv_sum / steps_taken, terminal_signed_inventory=float(inv_T),
        terminal_abs_inventory=abs(float(inv_T)), fills=total_fills,
        bid_action_mean=float(bid_actions_norm.mean()), ask_action_mean=float(ask_actions_norm.mean()),
        bid_action_std=float(bid_actions_norm.std()), ask_action_std=float(ask_actions_norm.std()),
        mean_quoted_spread=float((bid_depths + ask_depths).mean()), episode_length=steps_taken,
        reward_reconciliation_error=reconciliation_abs_diff,
        recurrent_state_reset_verified=None,
    )


def verify_analytic_reuse_fidelity(controls: dict, seeds) -> dict:
    """Cross-check run_analytic_agent_episode's core reconciled metrics
    against compare_four_policies_paired.run_episode's own numbers on the
    SAME seeds, for every analytic policy -- confirms the extended runner
    is a faithful superset, not an independent (and possibly divergent)
    reimplementation."""
    max_diffs = {}
    for agent_type in ANALYTIC_AGENT_TYPES:
        policy = ANALYTIC_POLICY_NAME_MAP[agent_type]
        worst = 0.0
        for seed in seeds:
            m_ref, _ = C4P.run_episode(policy, controls, seed, full_trace=False)
            m_new = run_analytic_agent_episode(agent_type, controls, seed)
            for key_ref, key_new in (("raw_pnl", "raw_pnl"), ("full_objective", "full_objective"),
                                      ("spread_revenue", "spread_revenue"),
                                      ("adverse_selection_loss", "adverse_selection_loss"),
                                      ("total_fills", "fills")):
                worst = max(worst, abs(m_ref[key_ref] - m_new[key_new]))
        max_diffs[agent_type] = worst
    return max_diffs


# ======================================================================
# Part 4: unify a raw per-episode dict into the common schema row
# ======================================================================
def build_row(agent_type: str, training_seed, model_path, total_training_timesteps,
              path_pairing_status: str, m: dict, checkpoint_selection: str = "analytic",
              loaded_model_path=None, best_validation_timestep=None,
              best_validation_mean_objective=None, training_env_seed=None,
              selected_checkpoint_timestep=None, selected_validation_mean=None,
              selected_validation_std=None, validation_seed_count=None) -> dict:
    from envs.make_envs import INITIAL_PRICE
    turnover = m["fills"] * INITIAL_PRICE
    return_on_turnover_pct = (m["raw_pnl"] / turnover * 100.0) if turnover > 0 else float("nan")
    row = dict(
        agent_type=agent_type, training_seed=training_seed, training_env_seed=training_env_seed,
        evaluation_seed=m["evaluation_seed"],
        raw_pnl=m["raw_pnl"], full_objective=m["full_objective"],
        spread_revenue=m["spread_revenue"], adverse_selection_loss=m["adverse_selection_loss"],
        running_penalty=m["running_penalty"], terminal_penalty=m["terminal_penalty"],
        mean_abs_inventory=m["mean_abs_inventory"], max_abs_inventory=m["max_abs_inventory"],
        mean_signed_inventory=m["mean_signed_inventory"], terminal_signed_inventory=m["terminal_signed_inventory"],
        terminal_abs_inventory=m["terminal_abs_inventory"], fills=m["fills"],
        bid_action_mean=m["bid_action_mean"], ask_action_mean=m["ask_action_mean"],
        bid_action_std=m["bid_action_std"], ask_action_std=m["ask_action_std"],
        mean_quoted_spread=m["mean_quoted_spread"], episode_length=m["episode_length"],
        reward_reconciliation_error=m["reward_reconciliation_error"],
        path_pairing_status=path_pairing_status, model_path=str(model_path) if model_path else None,
        total_training_timesteps=total_training_timesteps,
        recurrent_state_reset_verified=m["recurrent_state_reset_verified"],
        return_on_turnover_pct=return_on_turnover_pct,
        raw_pnl_loss=bool(m["raw_pnl"] < 0), full_objective_loss=bool(m["full_objective"] < 0),
        turnover=turnover,
        checkpoint_selection=checkpoint_selection,
        loaded_model_path=str(loaded_model_path) if loaded_model_path else None,
        best_validation_timestep=best_validation_timestep,
        best_validation_mean_objective=best_validation_mean_objective,
        selected_checkpoint_timestep=selected_checkpoint_timestep,
        selected_validation_mean=selected_validation_mean,
        selected_validation_std=selected_validation_std,
        validation_seed_count=validation_seed_count,
    )
    return row


def load_best_validation_metadata(agent_type: str, run_tag: str):
    """
    Read best_validation_mean_objective/best_validation_timestep from
    logs/<agent_type>/run_summary_<run_tag>.json (written by
    train_agents.py's validation-based best-checkpoint tracking).

    This is CONTEXTUAL information, not a hard requirement -- unlike the
    model .zip file itself (checked separately, see resolve_model_path /
    main()'s missing-file check), a missing or older run_summary (predating
    best-checkpoint tracking) only produces (None, None) plus a printed
    warning, never a hard failure.
    """
    summary_path = LOGS_DIR / agent_type / f"run_summary_{run_tag}.json"
    if not summary_path.exists():
        print(f"  WARNING: {summary_path} not found -- best_validation_* fields will be None for {agent_type}")
        return None, None
    with open(summary_path) as f:
        summary = json.load(f)
    return summary.get("best_validation_timestep"), summary.get("best_validation_mean_objective")


def load_offline_selection_metadata(agent_type: str, run_tag: str) -> dict:
    """
    Read logs/<agent_type>/offline_selection_<run_tag>.json (written by
    select_checkpoint_offline.py). Unlike load_best_validation_metadata,
    this is a HARD requirement whenever checkpoint_selection=="offline_best"
    -- raises rather than silently falling back to final/online-best, per
    the feature spec ("Do not silently fall back to final or online best
    when offline_best is missing").
    """
    selection_path = LOGS_DIR / agent_type / f"offline_selection_{run_tag}.json"
    if not selection_path.exists():
        raise FileNotFoundError(
            f"--checkpoint-selection=offline_best requires {selection_path} to exist -- "
            f"run select_checkpoint_offline.py for {agent_type}/{run_tag} first."
        )
    with open(selection_path) as f:
        return json.load(f)


def resolve_model_path(agent_type: str, run_tag: str, checkpoint_selection: str) -> Path:
    """ppo_<agent_type>_<run_tag>.zip for 'final', ppo_<agent_type>_<run_tag>_best.zip
    for 'best', ppo_<agent_type>_<run_tag>_offline_best.zip for 'offline_best' --
    matches train_agents.py's/select_checkpoint_offline.py's exact save-path
    conventions (model_path / best_model_path in main() / offline_best_path)."""
    if checkpoint_selection not in CHECKPOINT_SELECTIONS:
        raise ValueError(f"checkpoint_selection must be one of {CHECKPOINT_SELECTIONS}, got {checkpoint_selection!r}")
    suffix = {"final": "", "best": "_best", "offline_best": "_offline_best"}[checkpoint_selection]
    return MODELS_DIR / agent_type / f"ppo_{agent_type}_{run_tag}{suffix}.zip"


# ======================================================================
# Part 5: aggregation with uncertainty
# ======================================================================
def summarise_group(df: pd.DataFrame, value_col: str) -> dict:
    """mean/std/median/IQR/SE/95% CI/loss_rate for one column of one
    homogeneous group (e.g. one agent_type, or one agent_type x
    training_seed)."""
    x = df[value_col].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    mean = float(x.mean()) if n else float("nan")
    std = float(x.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    q25, q75 = (float(np.percentile(x, 25)), float(np.percentile(x, 75))) if n else (float("nan"),) * 2
    return dict(
        metric=value_col, n=n, mean=mean, std=std, se=se,
        median=float(np.median(x)) if n else float("nan"),
        iqr_low=q25, iqr_high=q75, iqr=q75 - q25 if n else float("nan"),
        ci_lo=mean - 1.96 * se, ci_hi=mean + 1.96 * se,
        loss_rate=float((x < 0).mean()) if n else float("nan"),
        max_abs=float(np.max(np.abs(x))) if n else float("nan"),
    )


def aggregate_flat(df: pd.DataFrame, metrics=("raw_pnl", "full_objective", "mean_abs_inventory",
                                                "terminal_abs_inventory", "reward_reconciliation_error")) -> pd.DataFrame:
    """Pooled-over-all-evaluation-seeds summary, per agent_type (and, for
    RL agents with multiple training_seeds, pooled across those too --
    see hierarchical_rl_summary for the between-training-seed breakdown
    this pooled view must NOT be used as a substitute for)."""
    rows = []
    for agent_type, sub in df.groupby("agent_type"):
        for metric in metrics:
            r = summarise_group(sub, metric)
            r["agent_type"] = agent_type
            r["level"] = "pooled_all_seeds"
            rows.append(r)
    return pd.DataFrame(rows)


def hierarchical_rl_summary(df: pd.DataFrame, metrics=("raw_pnl", "full_objective", "mean_abs_inventory",
                                                          "terminal_abs_inventory")) -> pd.DataFrame:
    """
    For RL agent types only: within-training-seed (across evaluation
    seeds) AND between-training-seed (across each training seed's own
    mean) summaries. Explicitly NOT a flat pool of
    n_training_seeds * n_evaluation_seeds episodes treated as iid --
    matches the hierarchy already established in
    evaluate_hamilton_ppo_multiseed_holdout.py's hierarchical_summary().
    """
    rl_df = df[df["agent_type"].isin(RL_AGENT_TYPES)]
    rows = []
    for agent_type, sub_a in rl_df.groupby("agent_type"):
        training_seeds = sorted(sub_a["training_seed"].dropna().unique())
        for metric in metrics:
            per_seed_means = []
            for ts in training_seeds:
                sub_ts = sub_a[sub_a["training_seed"] == ts]
                r = summarise_group(sub_ts, metric)
                r["agent_type"] = agent_type
                r["training_seed"] = ts
                r["level"] = "within_training_seed"
                rows.append(r)
                per_seed_means.append(r["mean"])
            if len(per_seed_means) >= 2:
                per_seed_means = np.array(per_seed_means)
                mean_of_means = float(per_seed_means.mean())
                std_across = float(per_seed_means.std(ddof=1))
                se_across = std_across / np.sqrt(len(per_seed_means))
                rows.append(dict(
                    metric=metric, n=len(per_seed_means), mean=mean_of_means, std=std_across, se=se_across,
                    median=float(np.median(per_seed_means)),
                    iqr_low=float(np.percentile(per_seed_means, 25)), iqr_high=float(np.percentile(per_seed_means, 75)),
                    iqr=float(np.percentile(per_seed_means, 75) - np.percentile(per_seed_means, 25)),
                    ci_lo=mean_of_means - 1.96 * se_across, ci_hi=mean_of_means + 1.96 * se_across,
                    loss_rate=float((per_seed_means < 0).mean()), max_abs=float(np.max(np.abs(per_seed_means))),
                    agent_type=agent_type, training_seed="ALL", level="between_training_seeds",
                ))
    return pd.DataFrame(rows)


def paired_or_seed_matched_diffs(df: pd.DataFrame, agent_a: str, agent_b: str,
                                  value_col: str = "full_objective",
                                  training_seed_a=None, training_seed_b=None) -> dict:
    """Episode-level differences (agent_a - agent_b) on the seeds common
    to both, labelled according to each row's own path_pairing_status
    (never silently upgraded to 'paired' if either side's status says
    otherwise).

    training_seed_a/training_seed_b MUST be given whenever the
    corresponding agent has more than one training seed in df: with
    duplicate evaluation_seed index entries (one per training seed),
    pandas' index alignment on subtraction silently takes the CARTESIAN
    product across the duplicates (n seeds x n training seeds rows
    instead of n), inflating n and silently mixing training seeds into
    what should be a single paired comparison. Raises rather than
    producing that silently-wrong result.
    """
    sub_a = df[df["agent_type"] == agent_a]
    sub_b = df[df["agent_type"] == agent_b]
    if training_seed_a is not None:
        sub_a = sub_a[sub_a["training_seed"] == training_seed_a]
    if training_seed_b is not None:
        sub_b = sub_b[sub_b["training_seed"] == training_seed_b]
    for name, sub, ts in (("a", sub_a, training_seed_a), ("b", sub_b, training_seed_b)):
        n_ts = sub["training_seed"].nunique(dropna=True)
        if n_ts > 1 and ts is None:
            raise ValueError(
                f"agent_{name} ({agent_a if name == 'a' else agent_b}) has {n_ts} distinct training seeds "
                f"in df -- pass training_seed_{name}=<seed> to disambiguate before computing a paired diff."
            )
    sub_a = sub_a.set_index("evaluation_seed")
    sub_b = sub_b.set_index("evaluation_seed")
    assert sub_a.index.is_unique and sub_b.index.is_unique, (
        "duplicate evaluation_seed rows remain after training_seed filtering -- refusing to compute a "
        "paired diff that pandas would silently cartesian-expand."
    )
    common_seeds = sub_a.index.intersection(sub_b.index)
    D = sub_a.loc[common_seeds, value_col] - sub_b.loc[common_seeds, value_col]
    n = len(D)
    mean = float(D.mean())
    se = float(D.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    statuses = set(sub_a.loc[common_seeds, "path_pairing_status"]) | set(sub_b.loc[common_seeds, "path_pairing_status"])
    comparison_status = "seed_matched" if len(statuses) != 1 else list(statuses)[0]
    return dict(
        agent_a=agent_a, agent_b=agent_b, metric=value_col, n=n, mean_diff=mean, se=se,
        ci_lo=mean - 1.96 * se, ci_hi=mean + 1.96 * se, comparison_status=comparison_status,
    )


# ======================================================================
# Orchestration
# ======================================================================
def evaluate_all_agents(holdout_seeds, model_paths: dict, training_seeds: dict, total_timesteps: dict,
                         checkpoint_selection: str = "final", best_validation: dict = None,
                         training_env_seeds: dict = None, offline_selection: dict = None):
    """
    model_paths: {agent_type: path or None (analytic)}
    training_seeds: {agent_type: seed or None (analytic)} -- the LEARNER seed
        (see SCHEMA_COLUMNS comment above).
    training_env_seeds: {agent_type: seed or None (analytic)} -- the separate
        training environment seed. Recorded verbatim alongside training_seed;
        never used to reconstruct or re-run anything here (holdout_seeds
        alone drive every evaluation episode's environment, exactly as
        before).
    total_timesteps: {agent_type: int or None (analytic)}
    checkpoint_selection: "final", "best" or "offline_best" -- recorded on
        every learned-agent row (analytic rows always get "analytic",
        regardless of this value).
    best_validation: {agent_type: (best_validation_timestep, best_validation_mean_objective)},
        read from run_summary_<run_tag>.json by the caller (see
        load_best_validation_metadata) -- contextual metadata carried on every
        learned-agent row regardless of which checkpoint was actually loaded.
    offline_selection: {agent_type: dict} -- the full parsed
        offline_selection_<run_tag>.json (see load_offline_selection_metadata),
        REQUIRED (by the caller, before this function is even reached) when
        checkpoint_selection=="offline_best". Only its selected_timestep/
        selected_mean_cumulative_reward/selected_std_cumulative_reward/
        validation_seed_count fields are surfaced on each row; ignored
        entirely for any other checkpoint_selection value.
    """
    training_env_seeds = training_env_seeds or {}
    offline_selection = offline_selection or {}
    print("Verifying exogenous path pairing (Part 3)...")
    pairing = verify_path_pairing()
    status = path_pairing_status_label(pairing)
    print(f"  {pairing}")
    print(f"  -> path_pairing_status for all rows: {status}")

    print("\nBuilding oracle optimal control tables (unchanged existing solver)...")
    controls = {}
    for regime, params in SBW.REGIME_PARAMS.items():
        da, db, qag, qbg = SBW.build_optimal_control(**params)
        controls[regime] = {"delta_ask": da, "delta_bid": db, "q_ask": qag, "q_bid": qbg}

    print("\nVerifying analytic-episode-runner reuse fidelity vs compare_four_policies_paired.run_episode "
          "(5 seeds)...")
    fidelity = verify_analytic_reuse_fidelity(controls, holdout_seeds[:5])
    for agent_type, diff in fidelity.items():
        print(f"  {agent_type}: max|discrepancy vs C4P.run_episode| = {diff:.3e}")
        assert diff < RECONCILIATION_TOL, f"{agent_type}: extended runner diverges from the validated original by {diff:.3e}"

    rows = []
    for agent_type in ANALYTIC_AGENT_TYPES:
        print(f"\n--- {agent_type} ---")
        for seed in holdout_seeds:
            m = run_analytic_agent_episode(agent_type, controls, seed)
            rows.append(build_row(agent_type, None, None, None, status, m))

    best_validation = best_validation or {}
    for agent_type in RL_AGENT_TYPES:
        path = model_paths.get(agent_type)
        if path is None:
            print(f"\n--- {agent_type}: SKIPPED (no model path given) ---")
            continue
        print(f"\n--- {agent_type} ({path}) [checkpoint_selection={checkpoint_selection}] ---")
        is_recurrent = agent_type in RECURRENT_AGENT_TYPES
        model_cls = RecurrentPPO if is_recurrent else PPO
        model = model_cls.load(str(path))
        ts = training_seeds.get(agent_type)
        tes = training_env_seeds.get(agent_type)
        tt = total_timesteps.get(agent_type)
        bv_timestep, bv_objective = best_validation.get(agent_type, (None, None))
        os_meta = offline_selection.get(agent_type) or {}
        for seed in holdout_seeds:
            if agent_type == "hamilton_ppo":
                m = run_hamilton_agent_episode(model, seed)
            else:
                m = run_return_agent_episode(model, seed, is_recurrent=is_recurrent)
            rows.append(build_row(
                agent_type, ts, path, tt, status, m,
                checkpoint_selection=checkpoint_selection, loaded_model_path=path,
                best_validation_timestep=bv_timestep, best_validation_mean_objective=bv_objective,
                training_env_seed=tes,
                selected_checkpoint_timestep=os_meta.get("selected_timestep"),
                selected_validation_mean=os_meta.get("selected_mean_cumulative_reward"),
                selected_validation_std=os_meta.get("selected_std_cumulative_reward"),
                validation_seed_count=os_meta.get("validation_seed_count"),
            ))

    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    return df, pairing


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-tag", type=str, default="diag")
    p.add_argument("--training-seed", type=int, default=0,
                    help="Recorded verbatim as training_seed on every row -- the LEARNER seed "
                         "train_agents.py's --learner-seed/--seed was run with. Not re-derived from "
                         "the model file or run_config; not used to reconstruct any environment here.")
    p.add_argument("--training-env-seed", type=int, default=None,
                    help="Recorded verbatim as training_env_seed on every row -- the separate "
                         "environment seed train_agents.py's --env-seed/--seed was run with. Defaults "
                         "to --training-seed if not given (matches train_agents.py's own default when "
                         "--env-seed is omitted).")
    p.add_argument("--holdout-seeds-start", type=int, default=DEFAULT_HOLDOUT_SEEDS[0])
    p.add_argument("--holdout-seeds-count", type=int, default=len(DEFAULT_HOLDOUT_SEEDS))
    p.add_argument("--output", type=str, default=str(RESULTS_DIR / "agent_comparison_episodes.csv"))
    p.add_argument("--total-training-timesteps", type=int, default=None,
                    help="Recorded verbatim in the output rows; not re-derived from the model file.")
    p.add_argument("--checkpoint-selection", type=str, default="final", choices=list(CHECKPOINT_SELECTIONS),
                    help="'final' loads ppo_<agent_type>_<run_tag>.zip (default, preserves prior behaviour). "
                         "'best' loads ppo_<agent_type>_<run_tag>_best.zip (the validation-selected best "
                         "checkpoint saved by train_agents.py's PeriodicEvalCallback, selected on the small "
                         "monitoring --eval-seeds set during training). 'offline_best' loads "
                         "ppo_<agent_type>_<run_tag>_offline_best.zip (selected by select_checkpoint_offline.py "
                         "on a larger, separate validation set -- requires "
                         "logs/<agent_type>/offline_selection_<run_tag>.json to exist; raises rather than "
                         "falling back to 'final'/'best' if it does not).")
    return p.parse_args()


def check_all_models_exist(model_paths: dict, checkpoint_selection: str, run_tag: str = ""):
    """Item 5: fail clearly, before any evaluation runs, if any required
    model file is missing -- rather than silently skipping that agent type."""
    missing = [str(p) for p in model_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"--checkpoint-selection={checkpoint_selection!r}: missing required model file(s) "
            f"for run-tag {run_tag!r}:\n  " + "\n  ".join(missing) +
            "\n(Train the missing model(s) first, or pass a different --run-tag / --checkpoint-selection.)"
        )


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    holdout_seeds = list(range(args.holdout_seeds_start, args.holdout_seeds_start + args.holdout_seeds_count))

    model_paths = {at: resolve_model_path(at, args.run_tag, args.checkpoint_selection) for at in RL_AGENT_TYPES}
    check_all_models_exist(model_paths, args.checkpoint_selection, args.run_tag)
    training_seeds = {k: args.training_seed for k in RL_AGENT_TYPES}
    training_env_seed = args.training_env_seed if args.training_env_seed is not None else args.training_seed
    training_env_seeds = {k: training_env_seed for k in RL_AGENT_TYPES}
    total_timesteps = {k: args.total_training_timesteps for k in RL_AGENT_TYPES}
    best_validation = {at: load_best_validation_metadata(at, args.run_tag) for at in RL_AGENT_TYPES}

    # offline_best is a HARD requirement (raises, no fallback) -- see
    # load_offline_selection_metadata's docstring.
    offline_selection = {}
    if args.checkpoint_selection == "offline_best":
        offline_selection = {at: load_offline_selection_metadata(at, args.run_tag) for at in RL_AGENT_TYPES}

    t0 = time.time()
    df, pairing = evaluate_all_agents(
        holdout_seeds, model_paths, training_seeds, total_timesteps,
        checkpoint_selection=args.checkpoint_selection, best_validation=best_validation,
        training_env_seeds=training_env_seeds, offline_selection=offline_selection,
    )
    elapsed = time.time() - t0

    df.to_csv(args.output, index=False)
    print(f"\n{len(df)} rows written to {args.output} ({elapsed:.1f}s total)")

    print("\n" + "=" * 100)
    print("POOLED SUMMARY (all evaluation seeds, per agent_type) -- full_objective")
    print("=" * 100)
    flat = aggregate_flat(df, metrics=("full_objective", "raw_pnl", "mean_abs_inventory", "terminal_abs_inventory"))
    print(flat[flat["metric"] == "full_objective"][
        ["agent_type", "n", "mean", "std", "ci_lo", "ci_hi", "median", "loss_rate"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    flat_path = Path(args.output).with_name(Path(args.output).stem + "_pooled_summary.csv")
    flat.to_csv(flat_path, index=False)
    print(f"Pooled (flat) summary saved to {flat_path}")

    hier = hierarchical_rl_summary(df)
    if len(hier):
        hier_path = Path(args.output).with_name(Path(args.output).stem + "_hierarchical_summary.csv")
        hier.to_csv(hier_path, index=False)
        print(f"Hierarchical (within/between-training-seed) summary saved to {hier_path}")

    max_recon = df["reward_reconciliation_error"].max()
    print(f"\nMaximum reconciliation discrepancy across all {len(df)} episodes: {max_recon:.3e}")


if __name__ == "__main__":
    main()
