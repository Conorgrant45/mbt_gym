"""
shared/hamilton_ppo_eval_lib.py
--------------------------
Shared, read-only-import evaluation library for the post-hoc Hamilton PPO
analysis (unseen-holdout evaluation, paired benchmarks, belief-input
ablation, policy-grid diagnostics). Imports building blocks from
train_hamilton_ppo.py, envs/hamilton_ppo_wrapper.py,
simulate_belief_weighted.py and compare_four_policies_paired.py
READ-ONLY -- none of those files, nor the trained model, are modified.

Key empirical fact this module relies on (verified directly, not assumed):
the midprice path (hence the return series fed to the Hamilton filter,
hence the belief trajectory) and the regime-switching path are BOTH
completely independent of the agent's actions for a given seed --
ArrivalJumpMidpriceModel.update() only reads `arrivals` (never `fills`
or `actions`), and RegimeSwitchingEnv's regime draw is a pure Markov-
chain step unconditional on the agent. Verified: running the SAME seed
with all-zero actions, uniform-random actions, and always-extreme
actions gives BIT-IDENTICAL belief and true_regime sequences. This means
a single "reference" rollout (any fixed action sequence) fully
determines the belief/regime path for a seed, independent of which
policy is later evaluated on it -- used here to make the belief-input
ablation (Section 3) and multi-policy comparisons efficient and exactly
reproducible.
"""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

import numpy as np

from envs.hamilton_ppo_wrapper import HamiltonPPOWrapper
from envs.make_envs import make_regime_envs, N_STEPS, TRANSITION_MATRIX
from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, ASSET_PRICE_INDEX, ASK_INDEX, BID_INDEX

from shared.train_hamilton_ppo import instrument_arrival_fill, PHI, ALPHA, DT, ACTION_BOUND_TOL

HOLDOUT_SEEDS = list(range(100_000, 100_200))       # item 1: 200 unseen seeds
BENCHMARK_SEEDS = list(range(100_000, 100_050))      # item 2/3: 50-seed subset of the holdout range
                                                       # (documented choice -- item 1 mandates >=200 for
                                                       # the holdout eval; items 2/3 don't mandate a count,
                                                       # and running 5 policies x 200 episodes / 4 belief
                                                       # treatments x 200 episodes would multiply runtime
                                                       # several-fold for limited extra statistical power)


def stationary_belief() -> float:
    P = np.array(TRANSITION_MATRIX)
    A = (P.T - np.eye(2))
    A[-1] = 1.0
    b = np.zeros(2)
    b[-1] = 1.0
    pi_stat = np.linalg.solve(A, b)
    return float(pi_stat[1])


def build_eval_env(seed: int, inventory_scale: float):
    """A FRESH base_env + wrapper for one episode -- never reuse an
    already-stepped wrapper via reset(seed=s) (see
    tests/test_hamilton_ppo_wrapper.py TestSeedReset)."""
    base_env = make_regime_envs(switch_within_episode=True, seed=seed)
    logs = instrument_arrival_fill(base_env)
    wrapper = HamiltonPPOWrapper(base_env=base_env, inventory_scale=inventory_scale)
    return wrapper, logs


def compute_reference_trajectory(seed: int, inventory_scale: float):
    """Action-independent belief/true_regime sequence for this seed (see
    module docstring). Length N_STEPS+1 (includes t=0 at reset)."""
    wrapper, _ = build_eval_env(seed, inventory_scale)
    obs, info = wrapper.reset(seed=seed)
    beliefs = [wrapper.belief]
    regimes = [wrapper.base_env.current_regime]
    zero_action = np.zeros(2, dtype=np.float32)
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = wrapper.step(zero_action)
        beliefs.append(wrapper.belief)
        regimes.append(info["true_regime"])
    return np.array(beliefs), np.array(regimes)


def make_belief_override(kind: str, true_beliefs: np.ndarray, seed: int) -> np.ndarray:
    """kind in {'correct', 'stationary', 'shuffled', 'complement'}."""
    if kind == "correct":
        return true_beliefs
    if kind == "stationary":
        return np.full_like(true_beliefs, stationary_belief())
    if kind == "shuffled":
        rng = np.random.default_rng(seed + 777_000_000)  # dedicated, independent stream
        return rng.permutation(true_beliefs)
    if kind == "complement":
        return 1.0 - true_beliefs
    raise ValueError(f"unknown belief override kind: {kind}")


class BucketAccumulator:
    """
    Accumulates per-step (inv_sign, belief_quintile, tau_quartile,
    true_regime) -> action running sums, for item 6's conditional
    saturation breakdown. true_regime is recorded for OFFLINE analysis
    only -- it is never read by add() from anything the policy saw; the
    caller is responsible for never having supplied it to the policy.
    """

    def __init__(self):
        self.rows = []  # list of dicts, one per step; small enough per-run, aggregated with pandas afterwards

    def add(self, inv_before: float, belief: float, tau: float, true_regime: int, action: np.ndarray):
        self.rows.append(dict(
            inv_sign="positive" if inv_before > 0 else ("negative" if inv_before < 0 else "zero"),
            belief=belief,
            tau=tau,
            true_regime=true_regime,
            bid=float(action[0]),
            ask=float(action[1]),
        ))


def run_eval_episode_detailed(model, seed: int, inventory_scale: float,
                               belief_override: np.ndarray = None,
                               deterministic: bool = True,
                               bucket_accumulator: "BucketAccumulator" = None) -> dict:
    """
    Full per-episode metrics (item 1's required field list). If
    belief_override is given (array of length N_STEPS+1), obs[2] is
    replaced by belief_override[t] before every model.predict() call --
    the true env belief/dynamics/reward, and everything else in the
    observation, are otherwise completely unaffected: only the POLICY
    INPUT's belief channel is patched, matching "only change the belief
    component supplied to the policy."
    """
    wrapper, logs = build_eval_env(seed, inventory_scale)
    obs, info = wrapper.reset(seed=seed)

    if belief_override is not None:
        obs = obs.copy()
        obs[2] = np.float32(belief_override[0])

    cash_0 = wrapper.base_env.raw_cash
    inv_0 = wrapper.base_env.raw_inventory
    mid_0 = wrapper.base_env.raw_midprice

    running_inventory_penalty = 0.0
    inv_sum = 0.0
    inv_abs_sum = 0.0
    belief_sum = 0.0  # TRUE belief (from the env filter), regardless of any override fed to the policy
    total_fills = 0.0
    cumulative_objective = 0.0
    actions = []

    q_t = inv_0  # inventory the CURRENT action is conditioned on
    t = 0
    terminated = truncated = False
    info_last = None
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float64)
        actions.append(action.copy())

        regime_before = wrapper.base_env.current_regime
        pre_len = len(logs[regime_before]["arrivals"])
        true_belief_this_step = wrapper.belief
        tau_this_step = 1.0 - t / N_STEPS

        obs, reward, terminated, truncated, info = wrapper.step(action)
        info_last = info

        arrivals = logs[regime_before]["arrivals"][pre_len]
        fills = logs[regime_before]["fills"][pre_len]
        buy_arrival = float(arrivals[0, ASK_INDEX])
        sell_arrival = float(arrivals[0, BID_INDEX])
        ask_fill_attempt = float(fills[0, ASK_INDEX])
        bid_fill_attempt = float(fills[0, BID_INDEX])
        total_fills += buy_arrival * ask_fill_attempt + sell_arrival * bid_fill_attempt

        inv_after = float(info["raw_state"][INVENTORY_INDEX])
        inv_sum += inv_after
        inv_abs_sum += abs(inv_after)
        running_inventory_penalty += PHI * (inv_after ** 2) * DT
        belief_sum += true_belief_this_step
        cumulative_objective += float(reward)

        if bucket_accumulator is not None:
            bucket_accumulator.add(
                inv_before=q_t, belief=true_belief_this_step, tau=tau_this_step,
                true_regime=info["true_regime"], action=action,
            )

        q_t = inv_after
        t += 1
        if belief_override is not None and t < len(belief_override) and not (terminated or truncated):
            obs = obs.copy()
            obs[2] = np.float32(belief_override[t])

    n_steps = t
    cash_T = float(info_last["raw_state"][CASH_INDEX])
    inv_T = float(info_last["raw_state"][INVENTORY_INDEX])
    mid_T = float(info_last["raw_state"][ASSET_PRICE_INDEX])

    raw_pnl = (cash_T + inv_T * mid_T) - (cash_0 + inv_0 * mid_0)
    terminal_inventory_penalty = ALPHA * (inv_T ** 2)
    full_objective = raw_pnl - running_inventory_penalty - terminal_inventory_penalty
    reconciliation_abs_diff = abs(full_objective - cumulative_objective)

    actions_arr = np.array(actions)
    bid_actions = actions_arr[:, 0]
    ask_actions = actions_arr[:, 1]
    low, high = -1.0, 1.0

    return dict(
        seed=seed,
        cumulative_objective=cumulative_objective,
        raw_pnl=raw_pnl,
        running_inventory_penalty=running_inventory_penalty,
        terminal_inventory_penalty=terminal_inventory_penalty,
        full_objective=full_objective,
        reconciliation_abs_diff=reconciliation_abs_diff,
        mean_signed_inventory=inv_sum / n_steps,
        mean_abs_inventory=inv_abs_sum / n_steps,
        terminal_signed_inventory=inv_T,
        terminal_abs_inventory=abs(inv_T),
        total_fills=total_fills,
        bid_action_mean=float(bid_actions.mean()),
        bid_action_std=float(bid_actions.std()),
        ask_action_mean=float(ask_actions.mean()),
        ask_action_std=float(ask_actions.std()),
        frac_bid_near_low=float(np.mean(np.abs(bid_actions - low) < ACTION_BOUND_TOL)),
        frac_bid_near_high=float(np.mean(np.abs(bid_actions - high) < ACTION_BOUND_TOL)),
        frac_ask_near_low=float(np.mean(np.abs(ask_actions - low) < ACTION_BOUND_TOL)),
        frac_ask_near_high=float(np.mean(np.abs(ask_actions - high) < ACTION_BOUND_TOL)),
        mean_belief=belief_sum / n_steps,
        episode_length=n_steps,
    )


MULTISEED_HOLDOUT_SEEDS = list(range(110_000, 110_100))  # 100 seeds, disjoint from training seeds (0-4),
                                                            # dev eval seeds (90001-90005), and the previous
                                                            # milestone's holdout (100000-100199)


def run_eval_episode_full(model, seed: int, inventory_scale: float,
                           deterministic: bool = True,
                           bucket_accumulator: "BucketAccumulator" = None) -> dict:
    """
    Extended per-episode metrics for the multiseed comparison -- adds
    adverse_selection_loss, spread_revenue, max_abs_inventory,
    frac_time_positive/negative, belief_std, and offline belief-
    classification accuracy on top of everything run_eval_episode_detailed
    (the single-seed-milestone version) already computes.

    Jump magnitudes (needed for adverse_selection_loss/spread_revenue) are
    captured via compare_four_policies_paired.instrument_env's
    JumpOnlyRNG wrapper -- imported and reused, not reimplemented; this is
    a strict superset of train_hamilton_ppo.instrument_arrival_fill (also
    wraps the midprice model's rng to log jump-exponential draws).
    """
    import shared.compare_four_policies_paired as C4P  # READ ONLY; local import avoids a hard dependency
                                                   # for callers that only need the lighter functions above
    import shared.simulate_belief_weighted as SBW  # READ ONLY

    wrapper, _ = build_eval_env(seed, inventory_scale)
    logs = C4P.instrument_env(wrapper.base_env, full_trace=False)
    obs, info = wrapper.reset(seed=seed)

    cash_0 = wrapper.base_env.raw_cash
    inv_0 = wrapper.base_env.raw_inventory
    mid_0 = wrapper.base_env.raw_midprice

    running_inventory_penalty = 0.0
    inv_sum = 0.0
    inv_abs_sum = 0.0
    inv_max_abs = 0.0
    time_positive = 0
    time_negative = 0
    belief_sum = 0.0
    belief_sq_sum = 0.0
    spread_revenue = 0.0
    adverse_selection_loss = 0.0
    total_fills = 0.0
    cumulative_objective = 0.0
    n_correct_regime_pred = 0
    actions = []

    q_t = inv_0
    t = 0
    terminated = truncated = False
    info_last = None
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float64)
        actions.append(action.copy())
        bid_action, ask_action = float(action[0]), float(action[1])

        regime_before = wrapper.base_env.current_regime
        pre_len_arr = len(logs[regime_before]["arrival_log"])
        pre_len_exp = len(logs[regime_before]["midprice_rng"].exponential_log)
        true_belief_this_step = wrapper.belief
        tau_this_step = 1.0 - t / N_STEPS

        obs, reward, terminated, truncated, info = wrapper.step(action)
        info_last = info

        arrivals = logs[regime_before]["arrival_log"][pre_len_arr]
        fills = logs[regime_before]["fill_log"][pre_len_arr]
        buy_arrival = float(arrivals[0, ASK_INDEX])
        sell_arrival = float(arrivals[0, BID_INDEX])
        ask_fill_attempt = float(fills[0, ASK_INDEX])
        bid_fill_attempt = float(fills[0, BID_INDEX])
        ask_fill_event = buy_arrival * ask_fill_attempt
        bid_fill_event = sell_arrival * bid_fill_attempt
        total_fills += ask_fill_event + bid_fill_event

        exp_calls = logs[regime_before]["midprice_rng"].exponential_log[pre_len_exp: pre_len_exp + 2]
        if len(exp_calls) == 2:
            jump_up = float(exp_calls[0][0, 0])
            jump_down = float(exp_calls[1][0, 0])
        else:
            jump_up = jump_down = 0.0

        # Real-unit depths, matching evaluate_hamilton_ppo_benchmarks.run_episode_ppo's
        # accounting exactly.
        ask_depth = (ask_action + 1.0) / 2.0 * SBW.MAX_DEPTH
        bid_depth = (bid_action + 1.0) / 2.0 * SBW.MAX_DEPTH
        spread_revenue += ask_depth * ask_fill_event + bid_depth * bid_fill_event
        if regime_before == 1:
            adverse_selection_loss += jump_up * ask_fill_event + jump_down * bid_fill_event

        inv_after = float(info["raw_state"][INVENTORY_INDEX])
        inv_sum += inv_after
        inv_abs_sum += abs(inv_after)
        inv_max_abs = max(inv_max_abs, abs(inv_after))
        if inv_after > 0:
            time_positive += 1
        elif inv_after < 0:
            time_negative += 1
        running_inventory_penalty += PHI * (inv_after ** 2) * DT
        belief_sum += true_belief_this_step
        belief_sq_sum += true_belief_this_step ** 2
        cumulative_objective += float(reward)
        n_correct_regime_pred += int((true_belief_this_step >= 0.5) == info["true_regime"])

        if bucket_accumulator is not None:
            bucket_accumulator.add(
                inv_before=q_t, belief=true_belief_this_step, tau=tau_this_step,
                true_regime=info["true_regime"], action=action,
            )

        q_t = inv_after
        t += 1

    n_steps = t
    cash_T = float(info_last["raw_state"][CASH_INDEX])
    inv_T = float(info_last["raw_state"][INVENTORY_INDEX])
    mid_T = float(info_last["raw_state"][ASSET_PRICE_INDEX])

    raw_pnl = (cash_T + inv_T * mid_T) - (cash_0 + inv_0 * mid_0)
    terminal_inventory_penalty = ALPHA * (inv_T ** 2)
    full_objective = raw_pnl - running_inventory_penalty - terminal_inventory_penalty
    reconciliation_abs_diff = abs(full_objective - cumulative_objective)

    actions_arr = np.array(actions)
    bid_actions = actions_arr[:, 0]
    ask_actions = actions_arr[:, 1]
    low, high = -1.0, 1.0

    belief_mean = belief_sum / n_steps
    belief_var = max(belief_sq_sum / n_steps - belief_mean ** 2, 0.0)

    return dict(
        seed=seed,
        cumulative_objective=cumulative_objective,
        raw_pnl=raw_pnl,
        running_inventory_penalty=running_inventory_penalty,
        terminal_inventory_penalty=terminal_inventory_penalty,
        adverse_selection_loss=adverse_selection_loss,
        spread_revenue=spread_revenue,
        full_objective=full_objective,
        reconciliation_abs_diff=reconciliation_abs_diff,
        total_fills=total_fills,
        mean_signed_inventory=inv_sum / n_steps,
        mean_abs_inventory=inv_abs_sum / n_steps,
        max_abs_inventory=inv_max_abs,
        terminal_signed_inventory=inv_T,
        terminal_abs_inventory=abs(inv_T),
        frac_time_positive=time_positive / n_steps,
        frac_time_negative=time_negative / n_steps,
        bid_action_mean=float(bid_actions.mean()),
        bid_action_std=float(bid_actions.std()),
        ask_action_mean=float(ask_actions.mean()),
        ask_action_std=float(ask_actions.std()),
        frac_bid_near_low=float(np.mean(np.abs(bid_actions - low) < ACTION_BOUND_TOL)),
        frac_bid_near_high=float(np.mean(np.abs(bid_actions - high) < ACTION_BOUND_TOL)),
        frac_ask_near_low=float(np.mean(np.abs(ask_actions - low) < ACTION_BOUND_TOL)),
        frac_ask_near_high=float(np.mean(np.abs(ask_actions - high) < ACTION_BOUND_TOL)),
        mean_belief=belief_mean,
        belief_std=float(np.sqrt(belief_var)),
        belief_accuracy_offline=n_correct_regime_pred / n_steps,
        episode_length=n_steps,
    )


def summarise_column(df, col: str) -> dict:
    """mean/std/median/95% CI (normal approx on the sample mean) for one column."""
    x = df[col].to_numpy(dtype=float)
    n = len(x)
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    return dict(
        metric=col, n=n, mean=mean, std=std, median=float(np.median(x)),
        ci_lo=mean - 1.96 * se, ci_hi=mean + 1.96 * se,
    )
