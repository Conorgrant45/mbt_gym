"""
ips_episode_runner.py
----------------------------
Per-episode evaluation for the inventory-penalty sensitivity experiment.
For PPO policies (feed-forward and recurrent), the frozen supervised clone
(duck-typed .predict() shim, identical convention to final_episode_runner.py)
and the analytical oracle/belief-weighted policies, produces BOTH:

  1. An episode-level summary dict -- the existing final_episode_runner.py
     field set (fills, spread revenue, adverse-selection loss, running/
     terminal penalty, EVENT-weighted mean inventory, ...), reused so this
     experiment stays comparable to the original one, PLUS the new
     TIME-weighted inventory statistics the brief requires (time-averaged
     signed/absolute inventory, integrated squared inventory, RMS inventory,
     boundary-contact count, bid/ask/total fill counts, terminal wealth
     decomposition) -- see module docstring section "NEW fields" below.

  2. A list of event-level row dicts, one per observable event (arrival or
     terminal), with the exact schema the brief specifies -- sufficient to
     reconstruct the complete piecewise-constant inventory path.

Every formula for the fields inherited from final_episode_runner.py
(spread_revenue, adverse_selection_loss, running/terminal penalty,
full_objective, reward reconciliation, ...) is copied VERBATIM from that
module -- not re-derived -- so the two experiments' episode-level numbers
remain directly comparable. The only structural difference is (a) the
environment is built via ips_common's phi/alpha-parameterised env builders
instead of final_common's fixed-default ones, and (b) every step's `info`
dict is also turned into an event-level row.

NEW fields (not present in final_episode_runner.py), all derived from the
fact that inventory is piecewise-constant between observable events (see
envs/event_driven_regime_env.py's module docstring): for the interval
immediately preceding event n, of length elapsed_inter_event_time, the
inventory held throughout that interval is exactly `inventory_before` of
event n (this is also exactly what the environment itself integrates to
compute that interval's running penalty, phi * inventory_before**2 * dtau --
see the running_penalty cross-check in tests/test_inventory_penalty_sensitivity.py):

    time_avg_signed_inventory    = (1/T) * sum_n inventory_before_n * dtau_n
    time_avg_abs_inventory       = (1/T) * sum_n |inventory_before_n| * dtau_n
    integrated_squared_inventory =         sum_n inventory_before_n**2 * dtau_n
    rms_inventory                = sqrt(integrated_squared_inventory / T)

mean_signed_inventory / mean_abs_inventory (no suffix) are RETAINED from
final_episode_runner.py with their ORIGINAL (event-weighted, i.e. an
unweighted average over decision events, not over time) definition --
renamed explicitly to *_event_weighted here as well so both definitions are
always available and neither silently replaces the other, per the brief.
"""
import numpy as np
import torch

import simulate_belief_weighted as SBW
import evaluate_agents_common as EAC
from evaluate_agents_event_driven import continuous_time_control, make_event_time_filter
from envs.event_driven_regime_env import MAX_INVENTORY
from mbt_gym.gym.index_names import TIME_INDEX

import ips_common as IC

NEAR_BOUND_TOL = 0.95

ARCHITECTURE_LABELS = {
    "hamilton_ppo": "belief_state_mlp",
    "return_mlp_ppo": "raw_return_mlp",
    "return_lstm_ppo": "raw_return_lstm",
    "oracle": "analytical_oracle",
    "belief_weighted": "analytical_belief_weighted",
    "frozen_clone": "supervised_clone_mlp",
}


class _InventoryAccumulator:
    """Per-episode accumulator: O(1) memory in the number of events (no
    full-trajectory storage needed for the SUMMARY -- the event-level rows
    themselves are collected separately by the caller, exactly once, from
    the same loop)."""

    def __init__(self):
        self.n_steps = 0
        self.n_arrivals = 0
        self.n_buy_arrivals = 0
        self.n_sell_arrivals = 0
        self.n_fills = 0
        self.n_buy_fills = 0
        self.n_sell_fills = 0
        self.spread_revenue = 0.0
        self.adverse_selection_loss = 0.0
        self.carry_jump_pnl = 0.0
        self.diffusion_pnl = 0.0
        self.running_penalty = 0.0
        self.inv_sum = 0.0
        self.inv_abs_sum = 0.0
        self.inv_max_abs = 0.0
        self.bid_depth_sum = 0.0
        self.ask_depth_sum = 0.0
        self.cumulative_reward = 0.0
        self.bid_actions = []
        self.ask_actions = []
        self.n_near_bound = 0

        # --- NEW: time-weighted inventory integrals ---
        self.integral_Q_dt = 0.0
        self.integral_absQ_dt = 0.0
        self.integral_Q2_dt = 0.0
        self.total_bid_fills = 0
        self.total_ask_fills = 0
        self.n_boundary_contacts = 0

    def add_step(self, info: dict, bid_action: float, ask_action: float, reward: float):
        self.bid_actions.append(bid_action)
        self.ask_actions.append(ask_action)
        if max(abs(bid_action), abs(ask_action)) > NEAR_BOUND_TOL:
            self.n_near_bound += 1

        bid_depth = float(info["bid_depth"])
        ask_depth = float(info["ask_depth"])
        self.bid_depth_sum += bid_depth
        self.ask_depth_sum += ask_depth

        inventory_before = float(info["inventory_before"])
        self.diffusion_pnl += inventory_before * float(info["brownian_increment"])
        self.carry_jump_pnl += inventory_before * float(info["jump_increment"])

        # --- NEW: time-weighted integrals, using the inventory held during
        # the interval immediately preceding this event (provably constant
        # over that interval -- see module docstring) and its realised
        # duration `elapsed_time` (== this event's Delta-tau). ---
        dtau = float(info["elapsed_time"])
        self.integral_Q_dt += inventory_before * dtau
        self.integral_absQ_dt += abs(inventory_before) * dtau
        self.integral_Q2_dt += (inventory_before ** 2) * dtau
        if abs(inventory_before) >= MAX_INVENTORY:
            self.n_boundary_contacts += 1

        fill_bid, fill_ask = info["fill_indicator"]
        fill_bid, fill_ask = int(fill_bid), int(fill_ask)
        self.total_bid_fills += fill_bid
        self.total_ask_fills += fill_ask

        if info["event_type"] == "arrival":
            self.n_arrivals += 1
            side = info["arrival_side"]
            if side == "buy":
                self.n_buy_arrivals += 1
                if fill_ask:
                    self.n_buy_fills += 1
                    self.n_fills += 1
            elif side == "sell":
                self.n_sell_arrivals += 1
                if fill_bid:
                    self.n_sell_fills += 1
                    self.n_fills += 1
            self.spread_revenue += ask_depth * fill_ask + bid_depth * fill_bid
            if info["regime_at_event"] == 1:
                self.adverse_selection_loss += abs(info["jump_increment"]) * (fill_ask + fill_bid)

        self.running_penalty += info["running_penalty_increment"]
        inv_after = float(info["inventory_after"])
        self.inv_sum += inv_after
        self.inv_abs_sum += abs(inv_after)
        self.inv_max_abs = max(self.inv_max_abs, abs(inv_after))
        self.cumulative_reward += float(reward)
        self.n_steps += 1

    def finalize(self, cash_0, inv_0, mid_0, info_last, seed) -> dict:
        cash_T, inv_T, mid_T = info_last["cash_after"], info_last["inventory_after"], info_last["price_after"]
        raw_pnl = (cash_T + inv_T * mid_T) - (cash_0 + inv_0 * mid_0)
        terminal_penalty = info_last["terminal_penalty_increment"]
        full_objective = raw_pnl - self.running_penalty - terminal_penalty
        reconciliation_abs_diff = abs(full_objective - self.cumulative_reward)

        decomposition = (self.spread_revenue - self.adverse_selection_loss + self.carry_jump_pnl
                          + self.diffusion_pnl - self.running_penalty - terminal_penalty)
        decomposition_residual = full_objective - decomposition

        bid_actions = np.array(self.bid_actions)
        ask_actions = np.array(self.ask_actions)
        n = self.n_steps
        episode_duration = float(info_last["raw_state"][TIME_INDEX])
        T = episode_duration if episode_duration > 0 else float("nan")

        return dict(
            evaluation_seed=seed,
            full_objective=float(full_objective), raw_pnl=float(raw_pnl),
            fills=self.n_fills, buy_side_fills=self.n_buy_fills, sell_side_fills=self.n_sell_fills,
            total_arrivals=self.n_arrivals, buy_side_arrivals=self.n_buy_arrivals,
            sell_side_arrivals=self.n_sell_arrivals,
            fill_to_arrival_ratio=(self.n_fills / self.n_arrivals) if self.n_arrivals else float("nan"),
            mean_quoted_spread=(self.bid_depth_sum + self.ask_depth_sum) / n,
            mean_bid_depth=self.bid_depth_sum / n, mean_ask_depth=self.ask_depth_sum / n,
            spread_revenue=float(self.spread_revenue),
            spread_revenue_per_fill=(self.spread_revenue / self.n_fills) if self.n_fills else float("nan"),
            adverse_selection_loss=float(self.adverse_selection_loss),
            carry_jump_pnl=float(self.carry_jump_pnl),
            diffusion_pnl=float(self.diffusion_pnl),
            decomposition_residual=float(decomposition_residual),
            running_penalty=float(self.running_penalty), terminal_penalty=float(terminal_penalty),
            # --- existing (event-weighted) thesis definitions, retained verbatim ---
            mean_abs_inventory=self.inv_abs_sum / n, terminal_abs_inventory=abs(float(inv_T)),
            mean_signed_inventory=self.inv_sum / n, terminal_signed_inventory=float(inv_T),
            mean_abs_inventory_event_weighted=self.inv_abs_sum / n,
            mean_signed_inventory_event_weighted=self.inv_sum / n,
            max_abs_inventory=self.inv_max_abs,
            mean_bid_action=float(bid_actions.mean()), mean_ask_action=float(ask_actions.mean()),
            action_std_bid=float(bid_actions.std()), action_std_ask=float(ask_actions.std()),
            near_bound_rate=(self.n_near_bound / n),
            episode_duration=episode_duration, n_decision_events=n,
            reward_reconciliation_error=float(reconciliation_abs_diff),
            # --- NEW: time-weighted inventory statistics ---
            time_avg_signed_inventory=float(self.integral_Q_dt / T),
            time_avg_abs_inventory=float(self.integral_absQ_dt / T),
            integrated_squared_inventory=float(self.integral_Q2_dt),
            rms_inventory=float(np.sqrt(max(self.integral_Q2_dt, 0.0) / T)),
            terminal_inventory=float(inv_T),
            abs_terminal_inventory=abs(float(inv_T)),
            squared_terminal_inventory=float(inv_T) ** 2,
            n_inventory_boundary_contacts=self.n_boundary_contacts,
            total_bid_fills=self.total_bid_fills, total_ask_fills=self.total_ask_fills,
            total_fills=self.total_bid_fills + self.total_ask_fills,
            terminal_mtm_wealth_pre_penalty=float(raw_pnl),
            running_penalty_contribution=float(self.running_penalty),
            terminal_penalty_contribution=float(terminal_penalty),
            final_realised_objective=float(full_objective),
        )


def _event_row(info: dict, reward: float, event_index: int, belief) -> dict:
    inv_before = float(info["inventory_before"])
    inv_after = float(info["inventory_after"])
    cash_before, cash_after = float(info["cash_before"]), float(info["cash_after"])
    price_before, price_after = float(info["price_before"]), float(info["price_after"])
    wealth_before = cash_before + inv_before * price_before
    wealth_after = cash_after + inv_after * price_after
    fill_bid, fill_ask = info["fill_indicator"]
    return dict(
        event_index=event_index,
        event_time=float(info["raw_state"][TIME_INDEX]),
        elapsed_inter_event_time=float(info["elapsed_time"]),
        inventory_before=inv_before,
        inventory_after=inv_after,
        bid_depth=float(info["bid_depth"]),
        ask_depth=float(info["ask_depth"]),
        fill_bid=int(fill_bid),
        fill_ask=int(fill_ask),
        latent_regime=int(info["regime_at_event"]),
        filtered_regime_belief=(float(belief) if belief is not None else float("nan")),
        wealth_change=float(wealth_after - wealth_before),
        running_penalty_contribution=float(info["running_penalty_increment"]),
        terminal_penalty_contribution=float(info["terminal_penalty_increment"]),
        realised_stage_reward=float(reward),
    )


def _finalize_rows(policy: str, architecture_label: str, calibration: str, learner_seed, holdout_episode_seed: int,
                    event_rows: list) -> list:
    for row in event_rows:
        row["policy"] = policy
        row["architecture"] = architecture_label
        row["penalty_calibration"] = calibration
        row["learner_seed"] = learner_seed
        row["holdout_episode_seed"] = holdout_episode_seed
    return event_rows


# ======================================================================
# PPO policies (feed-forward and recurrent) AND the frozen supervised clone
# (duck-typed .predict(), always run through the hamilton_ppo wrapper --
# identical convention to final_episode_runner.run_clone_episode).
# ======================================================================
def run_ppo_episode(model, architecture: str, seed: int, phi: float, alpha: float, deterministic: bool = True):
    wrapper = IC.build_eval_wrapper(architecture, seed, phi, alpha)

    obs, _ = wrapper.reset()
    cash_0 = wrapper.base_env.raw_cash
    inv_0 = wrapper.base_env.raw_inventory
    mid_0 = wrapper.base_env.raw_midprice

    recurrent = architecture in IC.RECURRENT_ARCHITECTURES
    lstm_states = None
    episode_start = np.array([True], dtype=bool)
    has_belief = (architecture == "hamilton_ppo")

    acc = _InventoryAccumulator()
    event_rows = []
    terminated = truncated = False
    info = None
    event_index = 0
    while not (terminated or truncated):
        if recurrent:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start,
                                                  deterministic=deterministic)
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float64).reshape(-1)

        obs, reward, terminated, truncated, info = wrapper.step(action)
        episode_start = np.array([False], dtype=bool)
        acc.add_step(info, float(action[0]), float(action[1]), reward)
        belief = wrapper.belief if has_belief else None
        event_rows.append(_event_row(info, reward, event_index, belief))
        event_index += 1

    summary = acc.finalize(cash_0, inv_0, mid_0, info, seed)
    return summary, event_rows


def run_clone_episode(clone_agent, seed: int, phi: float, alpha: float):
    return run_ppo_episode(clone_agent, "hamilton_ppo", seed, phi, alpha, deterministic=True)


# ======================================================================
# Analytical benchmarks (oracle / belief_weighted), via the raw environment
# ======================================================================
def run_analytic_episode(policy_name: str, controls: dict, seed: int, phi: float, alpha: float) -> tuple:
    policy = EAC.ANALYTIC_POLICY_NAME_MAP[policy_name]
    env = IC.build_base_env(seed, phi, alpha)
    filt = make_event_time_filter() if policy == "belief" else None

    raw_state = env.reset()
    if filt is not None:
        filt.reset(initial_price=env.raw_midprice)
    cash_0, inv_0, mid_0 = env.raw_cash, env.raw_inventory, env.raw_midprice

    q = inv_0
    t_elapsed = 0.0
    acc = _InventoryAccumulator()
    event_rows = []
    done = False
    info = None
    event_index = 0
    while not done:
        inv_sc = q
        if policy == "oracle":
            regime = env.current_regime
            ctrl = controls[regime]
            ask_depth, bid_depth = continuous_time_control(
                ctrl["delta_ask"], ctrl["delta_bid"], ctrl["q_ask"], ctrl["q_bid"], t_elapsed, inv_sc,
            )
            belief = None
        elif policy == "belief":
            belief = filt.belief
            c0, c1 = controls[0], controls[1]
            a0, b0 = continuous_time_control(c0["delta_ask"], c0["delta_bid"], c0["q_ask"], c0["q_bid"], t_elapsed, inv_sc)
            a1, b1 = continuous_time_control(c1["delta_ask"], c1["delta_bid"], c1["q_ask"], c1["q_bid"], t_elapsed, inv_sc)
            ask_depth = (1.0 - belief) * a0 + belief * a1
            bid_depth = (1.0 - belief) * b0 + belief * b1
        else:
            raise ValueError(f"run_analytic_episode only supports 'oracle'/'belief_weighted', got {policy_name!r}")

        bid_action = SBW.normalise_depth(bid_depth)
        ask_action = SBW.normalise_depth(ask_depth)

        raw_state, reward, done, info = env.step(np.array([bid_action, ask_action]))
        if filt is not None:
            filt.update(info["price_after"], info["elapsed_time"], info["event_type"] == "arrival")

        acc.add_step(info, bid_action, ask_action, reward)
        event_rows.append(_event_row(info, reward, event_index, belief))
        event_index += 1
        q = info["inventory_after"]
        t_elapsed = float(info["raw_state"][TIME_INDEX])

    summary = acc.finalize(cash_0, inv_0, mid_0, info, seed)
    return summary, event_rows


# ======================================================================
# Unified per-(policy, holdout path) row builder, used by ips_evaluate_holdout.py
# ======================================================================
def evaluate_one(policy: str, calibration: str, seed: int, phi: float, alpha: float, *,
                  model=None, controls=None, learner_seed=None) -> tuple:
    """Returns (episode_summary_dict, event_rows) for any of the 6 policy
    kinds this experiment evaluates. `policy` in
    {"hamilton_ppo","return_mlp_ppo","return_lstm_ppo"} requires `model`;
    policy in {"oracle","belief_weighted"} requires `controls`; policy ==
    "frozen_clone" requires `model` (a SupervisedCloneAgent, duck-typed)."""
    if policy in IC.ARCHITECTURES:
        summary, event_rows = run_ppo_episode(model, policy, seed, phi, alpha, deterministic=True)
    elif policy == "frozen_clone":
        summary, event_rows = run_clone_episode(model, seed, phi, alpha)
    elif policy in ("oracle", "belief_weighted"):
        summary, event_rows = run_analytic_episode(policy, controls, seed, phi, alpha)
    else:
        raise ValueError(f"Unsupported policy: {policy!r}")

    architecture_label = ARCHITECTURE_LABELS[policy]
    summary = dict(summary)
    summary.update(
        policy=policy, architecture=architecture_label, penalty_calibration=calibration,
        learner_seed=learner_seed, holdout_episode_seed=summary.pop("evaluation_seed"),
    )
    event_rows = _finalize_rows(policy, architecture_label, calibration, learner_seed, seed, event_rows)
    return summary, event_rows
