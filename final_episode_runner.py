"""
final_episode_runner.py
----------------------------
Rich, unified per-episode evaluation for the final architecture-comparison
experiment. Produces ONE dict per episode with every field required by the
brief (fills by side, arrivals by side, fill-to-arrival ratio, bid/ask
depths, spread revenue per fill, adverse-selection loss, running/terminal
penalty, inventory statistics, action statistics, near-bound rates,
episode duration/decision-event count) -- for PPO policies (feed-forward
and recurrent, via their respective event-driven wrappers) AND for the
analytical oracle/belief-weighted policies (via the raw event-driven
environment + the SAME continuous_time_control lookup Phase 2/3 already
validated) AND the frozen supervised clone (via the Hamilton wrapper, same
duck-typed .predict() shim used since Phase 4).

Reward-decomposition formulas (full_objective, spread_revenue,
adverse_selection_loss, running/terminal penalty, reward reconciliation)
are IDENTICAL to evaluate_agents_event_driven.py's existing, already-
validated runners -- reused, not re-derived. The only new work here is
accumulating the ADDITIONAL richness fields (buy/sell fill and arrival
breakdown, per-episode bid/ask depth series, action-bound diagnostics)
that those existing functions do not separately expose.

Recurrent (LSTM) handling matches train_agents.run_eval_episode exactly:
episode_start=True for the first prediction of every episode, LSTM hidden
state threaded across steps within the episode via model.predict(...,
state=lstm_states, episode_start=episode_start, ...).
"""
import numpy as np
import torch

import simulate_belief_weighted as SBW
import evaluate_agents_common as EAC
from evaluate_agents_event_driven import continuous_time_control, make_event_time_filter
from envs.event_driven_regime_env import EventDrivenRegimeSwitchingEnv
from envs.event_driven_hamilton_ppo_wrapper import EventDrivenHamiltonPPOWrapper
from envs.event_driven_return_ppo_wrapper import EventDrivenReturnPPOWrapper
from mbt_gym.gym.index_names import TIME_INDEX

import final_common as FC

NEAR_BOUND_TOL = 0.95


class _Accumulator:
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

    def add_step(self, info: dict, bid_action: float, ask_action: float, reward: float):
        self.bid_actions.append(bid_action)
        self.ask_actions.append(ask_action)
        if max(abs(bid_action), abs(ask_action)) > NEAR_BOUND_TOL:
            self.n_near_bound += 1

        bid_depth = float(info["bid_depth"])
        ask_depth = float(info["ask_depth"])
        self.bid_depth_sum += bid_depth
        self.ask_depth_sum += ask_depth

        # carry_jump_pnl / diffusion_pnl: mark-to-market P&L on inventory HELD
        # ENTERING this event (info["inventory_before"], i.e. pre-fill -- the
        # fill at this same event, if any, cannot retroactively change what
        # was earned/lost on the position already held through the price
        # move that just happened) from the jump and diffusion components of
        # the price move respectively (event_driven_regime_env.py separately
        # returns brownian_increment and jump_increment -- the exact
        # decomposition of this interval's price change, already summing to
        # it: price_after = price_before + brownian_increment + jump_increment).
        # No event_type/regime gate needed: jump_increment is exactly 0.0 for
        # every non-arrival event and every regime-0 arrival by construction
        # (event_driven_regime_env.py's _advance_to_next_event only assigns
        # it a nonzero value inside `if regime_at_event == 1` arrival
        # branches), so this sum is already restricted to exactly the
        # intended events without an explicit conditional.
        inventory_before = float(info["inventory_before"])
        self.diffusion_pnl += inventory_before * float(info["brownian_increment"])
        self.carry_jump_pnl += inventory_before * float(info["jump_increment"])

        fill_bid, fill_ask = info["fill_indicator"]
        fill_bid, fill_ask = int(fill_bid), int(fill_ask)

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

        # Second reconciliation check: does the full economic decomposition
        # (spread capture, adverse-selection loss on fills, carried jump P&L,
        # carried diffusion P&L, running/terminal penalty) sum to the same
        # full_objective? Read-only diagnostic -- does not feed back into
        # full_objective itself, which is computed above exactly as before.
        decomposition = (self.spread_revenue - self.adverse_selection_loss + self.carry_jump_pnl
                          + self.diffusion_pnl - self.running_penalty - terminal_penalty)
        decomposition_residual = full_objective - decomposition

        bid_actions = np.array(self.bid_actions)
        ask_actions = np.array(self.ask_actions)
        n = self.n_steps
        episode_duration = float(info_last["raw_state"][TIME_INDEX])

        return dict(
            evaluation_seed=seed,
            full_objective=float(full_objective), raw_pnl=float(raw_pnl),
            fills=self.n_fills, buy_side_fills=self.n_buy_fills, sell_side_fills=self.n_sell_fills,
            total_arrivals=self.n_arrivals, buy_side_arrivals=self.n_buy_arrivals, sell_side_arrivals=self.n_sell_arrivals,
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
            mean_abs_inventory=self.inv_abs_sum / n, terminal_abs_inventory=abs(float(inv_T)),
            mean_signed_inventory=self.inv_sum / n, terminal_signed_inventory=float(inv_T),
            max_abs_inventory=self.inv_max_abs,
            mean_bid_action=float(bid_actions.mean()), mean_ask_action=float(ask_actions.mean()),
            action_std_bid=float(bid_actions.std()), action_std_ask=float(ask_actions.std()),
            near_bound_rate=(self.n_near_bound / n),
            episode_duration=episode_duration, n_decision_events=n,
            reward_reconciliation_error=float(reconciliation_abs_diff),
        )


# ======================================================================
# PPO policies (feed-forward and recurrent), via their event-driven wrappers
# ======================================================================
def run_ppo_episode(model, architecture: str, seed: int, deterministic: bool) -> dict:
    wrapper_cls = EventDrivenHamiltonPPOWrapper if architecture == "hamilton_ppo" else EventDrivenReturnPPOWrapper
    base_env = EventDrivenRegimeSwitchingEnv(seed=seed)
    if architecture == "hamilton_ppo":
        wrapper = wrapper_cls(base_env=base_env, inventory_scale=FC.INVENTORY_SCALE)
    else:
        wrapper = wrapper_cls(base_env=base_env, inventory_scale=FC.INVENTORY_SCALE, return_scale=FC.RETURN_SCALE)

    obs, _ = wrapper.reset()
    cash_0 = wrapper.base_env.raw_cash
    inv_0 = wrapper.base_env.raw_inventory
    mid_0 = wrapper.base_env.raw_midprice

    recurrent = architecture in FC.RECURRENT_ARCHITECTURES
    lstm_states = None
    episode_start = np.array([True], dtype=bool)

    acc = _Accumulator()
    terminated = truncated = False
    info = None
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

    return acc.finalize(cash_0, inv_0, mid_0, info, seed)


# ======================================================================
# Analytical benchmarks (oracle / belief_weighted), via the raw environment
# ======================================================================
def run_analytic_episode(policy_name: str, controls: dict, seed: int) -> dict:
    policy = EAC.ANALYTIC_POLICY_NAME_MAP[policy_name]
    env = EventDrivenRegimeSwitchingEnv(seed=seed)
    filt = make_event_time_filter() if policy == "belief" else None

    raw_state = env.reset()
    if filt is not None:
        filt.reset(initial_price=env.raw_midprice)
    cash_0, inv_0, mid_0 = env.raw_cash, env.raw_inventory, env.raw_midprice

    q = inv_0
    t_elapsed = 0.0
    acc = _Accumulator()
    done = False
    info = None
    while not done:
        inv_sc = q
        if policy == "oracle":
            regime = env.current_regime
            ctrl = controls[regime]
            ask_depth, bid_depth = continuous_time_control(
                ctrl["delta_ask"], ctrl["delta_bid"], ctrl["q_ask"], ctrl["q_bid"], t_elapsed, inv_sc,
            )
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
        q = info["inventory_after"]
        t_elapsed = float(info["raw_state"][TIME_INDEX])

    return acc.finalize(cash_0, inv_0, mid_0, info, seed)


# ======================================================================
# Frozen supervised clone (duck-typed .predict() shim, via Hamilton wrapper)
# ======================================================================
def run_clone_episode(clone_agent, seed: int) -> dict:
    return run_ppo_episode(clone_agent, "hamilton_ppo", seed, deterministic=True)


# ======================================================================
# Combined deterministic + stochastic evaluation for one (policy, seed, path)
# ======================================================================
def evaluate_episode_pair(model, architecture: str, seed: int, stochastic_torch_seed: int = None) -> dict:
    """Runs the DETERMINISTIC episode (primary metrics) and a SEPARATE
    STOCHASTIC episode on the SAME path seed (diagnostic-only action
    statistics), merging them into one row: every primary field comes from
    the deterministic run; `stochastic_action_std_bid/ask` and
    `stochastic_near_bound_rate` come from the stochastic run."""
    det = run_ppo_episode(model, architecture, seed, deterministic=True)
    det["deterministic"] = True
    det["near_bound_rate_deterministic"] = det.pop("near_bound_rate")

    if stochastic_torch_seed is not None:
        torch.manual_seed(stochastic_torch_seed)
    stoch = run_ppo_episode(model, architecture, seed, deterministic=False)
    det["stochastic_action_std_bid"] = stoch["action_std_bid"]
    det["stochastic_action_std_ask"] = stoch["action_std_ask"]
    det["near_bound_rate_stochastic"] = stoch["near_bound_rate"]
    det["stochastic_full_objective"] = stoch["full_objective"]
    return det


def evaluate_benchmark_row(policy_name: str, controls: dict, seed: int) -> dict:
    """Deterministic-only (benchmarks have no stochastic sampling
    component) -- stochastic diagnostic fields are recorded as NaN with an
    explicit comment, never fabricated."""
    row = run_analytic_episode(policy_name, controls, seed)
    row["deterministic"] = True
    row["near_bound_rate_deterministic"] = row.pop("near_bound_rate")
    row["stochastic_action_std_bid"] = float("nan")
    row["stochastic_action_std_ask"] = float("nan")
    row["near_bound_rate_stochastic"] = float("nan")
    row["stochastic_full_objective"] = float("nan")
    return row


def evaluate_clone_row(clone_agent, seed: int) -> dict:
    row = run_clone_episode(clone_agent, seed)
    row["deterministic"] = True
    row["near_bound_rate_deterministic"] = row.pop("near_bound_rate")
    row["stochastic_action_std_bid"] = float("nan")
    row["stochastic_action_std_ask"] = float("nan")
    row["near_bound_rate_stochastic"] = float("nan")
    row["stochastic_full_objective"] = float("nan")
    return row
