"""
phase4_state_visitation_diagnostic.py
------------------------------------------
Phase 4, Section 5: runs every frozen policy (analytical belief-weighted,
all 15 fixed-step + 15 event-driven RL offline-best models across all three
architectures) on FRESH diagnostic seeds (210000-210199 -- disjoint from
every training/monitoring/validation/Phase-3-holdout range in this project;
see phase4_common.py's DIAGNOSTIC_SEEDS_START for the full inventory these
were checked against) and records the EMPIRICAL STATE-VISITATION
DISTRIBUTION (inventory, belief where applicable, remaining time, bid/ask
depth, quoted spread, expected fill probability, realised fills) using
incremental (bounded-memory) accumulators -- never storing full
trajectories.

These seeds are used ONLY to observe frozen policies -- never to select or
modify a model (Section 5's explicit discipline).

This is deliberately SEPARATE from phase4_action_surface_diagnostic.py's
common-grid comparison: the grid holds state fixed and asks "what would
this policy do here"; this script holds the policy fixed and asks "which
states does this policy actually spend its time in" -- together they
separate "policy surface" from "state visitation" (Section 5's explicit
goal).

Run from repo root:
    python phase4_state_visitation_diagnostic.py
"""
import time

import numpy as np
import pandas as pd

import phase4_common as P4
import simulate_belief_weighted as SBW
from train_agents import build_eval_env, RECURRENT_AGENT_TYPES
from envs.make_envs import make_regime_envs, N_STEPS, KAPPA
from envs.event_driven_regime_env import EventDrivenRegimeSwitchingEnv
from envs.event_driven_hamilton_ppo_wrapper import make_event_time_filter
from mbt_gym.gym.index_names import ASK_INDEX, BID_INDEX

AGENT_TYPES = ("hamilton_ppo", "return_mlp_ppo", "return_lstm_ppo")
ENVIRONMENT_TYPES = ("fixed", "event")

INVENTORY_EDGES = np.arange(-50, 51, 5)
BELIEF_EDGES = np.arange(0.0, 1.01, 0.1)


class VisitationAccumulator:
    def __init__(self):
        self.inventory = P4.IncrementalHistogram(INVENTORY_EDGES)
        self.belief = P4.IncrementalHistogram(BELIEF_EDGES)  # NaN-skipped for non-Hamilton/analytic policies
        self.remaining_time = P4.IncrementalHistogram(np.arange(0.0, 1.01, 0.1))
        self.bid_depth = P4.RunningStatsScalar()
        self.ask_depth = P4.RunningStatsScalar()
        self.quoted_spread = P4.RunningStatsScalar()
        self.p_fill_bid = P4.RunningStatsScalar()
        self.p_fill_ask = P4.RunningStatsScalar()
        self.p_fill_total = P4.RunningStatsScalar()
        self.n_fills = 0
        self.n_steps = 0

    def add_step(self, inventory, tau, bid_depth, ask_depth, fill_bid, fill_ask, belief=None):
        self.inventory.add(inventory)
        if belief is not None:
            self.belief.add(belief)
        self.remaining_time.add(tau)
        self.bid_depth.add(bid_depth)
        self.ask_depth.add(ask_depth)
        self.quoted_spread.add(bid_depth + ask_depth)
        p_bid = np.exp(-KAPPA * bid_depth)
        p_ask = np.exp(-KAPPA * ask_depth)
        self.p_fill_bid.add(p_bid)
        self.p_fill_ask.add(p_ask)
        self.p_fill_total.add(p_bid + p_ask)
        self.n_fills += int(fill_bid) + int(fill_ask)
        self.n_steps += 1

    def summary(self) -> dict:
        d = dict(
            n_steps=self.n_steps, realised_fill_rate_per_step=self.n_fills / self.n_steps if self.n_steps else np.nan,
            mean_bid_depth=self.bid_depth.mean, mean_ask_depth=self.ask_depth.mean,
            mean_quoted_spread=self.quoted_spread.mean,
            mean_p_fill_bid=self.p_fill_bid.mean, mean_p_fill_ask=self.p_fill_ask.mean,
            mean_p_fill_total=self.p_fill_total.mean,
        )
        d.update(self.inventory.as_dict("inventory"))
        d.update(self.belief.as_dict("belief"))
        d.update(self.remaining_time.as_dict("remaining_time"))
        return d


def run_fixed_rl(agent_type: str, seed: int, diag_seeds: list) -> VisitationAccumulator:
    is_recurrent = agent_type in RECURRENT_AGENT_TYPES
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    tag = f"offlinecv_env70000_learner_{seed}"
    path = f"models/{agent_type}/ppo_{agent_type}_{tag}_offline_best.zip"
    model = (RecurrentPPO if is_recurrent else PPO).load(path)

    acc = VisitationAccumulator()
    for dseed in diag_seeds:
        env = build_eval_env(agent_type, dseed, P4.INVENTORY_SCALE, 0.005, environment_type="fixed")
        obs, info = env.reset(seed=dseed)
        lstm_states = None
        episode_start = np.array([True], dtype=bool)
        done = False
        while not done:
            if is_recurrent:
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            episode_start = np.array([False], dtype=bool)
            bid_depth = (float(action[0]) + 1.0) / 2.0 * P4.MAX_DEPTH
            ask_depth = (float(action[1]) + 1.0) / 2.0 * P4.MAX_DEPTH
            inv_before = float(info["raw_state"][1]) if isinstance(info, dict) and "raw_state" in info else env.base_env.raw_inventory
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            belief = env.belief if agent_type == "hamilton_ppo" else None
            tau = float(obs[1])
            fill_bid = fill_ask = 0  # realised fills not directly exposed by these wrappers' info; approximate via inventory delta
            inv_after = float(info["raw_state"][1])
            if inv_after > inv_before:
                fill_bid = 1
            elif inv_after < inv_before:
                fill_ask = 1
            acc.add_step(inv_before, tau, bid_depth, ask_depth, fill_bid, fill_ask, belief=belief)
    return acc


def run_event_rl(agent_type: str, seed: int, diag_seeds: list) -> VisitationAccumulator:
    is_recurrent = agent_type in RECURRENT_AGENT_TYPES
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    tag = f"phase3_event_env70000_learner_{seed}"
    path = f"models/{agent_type}_event/ppo_{agent_type}_{tag}_offline_best.zip"
    model = (RecurrentPPO if is_recurrent else PPO).load(path)

    acc = VisitationAccumulator()
    for dseed in diag_seeds:
        env = build_eval_env(agent_type, dseed, P4.INVENTORY_SCALE, 0.005, environment_type="event")
        obs, info = env.reset(seed=dseed)
        lstm_states = None
        episode_start = np.array([True], dtype=bool)
        done = False
        while not done:
            if is_recurrent:
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            episode_start = np.array([False], dtype=bool)
            bid_depth = (float(action[0]) + 1.0) / 2.0 * P4.MAX_DEPTH
            ask_depth = (float(action[1]) + 1.0) / 2.0 * P4.MAX_DEPTH
            inv_before = float(info["inventory_after"]) if "inventory_after" in info else env.base_env.raw_inventory
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            belief = env.belief if agent_type == "hamilton_ppo" else None
            tau = float(obs[1])
            fill_bid, fill_ask = int(info["fill_indicator"][0]), int(info["fill_indicator"][1])
            acc.add_step(inv_before, tau, bid_depth, ask_depth, fill_bid, fill_ask, belief=belief)
    return acc


def run_fixed_analytical(diag_seeds: list, controls: dict) -> VisitationAccumulator:
    acc = VisitationAccumulator()
    for dseed in diag_seeds:
        np.random.seed(dseed)
        env = make_regime_envs(switch_within_episode=True, seed=dseed)
        filt = SBW.make_filter()
        filt.reset()
        obs = env.reset()
        mid = env.raw_midprice
        done = np.array([False])
        while not np.all(done):
            # BUG FIX: `inventory` here is reconstructed from the NORMALISED
            # observation (obs_flat[1] / SBW.INV_UNIT) for feeding the
            # analytical control -- correct for that purpose (get_control
            # rounds to the nearest integer grid cell, so tiny float noise
            # from the normalise/un-normalise round-trip is irrelevant
            # there). But comparing THIS round-tripped value against
            # info["raw_state"][1] (read directly, no round-trip) for fill
            # detection compares two representations that differ by
            # floating-point noise at the ~1e-9 level -- since real
            # inventory values are small integers, that noise makes
            # `inv_after != inv_before` true almost every step regardless of
            # whether a fill actually happened, wildly overcounting fills
            # (confirmed: 3169/4000 steps flagged as fills for a single
            # episode, vs the ~64/4000 known-correct rate from Phase 3).
            # Fixed: capture inv_before from env.raw_inventory directly (the
            # SAME raw accessor info["raw_state"][1] uses after the step),
            # never from the normalised-observation round-trip.
            obs_flat = np.array(obs).flatten()
            inventory = obs_flat[1] / SBW.INV_UNIT
            t_idx = env.current_step
            tau = 1.0 - t_idx / N_STEPS
            belief = filt.update(mid)
            bid_depth, ask_depth = P4.analytical_belief_weighted_depths(controls, tau, inventory, belief)
            action = np.array([[SBW.normalise_depth(bid_depth), SBW.normalise_depth(ask_depth)]])
            inv_before = env.raw_inventory
            obs, reward, done, info = env.step(action)
            mid = info["raw_midprice"]
            inv_after = float(info["raw_state"][1])
            fill_bid = int(inv_after > inv_before)
            fill_ask = int(inv_after < inv_before)
            acc.add_step(inv_before, tau, bid_depth, ask_depth, fill_bid, fill_ask, belief=belief)
    return acc


def run_event_analytical(diag_seeds: list, controls: dict) -> VisitationAccumulator:
    acc = VisitationAccumulator()
    for dseed in diag_seeds:
        env = EventDrivenRegimeSwitchingEnv(seed=dseed)
        filt = make_event_time_filter()
        env.reset()
        filt.reset(initial_price=env.raw_midprice)
        q = env.raw_inventory
        t_elapsed = 0.0
        done = False
        while not done:
            tau = 1.0 - t_elapsed / env.terminal_time
            belief = filt.belief
            bid_depth, ask_depth = P4.analytical_belief_weighted_depths(controls, tau, q, belief)
            action = np.array([SBW.normalise_depth(bid_depth), SBW.normalise_depth(ask_depth)])
            raw_state, reward, done, info = env.step(action)
            filt.update(info["price_after"], info["elapsed_time"], info["event_type"] == "arrival")
            fill_bid, fill_ask = int(info["fill_indicator"][0]), int(info["fill_indicator"][1])
            acc.add_step(q, tau, bid_depth, ask_depth, fill_bid, fill_ask, belief=belief)
            q = info["inventory_after"]
            t_elapsed = float(info["raw_state"][2])
    return acc


def main():
    diag_seeds = list(range(P4.DIAGNOSTIC_SEEDS_START, P4.DIAGNOSTIC_SEEDS_START + P4.DIAGNOSTIC_SEEDS_COUNT))
    print(f"Diagnostic seeds: {diag_seeds[0]}..{diag_seeds[-1]} ({len(diag_seeds)} total) -- fresh, never used before")

    controls = P4.build_analytical_controls()
    rows = []

    t0 = time.time()
    print("\nAnalytical belief-weighted policy (fixed-step)...")
    acc = run_fixed_analytical(diag_seeds, controls)
    rows.append(dict(agent_type="belief_weighted", environment_type="fixed", learner_seed=None, **acc.summary()))
    print(f"  done in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("Analytical belief-weighted policy (event-driven)...")
    acc = run_event_analytical(diag_seeds, controls)
    rows.append(dict(agent_type="belief_weighted", environment_type="event", learner_seed=None, **acc.summary()))
    print(f"  done in {time.time()-t0:.1f}s")

    for agent_type in AGENT_TYPES:
        for seed in P4.LEARNER_SEEDS:
            t0 = time.time()
            acc = run_fixed_rl(agent_type, seed, diag_seeds)
            rows.append(dict(agent_type=agent_type, environment_type="fixed", learner_seed=seed, **acc.summary()))
            print(f"  fixed/{agent_type}/seed{seed}: {time.time()-t0:.1f}s")

            t0 = time.time()
            acc = run_event_rl(agent_type, seed, diag_seeds)
            rows.append(dict(agent_type=agent_type, environment_type="event", learner_seed=seed, **acc.summary()))
            print(f"  event/{agent_type}/seed{seed}: {time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    out_path = P4.PHASE4_RESULTS_DIR / "phase4_state_visitation.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{len(df)} rows written to {out_path}")
    print(df[["agent_type", "environment_type", "learner_seed", "n_steps", "realised_fill_rate_per_step",
              "mean_quoted_spread", "mean_p_fill_total", "inventory_mean", "inventory_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
