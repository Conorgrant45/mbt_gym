"""
shared/phase4_common.py
--------------------
Shared utilities for Phase 4 (Hamilton PPO policy diagnostics). Read-only
reuse of simulate_belief_weighted.py's solver/control-lookup and
envs/make_envs.py's constants -- no re-derivation.

Model provenance (Section 1): the 5 fixed-step + 5 event-driven Hamilton PPO
`_offline_best.zip` models are exactly the ones produced and verified in
Phase 3 (`offlinecv_env70000_learner_{0..4}` /
`phase3_event_env70000_learner_{0..4}`). This module re-verifies their
metadata directly (architecture, total_timesteps, environment_type,
selected checkpoint, path suffix) rather than assuming Phase 3's checks
still hold.
"""
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

import shared.simulate_belief_weighted as SBW
from envs.make_envs import (
    N_STEPS, STEP_SIZE, TERMINAL_TIME, KAPPA, PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION,
)
from envs.hamilton_ppo_wrapper import DEFAULT_INVENTORY_SCALE

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE4_RESULTS_DIR = REPO_ROOT / "results" / "phase4_policy_diagnostic"
PHASE4_LOGS_DIR = REPO_ROOT / "logs" / "phase4_policy_diagnostic"
PLOTS_DIR = PHASE4_RESULTS_DIR / "plots"

LEARNER_SEEDS = (0, 1, 2, 3, 4)
INVENTORY_SCALE = DEFAULT_INVENTORY_SCALE  # 10.0, production value, unchanged
MAX_DEPTH = SBW.MAX_DEPTH

# Fresh diagnostic seeds: disjoint from EVERY previous range in this project
# (see phase3_orchestrate_event_driven_batch.py's own inventory of ranges,
# plus Phase 3's own training/validation/holdout 0-4/70000/195001-195050/
# 200000-200199). Chosen here, well clear of all of them.
DIAGNOSTIC_SEEDS_START = 210_000
# 50 (not 200): state-visitation instrumentation runs a manual per-step
# Python loop (histogram/running-stats bookkeeping every step) rather than
# the tighter existing evaluate_* functions, and fixed-step episodes are
# 4000 steps each x 15 models -- 50 seeds keeps total wall-clock in the
# tens-of-minutes range on this 8GB machine while still giving a
# statistically meaningful empirical distribution (matches this project's
# existing validation-set convention of 50 episodes, e.g.
# select_checkpoint_offline.py).
DIAGNOSTIC_SEEDS_COUNT = 50


def fixed_model_path(seed: int) -> Path:
    tag = f"offlinecv_env70000_learner_{seed}"
    return REPO_ROOT / "models" / "hamilton_ppo" / f"ppo_hamilton_ppo_{tag}_offline_best.zip"


def event_model_path(seed: int) -> Path:
    tag = f"phase3_event_env70000_learner_{seed}"
    return REPO_ROOT / "models" / "hamilton_ppo_event" / f"ppo_hamilton_ppo_{tag}_offline_best.zip"


def fixed_log_dir() -> Path:
    return REPO_ROOT / "logs" / "hamilton_ppo"


def event_log_dir() -> Path:
    return REPO_ROOT / "logs" / "hamilton_ppo_event"


def verify_and_load_hamilton_models() -> dict:
    """Section 1: verify provenance (learner seed, environment type,
    selected checkpoint timestep, offline_best path suffix, observation/
    action definition via architecture introspection, network architecture,
    training transition count), then load. Returns
    {(environment_type, learner_seed): {"model": PPO, "metadata": dict}}.
    Raises AssertionError on any provenance mismatch -- this diagnostic must
    not silently run on the wrong model."""
    models = {}
    for env_type, path_fn, tag_fn, log_dir_fn in (
        ("fixed", fixed_model_path, lambda s: f"offlinecv_env70000_learner_{s}", fixed_log_dir),
        ("event", event_model_path, lambda s: f"phase3_event_env70000_learner_{s}", event_log_dir),
    ):
        for seed in LEARNER_SEEDS:
            path = path_fn(seed)
            tag = tag_fn(seed)
            log_dir = log_dir_fn()
            assert path.exists(), f"Missing Hamilton PPO offline-best model: {path}"
            assert str(path).endswith("_offline_best.zip"), f"Not an offline-best path: {path}"

            cfg = json.load(open(log_dir / f"run_config_{tag}.json"))
            sel = json.load(open(log_dir / f"offline_selection_{tag}.json"))

            actual_env_type = cfg.get("environment_type", "fixed")
            assert actual_env_type == env_type, f"{path}: environment_type={actual_env_type}, expected {env_type}"
            assert cfg["learner_seed"] == seed, f"{path}: learner_seed={cfg['learner_seed']}, expected {seed}"
            assert cfg["cli_args"]["total_timesteps"] == 200_000, f"{path}: total_timesteps mismatch"
            assert cfg["architecture"]["actor_hidden_layers"] == [64, 64], f"{path}: architecture mismatch"
            assert cfg["architecture"]["activation_fn"] == "Tanh", f"{path}: activation mismatch"
            manifest = json.load(open(log_dir / f"checkpoint_manifest_{tag}.json"))
            candidate_timesteps = [c["timestep"] for c in manifest["checkpoints"]]
            assert sel["selected_timestep"] in candidate_timesteps, (
                f"{path}: selected_timestep {sel['selected_timestep']} not among candidates {candidate_timesteps}"
            )
            assert sel["selected_timestep"] != 0, f"{path}: refuses a trivial/untrained selection"

            model = PPO.load(str(path))
            obs_shape = model.observation_space.shape
            act_shape = model.action_space.shape
            assert obs_shape == (3,), f"{path}: Hamilton PPO observation shape {obs_shape}, expected (3,)"
            assert act_shape == (2,), f"{path}: action shape {act_shape}, expected (2,)"

            models[(env_type, seed)] = dict(
                model=model,
                metadata=dict(
                    environment_type=env_type, learner_seed=seed, tag=tag,
                    selected_checkpoint_timestep=sel["selected_timestep"],
                    selected_validation_mean_objective=sel["selected_mean_cumulative_reward"],
                    model_path=str(path),
                    architecture=cfg["architecture"],
                    total_training_timesteps=cfg["cli_args"]["total_timesteps"],
                ),
            )
    return models


def build_analytical_controls() -> dict:
    """The exact same solver used throughout Phases 1-3 -- unchanged,
    read-only."""
    controls = {}
    for regime, params in SBW.REGIME_PARAMS.items():
        da, db, qag, qbg = SBW.build_optimal_control(**params)
        controls[regime] = dict(delta_ask=da, delta_bid=db, q_ask=qag, q_bid=qbg)
    return controls


def analytical_belief_weighted_depths(controls: dict, tau: float, q: float, belief: float) -> tuple:
    """Returns (bid_depth, ask_depth) for the analytical belief-weighted
    policy at (tau, q, belief). Mirrors evaluate_agents_event_driven.py's
    continuous_time_control's nearest-grid-point convention -- SAME formula,
    not re-derived -- but parameterised by tau (remaining-time fraction)
    directly, since the grid is defined over tau, not elapsed time."""
    elapsed_time = (1.0 - tau) * TERMINAL_TIME
    t_idx = int(np.clip(round(elapsed_time / STEP_SIZE), 0, N_STEPS - 1))
    c0, c1 = controls[0], controls[1]
    a0, b0 = SBW.get_control(c0["delta_ask"], c0["delta_bid"], c0["q_ask"], c0["q_bid"], t_idx, q)
    a1, b1 = SBW.get_control(c1["delta_ask"], c1["delta_bid"], c1["q_ask"], c1["q_bid"], t_idx, q)
    ask_depth = (1.0 - belief) * a0 + belief * a1
    bid_depth = (1.0 - belief) * b0 + belief * b1
    return bid_depth, ask_depth


def hamilton_ppo_depths(model, q: float, tau: float, belief: float) -> tuple:
    """Exact production observation ([tanh(q/scale), tau, belief]) and
    action transformation ((action+1)/2*MAX_DEPTH) -- identical formulas to
    envs/hamilton_ppo_wrapper.py and envs/event_driven_hamilton_ppo_wrapper.py,
    applied exactly once (see tests/test_phase4_policy_diagnostic.py)."""
    obs = np.array([np.tanh(q / INVENTORY_SCALE), tau, belief], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    bid_action, ask_action = float(action[0]), float(action[1])
    bid_depth = (bid_action + 1.0) / 2.0 * MAX_DEPTH
    ask_depth = (ask_action + 1.0) / 2.0 * MAX_DEPTH
    return bid_depth, ask_depth


def fill_probability(depth: float, kappa: float = KAPPA) -> float:
    return float(np.exp(-kappa * depth))


class RunningStatsScalar:
    """O(1)-memory incremental mean/std accumulator for a single scalar
    quantity (see train_agents.RunningStats for the vector version already
    used by the event-driven evaluator)."""

    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0

    def add(self, value: float):
        value = float(value)
        self.n += 1
        self.sum += value
        self.sumsq += value ** 2

    @property
    def mean(self) -> float:
        return self.sum / self.n if self.n else float("nan")

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return float(np.sqrt(max(self.sumsq / self.n - self.mean ** 2, 0.0)))


class IncrementalHistogram:
    """Fixed-bin-edge histogram accumulator -- O(n_bins) memory regardless
    of how many .add() calls are made, so the empirical distribution of a
    state variable can be recorded across arbitrarily many rollout steps
    without storing per-step values (Section 5's 'use incremental
    statistics and do not store unnecessary full trajectories')."""

    def __init__(self, edges: np.ndarray):
        self.edges = np.asarray(edges, dtype=float)
        self.counts = np.zeros(len(edges) - 1, dtype=np.int64)
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0

    def add(self, value: float):
        value = float(value)
        idx = np.clip(np.searchsorted(self.edges, value, side="right") - 1, 0, len(self.counts) - 1)
        self.counts[idx] += 1
        self.n += 1
        self.sum += value
        self.sumsq += value ** 2

    @property
    def mean(self) -> float:
        return self.sum / self.n if self.n else float("nan")

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        var = max(self.sumsq / self.n - self.mean ** 2, 0.0)
        return float(np.sqrt(var))

    def as_dict(self, prefix: str) -> dict:
        d = {f"{prefix}_mean": self.mean, f"{prefix}_std": self.std, f"{prefix}_n": self.n}
        for i, (lo, hi) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            d[f"{prefix}_hist_[{lo:g},{hi:g})"] = int(self.counts[i])
        return d


def default_grid():
    """Section 2: dense diagnostic grid, calibrated against the actual
    observed inventory range in Phase 3's holdout data (max |inventory| was
    46 fixed-step / 26 event-driven, see event_driven_phase3_results.md) --
    extended slightly beyond the task's own {-30,...,30} example to cover
    the realised tail."""
    q_grid = np.arange(-40, 41, 2, dtype=float)     # 41 points
    tau_grid = np.round(np.arange(0.0, 1.01, 0.1), 2)  # 11 points
    b_grid = np.round(np.arange(0.0, 1.01, 0.05), 2)   # 21 points
    return q_grid, tau_grid, b_grid
