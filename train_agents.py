"""
train_agents.py
------------------------
Common training entry point for all three PPO agent types:

    --agent-type hamilton_ppo      (Hamilton belief-state feed-forward PPO)
    --agent-type return_mlp_ppo    (raw-return feed-forward PPO, no memory)
    --agent-type return_lstm_ppo   (raw-return recurrent PPO, sb3-contrib)

Does NOT replace train_hamilton_ppo.py (kept unmodified). Reuses its
conventions (architecture introspection, NaN-parameter checks, dependency
logging, run-config/run-summary JSON, save/reload equivalence, fixed
held-out eval seeds) generalised across all three agent types, and its
market-dynamics / reward / observation-wrapper imports UNCHANGED.

This is a Phase-2 (structural validation) training script: correctness,
not final dissertation hyperparameters. See module-level DEFAULTS comment
below for exactly which settings mirror train_hamilton_ppo.py and which
are recurrent-specific provisional choices.

Run from repo root:
    python train_agents.py --agent-type hamilton_ppo    --total-timesteps 20000 --seed 0
    python train_agents.py --agent-type return_mlp_ppo   --total-timesteps 20000 --seed 0
    python train_agents.py --agent-type return_lstm_ppo  --total-timesteps 20000 --seed 0
"""

import argparse
import hashlib
import io
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gym as legacy_gym
import gymnasium
import stable_baselines3
import sb3_contrib

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO

from envs.hamilton_ppo_wrapper import HamiltonPPOWrapper, DEFAULT_INVENTORY_SCALE
from envs.return_ppo_wrapper import ReturnPPOWrapper, DEFAULT_RETURN_SCALE
from envs.make_envs import make_regime_envs, N_STEPS
from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, ASSET_PRICE_INDEX

REPO_ROOT = Path(__file__).resolve().parent

AGENT_TYPES = ("hamilton_ppo", "return_mlp_ppo", "return_lstm_ppo")
RECURRENT_AGENT_TYPES = ("return_lstm_ppo",)

# ----------------------------------------------------------------------
# DEFAULTS
#
# hamilton_ppo / return_mlp_ppo (feed-forward): n_steps=4000 (one full
# episode per rollout), batch_size=400 -- IDENTICAL to train_hamilton_ppo.
# py's existing defaults, preserved exactly (not re-derived here).
#
# return_lstm_ppo (recurrent): n_steps=4000, batch_size=400 -- REVISED
# after a targeted recurrent-PPO diagnostic (60k-step runs, seeds 0-3 at
# 400/400, then seeds 0-1 at 4000/400). At 400/400 (10 rollouts per
# 4000-step episode), seeds 0 and 2 both showed a severe, fully-collapsed
# policy (terminal inventory saturated at the same sign on 100%/0% of 30
# holdout episodes respectively, magnitude ~77-86 shares, opposite signs;
# ask/bid action std near zero, one component pinned exactly at a Box
# bound; critic explained_variance ~1e-5-1e-6 -- indistinguishable from
# noise -- vs 0.12-0.47 for the feed-forward agents; full_objective std
# 573-623 vs 39-110 for every other agent) while seeds 1 and 3 showed the
# same critic-failure and reduced-action-std symptoms without the
# catastrophic inventory outcome. Switching to n_steps=4000 (one rollout
# per episode, still batch_size=400 so gradient-update density stays at
# (4000/400)*10 = 100 per 4000 env steps -- unchanged from 400/400)
# resolved the catastrophic pattern on both seeds retested (0 and 1):
# full_objective std fell to 60-67 (in the feed-forward range), terminal
# inventory saturation disappeared (a moderate, non-saturated directional
# bias remained, comparable in kind to what the feed-forward agents
# already show), and explained_variance rose ~100-1000x (still below
# feed-forward levels, but no longer indistinguishable from zero). Only
# seeds 0 and 1 were retested at 4000/400 -- the full 5-seed experiment
# is the first confirmation at target scale, matching how this project
# has always used the final multiseed run as the actual arbiter of
# seed-to-seed robustness.
#
# This is safe because of how sb3-contrib actually carries state (read
# from sb3_contrib/ppo_recurrent/ppo_recurrent.py and common/recurrent/
# buffers.py, not assumed): self._last_lstm_states is carried across
# successive collect_rollouts() calls (deepcopy()'d in, reassigned at the
# end of the loop) and is ONLY masked to zero at genuine episode
# boundaries via episode_starts inside _process_sequence() -- this holds
# regardless of n_steps, so n_steps=4000 does not change the state-
# carrying mechanism itself, only how many rollouts span one episode (1
# instead of 10) and the resulting sequence/padding structure each
# minibatch is built from.
# ----------------------------------------------------------------------
FEEDFORWARD_DEFAULT_N_STEPS = 4_000
FEEDFORWARD_DEFAULT_BATCH_SIZE = 400
RECURRENT_DEFAULT_N_STEPS = 4_000
RECURRENT_DEFAULT_BATCH_SIZE = 400
DEFAULT_LSTM_HIDDEN_SIZE = 64
DEFAULT_N_LSTM_LAYERS = 1

DEFAULT_EVAL_SEEDS = [90001, 90002, 90003, 90004, 90005]
RECONCILIATION_TOL = 1e-6


# ======================================================================
# Environment construction (single source per agent type -- no
# duplicated make_regime_envs()/wrapper-construction logic)
# ======================================================================
def build_train_env(agent_type: str, seed: int, inventory_scale: float, return_scale: float):
    if agent_type == "hamilton_ppo":
        return HamiltonPPOWrapper(inventory_scale=inventory_scale, seed=seed)
    elif agent_type in ("return_mlp_ppo", "return_lstm_ppo"):
        return ReturnPPOWrapper(inventory_scale=inventory_scale, return_scale=return_scale, seed=seed)
    raise ValueError(f"Unknown agent_type: {agent_type!r} (expected one of {AGENT_TYPES})")


def build_eval_env(agent_type: str, seed: int, inventory_scale: float, return_scale: float):
    """A FRESH base_env + wrapper for this one episode -- never reuse an
    already-stepped wrapper via reset(seed=s) for reproducibility (see
    tests/test_hamilton_ppo_wrapper.py's TestSeedReset for why)."""
    base_env = make_regime_envs(switch_within_episode=True, seed=seed)
    if agent_type == "hamilton_ppo":
        wrapper = HamiltonPPOWrapper(base_env=base_env, inventory_scale=inventory_scale)
    elif agent_type in ("return_mlp_ppo", "return_lstm_ppo"):
        wrapper = ReturnPPOWrapper(base_env=base_env, inventory_scale=inventory_scale, return_scale=return_scale)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type!r}")
    return wrapper


def make_train_env_fn(agent_type: str, inventory_scale: float, return_scale: float, seed: int):
    def _init():
        env = build_train_env(agent_type, seed, inventory_scale, return_scale)
        return Monitor(env)
    return _init


# ======================================================================
# Recurrent-aware single-episode evaluation
# ======================================================================
def run_eval_episode(model, env, seed: int, is_recurrent: bool, deterministic: bool = True) -> dict:
    """
    Run one full episode to completion with deterministic actions.
    Recurrent-aware: carries LSTM hidden/cell state across steps within
    the episode, and always starts a fresh episode with episode_start=True
    (state effectively re-initialised to zeros inside model.predict when
    state=None, matching a genuine new-episode reset -- see
    RecurrentActorCriticPolicy.predict()).

    reset(seed=seed) -- not reset() -- is required here: RegimeSwitchingEnv's
    regime-transition draws read from the LEGACY GLOBAL np.random stream
    (see envs/make_envs.py's make_regime_envs docstring), which build_eval_env's
    make_regime_envs(seed=seed) call does NOT reseed (it only seeds the
    per-model midprice/arrival/fill RNGs via SeedSequence). Without
    reset(seed=seed) here, each evaluation episode's regime path would
    depend on whatever uncontrolled global-stream position preceded it
    (e.g. left over from training rollouts or earlier eval episodes),
    making baseline/final/reload evaluations on "the same seed"
    silently non-reproducible -- exactly the bug this reset(seed=...)
    call fixes (caught by the save/reload equivalence check during Phase 2
    smoke testing).
    """
    obs, info = env.reset(seed=seed)
    cash_0 = env.base_env.raw_cash
    inv_0 = env.base_env.raw_inventory
    mid_0 = env.base_env.raw_midprice

    lstm_states = None
    episode_start = np.array([True], dtype=bool)

    cumulative_reward = 0.0
    actions_taken = []
    n_steps = 0
    terminated = truncated = False
    info = None
    while not (terminated or truncated):
        if is_recurrent:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=deterministic,
            )
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
        actions_taken.append(np.asarray(action, dtype=np.float64).copy())

        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += float(reward)
        episode_start = np.array([False], dtype=bool)
        n_steps += 1

    cash_T = float(info["raw_state"][CASH_INDEX])
    inv_T = float(info["raw_state"][INVENTORY_INDEX])
    mid_T = float(info["raw_state"][ASSET_PRICE_INDEX])
    raw_pnl = (cash_T + inv_T * mid_T) - (cash_0 + inv_0 * mid_0)

    return dict(
        steps=n_steps,
        cumulative_reward=cumulative_reward,
        raw_pnl=raw_pnl,
        terminal_abs_inventory=abs(inv_T),
        actions=np.array(actions_taken),
    )


def evaluate_policy(model, agent_type: str, eval_seeds, inventory_scale: float, return_scale: float,
                     deterministic: bool = True):
    is_recurrent = agent_type in RECURRENT_AGENT_TYPES
    records = []
    for seed in eval_seeds:
        env = build_eval_env(agent_type, seed, inventory_scale, return_scale)
        r = run_eval_episode(model, env, seed, is_recurrent, deterministic)
        assert r["steps"] == N_STEPS, f"eval episode (seed={seed}) did not run the full {N_STEPS} steps"
        r["seed"] = seed
        records.append(r)

    rewards = np.array([r["cumulative_reward"] for r in records])
    pnls = np.array([r["raw_pnl"] for r in records])
    term_invs = np.array([r["terminal_abs_inventory"] for r in records])
    all_actions = np.concatenate([r["actions"] for r in records], axis=0)

    summary = dict(
        n_episodes=len(records),
        mean_cumulative_reward=float(rewards.mean()),
        std_cumulative_reward=float(rewards.std()),
        mean_raw_pnl=float(pnls.mean()),
        mean_terminal_abs_inventory=float(term_invs.mean()),
        action_mean_bid=float(all_actions[:, 0].mean()),
        action_mean_ask=float(all_actions[:, 1].mean()),
        action_std_bid=float(all_actions[:, 0].std()),
        action_std_ask=float(all_actions[:, 1].std()),
    )
    return summary, records


# ======================================================================
# Periodic evaluation callback (recurrent-aware)
#
# Evaluates from _on_rollout_start(), NOT _on_step(). Verified directly
# from the installed stable_baselines3.common.on_policy_algorithm source
# (OnPolicyAlgorithm.learn()'s main loop):
#
#     while self.num_timesteps < total_timesteps:
#         continue_training = self.collect_rollouts(...)   # calls
#             #   callback.on_rollout_start()   <- fires FIRST
#             #   n_steps x callback.on_step()
#             #   callback.on_rollout_end()     <- fires LAST
#         self.train()                          # <- runs AFTER collect_rollouts() returns
#
# So on_rollout_start() for rollout i+1 always fires strictly after
# rollout i's train() call has completed (train() runs between
# collect_rollouts() returning for rollout i and collect_rollouts()
# starting for rollout i+1, and on_rollout_start() is the very first hook
# inside collect_rollouts()). Evaluating there means a row labelled
# total_timesteps=N reflects policy weights updated using ALL rollouts
# through N -- not, as with the previous _on_step()-based version, weights
# from one rollout earlier while already claiming credit for N timesteps
# of data.
#
# self.model.num_timesteps (not self.num_timesteps) is read directly:
# BaseCallback only refreshes its own cached self.num_timesteps inside
# on_step() (see callbacks.py), never inside on_rollout_start() -- reading
# the model's own live counter is the authoritative, unambiguous source
# at this call site regardless of that caching detail.
#
# The very first on_rollout_start() call (at the start of training,
# self.model.num_timesteps == 0) fires before ANY train() has run --
# explicitly skipped here so this callback never evaluates before a
# completed PPO update. The separately-computed baseline_untrained row
# in main() already covers t=0 with its own, differently-timed evaluation
# (built before .learn() is even called).
# ======================================================================
class PeriodicEvalCallback(BaseCallback):
    """
    best_model_path (optional): if given, the FULL model (identical
    save format to the ordinary final-model save -- self.model.save(...),
    the same call main() uses for the final model, so a recurrent model's
    complete trained policy parameters are saved/reloadable exactly as the
    final model is) is saved to this path every time a periodic evaluation
    (never the separately-computed, pre-training baseline at timestep
    zero -- that never reaches this callback at all, see the class-level
    lifecycle docstring above) produces a NEW best mean_cumulative_reward
    on the fixed eval_seeds. This does not implement early stopping:
    training always continues to the requested budget regardless of
    whether a new best is found; only the best-seen checkpoint on disk is
    updated.

    checkpoints_dir/run_tag/overwrite_checkpoints/checkpoint_manifest (all
    optional; save-all-checkpoints stays fully disabled unless
    checkpoints_dir is given): if checkpoints_dir is not None, EVERY
    periodic evaluation (again, never the pre-training baseline) also
    saves a candidate checkpoint via save_candidate_checkpoint() -- see
    that function's docstring for exactly why this timing is correct (the
    same _on_rollout_start()/_run_eval() call site as best-checkpoint
    saving, strictly after that rollout's train() has completed) and for
    why the FINAL total_timesteps candidate is NOT saved from here (main()
    saves it directly after model.learn() returns). checkpoint_manifest,
    if given, is a caller-owned list (same convention as training_curve)
    that each saved candidate's manifest entry is appended to.
    """

    def __init__(self, agent_type, eval_seeds, inventory_scale, return_scale, eval_freq,
                 training_curve: list, verbose: int = 1, best_model_path: str = None,
                 checkpoints_dir=None, run_tag: str = None, overwrite_checkpoints: bool = False,
                 checkpoint_manifest: list = None):
        super().__init__(verbose)
        self.agent_type = agent_type
        self.eval_seeds = eval_seeds
        self.inventory_scale = inventory_scale
        self.return_scale = return_scale
        self.eval_freq = eval_freq
        self.training_curve = training_curve
        self.best_model_path = best_model_path
        self._last_eval_step = 0

        self.checkpoints_dir = checkpoints_dir
        self.run_tag = run_tag
        self.overwrite_checkpoints = overwrite_checkpoints
        self.checkpoint_manifest = checkpoint_manifest

        # Best-checkpoint tracking -- populated only by genuine periodic
        # evaluations (never the pre-training baseline). None until the
        # first periodic evaluation fires.
        self.best_mean_objective = None
        self.best_std_objective = None
        self.best_timestep = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        completed_timesteps = self.model.num_timesteps
        if completed_timesteps == 0:
            return  # no train() has completed yet -- nothing to evaluate
        if completed_timesteps - self._last_eval_step >= self.eval_freq:
            self._last_eval_step = completed_timesteps
            self._run_eval(completed_timesteps)

    def _run_eval(self, completed_timesteps: int):
        summary, _ = evaluate_policy(
            self.model, self.agent_type, self.eval_seeds, self.inventory_scale, self.return_scale,
            deterministic=True,
        )
        # completed_timesteps: environment samples whose associated PPO
        # training update(s) have actually finished by the time this
        # evaluation runs -- see class docstring above.
        summary["total_timesteps"] = completed_timesteps
        summary["tag"] = f"timestep_{completed_timesteps}"
        self.training_curve.append(summary)
        if self.verbose:
            print(f"\n[periodic eval @ {completed_timesteps} completed training timesteps] "
                  f"mean_reward={summary['mean_cumulative_reward']:.4f} "
                  f"(std={summary['std_cumulative_reward']:.4f})  "
                  f"mean_raw_pnl={summary['mean_raw_pnl']:.4f}  "
                  f"mean_terminal|inv|={summary['mean_terminal_abs_inventory']:.3f}\n")
        for key in ("mean_cumulative_reward", "std_cumulative_reward", "mean_raw_pnl",
                    "mean_terminal_abs_inventory"):
            self.logger.record(f"eval/{key}", summary[key])

        is_new_best = self.best_mean_objective is None or summary["mean_cumulative_reward"] > self.best_mean_objective
        if is_new_best:
            self.best_mean_objective = summary["mean_cumulative_reward"]
            self.best_std_objective = summary["std_cumulative_reward"]
            self.best_timestep = completed_timesteps
            if self.best_model_path is not None:
                self.model.save(self.best_model_path)
                if self.verbose:
                    print(f"  new best validation mean_cumulative_reward={self.best_mean_objective:.4f} "
                          f"@ {completed_timesteps} timesteps -- saved to {self.best_model_path}")

        if self.checkpoints_dir is not None:
            entry = save_candidate_checkpoint(
                self.model, self.checkpoints_dir, self.agent_type, self.run_tag,
                completed_timesteps, self.overwrite_checkpoints, is_final=False,
            )
            if self.checkpoint_manifest is not None:
                self.checkpoint_manifest.append(entry)
            if self.verbose:
                print(f"  save-all-checkpoints: candidate saved to {entry['path']}")


# ======================================================================
# Config / introspection helpers
# ======================================================================
def get_dependency_versions():
    return dict(
        python=sys.version,
        platform=platform.platform(),
        gym=legacy_gym.__version__,
        gymnasium=gymnasium.__version__,
        stable_baselines3=stable_baselines3.__version__,
        sb3_contrib=sb3_contrib.__version__,
        torch=torch.__version__,
        numpy=np.__version__,
        pandas=pd.__version__,
    )


def introspect_architecture(model, is_recurrent: bool):
    """Confirm (not just declare) the actual built actor/critic layer
    shapes, read directly from the constructed torch modules."""
    def linear_out_features(seq_module):
        return [m.out_features for m in seq_module if isinstance(m, torch.nn.Linear)]

    mlp_extractor = model.policy.mlp_extractor
    arch = dict(
        actor_hidden_layers=linear_out_features(mlp_extractor.policy_net),
        critic_hidden_layers=linear_out_features(mlp_extractor.value_net),
        actor_output_dim=model.policy.action_net.out_features,
        critic_output_dim=model.policy.value_net.out_features,
        activation_fn=model.policy.activation_fn.__name__,
        total_parameters=sum(p.numel() for p in model.policy.parameters()),
    )
    if is_recurrent:
        lstm_actor = model.policy.lstm_actor
        lstm_critic = model.policy.lstm_critic
        arch.update(
            lstm_hidden_size=lstm_actor.hidden_size,
            n_lstm_layers=lstm_actor.num_layers,
            shared_lstm=model.policy.shared_lstm,
            enable_critic_lstm=model.policy.enable_critic_lstm,
            critic_has_separate_lstm=lstm_critic is not None,
        )
    return arch


def assert_no_nan_params(model):
    for name, param in model.policy.named_parameters():
        assert torch.isfinite(param).all(), f"Non-finite value found in policy parameter '{name}'"


def hash_policy_state_dict(model) -> str:
    """SHA-256 of the policy's state_dict, exactly as it stands at call
    time -- used to fingerprint save-all-checkpoints candidates (and, in
    select_checkpoint_offline.py, to verify a copied model is identical to
    its source). Full hex digest (not truncated) -- this is a
    correctness-verification artifact, not a console diagnostic label."""
    buf = io.BytesIO()
    torch.save(model.policy.state_dict(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


# ======================================================================
# CLI
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent-type", type=str, required=True, choices=AGENT_TYPES)
    p.add_argument("--seed", type=int, default=None,
                    help="[Retained for backwards compatibility] Used as BOTH --learner-seed and "
                         "--env-seed when neither is given explicitly -- see resolve_seeds(). Prefer "
                         "--learner-seed/--env-seed for new runs where the two should differ.")
    p.add_argument("--learner-seed", type=int, default=None,
                    help="Seeds PPO/RecurrentPPO construction only: policy/value-network initialisation, "
                         "action sampling, minibatch ordering, and the torch/numpy/random state SB3 seeds "
                         "internally (see stable_baselines3.common.utils.set_random_seed). Never reaches "
                         "the training environment's exogenous RNG streams (see --env-seed). Defaults to "
                         "--seed if not given.")
    p.add_argument("--env-seed", type=int, default=None,
                    help="Seeds ONLY the exogenous training environment (initial regime, regime-transition "
                         "draws, Brownian increments, arrivals, jumps, fill draws) via "
                         "make_regime_envs(seed=env_seed) -- see build_train_env(). Defaults to the "
                         "resolved learner_seed if not given (preserves the old single --seed behaviour).")
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=None,
                    help="Rollout length. Default: 4000 for feed-forward agents (one full episode), "
                         "4000 for return_lstm_ppo too (one rollout per episode; resolved a training-seed "
                         "collapse seen at n_steps=400 -- see module DEFAULTS comment) unless overridden.")
    p.add_argument("--batch-size", type=int, default=None,
                    help="Default: 400 for feed-forward agents, 400 for return_lstm_ppo unless overridden.")
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--net-arch", type=int, nargs="+", default=[64, 64])
    p.add_argument("--lstm-hidden-size", type=int, default=DEFAULT_LSTM_HIDDEN_SIZE,
                    help="Only used when --agent-type return_lstm_ppo.")
    p.add_argument("--n-lstm-layers", type=int, default=DEFAULT_N_LSTM_LAYERS,
                    help="Only used when --agent-type return_lstm_ppo.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--inventory-scale", type=float, default=DEFAULT_INVENTORY_SCALE)
    p.add_argument("--return-scale", type=float, default=DEFAULT_RETURN_SCALE,
                    help="Only used when --agent-type is return_mlp_ppo or return_lstm_ppo.")
    p.add_argument("--eval-seeds", type=int, nargs="+", default=DEFAULT_EVAL_SEEDS)
    p.add_argument("--eval-freq", type=int, default=40_000)
    p.add_argument("--checkpoint-frequency", type=int, default=0,
                    help="If >0, save an intermediate model checkpoint every N timesteps. 0 disables checkpointing.")
    p.add_argument("--save-all-checkpoints", action="store_true", default=False,
                    help="Save a full candidate checkpoint after each completed rollout that crosses the "
                         "--eval-freq threshold (the same cadence as periodic evaluation / online-best "
                         "tracking), plus one at --total-timesteps, to "
                         "models/<agent_type>/checkpoints/<run-tag>/ -- for later offline selection via "
                         "select_checkpoint_offline.py on a larger validation set. Independent of, and does "
                         "not affect, --checkpoint-frequency's existing raw-step CheckpointCallback.")
    p.add_argument("--overwrite-checkpoints", action="store_true", default=False,
                    help="Allow --save-all-checkpoints to overwrite candidate checkpoint files that already "
                         "exist for this --run-tag. Default false: a pre-existing candidate file raises "
                         "before training starts.")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Directory for saved models. Default: models/<agent_type>/")
    p.add_argument("--log-dir", type=str, default=None,
                    help="Directory for config/summary/log files. Default: logs/<agent_type>/")
    p.add_argument("--model-path", type=str, default=None,
                    help="Full model save path (without .zip). Default: <output-dir>/ppo_<agent_type>_<run-tag>")
    p.add_argument("--run-tag", type=str, default="v1")
    return p.parse_args()


def rollout_spans_full_episode(n_steps: int, episode_length: int) -> bool:
    """Whether one rollout (n_steps env-steps) covers a whole number of
    episodes -- agent-type-agnostic by construction, with no per-agent
    branching: the market environment's episode length is fixed
    regardless of which algorithm/policy is stepping it, so this
    condition means the same thing for every agent type."""
    return n_steps % episode_length == 0


def resolve_seeds(args) -> tuple:
    """
    Resolve (learner_seed, env_seed) from --seed (legacy)/--learner-seed/--env-seed.

    learner_seed controls ONLY PPO/RecurrentPPO's own construction -- passed
    as its seed=... kwarg (policy/value-network initialisation, action
    sampling, minibatch ordering, and the torch/numpy/random global state
    SB3 seeds internally via set_random_seed(); see common_kwargs in main()).

    env_seed controls ONLY the exogenous training environment -- passed to
    build_train_env()/make_regime_envs(seed=env_seed) (initial regime,
    regime-transition draws, Brownian increments, arrivals, jumps, fill
    draws), each already isolated in its own local np.random.Generator (see
    envs/make_envs.py, envs/regime_env.py, tests/test_rng_isolation.py) --
    nothing learner_seed touches can ever reach these streams.

    Resolution order (agent-type-agnostic, pure function -- no I/O, no
    environment/model construction):
      1. --seed and --learner-seed conflicting with each other -> error.
      2. learner_seed = --learner-seed if given, else --seed.
      3. Neither given -> error.
      4. env_seed = --env-seed if given, else learner_seed (old --seed-only
         behaviour: one seed drives both).
    """
    if args.learner_seed is not None and args.seed is not None and args.learner_seed != args.seed:
        raise ValueError(
            f"--seed={args.seed} and --learner-seed={args.learner_seed} were both supplied with "
            f"conflicting values -- pass only one, or make them agree."
        )
    if args.learner_seed is not None:
        learner_seed = args.learner_seed
    elif args.seed is not None:
        learner_seed = args.seed
    else:
        raise ValueError("Must supply --learner-seed (preferred) or --seed.")
    env_seed = args.env_seed if args.env_seed is not None else learner_seed
    return learner_seed, env_seed


# ======================================================================
# --save-all-checkpoints: offline checkpoint-selection support.
#
# Candidate checkpoints are saved at the SAME cadence as periodic
# evaluation and online-best-checkpoint tracking (PeriodicEvalCallback's
# _on_rollout_start() threshold-crossing check -- see that class's
# docstring), plus total_timesteps itself. total_timesteps is included
# unconditionally because OnPolicyAlgorithm.learn()'s loop
# (`while self.num_timesteps < total_timesteps: collect_rollouts(); train()`)
# exits immediately once the final train() completes -- there is no
# subsequent on_rollout_start() call in which the final rollout's own
# completion could ever be evaluated/checkpointed by the periodic
# mechanism. main() saves that last candidate explicitly, after
# model.learn() returns.
# ======================================================================
def compute_checkpoint_timesteps(n_steps: int, eval_freq: int, total_timesteps: int) -> list:
    """
    Pure function: the exact set of timesteps save-all-checkpoints saves a
    candidate at, mirroring PeriodicEvalCallback._on_rollout_start()'s own
    threshold-crossing loop (rollout boundaries -- multiples of n_steps --
    where completed_timesteps - last_eval_step >= eval_freq) rather than
    assuming eval_freq is itself a multiple of n_steps. total_timesteps is
    always appended if not already the last candidate (see module
    docstring above for why).

    Example: n_steps=4000, eval_freq=16000, total_timesteps=200000 ->
    [16000, 32000, ..., 192000, 200000] (200000 is NOT a multiple of
    16000, so it is appended separately -- matching the exact example in
    this feature's specification).
    """
    candidates = []
    last_eval_step = 0
    completed = 0
    while completed < total_timesteps:
        completed += n_steps
        if completed - last_eval_step >= eval_freq:
            last_eval_step = completed
            candidates.append(completed)
    if not candidates or candidates[-1] != total_timesteps:
        candidates.append(total_timesteps)
    return candidates


def candidate_checkpoint_path(checkpoints_dir, agent_type: str, run_tag: str, timestep: int) -> Path:
    """Path WITHOUT a .zip suffix (matches this file's model.save(...)
    convention elsewhere, e.g. best_model_path) for the save-all-checkpoints
    candidate at this timestep. The file SB3 actually writes is this path
    + '.zip'. Zero-padded to 9 digits (matches the feature spec's exact
    example: t000016000.zip, ..., t000200000.zip)."""
    return Path(checkpoints_dir) / f"ppo_{agent_type}_{run_tag}_t{timestep:09d}"


def save_candidate_checkpoint(model, checkpoints_dir, agent_type: str, run_tag: str,
                               timestep: int, overwrite: bool, is_final: bool) -> dict:
    """
    Save one save-all-checkpoints candidate and return its checkpoint-manifest
    entry. Shared by PeriodicEvalCallback (periodic candidates, saved from
    _on_rollout_start()/_run_eval() -- i.e. strictly after the PPO update
    for the just-completed rollout, never before) and main() (the final
    -timestep candidate -- see module docstring above for why that one
    cannot go through the callback).

    Raises FileExistsError (not a silent overwrite) if the target file
    already exists and overwrite=False.
    """
    base_path = candidate_checkpoint_path(checkpoints_dir, agent_type, run_tag, timestep)
    zip_path = Path(str(base_path) + ".zip")
    if zip_path.exists() and not overwrite:
        raise FileExistsError(
            f"Candidate checkpoint already exists: {zip_path}\n"
            f"Pass --overwrite-checkpoints to overwrite, or use a different --run-tag."
        )
    model.save(str(base_path))
    return dict(
        timestep=timestep,
        path=str(zip_path),
        saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        is_final=is_final,
        policy_hash=hash_policy_state_dict(model),
    )


def resolve_rollout_geometry(args):
    is_recurrent = args.agent_type in RECURRENT_AGENT_TYPES
    n_steps = args.n_steps
    batch_size = args.batch_size
    if n_steps is None:
        n_steps = RECURRENT_DEFAULT_N_STEPS if is_recurrent else FEEDFORWARD_DEFAULT_N_STEPS
    if batch_size is None:
        batch_size = RECURRENT_DEFAULT_BATCH_SIZE if is_recurrent else FEEDFORWARD_DEFAULT_BATCH_SIZE
    return n_steps, batch_size


def main():
    args = parse_args()
    t_run_start = time.time()
    is_recurrent = args.agent_type in RECURRENT_AGENT_TYPES

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "models" / args.agent_type
    log_dir = Path(args.log_dir) if args.log_dir else REPO_ROOT / "logs" / args.agent_type
    model_path = args.model_path or str(output_dir / f"ppo_{args.agent_type}_{args.run_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    n_steps, batch_size = resolve_rollout_geometry(args)
    learner_seed, env_seed = resolve_seeds(args)

    # --- save-all-checkpoints: fail fast, before spending any training
    # compute, if any candidate checkpoint file for this run-tag already
    # exists and --overwrite-checkpoints was not given. (save_candidate_
    # checkpoint() itself re-checks at save time too -- this is a
    # fail-fast pre-flight, not a substitute for that check.) ---
    checkpoints_dir = output_dir / "checkpoints" / args.run_tag
    candidate_timesteps = []
    if args.save_all_checkpoints:
        candidate_timesteps = compute_checkpoint_timesteps(n_steps, args.eval_freq, args.total_timesteps)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        pre_existing = [
            str(Path(str(candidate_checkpoint_path(checkpoints_dir, args.agent_type, args.run_tag, t)) + ".zip"))
            for t in candidate_timesteps
            if Path(str(candidate_checkpoint_path(checkpoints_dir, args.agent_type, args.run_tag, t)) + ".zip").exists()
        ]
        if pre_existing and not args.overwrite_checkpoints:
            raise FileExistsError(
                f"--save-all-checkpoints: candidate checkpoint file(s) already exist for run-tag "
                f"{args.run_tag!r}:\n  " + "\n  ".join(pre_existing) +
                f"\nPass --overwrite-checkpoints to overwrite, or use a different --run-tag."
            )

    dep_versions = get_dependency_versions()

    print("=" * 78)
    print(f"train_agents.py [{args.agent_type} / {args.run_tag}] -- structural-validation run "
          f"(Phase 2), NOT a final dissertation result")
    print("=" * 78)
    for k, v in vars(args).items():
        print(f"  {k} = {v}")
    print(f"  resolved n_steps = {n_steps}")
    print(f"  resolved batch_size = {batch_size}")
    print(f"  resolved learner_seed = {learner_seed}")
    print(f"  resolved training_env_seed = {env_seed}")
    print("=" * 78)

    # --- Build training env ---
    # learner_seed seeds ONLY learner-side global numpy state here (belt-
    # and-suspenders alongside PPO's own set_random_seed(learner_seed) call
    # inside its constructor below) -- it has NO effect on the environment:
    # every stochastic component built by make_train_env_fn(..., env_seed)
    # below draws from its own isolated np.random.Generator, seeded solely
    # from env_seed (see envs/make_envs.py, envs/regime_env.py). Env
    # construction is intentionally ordered after this call to make that
    # independence obvious, not because it is load-bearing.
    np.random.seed(learner_seed)
    vec_env = DummyVecEnv([make_train_env_fn(args.agent_type, args.inventory_scale, args.return_scale, env_seed)])
    action_low = np.asarray(vec_env.action_space.low, dtype=np.float64)
    action_high = np.asarray(vec_env.action_space.high, dtype=np.float64)

    common_kwargs = dict(
        learning_rate=args.learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        seed=learner_seed,
        verbose=1,
    )

    if is_recurrent:
        policy_kwargs = dict(
            net_arch=list(args.net_arch),
            lstm_hidden_size=args.lstm_hidden_size,
            n_lstm_layers=args.n_lstm_layers,
        )
        model = RecurrentPPO("MlpLstmPolicy", vec_env, policy_kwargs=policy_kwargs, **common_kwargs)
    else:
        policy_kwargs = dict(net_arch=list(args.net_arch))
        model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, **common_kwargs)

    architecture = introspect_architecture(model, is_recurrent)
    print("\nConfirmed actor/critic architecture (introspected from the built model):")
    for k, v in architecture.items():
        print(f"  {k} = {v}")

    n_envs = vec_env.num_envs
    n_minibatches = (n_steps * n_envs) // batch_size
    rollout_geometry = dict(
        n_steps=n_steps,
        batch_size=batch_size,
        n_envs=n_envs,
        buffer_size=n_steps * n_envs,
        episode_length=N_STEPS,
        rollouts_span_full_episode=rollout_spans_full_episode(n_steps, N_STEPS),
        n_minibatches_per_epoch=n_minibatches,
        n_gradient_updates_per_rollout=n_minibatches * args.n_epochs,
        n_rollouts_total=args.total_timesteps // n_steps,
    )
    print("\nRollout geometry:")
    for k, v in rollout_geometry.items():
        print(f"  {k} = {v}")

    run_config = dict(
        agent_type=args.agent_type,
        run_tag=args.run_tag,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        cli_args=vars(args),
        resolved_n_steps=n_steps,
        resolved_batch_size=batch_size,
        learner_seed=learner_seed,
        training_env_seed=env_seed,
        architecture=architecture,
        rollout_geometry=rollout_geometry,
        dependency_versions=dep_versions,
        resolved_device=str(model.device),
        observation_scaling=dict(
            inventory_scale=args.inventory_scale,
            inventory_transform="tanh(raw_inventory / inventory_scale)",
            return_scale=(args.return_scale if args.agent_type != "hamilton_ppo" else None),
            return_transform=("tanh(raw_return / return_scale)" if args.agent_type != "hamilton_ppo" else None),
        ),
        eval_seeds=args.eval_seeds,
    )
    config_path = log_dir / f"run_config_{args.run_tag}.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2, default=str)
    print(f"\nRun config written to {config_path}")

    # --- Untrained-policy baseline evaluation ---
    print("\nRunning untrained-policy baseline evaluation on fixed eval seeds...")
    training_curve = []
    baseline_summary, _ = evaluate_policy(
        model, args.agent_type, args.eval_seeds, args.inventory_scale, args.return_scale, deterministic=True,
    )
    baseline_summary["total_timesteps"] = 0
    baseline_summary["tag"] = "baseline_untrained"
    training_curve.append(baseline_summary)
    print(f"  untrained: mean_reward={baseline_summary['mean_cumulative_reward']:.4f} "
          f"(std={baseline_summary['std_cumulative_reward']:.4f})  "
          f"mean_raw_pnl={baseline_summary['mean_raw_pnl']:.4f}  "
          f"mean_terminal|inv|={baseline_summary['mean_terminal_abs_inventory']:.3f}")

    # --- Callbacks: periodic eval (+ validation-based best-checkpoint saving,
    # + optional save-all-checkpoints) + optional --checkpoint-frequency checkpointing ---
    best_model_path = str(output_dir / f"ppo_{args.agent_type}_{args.run_tag}_best")
    checkpoint_manifest_entries = []
    eval_callback = PeriodicEvalCallback(
        agent_type=args.agent_type, eval_seeds=args.eval_seeds,
        inventory_scale=args.inventory_scale, return_scale=args.return_scale,
        eval_freq=args.eval_freq, training_curve=training_curve,
        best_model_path=best_model_path,
        checkpoints_dir=(checkpoints_dir if args.save_all_checkpoints else None),
        run_tag=args.run_tag, overwrite_checkpoints=args.overwrite_checkpoints,
        checkpoint_manifest=checkpoint_manifest_entries,
    )
    callbacks = [eval_callback]
    if args.checkpoint_frequency > 0:
        from stable_baselines3.common.callbacks import CheckpointCallback
        callbacks.append(CheckpointCallback(
            save_freq=args.checkpoint_frequency, save_path=str(output_dir),
            name_prefix=f"ppo_{args.agent_type}_{args.run_tag}_ckpt",
        ))

    print("\nTraining...")
    t_train_start = time.time()
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=False)
    train_elapsed = time.time() - t_train_start
    print(f"Training finished without crashing in {train_elapsed:.1f}s "
          f"({args.total_timesteps/train_elapsed:.0f} steps/s).")

    # --- Finite-loss / NaN-parameter checks ---
    logged = dict(model.logger.name_to_value)
    loss_entries = {k: v for k, v in logged.items() if "loss" in k or "kl" in k or "entropy" in k}
    print("\nFinal logged training metrics:")
    for k, v in sorted(loss_entries.items()):
        print(f"  {k} = {v}")
    numerical_instability = [f"{k}={v}" for k, v in loss_entries.items() if not np.isfinite(v)]
    assert not numerical_instability, f"Non-finite training metric(s) detected: {numerical_instability}"
    print("All logged losses/metrics are finite.")

    assert_no_nan_params(model)
    print("No NaNs/infs found in policy parameters.")

    # --- Save ---
    model.save(model_path)
    print(f"\nModel saved to {model_path}.zip")

    # --- save-all-checkpoints: the final-timestep candidate. Not saved by
    # the callback (see compute_checkpoint_timesteps()'s module docstring
    # for why on_rollout_start() never fires again after the last rollout's
    # train() completes) -- saved here instead, immediately after training
    # is confirmed finite/NaN-free, i.e. exactly "the policy after the PPO
    # update associated with the rollout ending at total_timesteps". If the
    # periodic mechanism already happened to save this exact timestep
    # (e.g. eval_freq evenly divides total_timesteps), just relabel that
    # entry as final rather than saving (and hashing) it twice. ---
    if args.save_all_checkpoints:
        already_saved = next((e for e in checkpoint_manifest_entries if e["timestep"] == args.total_timesteps), None)
        if already_saved is not None:
            already_saved["is_final"] = True
        else:
            final_entry = save_candidate_checkpoint(
                model, checkpoints_dir, args.agent_type, args.run_tag,
                args.total_timesteps, args.overwrite_checkpoints, is_final=True,
            )
            checkpoint_manifest_entries.append(final_entry)
        checkpoint_manifest_entries.sort(key=lambda e: e["timestep"])

        checkpoint_manifest = dict(
            agent_type=args.agent_type,
            run_tag=args.run_tag,
            learner_seed=learner_seed,
            training_env_seed=env_seed,
            total_timesteps=args.total_timesteps,
            checkpoint_frequency=args.eval_freq,
            checkpoints=checkpoint_manifest_entries,
        )
        manifest_path = log_dir / f"checkpoint_manifest_{args.run_tag}.json"
        with open(manifest_path, "w") as f:
            json.dump(checkpoint_manifest, f, indent=2, default=str)
        print(f"Checkpoint manifest ({len(checkpoint_manifest_entries)} candidates) saved to {manifest_path}")

    # --- Final (post-training) evaluation ---
    print("\nRunning final post-training evaluation on fixed eval seeds...")
    final_summary, final_records = evaluate_policy(
        model, args.agent_type, args.eval_seeds, args.inventory_scale, args.return_scale, deterministic=True,
    )
    final_summary["total_timesteps"] = args.total_timesteps
    final_summary["tag"] = "final_post_training"
    training_curve.append(final_summary)
    print(f"  final: mean_reward={final_summary['mean_cumulative_reward']:.4f} "
          f"(std={final_summary['std_cumulative_reward']:.4f})  "
          f"mean_raw_pnl={final_summary['mean_raw_pnl']:.4f}  "
          f"mean_terminal|inv|={final_summary['mean_terminal_abs_inventory']:.3f}")

    improvement = final_summary["mean_cumulative_reward"] - baseline_summary["mean_cumulative_reward"]
    print(f"\nImprovement over untrained baseline (mean_cumulative_reward): {improvement:+.4f}")

    # --- Save / reload equivalence ---
    print("\nChecking save/reload equivalence...")
    load_cls = RecurrentPPO if is_recurrent else PPO
    loaded_model = load_cls.load(model_path, env=vec_env)
    assert_no_nan_params(loaded_model)

    reload_summary, reload_records = evaluate_policy(
        loaded_model, args.agent_type, args.eval_seeds, args.inventory_scale, args.return_scale, deterministic=True,
    )

    equivalence_ok = True
    max_action_diff = 0.0
    max_reward_diff = 0.0
    for r_final, r_reload in zip(final_records, reload_records):
        assert r_final["seed"] == r_reload["seed"]
        action_diff = float(np.max(np.abs(r_final["actions"] - r_reload["actions"])))
        reward_diff = abs(r_final["cumulative_reward"] - r_reload["cumulative_reward"])
        max_action_diff = max(max_action_diff, action_diff)
        max_reward_diff = max(max_reward_diff, reward_diff)
        if action_diff > 1e-6 or reward_diff > 1e-6:
            equivalence_ok = False

    print(f"  max |action difference| across all eval episodes: {max_action_diff:.3e}")
    print(f"  max |cumulative_reward difference|:                {max_reward_diff:.3e}")
    print("  OK: save/reload equivalence confirmed." if equivalence_ok else
          "  WARNING: save/reload produced different results -- see diffs above.")

    # --- Save results ---
    training_curve_df = pd.DataFrame([{k: v for k, v in row.items()} for row in training_curve])
    training_curve_df["learner_seed"] = learner_seed
    training_curve_df["training_env_seed"] = env_seed
    training_curve_path = REPO_ROOT / "results" / f"{args.agent_type}_training_curve_{args.run_tag}.csv"
    training_curve_path.parent.mkdir(parents=True, exist_ok=True)
    training_curve_df.to_csv(training_curve_path, index=False)
    print(f"\nTraining curve (baseline + periodic + final) saved to {training_curve_path}")

    per_episode_rows = []
    for tag, records in (("final_post_training", final_records), ("reloaded_model", reload_records)):
        for r in records:
            row = {k: v for k, v in r.items() if k != "actions"}
            row["tag"] = tag
            per_episode_rows.append(row)
    per_episode_df = pd.DataFrame(per_episode_rows)
    per_episode_df["learner_seed"] = learner_seed
    per_episode_df["training_env_seed"] = env_seed
    per_episode_path = REPO_ROOT / "results" / f"{args.agent_type}_eval_episodes_{args.run_tag}.csv"
    per_episode_df.to_csv(per_episode_path, index=False)
    print(f"Per-episode evaluation results saved to {per_episode_path}")

    total_elapsed = time.time() - t_run_start
    run_summary = dict(
        agent_type=args.agent_type,
        run_tag=args.run_tag,
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        total_elapsed_seconds=total_elapsed,
        train_elapsed_seconds=train_elapsed,
        learner_seed=learner_seed,
        training_env_seed=env_seed,
        architecture=architecture,
        rollout_geometry=rollout_geometry,
        baseline_summary=baseline_summary,
        final_summary=final_summary,
        reload_summary=reload_summary,
        improvement_over_baseline_mean_reward=improvement,
        save_reload_equivalence_ok=equivalence_ok,
        max_action_diff_save_reload=max_action_diff,
        max_reward_diff_save_reload=max_reward_diff,
        numerical_instability=numerical_instability,
        final_training_metrics=loss_entries,
        best_validation_mean_objective=eval_callback.best_mean_objective,
        best_validation_std_objective=eval_callback.best_std_objective,
        best_validation_timestep=eval_callback.best_timestep,
        best_model_path=(f"{best_model_path}.zip" if eval_callback.best_mean_objective is not None else None),
        validation_seeds=args.eval_seeds,
    )
    summary_path = log_dir / f"run_summary_{args.run_tag}.json"
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=2, default=str)
    print(f"Run summary saved to {summary_path}")

    print("\n" + "=" * 78)
    print("Run complete: PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
