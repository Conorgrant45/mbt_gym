"""
final_common.py
--------------------
Shared configuration and utilities for the final long-budget experiment
comparing all three event-driven PPO architectures (Hamilton belief-state
MLP, raw-return MLP, raw-return LSTM) under the corrected reduced-exploration
setting (log_std_init=-1.5), 1,000,000 transitions, 5 identical learner
seeds each.

Reuses, READ-ONLY: train_agents.py (env/model construction conventions,
recurrent-aware evaluation pattern, RunningStats), phase5_common.py (PPO
kwargs, InstrumentedPPO), phase6_common.py (checkpoint-selection-style
helpers, hash utilities), phase7_common.py / phase7_post_training_common.py
(Hamilton Group B config, checkpoint cadence, the master list of every
seed range used in this project through Phase 7). No environment, reward,
observation, action-mapping, or PPO-hyperparameter change anywhere in this
module -- the ONLY deliberate experimental choice repeated here is
log_std_init=-1.5 for all three architectures, which Phase 7 already
established for Hamilton and this experiment extends to the other two.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO

import phase5_common as P5
import phase6_common as P6
import phase7_common as P7
import phase7_post_training_common as P7PC
from train_agents import (
    build_train_env, build_eval_env, compute_checkpoint_timesteps, get_dependency_versions,
    FEEDFORWARD_DEFAULT_N_STEPS, FEEDFORWARD_DEFAULT_BATCH_SIZE,
    RECURRENT_DEFAULT_N_STEPS, RECURRENT_DEFAULT_BATCH_SIZE,
    DEFAULT_LSTM_HIDDEN_SIZE, DEFAULT_N_LSTM_LAYERS,
)
from envs.hamilton_ppo_wrapper import DEFAULT_INVENTORY_SCALE
from envs.return_ppo_wrapper import DEFAULT_RETURN_SCALE

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME = "final_reduced_exploration_architecture_comparison"
RESULTS_DIR = REPO_ROOT / "results" / EXPERIMENT_NAME
LOGS_DIR = REPO_ROOT / "logs" / EXPERIMENT_NAME
MODELS_DIR = REPO_ROOT / "models" / EXPERIMENT_NAME
PLOTS_DIR = RESULTS_DIR / "plots"

# ======================================================================
# Architectures
# ======================================================================
ARCHITECTURES = ("hamilton_ppo", "return_mlp_ppo", "return_lstm_ppo")
RECURRENT_ARCHITECTURES = ("return_lstm_ppo",)
ENVIRONMENT_TYPE = "event"  # event-driven only, per this experiment's brief

LEARNER_SEEDS = (0, 1, 2, 3, 4)
TRAIN_ENV_SEED = P7.TRAIN_ENV_SEED  # 70,000 -- identical to Phases 6-7
assert TRAIN_ENV_SEED == 70_000

TOTAL_TRANSITIONS = 1_000_000
LOG_STD_INIT = -1.5  # the corrected reduced-exploration setting, ALL three architectures

INVENTORY_SCALE = DEFAULT_INVENTORY_SCALE
RETURN_SCALE = DEFAULT_RETURN_SCALE

# PPO/RecurrentPPO hyperparameters: reused VERBATIM from phase5_common.py
# (itself reproduced verbatim from train_agents.py's own defaults) -- never
# re-specified independently, so there is no possibility of silent drift.
PPO_KWARGS = dict(P5.PPO_KWARGS)
NET_ARCH = list(P5.NET_ARCH)
LSTM_HIDDEN_SIZE = DEFAULT_LSTM_HIDDEN_SIZE
N_LSTM_LAYERS = DEFAULT_N_LSTM_LAYERS
assert PPO_KWARGS["n_steps"] == FEEDFORWARD_DEFAULT_N_STEPS == RECURRENT_DEFAULT_N_STEPS == 4_000
assert PPO_KWARGS["batch_size"] == FEEDFORWARD_DEFAULT_BATCH_SIZE == RECURRENT_DEFAULT_BATCH_SIZE == 400
assert PPO_KWARGS["gamma"] == 1.0

# Checkpoint cadence: IDENTICAL to Phase 7's own Hamilton cadence (13
# checkpoints every 16,000 transitions to 200,000, then 8 more every
# 100,000 to 1,000,000 -- 21 total), used UNCHANGED for all three
# architectures in this experiment (task requirement: "same checkpoint
# cadence for every architecture" + "match the existing Phase 7 Hamilton
# checkpoint cadence wherever possible").
CHECKPOINT_TIMESTEPS = list(P7.CHECKPOINT_TIMESTEPS)
assert CHECKPOINT_TIMESTEPS[-1] == TOTAL_TRANSITIONS == 1_000_000
assert 200_000 in CHECKPOINT_TIMESTEPS
FIXED_200K_1M_TIMESTEPS = [200_000, 1_000_000]

# ======================================================================
# Seed ranges: fresh, disjoint from EVERY previously-used range in this
# project through Phase 7 (learner seeds, training-env seed, every
# validation/holdout/diagnostic range documented in
# phase7_post_training_common.PRIOR_SEED_RANGES, PLUS Phase 7's own new
# unseen-holdout range 270000-270499 added by the post-training analysis).
# ======================================================================
PRIOR_SEED_RANGES = dict(P7PC.PRIOR_SEED_RANGES)
PRIOR_SEED_RANGES["phase7_post_training_unseen_holdout"] = set(P7PC.NEW_HOLDOUT_SEEDS)

VALIDATION_SEEDS = list(range(280_000, 280_050))     # 50 fresh (next block after 270000-270499)
HOLDOUT_SEEDS = list(range(290_000, 290_500))        # 500 fresh


def all_prior_seeds_flat() -> set:
    combined = set()
    for s in PRIOR_SEED_RANGES.values():
        combined |= s
    return combined


def verify_seed_range_disjoint(seed_range: list, label: str) -> dict:
    new_set = set(seed_range)
    overlaps = {name: sorted(new_set & rng) for name, rng in PRIOR_SEED_RANGES.items() if new_set & rng}
    return dict(label=label, disjoint=(len(overlaps) == 0), n_seeds=len(new_set), overlaps=overlaps)


def verify_validation_holdout_disjoint_from_each_other() -> bool:
    return set(VALIDATION_SEEDS).isdisjoint(set(HOLDOUT_SEEDS))


# ======================================================================
# Environment / model construction (single source, generalised across all
# three architectures -- reuses train_agents.py's build_train_env/
# build_eval_env unmodified, event-driven only)
# ======================================================================
def build_training_vec_env(architecture: str, env_seed: int = TRAIN_ENV_SEED) -> DummyVecEnv:
    def _init():
        env = build_train_env(architecture, env_seed, INVENTORY_SCALE, RETURN_SCALE,
                               environment_type=ENVIRONMENT_TYPE)
        return Monitor(env)
    return DummyVecEnv([_init])


def build_evaluation_env(architecture: str, seed: int):
    return build_eval_env(architecture, seed, INVENTORY_SCALE, RETURN_SCALE, environment_type=ENVIRONMENT_TYPE)


def is_recurrent(architecture: str) -> bool:
    return architecture in RECURRENT_ARCHITECTURES


def build_model(architecture: str, learner_seed: int, model_cls=None, env_seed: int = TRAIN_ENV_SEED):
    """Builds a fresh (randomly initialised) model for `architecture` with
    log_std_init=-1.5, using IDENTICAL hyperparameters to train_agents.py's
    common_kwargs/policy_kwargs. `model_cls` defaults to PPO/RecurrentPPO;
    pass an instrumented subclass (final_instrumented_ppo.py) to get
    per-update diagnostics with no change to the underlying optimisation."""
    vec_env = build_training_vec_env(architecture, env_seed)
    recurrent = is_recurrent(architecture)
    default_cls = RecurrentPPO if recurrent else PPO
    cls = model_cls if model_cls is not None else default_cls

    common_kwargs = dict(PPO_KWARGS)
    common_kwargs["seed"] = learner_seed
    common_kwargs["verbose"] = 0

    if recurrent:
        policy_kwargs = dict(net_arch=list(NET_ARCH), lstm_hidden_size=LSTM_HIDDEN_SIZE,
                              n_lstm_layers=N_LSTM_LAYERS, log_std_init=LOG_STD_INIT)
        model = cls("MlpLstmPolicy", vec_env, policy_kwargs=policy_kwargs, **common_kwargs)
    else:
        policy_kwargs = dict(net_arch=list(NET_ARCH), log_std_init=LOG_STD_INIT)
        model = cls("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, **common_kwargs)
    return model, vec_env


# ======================================================================
# Paths
# ======================================================================
def run_dir(architecture: str, learner_seed: int) -> Path:
    return MODELS_DIR / f"{architecture}_seed{learner_seed}"


def initial_checkpoint_path(architecture: str, learner_seed: int) -> Path:
    return run_dir(architecture, learner_seed) / f"ppo_{architecture}_seed{learner_seed}_t000000000_initial"


def checkpoint_path(architecture: str, learner_seed: int, timestep: int) -> Path:
    return run_dir(architecture, learner_seed) / f"ppo_{architecture}_seed{learner_seed}_t{timestep:09d}.zip"


def final_checkpoint_path(architecture: str, learner_seed: int) -> Path:
    return checkpoint_path(architecture, learner_seed, TOTAL_TRANSITIONS)


def run_manifest_path(architecture: str, learner_seed: int) -> Path:
    return run_dir(architecture, learner_seed) / "run_manifest.json"


def hash_file_bytes(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ======================================================================
# Hamilton / Phase 7 reuse-compatibility check
# ======================================================================
def check_hamilton_reuse_compatibility() -> dict:
    """Compares Phase 7's config manifest field-by-field against this
    experiment's own Hamilton configuration. Returns a dict recording each
    field's match/mismatch and an overall `compatible` flag. This is a
    programmatic check, not an assumption -- if Phase 7's manifest or
    checkpoints are missing, or ANY field differs, `compatible` is False
    and Hamilton must be trained fresh in this experiment instead of reused."""
    manifest_path = P7.RESULTS_DIR / "phase7_config_manifest.json"
    if not manifest_path.exists():
        return dict(compatible=False, reason=f"Phase 7 config manifest not found: {manifest_path}")
    phase7_manifest = json.loads(manifest_path.read_text())

    checks = {}
    checks["environment_type_event"] = True  # Phase 7 is event-driven-only by construction
    checks["training_env_seed"] = (phase7_manifest["training_env_seed"] == TRAIN_ENV_SEED)
    checks["total_transitions"] = (phase7_manifest["total_transitions"] == TOTAL_TRANSITIONS)
    checks["learner_seeds"] = (list(phase7_manifest["learner_seeds"]) == list(LEARNER_SEEDS))
    checks["log_std_init"] = (phase7_manifest["group_config"]["log_std_init"] == LOG_STD_INIT)
    checks["init_type_random"] = (phase7_manifest["group_config"]["init"] == "random")
    checks["checkpoint_timesteps"] = (list(phase7_manifest["checkpoint_timesteps"]) == list(CHECKPOINT_TIMESTEPS))
    checks["net_arch"] = (list(phase7_manifest["net_arch"]) == list(NET_ARCH))

    p7_ppo = phase7_manifest["ppo_kwargs"]
    for key in ("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
                "clip_range", "ent_coef", "vf_coef", "max_grad_norm"):
        checks[f"ppo_kwargs.{key}"] = (p7_ppo.get(key) == PPO_KWARGS.get(key))

    # Checkpoint files must actually exist on disk and match their recorded hash.
    checkpoints_ok = True
    checkpoint_details = {}
    for seed in LEARNER_SEEDS:
        rm = json.loads(P7.run_manifest_path(seed).read_text()) if P7.run_manifest_path(seed).exists() else None
        if rm is None:
            checkpoints_ok = False
            checkpoint_details[seed] = "run_manifest.json missing"
            continue
        seed_ok = True
        for cand in rm.get("checkpoints", []):
            p = Path(cand["path"])
            if not p.exists() or hash_file_bytes(p) != cand["policy_hash"]:
                seed_ok = False
        checkpoints_ok = checkpoints_ok and seed_ok
        checkpoint_details[seed] = "all checkpoint hashes verified" if seed_ok else "hash mismatch or missing file"
    checks["all_checkpoint_files_present_and_hash_verified"] = checkpoints_ok

    compatible = all(checks.values())
    return dict(
        compatible=compatible, field_checks=checks, checkpoint_details=checkpoint_details,
        phase7_manifest_path=str(manifest_path),
        reason=None if compatible else "one or more configuration fields or checkpoint hashes did not match",
    )


def get_actor_params(model) -> list:
    """Actor-side parameters, generalised across feed-forward (PPO) and
    recurrent (RecurrentPPO) policies: mlp_extractor.policy_net + action_net
    + log_std, PLUS lstm_actor when present (RecurrentActorCriticPolicy with
    shared_lstm=False, the sb3-contrib default, keeps separate lstm_actor/
    lstm_critic modules -- see sb3_contrib.common.recurrent.policies)."""
    params = list(model.policy.mlp_extractor.policy_net.parameters())
    params += list(model.policy.action_net.parameters())
    params += [model.policy.log_std]
    if hasattr(model.policy, "lstm_actor"):
        params += list(model.policy.lstm_actor.parameters())
    return params


def get_critic_params(model) -> list:
    params = list(model.policy.mlp_extractor.value_net.parameters())
    params += list(model.policy.value_net.parameters())
    if hasattr(model.policy, "lstm_critic") and model.policy.lstm_critic is not None:
        params += list(model.policy.lstm_critic.parameters())
    return params


def flat_param_vector(params: list) -> np.ndarray:
    with torch.no_grad():
        return torch.cat([p.detach().flatten() for p in params]).cpu().numpy()


def hamilton_reused_checkpoint_path(learner_seed: int, timestep: int) -> Path:
    """Path to the ORIGINAL Phase 7 checkpoint file (read-only; never
    copied or moved -- this experiment's own models/ directory contains no
    Hamilton files at all when reuse applies, only a pointer recorded in
    this experiment's run_manifest.json)."""
    return P7.checkpoint_path(learner_seed, timestep)
