"""
ips/ips_common.py
--------------------
Shared configuration for the inventory-penalty sensitivity experiment
(supervisor-requested controlled comparison of the running inventory
penalty phi=PER_STEP_INVENTORY_AVERSION / terminal inventory penalty
alpha=TERMINAL_INVENTORY_AVERSION at 10x their original values: 0.01->0.10,
0.001->0.010, preserving their relative weighting).

Every other environment parameter, PPO hyperparameter, network architecture,
and experimental convention (architectures, learner seeds, training-env
seed, transition budget, checkpoint semantics, holdout seeds) is REUSED
VERBATIM from final_common.py ("the original experiment") -- imported, never
retyped -- so there is no possibility of silent drift between the two
calibrations on anything except phi/alpha.

Reuses, READ-ONLY, and modifies NOTHING in: final_common.py (architecture /
seed / PPO-hyperparameter / holdout-seed conventions -- the SAME 500 holdout
paths as the original experiment are reused verbatim, per the brief's
instruction that both calibrations must be scored on identical holdout
paths), phase4_common.py / simulate_belief_weighted.py (the analytical
control solver -- SBW.build_optimal_control already takes phi/alpha as
explicit keyword arguments, so it is called here unmodified with overridden
values), phase4_supervised_clone.py (clone architecture/training loop,
already parameterised by a `controls` dict -- reused unmodified), the
event-driven wrapper classes EventDrivenHamiltonPPOWrapper /
EventDrivenReturnPPOWrapper (both already accept a pre-built `base_env=`;
passing one built with custom phi/alpha requires ZERO modification to
either wrapper or to EventDrivenRegimeSwitchingEnv, which already exposes
phi/alpha as constructor kwargs).

No existing file is modified anywhere by this experiment.
"""
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

import hashlib
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO

import final.final_common as FC
import shared.phase5_common as P5
import phase7_post_training.phase7_post_training_common as P7PC
import shared.simulate_belief_weighted as SBW
from envs.event_driven_regime_env import EventDrivenRegimeSwitchingEnv, MAX_INVENTORY
from envs.event_driven_hamilton_ppo_wrapper import EventDrivenHamiltonPPOWrapper
from envs.event_driven_return_ppo_wrapper import EventDrivenReturnPPOWrapper
from envs.make_envs import PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_NAME = "inventory_penalty_sensitivity"
RESULTS_DIR = REPO_ROOT / "results" / EXPERIMENT_NAME
LOGS_DIR = REPO_ROOT / "logs" / EXPERIMENT_NAME
MODELS_DIR = REPO_ROOT / "models" / EXPERIMENT_NAME
EVENT_LEVEL_DIR = RESULTS_DIR / "event_level"

# New-output-paths-cannot-collide-with-the-original-experiment check, run at
# import time so any accidental future edit that aliases a path is caught
# immediately rather than discovered after overwriting something.
assert RESULTS_DIR != FC.RESULTS_DIR
assert MODELS_DIR != FC.MODELS_DIR
assert LOGS_DIR != FC.LOGS_DIR

# ======================================================================
# Penalty calibrations
# ======================================================================
assert PER_STEP_INVENTORY_AVERSION == 0.01, (
    f"envs.make_envs.PER_STEP_INVENTORY_AVERSION has drifted from the 0.01 this experiment assumes as "
    f"'the original calibration' (got {PER_STEP_INVENTORY_AVERSION})"
)
assert TERMINAL_INVENTORY_AVERSION == 0.001, (
    f"envs.make_envs.TERMINAL_INVENTORY_AVERSION has drifted from the 0.001 this experiment assumes as "
    f"'the original calibration' (got {TERMINAL_INVENTORY_AVERSION})"
)
PHI_ORIGINAL = PER_STEP_INVENTORY_AVERSION
ALPHA_ORIGINAL = TERMINAL_INVENTORY_AVERSION
PHI_HIGH = 0.10
ALPHA_HIGH = 0.010
assert PHI_HIGH == PHI_ORIGINAL * 10.0
assert ALPHA_HIGH == ALPHA_ORIGINAL * 10.0

CALIBRATIONS = {
    "original": dict(phi=PHI_ORIGINAL, alpha=ALPHA_ORIGINAL),
    "high_penalty": dict(phi=PHI_HIGH, alpha=ALPHA_HIGH),
}
CALIBRATION_NAMES = tuple(CALIBRATIONS.keys())

# ======================================================================
# Reused-verbatim experimental conventions -- IDENTICAL to final_common.py,
# imported not retyped.
# ======================================================================
ARCHITECTURES = FC.ARCHITECTURES
RECURRENT_ARCHITECTURES = FC.RECURRENT_ARCHITECTURES
ENVIRONMENT_TYPE = FC.ENVIRONMENT_TYPE
LEARNER_SEEDS = FC.LEARNER_SEEDS
TRAIN_ENV_SEED = FC.TRAIN_ENV_SEED
assert TRAIN_ENV_SEED == 70_000
TOTAL_TRANSITIONS = FC.TOTAL_TRANSITIONS
assert TOTAL_TRANSITIONS == 1_000_000
LOG_STD_INIT = FC.LOG_STD_INIT
INVENTORY_SCALE = FC.INVENTORY_SCALE
RETURN_SCALE = FC.RETURN_SCALE
PPO_KWARGS = dict(FC.PPO_KWARGS)
NET_ARCH = list(FC.NET_ARCH)
LSTM_HIDDEN_SIZE = FC.LSTM_HIDDEN_SIZE
N_LSTM_LAYERS = FC.N_LSTM_LAYERS

# The SAME 500 holdout paths/seeds as the original experiment (task
# instruction #7: "Evaluate every policy on the same 500 holdout paths and
# holdout seeds used in the original experiment") -- reused VERBATIM, never
# re-derived, so both calibrations are scored on identical exogenous paths.
HOLDOUT_SEEDS = list(FC.HOLDOUT_SEEDS)
assert HOLDOUT_SEEDS == list(range(290_000, 290_500))
assert len(HOLDOUT_SEEDS) == 500

# ======================================================================
# Fresh seed range needed ONLY for the higher-penalty clone's train/val/test
# data collection. NOTE (verified from phase4_supervised_clone.collect_dataset):
# the (state, action) trajectory this collects depends on which `controls`
# table chooses the actions and on the environment's price/fill/regime
# dynamics (kappa, sigma, epsilon, lambda, transition generator) -- NONE of
# which depend on phi/alpha. phi/alpha only ever enter through the reward,
# which collect_dataset never reads. A fresh, disjoint seed range is used
# anyway so this experiment never reuses a Phase-4 (q,tau,belief) sample
# under a different target policy.
# ======================================================================
PRIOR_SEED_RANGES = dict(P7PC.PRIOR_SEED_RANGES)
PRIOR_SEED_RANGES["phase7_post_training_unseen_holdout"] = set(P7PC.NEW_HOLDOUT_SEEDS)
PRIOR_SEED_RANGES["final_validation_seeds"] = set(FC.VALIDATION_SEEDS)
PRIOR_SEED_RANGES["final_holdout_seeds"] = set(FC.HOLDOUT_SEEDS)

CLONE_SEED_TRAIN_START = 300_000
CLONE_SEED_VAL_START = 301_000
CLONE_SEED_TEST_START = 302_000
CLONE_N_TRAIN_EPISODES = 100
CLONE_N_VAL_EPISODES = 30
CLONE_N_TEST_EPISODES = 40


def _seed_block(start: int, n: int) -> set:
    return set(range(start, start + n))


NEW_SEED_BLOCKS = {
    "ips_clone_train": _seed_block(CLONE_SEED_TRAIN_START, CLONE_N_TRAIN_EPISODES),
    "ips_clone_val": _seed_block(CLONE_SEED_VAL_START, CLONE_N_VAL_EPISODES),
    "ips_clone_test": _seed_block(CLONE_SEED_TEST_START, CLONE_N_TEST_EPISODES),
}


def verify_new_seeds_disjoint() -> dict:
    """Checked at test-time (and again defensively in ips_train_clone.py's
    main()): the fresh clone-data seed blocks above must not collide with
    ANY seed range used anywhere else in this project (including this
    experiment's own reused HOLDOUT_SEEDS)."""
    combined_prior = set()
    for s in PRIOR_SEED_RANGES.values():
        combined_prior |= s
    overlaps = {}
    for name, block in NEW_SEED_BLOCKS.items():
        hit = block & combined_prior
        if hit:
            overlaps[name] = sorted(hit)
    mutual = set()
    for block in NEW_SEED_BLOCKS.values():
        mutual |= block
    holdout_overlap = mutual & set(HOLDOUT_SEEDS)
    return dict(
        disjoint_from_prior=(len(overlaps) == 0), overlaps=overlaps,
        disjoint_from_holdout=(len(holdout_overlap) == 0), holdout_overlap=sorted(holdout_overlap),
    )


# ======================================================================
# Environment / wrapper construction, parameterised by phi/alpha -- the ONE
# deliberate change vs. final_common.py's own build_training_vec_env /
# build_evaluation_env, which construct the wrapper with NO base_env and so
# always get the module-default penalty. Both wrapper classes already
# accept a pre-built base_env, so no wrapper code is touched; only a custom
# EventDrivenRegimeSwitchingEnv(phi=..., alpha=...) is built here and handed
# in.
# ======================================================================
def build_base_env(seed: int, phi: float, alpha: float) -> EventDrivenRegimeSwitchingEnv:
    return EventDrivenRegimeSwitchingEnv(seed=seed, phi=phi, alpha=alpha)


def build_wrapper(architecture: str, base_env: EventDrivenRegimeSwitchingEnv):
    if architecture == "hamilton_ppo":
        return EventDrivenHamiltonPPOWrapper(base_env=base_env, inventory_scale=INVENTORY_SCALE)
    elif architecture in ("return_mlp_ppo", "return_lstm_ppo"):
        return EventDrivenReturnPPOWrapper(base_env=base_env, inventory_scale=INVENTORY_SCALE,
                                            return_scale=RETURN_SCALE)
    raise ValueError(f"Unknown architecture: {architecture!r}")


def build_train_wrapper(architecture: str, seed: int, phi: float, alpha: float):
    return build_wrapper(architecture, build_base_env(seed, phi, alpha))


def build_eval_wrapper(architecture: str, seed: int, phi: float, alpha: float):
    return build_wrapper(architecture, build_base_env(seed, phi, alpha))


def build_training_vec_env(architecture: str, phi: float, alpha: float,
                            env_seed: int = TRAIN_ENV_SEED) -> DummyVecEnv:
    def _init():
        env = build_train_wrapper(architecture, env_seed, phi, alpha)
        return Monitor(env)
    return DummyVecEnv([_init])


def is_recurrent(architecture: str) -> bool:
    return architecture in RECURRENT_ARCHITECTURES


def build_model(architecture: str, learner_seed: int, phi: float, alpha: float, model_cls=None,
                 env_seed: int = TRAIN_ENV_SEED):
    """Fresh (randomly initialised) model for `architecture` under the given
    (phi, alpha) reward calibration, using IDENTICAL hyperparameters to
    final_common.build_model (log_std_init=-1.5, same PPO_KWARGS/NET_ARCH) --
    the only difference is the reward the training environment computes."""
    vec_env = build_training_vec_env(architecture, phi, alpha, env_seed)
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
# Analytical controls, parameterised by phi/alpha. Reuses
# SBW.build_optimal_control UNMODIFIED (it already takes phi/alpha as
# explicit keyword arguments) -- only the phi/alpha entries of the
# REGIME_PARAMS dict passed in are overridden here; kappa/lam/eps (the
# regime-defining parameters, untouched by this experiment) come straight
# from SBW.REGIME_PARAMS.
# ======================================================================
def build_analytical_controls(phi: float, alpha: float) -> dict:
    controls = {}
    for regime, base_params in SBW.REGIME_PARAMS.items():
        params = dict(base_params)
        params["phi"] = phi
        params["alpha"] = alpha
        d_ask, d_bid, q_ask, q_bid = SBW.build_optimal_control(**params)
        controls[regime] = dict(delta_ask=d_ask, delta_bid=d_bid, q_ask=q_ask, q_bid=q_bid)
    return controls


# ======================================================================
# Paths (high-penalty training outputs only -- the original calibration's
# PPO checkpoints/clone are REUSED READ-ONLY from the existing experiments,
# see original_checkpoint_path()/original_clone_path() below, never copied
# or retrained).
# ======================================================================
def run_dir(architecture: str, learner_seed: int) -> Path:
    return MODELS_DIR / f"{architecture}_seed{learner_seed}"


def initial_checkpoint_path(architecture: str, learner_seed: int) -> Path:
    return run_dir(architecture, learner_seed) / f"ppo_{architecture}_seed{learner_seed}_t000000000_initial"


def final_checkpoint_path(architecture: str, learner_seed: int) -> Path:
    return run_dir(architecture, learner_seed) / f"ppo_{architecture}_seed{learner_seed}_t{TOTAL_TRANSITIONS:09d}.zip"


def run_manifest_path(architecture: str, learner_seed: int) -> Path:
    return run_dir(architecture, learner_seed) / "run_manifest.json"


def hash_file_bytes(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clone_checkpoint_path() -> Path:
    return LOGS_DIR / "supervised_clone_high_penalty_best.pt"


# ======================================================================
# Original-calibration checkpoint/clone reuse (READ-ONLY -- never copied,
# moved, or retrained). Points directly at the EXISTING
# final_reduced_exploration_architecture_comparison models/logs ("the
# original experiment" this task requires be preserved untouched) and at
# the existing Phase 4 frozen clone.
# ======================================================================
def original_checkpoint_path(architecture: str, learner_seed: int) -> Path:
    if architecture == "hamilton_ppo":
        return FC.hamilton_reused_checkpoint_path(learner_seed, FC.TOTAL_TRANSITIONS)
    return FC.checkpoint_path(architecture, learner_seed, FC.TOTAL_TRANSITIONS)


def original_clone_path() -> Path:
    return P5.SUPERVISED_CLONE_PATH


def get_dependency_versions():
    from shared.train_agents import get_dependency_versions as _gdv
    return _gdv()
