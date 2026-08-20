"""
shared/phase7_common.py
--------------------
Shared utilities for Phase 7 (Group B, 1,000,000-transition convergence
diagnostic). Group B = random actor initialisation, log_std_init=-1.5,
event-driven Hamilton PPO -- EXACTLY Phase 6's Group B definition, reused
read-only from phase6_common.py/phase5_common.py, not re-derived.

The ONLY experimental changes relative to Phase 6 Group B are: (1) total
transitions 200,000 -> 1,000,000; (2) additional diagnostic logging
(phase7_instrumented_ppo.py); (3) additional checkpoint persistence/
cadence beyond 200,000. No change to the environment, reward, Hamilton
filter, observation space, action mapping, network architecture, or any
PPO hyperparameter.
"""
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

import shared.phase5_common as P5
import shared.phase6_common as P6
from shared.train_agents import compute_checkpoint_timesteps, get_dependency_versions

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "phase7_groupB_1m_convergence"
LOGS_DIR = REPO_ROOT / "logs" / "phase7_groupB_1m_convergence"
MODELS_DIR = REPO_ROOT / "models" / "phase7_groupB_1m_convergence"
PLOTS_DIR = RESULTS_DIR / "plots"
TB_DIR = LOGS_DIR / "tensorboard"

# --- Group B definition: IDENTICAL to phase6_common.GROUPS["B"] ---
GROUP_B_CONFIG = P6.GROUPS["B"]
assert GROUP_B_CONFIG == dict(init="random", log_std_init=-1.5), (
    f"phase6_common.GROUPS['B'] has changed unexpectedly: {GROUP_B_CONFIG}"
)

LEARNER_SEEDS = (0, 1, 2, 3, 4)
TRAIN_ENV_SEED = P6.TRAIN_ENV_SEED  # 70,000 -- same fixed training-environment seed as Phase 6
assert TRAIN_ENV_SEED == 70_000

TOTAL_TRANSITIONS = 1_000_000  # the ONLY transition-count change vs Phase 6 (200,000)

# PPO hyperparameters: IDENTICAL to Phase 6 (and Phase 5's train_agents.py-derived
# defaults) -- reused verbatim, never re-specified independently, so there is no
# possibility of silent drift between what Phase 6 used and what this phase uses.
PPO_KWARGS = dict(P5.PPO_KWARGS)
NET_ARCH = list(P5.NET_ARCH)
assert PPO_KWARGS["gamma"] == 1.0, "gamma must remain 1.0 -- Phase 7 changes only the transition budget"
assert PPO_KWARGS["n_steps"] == 4_000
assert PPO_KWARGS["batch_size"] == 400
assert PPO_KWARGS["n_epochs"] == 10
assert PPO_KWARGS["gae_lambda"] == 0.95
assert PPO_KWARGS["clip_range"] == 0.2
assert PPO_KWARGS["ent_coef"] == 0.0
assert PPO_KWARGS["vf_coef"] == 0.5
assert PPO_KWARGS["max_grad_norm"] == 0.5

# Checkpoint cadence: Phase 6's OWN cadence (every 16,000 transitions) preserved
# EXACTLY over the comparable 0-200,000 range (13 checkpoints, identical
# timesteps to Phase 6's own CHECKPOINT_TIMESTEPS) so the two phases are
# directly comparable at every shared checkpoint; a coarser cadence (every
# 100,000 transitions) covers the new 300,000-1,000,000 range -- preserving
# the exact Phase 6 cadence throughout would mean 62 candidate checkpoints
# per run (5x Phase 6's own count) for no diagnostic benefit over a region
# where the underlying question is "does it plateau/rise/fall", which a
# coarser grid answers just as well at a fraction of the evaluation cost
# (see PRE-RUN AUDIT / phase7_report Section 2 for the explicit cost
# accounting this trades off).
_PHASE6_STYLE_CHECKPOINTS = compute_checkpoint_timesteps(PPO_KWARGS["n_steps"], 16_000, 200_000)
assert _PHASE6_STYLE_CHECKPOINTS == P6.CHECKPOINT_TIMESTEPS, "Phase 7's 0-200k cadence must match Phase 6 exactly"
_EXTENDED_CHECKPOINTS = list(range(300_000, TOTAL_TRANSITIONS + 1, 100_000))
CHECKPOINT_TIMESTEPS = _PHASE6_STYLE_CHECKPOINTS + _EXTENDED_CHECKPOINTS

# Fresh validation seeds: disjoint from Phase 6's own validation (240000-240049)
# AND holdout (250000-250199) ranges, and from every other range enumerated in
# phase6_common.py's own docstring. Phase 7 does not touch Phase 6's holdout
# set at all (EVALUATION item 4) -- there is no separate Phase-7 holdout set;
# the fixed validation set below is used for every checkpoint's evaluation
# throughout training, matching Phase 6's own convention of one fixed
# validation set reused at every checkpoint.
VALIDATION_SEEDS = list(range(260_000, 260_050))  # 50 fresh

STOCHASTIC_TORCH_SEED_BASE = P5.STOCHASTIC_TORCH_SEED_BASE  # 900,000, reused for reproducible stochastic eval


def run_dir(learner_seed: int) -> Path:
    return MODELS_DIR / f"groupB_seed{learner_seed}"


def checkpoint_path(learner_seed: int, timestep: int) -> Path:
    return run_dir(learner_seed) / f"ppo_hamilton_ppo_groupB_seed{learner_seed}_t{timestep:09d}.zip"


def final_checkpoint_path(learner_seed: int) -> Path:
    return checkpoint_path(learner_seed, TOTAL_TRANSITIONS)


def run_manifest_path(learner_seed: int) -> Path:
    return run_dir(learner_seed) / "run_manifest.json"


def checkpoint_validation_path(learner_seed: int) -> Path:
    return run_dir(learner_seed) / "checkpoint_validation.csv"


def training_diagnostics_path(learner_seed: int) -> Path:
    return run_dir(learner_seed) / "training_diagnostics.csv"


def hash_file_bytes(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_group_b_model(learner_seed: int, model_cls=PPO):
    """Random actor initialisation, log_std_init=-1.5 -- Phase 6 Group B,
    exactly. No clone weights are loaded anywhere in this function."""
    model, vec_env = P5.build_hamilton_ppo(
        learner_seed=learner_seed, env_seed=TRAIN_ENV_SEED, model_cls=model_cls,
        log_std_init=GROUP_B_CONFIG["log_std_init"],
    )
    return model, vec_env


def evaluate_checkpoint_model(model, seeds=VALIDATION_SEEDS, torch_seed: int = None) -> dict:
    """Identical accounting to phase6_common.evaluate_checkpoint_model --
    reused, not reimplemented."""
    return P6.evaluate_checkpoint_model(model, seeds=seeds, torch_seed=torch_seed)


def run_is_complete(learner_seed: int) -> bool:
    """A run is complete only if every checkpoint file up to and including
    the final 1,000,000-transition checkpoint exists and matches its
    recorded hash, and checkpoint_validation.csv has exactly
    len(CHECKPOINT_TIMESTEPS) rows. Per instruction #9 / Section CHECKPOINTS
    item 6: an incomplete run is NEVER reported as completed -- if this
    returns False, the run is restarted FROM SCRATCH (fresh random
    initialisation), never resumed mid-training from a partial checkpoint
    (see phase7_run_training.py's module docstring for why an exact
    environment-state resume is not attempted)."""
    rm_path = run_manifest_path(learner_seed)
    val_path = checkpoint_validation_path(learner_seed)
    if not (rm_path.exists() and val_path.exists()):
        return False
    try:
        run_manifest = json.loads(rm_path.read_text())
        val_df = pd.read_csv(val_path)
    except (json.JSONDecodeError, pd.errors.EmptyDataError):
        return False
    if len(val_df) != len(CHECKPOINT_TIMESTEPS):
        return False
    if not run_manifest.get("final_transitions_reached", 0) == TOTAL_TRANSITIONS:
        return False
    for cand in run_manifest.get("checkpoints", []):
        p = Path(cand["path"])
        if not p.exists() or hash_file_bytes(p) != cand["policy_hash"]:
            return False
    fcp = final_checkpoint_path(learner_seed)
    if not fcp.exists():
        return False
    return True
