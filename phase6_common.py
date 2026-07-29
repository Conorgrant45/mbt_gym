"""
phase6_common.py
--------------------
Shared utilities for Phase 6 (2x2 optimisation experiment: actor
initialisation x exploration variance). Read-only reuse of phase5_common.py
(PPO construction, clone-weight transfer + verification, parameter-vector
helpers) and train_agents.py (compute_checkpoint_timesteps,
hash_policy_state_dict, get_dependency_versions) -- no re-derivation.

No change to the environment, reward, Hamilton filter, observation space,
action mapping, network architecture, or any PPO hyperparameter OTHER than
log_std_init (the one factor this experiment deliberately varies) anywhere
in this module.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

import phase5_common as P5
from train_agents import compute_checkpoint_timesteps, hash_policy_state_dict, get_dependency_versions
from envs.event_driven_hamilton_ppo_wrapper import EventDrivenHamiltonPPOWrapper, DEFAULT_INVENTORY_SCALE

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results" / "phase6_optimisation_experiment"
LOGS_DIR = REPO_ROOT / "logs" / "phase6_optimisation_experiment"
MODELS_DIR = REPO_ROOT / "models" / "phase6_optimisation_experiment"
PLOTS_DIR = RESULTS_DIR / "plots"

SUPERVISED_CLONE_PATH = P5.SUPERVISED_CLONE_PATH

# --- 2x2 design ---
GROUPS = {
    "A": dict(init="random", log_std_init=0.0),
    "B": dict(init="random", log_std_init=-1.5),
    "C": dict(init="clone", log_std_init=0.0),
    "D": dict(init="clone", log_std_init=-1.5),
}
LEARNER_SEEDS = (0, 1, 2, 3, 4)
TRAIN_ENV_SEED = 70_000
TOTAL_TRANSITIONS = 200_000
N_STEPS = P5.PPO_KWARGS["n_steps"]  # 4000, unchanged
EVAL_FREQ_FOR_CHECKPOINTS = 16_000
CHECKPOINT_TIMESTEPS = compute_checkpoint_timesteps(N_STEPS, EVAL_FREQ_FOR_CHECKPOINTS, TOTAL_TRANSITIONS)

# Fresh seed ranges (Section 5): checked disjoint from every previous range
# in this project -- learner seeds 0-4, training-env seed 70000, dev/monitor
# eval 90001-90005, fixed offline-selection validation 91001-91050,
# hamilton_ppo_eval_lib holdout 100000-100199, multiseed holdout
# 110000-110099, evaluate_agents_common holdout 120000-120099,
# evaluate_agents_event_driven holdout 130000-130099, Phase 3 validation
# 195001-195050, Phase 3 holdout 200000-200199, Phase 4 diagnostic seeds
# 210000-210049, Phase 4 supervised-clone train/val/test seeds
# 220000-222999, Phase 4 policy-evaluation seeds 225001-225050, Phase 5
# diagnostic validation seeds 230000-230019, Phase 5 critic/GAE diagnostic
# env seed 231000. See test_phase6_optimisation_experiment.py's disjointness
# test for the full enumerated list this is checked against.
VALIDATION_SEEDS = list(range(240_000, 240_050))    # 50 fresh
HOLDOUT_SEEDS = list(range(250_000, 250_200))        # 200 fresh

TIE_BREAK_RULE = "earlier_timestep"
SELECTION_METRIC = "det_mean_objective"
ACTOR_CLONE_VERIFY_TOL = 1e-4


def run_dir(group: str, learner_seed: int) -> Path:
    return MODELS_DIR / f"group{group}_seed{learner_seed}"


def checkpoint_path(group: str, learner_seed: int, timestep: int) -> Path:
    return run_dir(group, learner_seed) / f"ppo_hamilton_ppo_group{group}_seed{learner_seed}_t{timestep:09d}.zip"


def offline_best_path(group: str, learner_seed: int) -> Path:
    return run_dir(group, learner_seed) / f"ppo_hamilton_ppo_group{group}_seed{learner_seed}_offline_best.zip"


def run_manifest_path(group: str, learner_seed: int) -> Path:
    return run_dir(group, learner_seed) / "run_manifest.json"


def offline_selection_path(group: str, learner_seed: int) -> Path:
    return run_dir(group, learner_seed) / "offline_selection.json"


def checkpoint_validation_path(group: str, learner_seed: int) -> Path:
    return run_dir(group, learner_seed) / "checkpoint_validation.csv"


def hash_file_bytes(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_group_model(group: str, learner_seed: int, model_cls=PPO):
    """Constructs a Hamilton PPO model per this group's (init, log_std_init)
    definition. For clone-initialised groups (C, D), copies the supervised
    clone's actor weights in immediately and returns a verification report
    (None for random-init groups A/B)."""
    cfg = GROUPS[group]
    model, vec_env = P5.build_hamilton_ppo(
        learner_seed=learner_seed, env_seed=TRAIN_ENV_SEED, model_cls=model_cls,
        log_std_init=cfg["log_std_init"],
    )
    verify_report = None
    if cfg["init"] == "clone":
        clone_net = P5.load_supervised_clone_net()
        P5.copy_supervised_actor_weights(model, clone_net)
        grid_obs = P5.dense_grid_observations()
        verify_report = P5.verify_actor_matches_clone(model, clone_net, grid_obs, tol=ACTOR_CLONE_VERIFY_TOL)
        assert verify_report["passed"], (
            f"group {group} seed {learner_seed}: clone-initialised actor does not match the "
            f"supervised clone within tolerance: {verify_report}"
        )
    return model, vec_env, verify_report


def evaluate_checkpoint_model(model, seeds=VALIDATION_SEEDS, torch_seed: int = None) -> dict:
    """Deterministic + stochastic mean objective and behavioural summary on
    the given (validation or holdout) seeds, reusing
    phase5_common.run_event_episode's per-episode accounting unchanged."""
    det_records = [P5.run_event_episode(model, s, deterministic=True) for s in seeds]
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
    stoch_records = [P5.run_event_episode(model, s, deterministic=False) for s in seeds]

    log_std = model.policy.log_std.detach().numpy().copy()
    action_std = np.exp(log_std)

    def agg(records, prefix):
        obj = np.array([r["full_objective"] for r in records])
        return {
            f"{prefix}_mean_objective": float(obj.mean()),
            f"{prefix}_std_objective": float(obj.std(ddof=1)) if len(obj) > 1 else 0.0,
            f"{prefix}_se_objective": float(obj.std(ddof=1) / np.sqrt(len(obj))) if len(obj) > 1 else 0.0,
            f"{prefix}_mean_raw_pnl": float(np.mean([r["raw_pnl"] for r in records])),
            f"{prefix}_loss_rate": float(np.mean(obj < 0)),
            f"{prefix}_mean_quoted_spread": float(np.mean([r["mean_quoted_spread"] for r in records])),
            f"{prefix}_median_quoted_spread": float(np.median([r["mean_quoted_spread"] for r in records])),
            f"{prefix}_mean_fills": float(np.mean([r["fills"] for r in records])),
            f"{prefix}_mean_abs_inventory": float(np.mean([r["mean_abs_inventory"] for r in records])),
            f"{prefix}_mean_signed_inventory": float(np.mean([r["mean_signed_inventory"] for r in records])),
            f"{prefix}_terminal_abs_inventory": float(np.mean([abs(r["terminal_signed_inventory"]) for r in records])),
            f"{prefix}_terminal_signed_inventory": float(np.mean([r["terminal_signed_inventory"] for r in records])),
            f"{prefix}_running_penalty": float(np.mean([r["running_penalty"] for r in records])),
            f"{prefix}_terminal_penalty": float(np.mean([r["terminal_penalty"] for r in records])),
            f"{prefix}_frac_near_bound": float(np.mean([r["frac_near_bound"] for r in records])),
        }

    result = dict(log_std_bid=float(log_std[0]), log_std_ask=float(log_std[1]),
                  action_std_bid=float(action_std[0]), action_std_ask=float(action_std[1]),
                  n_eval_seeds=len(seeds))
    result.update(agg(det_records, "det"))
    result.update(agg(stoch_records, "stoch"))
    return result


def select_best_checkpoint(checkpoint_rows: list, metric: str = SELECTION_METRIC) -> dict:
    ranked = sorted(checkpoint_rows, key=lambda r: (-r[metric], r["timestep"]))
    return ranked[0]


def copy_and_verify(source_path, dest_path: Path, overwrite: bool = True) -> dict:
    if dest_path.exists() and not overwrite:
        raise FileExistsError(f"offline_best model already exists: {dest_path}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, dest_path)
    source_hash = hash_file_bytes(source_path)
    copied_hash = hash_file_bytes(dest_path)
    if source_hash != copied_hash:
        raise RuntimeError(f"Copied offline-best model hash mismatch: source={source_hash} copied={copied_hash}")
    return dict(source_hash=source_hash, copied_hash=copied_hash)


def run_is_complete(group: str, learner_seed: int) -> bool:
    """Resumability check (Section 9): a run is considered complete only if
    every checkpoint file exists AND matches its recorded hash in
    run_manifest.json, the checkpoint_validation.csv has exactly
    len(CHECKPOINT_TIMESTEPS) rows, and the offline_selection.json records
    hash_verified=True with a copied_hash matching the CURRENT offline_best
    file on disk. Any failure here means the run is re-run FROM SCRATCH
    (this phase resumes at (group, learner_seed)-run granularity, not
    mid-training -- SB3 does not cleanly support resuming optimizer state
    from a saved checkpoint, and a full 200,000-transition run is cheap
    enough, relative to the whole 20-run experiment, that restarting it is
    the simplest correct option)."""
    rm_path = run_manifest_path(group, learner_seed)
    sel_path = offline_selection_path(group, learner_seed)
    val_path = checkpoint_validation_path(group, learner_seed)
    if not (rm_path.exists() and sel_path.exists() and val_path.exists()):
        return False
    try:
        run_manifest = json.loads(rm_path.read_text())
        selection = json.loads(sel_path.read_text())
        val_df = pd.read_csv(val_path)
    except (json.JSONDecodeError, pd.errors.EmptyDataError):
        return False

    if len(val_df) != len(CHECKPOINT_TIMESTEPS):
        return False
    for cand in run_manifest.get("checkpoints", []):
        p = Path(cand["path"])
        if not p.exists() or hash_file_bytes(p) != cand["policy_hash"]:
            return False
    if not selection.get("hash_verified", False):
        return False
    ob_path = offline_best_path(group, learner_seed)
    if not ob_path.exists() or hash_file_bytes(ob_path) != selection.get("copied_hash"):
        return False
    return True
