"""
phase7_post_training/phase7_post_training_common.py
-----------------------------------
Shared utilities for the Phase 7 post-training analyses (validation
learning-curve plot + untouched paired holdout evaluation). Read-only
reuse of phase6_common.py (checkpoint-selection rule) and phase7_common.py
(Group B configuration, checkpoint paths, validation seeds) -- no
retraining, no change to checkpoint selection logic, no modification of any
existing Phase 7 file.

All new outputs from these analyses are written under
results/phase7_groupB_1m_convergence/post_training_analysis/ ONLY.
"""
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

import shared.phase6_common as P6
import shared.phase7_common as P7

POST_DIR = P7.RESULTS_DIR / "post_training_analysis"

# ======================================================================
# Master list of every seed range used anywhere in this project up to and
# including Phase 7 (compiled directly from source, not from memory --
# see the module docstring of each file cited below). Used to (a) verify
# the new unseen-holdout range is disjoint from all of them, and (b) as
# the canonical reference for any future phase's own disjointness check.
# ======================================================================
PRIOR_SEED_RANGES = {
    "learner_seeds": set(range(0, 5)),
    "training_env_seed": {70_000},
    "dev_monitor_eval_seeds (train_hamilton_ppo.py/train_agents.py DEFAULT_EVAL_SEEDS)": set(range(90_001, 90_006)),
    "fixed_offline_selection_validation (select_checkpoint_offline.py example)": set(range(91_001, 91_051)),
    "hamilton_ppo_eval_lib.HOLDOUT_SEEDS": set(range(100_000, 100_200)),
    "multiseed_holdout (evaluate_hamilton_ppo_multiseed_holdout.py)": set(range(110_000, 110_100)),
    "evaluate_agents_common.DEFAULT_HOLDOUT_SEEDS": set(range(120_000, 120_100)),
    "evaluate_agents_event_driven.DEFAULT_HOLDOUT_SEEDS": set(range(130_000, 130_100)),
    "phase3_event_driven_validation": set(range(195_001, 195_051)),
    "phase3_holdout": set(range(200_000, 200_200)),
    "phase4_diagnostic_validation_seeds": set(range(210_000, 210_050)),
    "phase4_supervised_clone_train_val_test": set(range(220_000, 223_000)),
    "phase4_policy_evaluation_seeds": set(range(225_001, 225_051)),
    "phase5_diagnostic_validation_seeds": set(range(230_000, 230_020)),
    "phase5_critic_gae_diagnostic_env_seed": {231_000},
    "phase6_validation_seeds": set(P6.VALIDATION_SEEDS),
    "phase6_holdout_seeds": set(P6.HOLDOUT_SEEDS),
    "phase7_validation_seeds": set(P7.VALIDATION_SEEDS),
}

# New unseen holdout range for this analysis: 500 fresh seeds, chosen well
# clear of every range above (next available block after Phase 7's own
# validation range 260000-260049, with a gap).
NEW_HOLDOUT_SEEDS = list(range(270_000, 270_500))  # 500 seeds


def all_prior_seeds_flat() -> set:
    combined = set()
    for s in PRIOR_SEED_RANGES.values():
        combined |= s
    return combined


def verify_new_holdout_disjoint() -> dict:
    new_set = set(NEW_HOLDOUT_SEEDS)
    overlaps = {name: sorted(new_set & rng) for name, rng in PRIOR_SEED_RANGES.items() if new_set & rng}
    return dict(
        disjoint=(len(overlaps) == 0),
        n_new_holdout_seeds=len(new_set),
        overlaps=overlaps,
    )


def select_checkpoint_for_seed(seed: int, checkpoint_df: pd.DataFrame) -> dict:
    """Applies Phase 6's EXACT selection rule (argmax deterministic mean
    validation objective, ties broken toward the earlier timestep) to this
    seed's Phase 7 checkpoint rows -- Phase 7 itself never froze a
    'selected' checkpoint, so this performs that selection now, using ONLY
    the existing validation records (phase7_checkpoint_summary.csv),
    exactly as Phase 6 would have. Never touches holdout data."""
    rows = checkpoint_df[checkpoint_df["learner_seed"] == seed].to_dict("records")
    assert len(rows) == len(P7.CHECKPOINT_TIMESTEPS), (
        f"seed {seed}: expected {len(P7.CHECKPOINT_TIMESTEPS)} checkpoint rows, got {len(rows)}"
    )
    selected = P6.select_best_checkpoint(rows, metric=P6.SELECTION_METRIC)
    return selected


def load_frozen_checkpoint(seed: int, timestep: int) -> PPO:
    path = P7.checkpoint_path(seed, timestep)
    assert path.exists(), f"checkpoint file missing: {path}"
    return PPO.load(str(path))
