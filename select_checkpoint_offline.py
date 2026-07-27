"""
select_checkpoint_offline.py
------------------------------
Offline checkpoint selection: decouples training from checkpoint
selection. train_agents.py --save-all-checkpoints saves every candidate
checkpoint at the periodic-evaluation cadence (plus one at
total_timesteps) and writes a checkpoint manifest describing them. This
script reads that manifest, evaluates EVERY candidate deterministically on
a larger, fixed VALIDATION seed set, and selects the checkpoint with the
highest mean cumulative objective -- entirely independent of, and never
influenced by, the small monitoring set (--eval-seeds) used during
training itself.

Reuses, READ-ONLY, from train_agents.py: evaluate_policy/run_eval_episode
(the SAME recurrent-aware, per-episode evaluation harness already used by
PeriodicEvalCallback's periodic evaluation and online best-checkpoint
selection -- feed-forward and recurrent policies, LSTM state reset at
episode boundaries, are handled identically here because this script never
reimplements that logic), hash_policy_state_dict, RECURRENT_AGENT_TYPES,
DEFAULT_INVENTORY_SCALE, get_dependency_versions.

Deliberately does NOT import evaluate_agents_common.py's holdout-seed range
(DEFAULT_HOLDOUT_SEEDS) or accept any holdout/test-seed CLI argument -- the
only notion of "seed" here is the validation range given via
--validation-seeds-start/--validation-seeds-count. The project's eventual
holdout range must never appear in this script.

Run from repo root:
    python select_checkpoint_offline.py --agent-type hamilton_ppo --run-tag v1 \\
        --learner-seed 0 --training-env-seed 70000 \\
        --validation-seeds-start 91001 --validation-seeds-count 50
"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from train_agents import (
    AGENT_TYPES, RECURRENT_AGENT_TYPES, DEFAULT_INVENTORY_SCALE,
    evaluate_policy, hash_policy_state_dict, get_dependency_versions,
)
from envs.return_ppo_wrapper import DEFAULT_RETURN_SCALE

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models"
LOGS_DIR = REPO_ROOT / "logs"

# Only mean_objective is supported -- no risk-adjusted (e.g. Sharpe-like)
# criterion in this implementation, per the feature specification.
SUPPORTED_SELECTION_METRICS = ("mean_objective",)
TIE_BREAK_RULE = "earlier_timestep"


def hash_file_bytes(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(agent_type: str, run_tag: str) -> dict:
    manifest_path = LOGS_DIR / agent_type / f"checkpoint_manifest_{run_tag}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No checkpoint manifest found at {manifest_path} -- run train_agents.py with "
            f"--save-all-checkpoints for agent-type={agent_type!r}, run-tag={run_tag!r} first."
        )
    with open(manifest_path) as f:
        return json.load(f)


def verify_candidate_files_exist(manifest: dict):
    """Item 2: fail clearly, before any evaluation runs, if the manifest
    references a checkpoint file that no longer exists on disk -- rather
    than silently skipping it (same fail-fast convention as
    evaluate_agents_common.py's check_all_models_exist)."""
    missing = [c["path"] for c in manifest["checkpoints"] if not Path(c["path"]).exists()]
    if missing:
        raise FileNotFoundError(
            f"checkpoint_manifest lists {len(missing)} candidate file(s) that do not exist on disk:\n  " +
            "\n  ".join(missing) +
            f"\n(The manifest may be stale, or files were moved/deleted -- re-run training or fix paths.)"
        )


def evaluate_all_candidates(manifest: dict, validation_seeds: list) -> tuple:
    """Evaluate every candidate checkpoint listed in the manifest,
    deterministically, on exactly the same validation_seeds. Returns
    (checkpoint_rows, episode_rows)."""
    agent_type = manifest["agent_type"]
    is_recurrent = agent_type in RECURRENT_AGENT_TYPES
    model_cls = RecurrentPPO if is_recurrent else PPO

    checkpoint_rows = []
    episode_rows = []
    for cand in sorted(manifest["checkpoints"], key=lambda c: c["timestep"]):
        model = model_cls.load(cand["path"])
        summary, records = evaluate_policy(
            model, agent_type, validation_seeds, DEFAULT_INVENTORY_SCALE, DEFAULT_RETURN_SCALE,
            deterministic=True,
        )
        rewards = np.array([r["cumulative_reward"] for r in records])
        loss_rate = float((rewards < 0).mean())
        policy_hash = hash_policy_state_dict(model)

        checkpoint_rows.append(dict(
            agent_type=agent_type, run_tag=manifest["run_tag"],
            learner_seed=manifest["learner_seed"], training_env_seed=manifest["training_env_seed"],
            checkpoint_timestep=cand["timestep"], checkpoint_path=cand["path"],
            n_validation_episodes=summary["n_episodes"],
            validation_seed_start=validation_seeds[0], validation_seed_count=len(validation_seeds),
            mean_cumulative_reward=summary["mean_cumulative_reward"],
            std_cumulative_reward=summary["std_cumulative_reward"],
            mean_raw_pnl=summary["mean_raw_pnl"],
            mean_terminal_abs_inventory=summary["mean_terminal_abs_inventory"],
            action_mean_bid=summary["action_mean_bid"], action_mean_ask=summary["action_mean_ask"],
            action_std_bid=summary["action_std_bid"], action_std_ask=summary["action_std_ask"],
            loss_rate=loss_rate,
            selected=False,  # filled in after selection, below
            policy_hash=policy_hash,
        ))
        for r in records:
            episode_rows.append(dict(
                agent_type=agent_type, run_tag=manifest["run_tag"],
                checkpoint_timestep=cand["timestep"], checkpoint_path=cand["path"],
                evaluation_seed=r["seed"], cumulative_reward=r["cumulative_reward"],
                raw_pnl=r["raw_pnl"], terminal_abs_inventory=r["terminal_abs_inventory"],
                steps=r["steps"],
            ))
    return checkpoint_rows, episode_rows


def select_best(checkpoint_rows: list) -> dict:
    """Highest mean_cumulative_reward; exact ties resolved in favour of the
    EARLIER timestep (item 9/11)."""
    ranked = sorted(checkpoint_rows, key=lambda r: (-r["mean_cumulative_reward"], r["checkpoint_timestep"]))
    return ranked[0]


def copy_and_verify(source_path, dest_path: Path, overwrite: bool) -> dict:
    """Item 10/11/12: copy the selected checkpoint to the offline_best path
    and verify the copy is byte-identical to its source via independent
    SHA-256 hashes of both files' contents."""
    if dest_path.exists() and not overwrite:
        raise FileExistsError(
            f"offline_best model already exists: {dest_path}\nPass --overwrite-selection to overwrite."
        )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, dest_path)
    source_hash = hash_file_bytes(source_path)
    copied_hash = hash_file_bytes(dest_path)
    if source_hash != copied_hash:
        raise RuntimeError(
            f"Copied offline-best model hash does not match its source checkpoint "
            f"(source={source_hash}, copied={copied_hash}) -- copy is corrupt."
        )
    return dict(source_hash=source_hash, copied_hash=copied_hash)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent-type", type=str, required=True, choices=AGENT_TYPES)
    p.add_argument("--run-tag", type=str, required=True)
    p.add_argument("--learner-seed", type=int, required=True,
                    help="Must match the checkpoint manifest's own learner_seed -- cross-checked, not used "
                         "to reconstruct anything (candidates are already-trained model files).")
    p.add_argument("--training-env-seed", type=int, required=True,
                    help="Must match the checkpoint manifest's own training_env_seed -- cross-checked.")
    p.add_argument("--validation-seeds-start", type=int, required=True)
    p.add_argument("--validation-seeds-count", type=int, required=True)
    p.add_argument("--selection-metric", type=str, default="mean_objective",
                    choices=list(SUPPORTED_SELECTION_METRICS))
    p.add_argument("--overwrite-selection", action="store_true", default=False,
                    help="Allow overwriting a pre-existing offline_best model / offline_selection JSON for "
                         "this agent-type/run-tag. Default false: existing output(s) raise before evaluating.")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    manifest = load_manifest(args.agent_type, args.run_tag)
    if manifest["learner_seed"] != args.learner_seed:
        raise ValueError(
            f"--learner-seed={args.learner_seed} does not match the checkpoint manifest's own "
            f"learner_seed={manifest['learner_seed']} for {args.agent_type}/{args.run_tag}."
        )
    if manifest["training_env_seed"] != args.training_env_seed:
        raise ValueError(
            f"--training-env-seed={args.training_env_seed} does not match the checkpoint manifest's own "
            f"training_env_seed={manifest['training_env_seed']} for {args.agent_type}/{args.run_tag}."
        )
    verify_candidate_files_exist(manifest)

    offline_best_path = MODELS_DIR / args.agent_type / f"ppo_{args.agent_type}_{args.run_tag}_offline_best.zip"
    selection_json_path = LOGS_DIR / args.agent_type / f"offline_selection_{args.run_tag}.json"
    if not args.overwrite_selection:
        existing = [p for p in (offline_best_path, selection_json_path) if p.exists()]
        if existing:
            raise FileExistsError(
                "Offline-selection output(s) already exist:\n  " + "\n  ".join(str(p) for p in existing) +
                "\nPass --overwrite-selection to overwrite."
            )

    # NOT holdout/test seeds -- see module docstring.
    validation_seeds = list(range(args.validation_seeds_start,
                                   args.validation_seeds_start + args.validation_seeds_count))

    print(f"Evaluating {len(manifest['checkpoints'])} candidate checkpoints for "
          f"{args.agent_type}/{args.run_tag} on {len(validation_seeds)} validation seeds "
          f"({validation_seeds[0]}..{validation_seeds[-1]})...")
    checkpoint_rows, episode_rows = evaluate_all_candidates(manifest, validation_seeds)

    selected = select_best(checkpoint_rows)
    for row in checkpoint_rows:
        row["selected"] = (row["checkpoint_timestep"] == selected["checkpoint_timestep"])

    print(f"Selected checkpoint: timestep={selected['checkpoint_timestep']} "
          f"mean_cumulative_reward={selected['mean_cumulative_reward']:.4f} "
          f"(std={selected['std_cumulative_reward']:.4f})")

    hash_check = copy_and_verify(selected["checkpoint_path"], offline_best_path, args.overwrite_selection)
    print(f"Copied selected checkpoint to {offline_best_path} "
          f"(hash verified: {hash_check['source_hash'] == hash_check['copied_hash']})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_csv_path = RESULTS_DIR / f"{args.agent_type}_checkpoint_validation_{args.run_tag}.csv"
    pd.DataFrame(checkpoint_rows).to_csv(checkpoint_csv_path, index=False)
    print(f"Per-checkpoint validation results saved to {checkpoint_csv_path}")

    episode_csv_path = RESULTS_DIR / f"{args.agent_type}_checkpoint_validation_episodes_{args.run_tag}.csv"
    pd.DataFrame(episode_rows).to_csv(episode_csv_path, index=False)
    print(f"Per-checkpoint per-episode validation results saved to {episode_csv_path}")

    selection_json_path.parent.mkdir(parents=True, exist_ok=True)
    selection_record = dict(
        agent_type=args.agent_type, run_tag=args.run_tag,
        learner_seed=args.learner_seed, training_env_seed=args.training_env_seed,
        selection_metric=args.selection_metric, tie_break_rule=TIE_BREAK_RULE,
        validation_seed_start=args.validation_seeds_start, validation_seed_count=args.validation_seeds_count,
        validation_seeds=validation_seeds,
        candidate_timesteps=[c["timestep"] for c in sorted(manifest["checkpoints"], key=lambda c: c["timestep"])],
        selected_timestep=selected["checkpoint_timestep"],
        selected_source_path=selected["checkpoint_path"],
        offline_best_path=str(offline_best_path),
        selected_mean_cumulative_reward=selected["mean_cumulative_reward"],
        selected_std_cumulative_reward=selected["std_cumulative_reward"],
        source_policy_hash=hash_check["source_hash"],
        copied_policy_hash=hash_check["copied_hash"],
        hash_verified=(hash_check["source_hash"] == hash_check["copied_hash"]),
        dependency_versions=get_dependency_versions(),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        elapsed_seconds=time.time() - t0,
    )
    with open(selection_json_path, "w") as f:
        json.dump(selection_record, f, indent=2, default=str)
    print(f"Offline-selection record saved to {selection_json_path}")

    print("\nOffline checkpoint selection complete: PASSED")


if __name__ == "__main__":
    main()
