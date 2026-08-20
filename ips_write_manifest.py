"""
ips_write_manifest.py
--------------------------------
Consolidates the inventory-penalty sensitivity experiment's configuration,
seeds, checkpoint paths (with hashes where files exist), clone paths, and
code version into ONE manifest file, read-only over whatever has actually
been produced on disk so far (never fabricates a path or hash for a file
that does not exist -- `status` fields make partial progress explicit).

Safe to re-run at any point during or after the experiment (idempotent,
purely additive information gathering).

Run from repo root:
    python ips_write_manifest.py
"""
import json
import subprocess
import time

import ips_common as IC

CALIBRATIONS = ("original", "high_penalty")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=IC.REPO_ROOT).decode().strip()
    except Exception as e:
        return f"UNKNOWN ({e})"


def git_branch() -> str:
    try:
        return subprocess.check_output(["git", "branch", "--show-current"], cwd=IC.REPO_ROOT).decode().strip()
    except Exception as e:
        return f"UNKNOWN ({e})"


def checkpoint_entry(architecture: str, seed: int, calibration: str) -> dict:
    path = (IC.original_checkpoint_path(architecture, seed) if calibration == "original"
            else IC.final_checkpoint_path(architecture, seed))
    exists = path.exists()
    return dict(
        architecture=architecture, learner_seed=seed, calibration=calibration, path=str(path),
        exists=exists, policy_hash=(IC.hash_file_bytes(path) if exists else None),
    )


def clone_entry(calibration: str) -> dict:
    path = IC.original_clone_path() if calibration == "original" else IC.clone_checkpoint_path()
    exists = path.exists()
    return dict(calibration=calibration, path=str(path), exists=exists,
                policy_hash=(IC.hash_file_bytes(path) if exists else None))


def main():
    IC.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoints = [checkpoint_entry(arch, seed, cal)
                   for cal in CALIBRATIONS for arch in IC.ARCHITECTURES for seed in IC.LEARNER_SEEDS]
    n_ready = {cal: sum(1 for c in checkpoints if c["calibration"] == cal and c["exists"]) for cal in CALIBRATIONS}

    clones = [clone_entry(cal) for cal in CALIBRATIONS]

    holdout_evaluated = {}
    for cal in CALIBRATIONS:
        d = IC.EVENT_LEVEL_DIR / cal
        n = len(list(d.glob("*_episode_summary.parquet"))) if d.exists() else 0
        holdout_evaluated[cal] = dict(n_policy_series_evaluated=n, expected=len(IC.ARCHITECTURES) * len(IC.LEARNER_SEEDS) + 3)

    manifest = dict(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git_commit=git_commit(), git_branch=git_branch(),
        dependency_versions=IC.get_dependency_versions(),
        experiment_name=IC.EXPERIMENT_NAME,

        penalty_calibrations=IC.CALIBRATIONS,

        architectures=list(IC.ARCHITECTURES),
        recurrent_architectures=list(IC.RECURRENT_ARCHITECTURES),
        environment_type=IC.ENVIRONMENT_TYPE,
        learner_seeds=list(IC.LEARNER_SEEDS),
        training_env_seed=IC.TRAIN_ENV_SEED,
        total_transitions=IC.TOTAL_TRANSITIONS,
        log_std_init=IC.LOG_STD_INIT,
        ppo_kwargs=IC.PPO_KWARGS,
        net_arch=IC.NET_ARCH,
        lstm_hidden_size=IC.LSTM_HIDDEN_SIZE, n_lstm_layers=IC.N_LSTM_LAYERS,
        inventory_scale=IC.INVENTORY_SCALE, return_scale=IC.RETURN_SCALE,

        holdout_seed_start=IC.HOLDOUT_SEEDS[0], holdout_seed_count=len(IC.HOLDOUT_SEEDS),
        holdout_identical_to_original_experiment=(IC.HOLDOUT_SEEDS == list(IC.FC.HOLDOUT_SEEDS)),

        clone_seed_ranges=dict(
            train=[IC.CLONE_SEED_TRAIN_START, IC.CLONE_SEED_TRAIN_START + IC.CLONE_N_TRAIN_EPISODES - 1],
            val=[IC.CLONE_SEED_VAL_START, IC.CLONE_SEED_VAL_START + IC.CLONE_N_VAL_EPISODES - 1],
            test=[IC.CLONE_SEED_TEST_START, IC.CLONE_SEED_TEST_START + IC.CLONE_N_TEST_EPISODES - 1],
        ),
        seed_disjointness_check=IC.verify_new_seeds_disjoint(),

        checkpoints=checkpoints,
        n_ppo_checkpoints_ready=n_ready,
        n_ppo_checkpoints_expected_per_calibration=len(IC.ARCHITECTURES) * len(IC.LEARNER_SEEDS),
        clones=clones,
        holdout_evaluation_progress=holdout_evaluated,
    )

    manifest_path = IC.RESULTS_DIR / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Saved {manifest_path}")
    print(f"PPO checkpoints ready: {n_ready} / {manifest['n_ppo_checkpoints_expected_per_calibration']} expected each")
    print(f"Clones ready: {[(c['calibration'], c['exists']) for c in clones]}")
    print(f"Holdout evaluation progress: {holdout_evaluated}")


if __name__ == "__main__":
    main()
