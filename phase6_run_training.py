"""
phase6_run_training.py
----------------------------
Phase 6: resumable orchestrator for the 2x2 (actor initialisation x
exploration variance) optimisation experiment. 4 groups x 5 learner seeds =
20 sequential 200,000-transition Hamilton PPO training runs (event-driven
environment only), each with 13 candidate checkpoints (16k, 32k, ..., 192k,
200k), offline-selected by deterministic mean validation objective on a
fresh 50-seed validation set, copied+hash-verified to *_offline_best.zip.

Resumable at (group, learner_seed)-run granularity (see
phase6_common.run_is_complete's docstring for why: SB3 does not cleanly
support resuming optimizer state mid-training, and a single run is cheap
relative to the whole 20-run experiment). Re-running this script skips any
(group, seed) whose artefacts + hashes already verify complete, and
restarts from scratch any that do not.

Run from repo root:
    python phase6_run_training.py
"""
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import phase5_common as P5
import phase6_common as P6
from train_agents import get_dependency_versions


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=P6.REPO_ROOT).decode().strip()
    except Exception as e:
        return f"UNKNOWN ({e})"


def write_experiment_manifest():
    manifest = dict(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git_commit=git_commit(),
        dependency_versions=get_dependency_versions(),
        groups=P6.GROUPS,
        learner_seeds=list(P6.LEARNER_SEEDS),
        training_env_seed=P6.TRAIN_ENV_SEED,
        total_transitions=P6.TOTAL_TRANSITIONS,
        checkpoint_timesteps=P6.CHECKPOINT_TIMESTEPS,
        ppo_kwargs=P5.PPO_KWARGS,
        net_arch=P5.NET_ARCH,
        validation_seed_start=P6.VALIDATION_SEEDS[0], validation_seed_count=len(P6.VALIDATION_SEEDS),
        holdout_seed_start=P6.HOLDOUT_SEEDS[0], holdout_seed_count=len(P6.HOLDOUT_SEEDS),
        selection_metric=P6.SELECTION_METRIC, tie_break_rule=P6.TIE_BREAK_RULE,
        supervised_clone_path=str(P6.SUPERVISED_CLONE_PATH),
    )
    P6.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = P6.RESULTS_DIR / "phase6_experiment_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Experiment manifest written to {path}")
    return manifest


def append_status_row(row: dict):
    P6.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = P6.RESULTS_DIR / "phase6_run_status.csv"
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def run_one(group: str, learner_seed: int):
    t0 = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_dir = P6.run_dir(group, learner_seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nGroup {group} (init={P6.GROUPS[group]['init']}, "
          f"log_std_init={P6.GROUPS[group]['log_std_init']}), learner_seed={learner_seed}\n{'='*70}")

    model, vec_env, verify_report = P6.build_group_model(group, learner_seed)

    checkpoint_manifest_entries = []
    checkpoint_rows = []

    done = 0
    for target_t in P6.CHECKPOINT_TIMESTEPS:
        increment = target_t - done
        model.learn(total_timesteps=increment, reset_num_timesteps=False)
        done = model.num_timesteps
        assert done == target_t, f"checkpoint timestep mismatch: expected {target_t}, got {done}"

        cp_path = P6.checkpoint_path(group, learner_seed, done)
        model.save(str(cp_path).replace(".zip", ""))
        policy_hash = P6.hash_file_bytes(cp_path)
        checkpoint_manifest_entries.append(dict(timestep=done, path=str(cp_path), policy_hash=policy_hash))

        eval_result = P6.evaluate_checkpoint_model(model, seeds=P6.VALIDATION_SEEDS,
                                                     torch_seed=900_000 + done)
        row = dict(group=group, learner_seed=learner_seed, timestep=done, policy_hash=policy_hash)
        row.update(eval_result)
        checkpoint_rows.append(row)
        print(f"  t={done:>7}: det_obj={row['det_mean_objective']:.3f} "
              f"stoch_obj={row['stoch_mean_objective']:.3f} log_std_mean={(row['log_std_bid']+row['log_std_ask'])/2:.3f}")

    val_df = pd.DataFrame(checkpoint_rows)
    val_df.to_csv(P6.checkpoint_validation_path(group, learner_seed), index=False)

    selected = P6.select_best_checkpoint(checkpoint_rows)
    ob_path = P6.offline_best_path(group, learner_seed)
    hash_check = P6.copy_and_verify(selected_source_path(group, learner_seed, selected["timestep"]), ob_path,
                                     overwrite=True)

    run_manifest = dict(
        group=group, learner_seed=learner_seed, training_env_seed=P6.TRAIN_ENV_SEED,
        log_std_init=P6.GROUPS[group]["log_std_init"], init_type=P6.GROUPS[group]["init"],
        checkpoints=checkpoint_manifest_entries,
        clone_verification=verify_report,
    )
    P6.run_manifest_path(group, learner_seed).write_text(json.dumps(run_manifest, indent=2, default=str))

    selection_record = dict(
        group=group, learner_seed=learner_seed, selection_metric=P6.SELECTION_METRIC,
        tie_break_rule=P6.TIE_BREAK_RULE,
        validation_seed_start=P6.VALIDATION_SEEDS[0], validation_seed_count=len(P6.VALIDATION_SEEDS),
        selected_timestep=selected["timestep"], selected_det_mean_objective=selected["det_mean_objective"],
        selected_stoch_mean_objective=selected["stoch_mean_objective"],
        offline_best_path=str(ob_path),
        source_hash=hash_check["source_hash"], copied_hash=hash_check["copied_hash"],
        hash_verified=(hash_check["source_hash"] == hash_check["copied_hash"]),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    P6.offline_selection_path(group, learner_seed).write_text(json.dumps(selection_record, indent=2, default=str))

    elapsed = time.time() - t0
    finish_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    clone_diff = verify_report["max_abs_diff"] if verify_report else None
    append_status_row(dict(
        group=group, learner_seed=learner_seed, status="completed",
        start_time=start_iso, finish_time=finish_iso, duration_seconds=elapsed,
        clone_verification_max_abs_diff=clone_diff,
        selected_timestep=selected["timestep"], selected_det_mean_objective=selected["det_mean_objective"],
        offline_best_hash=hash_check["copied_hash"],
    ))
    print(f"Group {group} seed {learner_seed} complete in {elapsed:.1f}s "
          f"(selected t={selected['timestep']}, det_obj={selected['det_mean_objective']:.3f})")

    del model, vec_env
    gc.collect()


def selected_source_path(group, learner_seed, timestep):
    return P6.checkpoint_path(group, learner_seed, timestep)


def main():
    if not (P6.RESULTS_DIR / "phase6_experiment_manifest.json").exists():
        write_experiment_manifest()
    else:
        print("Experiment manifest already exists -- not overwriting (resumed run).")

    t_start = time.time()
    for group in ("A", "B", "C", "D"):
        for seed in P6.LEARNER_SEEDS:
            if P6.run_is_complete(group, seed):
                print(f"SKIP: group {group} seed {seed} already complete (verified via hashes).")
                continue
            try:
                run_one(group, seed)
            except Exception as e:
                append_status_row(dict(
                    group=group, learner_seed=seed, status="FAILED",
                    start_time=time.strftime("%Y-%m-%dT%H:%M:%S"), finish_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    duration_seconds=None, clone_verification_max_abs_diff=None,
                    selected_timestep=None, selected_det_mean_objective=None, offline_best_hash=None,
                ))
                print(f"FAILED: group {group} seed {seed}: {e}", file=sys.stderr)
                raise

    total_elapsed = time.time() - t_start
    print(f"\nAll groups/seeds processed in {total_elapsed:.1f}s")

    # Consolidate all runs' checkpoint_validation.csv into one summary file.
    all_rows = []
    for group in ("A", "B", "C", "D"):
        for seed in P6.LEARNER_SEEDS:
            p = P6.checkpoint_validation_path(group, seed)
            if p.exists():
                all_rows.append(pd.read_csv(p))
    if all_rows:
        summary_df = pd.concat(all_rows, ignore_index=True)
        summary_df.to_csv(P6.RESULTS_DIR / "phase6_checkpoint_summary.csv", index=False)
        print(f"Saved phase6_checkpoint_summary.csv ({len(summary_df)} rows)")


if __name__ == "__main__":
    main()
