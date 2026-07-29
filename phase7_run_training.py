"""
phase7_run_training.py
----------------------------
Phase 7: Group B (random init, log_std_init=-1.5) event-driven Hamilton PPO,
5 learner seeds, 1,000,000 transitions each, run SEQUENTIALLY (one process,
one seed at a time). The only experimental change relative to
phase6_run_training.py's Group B branch is the transition budget (200,000
-> 1,000,000), the extended checkpoint cadence beyond 200,000, and the
additional per-update diagnostic logging (phase7_instrumented_ppo.py).

Resumability: at the SEED-run granularity, restart-from-scratch on
interruption (see phase7_common.run_is_complete's docstring for why an
exact environment-state resume is not attempted here, matching Phase 6's
own documented precedent). Every checkpoint's evaluation and every PPO
update's diagnostics are written to disk incrementally (per learner_seed's
own checkpoint_validation.csv / training_diagnostics.csv), so an
interrupted run can always be DIAGNOSED even though it will be RESTARTED,
never silently reported as complete (phase7_run_status.csv only gains a row
for a learner_seed once phase7_common.run_is_complete confirms it in full).

Run from repo root:
    python phase7_run_training.py
"""
import json
import subprocess
import sys
import time

import pandas as pd

import phase5_common as P5
import phase7_common as P7
from phase7_instrumented_ppo import Phase7InstrumentedPPO
from train_agents import get_dependency_versions


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=P7.REPO_ROOT).decode().strip()
    except Exception as e:
        return f"UNKNOWN ({e})"


def write_config_manifest():
    manifest = dict(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git_commit=git_commit(),
        dependency_versions=get_dependency_versions(),
        group="B", group_config=P7.GROUP_B_CONFIG,
        learner_seeds=list(P7.LEARNER_SEEDS),
        training_env_seed=P7.TRAIN_ENV_SEED,
        total_transitions=P7.TOTAL_TRANSITIONS,
        phase6_total_transitions=200_000,
        checkpoint_timesteps=P7.CHECKPOINT_TIMESTEPS,
        ppo_kwargs=P7.PPO_KWARGS,
        net_arch=P7.NET_ARCH,
        validation_seed_start=P7.VALIDATION_SEEDS[0], validation_seed_count=len(P7.VALIDATION_SEEDS),
        phase6_validation_seed_range=[240_000, 240_049],
        phase6_holdout_seed_range=[250_000, 250_199],
        note_phase6_holdout_untouched="Phase 7 never reads or evaluates on Phase 6's holdout seeds 250000-250199.",
        tensorboard_available=False,
        tensorboard_note="tensorboard package is not installed in this environment; per instruction #10 "
                          "(do not install/upgrade packages unless absolutely necessary), TensorBoard logging "
                          "is skipped and all diagnostics are written to CSV instead (phase7_training_diagnostics.csv, "
                          "phase7_checkpoint_summary.csv), matching the brief's own fallback instruction.",
        resumability_convention="seed-run granularity restart-from-scratch on interruption; see "
                                 "phase7_common.run_is_complete docstring.",
    )
    P7.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = P7.RESULTS_DIR / "phase7_config_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Config manifest written to {path}")
    return manifest


def append_status_row(row: dict):
    P7.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = P7.RESULTS_DIR / "phase7_run_status.csv"
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def run_one_seed(learner_seed: int):
    t0 = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_dir = P7.run_dir(learner_seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nGroup B (random init, log_std_init={P7.GROUP_B_CONFIG['log_std_init']}), "
          f"learner_seed={learner_seed}, target={P7.TOTAL_TRANSITIONS} transitions\n{'='*70}")

    model, _ = P7.build_group_b_model(learner_seed, model_cls=Phase7InstrumentedPPO)

    checkpoint_manifest_entries = []
    checkpoint_rows = []

    done = 0
    for target_t in P7.CHECKPOINT_TIMESTEPS:
        increment = target_t - done
        model.learn(total_timesteps=increment, reset_num_timesteps=False)
        done = model.num_timesteps
        assert done == target_t, f"checkpoint timestep mismatch: expected {target_t}, got {done}"

        cp_path = P7.checkpoint_path(learner_seed, done)
        model.save(str(cp_path).replace(".zip", ""))
        policy_hash = P7.hash_file_bytes(cp_path)
        checkpoint_manifest_entries.append(dict(timestep=done, path=str(cp_path), policy_hash=policy_hash))

        eval_result = P7.evaluate_checkpoint_model(model, seeds=P7.VALIDATION_SEEDS,
                                                     torch_seed=P7.STOCHASTIC_TORCH_SEED_BASE + done)
        row = dict(learner_seed=learner_seed, timestep=done, policy_hash=policy_hash,
                   wall_clock_elapsed_seconds=time.time() - t0)
        row.update(eval_result)
        checkpoint_rows.append(row)
        print(f"  t={done:>8}: det_obj={row['det_mean_objective']:.3f} "
              f"stoch_obj={row['stoch_mean_objective']:.3f} "
              f"log_std=({row['log_std_bid']:.3f},{row['log_std_ask']:.3f}) "
              f"det_frac_near_bound={row['det_frac_near_bound']:.3f} "
              f"stoch_frac_near_bound={row['stoch_frac_near_bound']:.3f} "
              f"[{time.time()-t0:.0f}s elapsed]")

        # --- Incremental persistence (CHECKPOINTS item 5): write status
        # after EVERY checkpoint, not just at the end, so an interrupted run
        # can be diagnosed from disk alone. ---
        pd.DataFrame(checkpoint_rows).to_csv(P7.checkpoint_validation_path(learner_seed), index=False)
        pd.DataFrame(model.update_records).to_csv(P7.training_diagnostics_path(learner_seed), index=False)
        run_manifest = dict(
            group="B", learner_seed=learner_seed, training_env_seed=P7.TRAIN_ENV_SEED,
            log_std_init=P7.GROUP_B_CONFIG["log_std_init"], init_type="random",
            checkpoints=checkpoint_manifest_entries,
            final_transitions_reached=done,
            last_updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        P7.run_manifest_path(learner_seed).write_text(json.dumps(run_manifest, indent=2, default=str))

    elapsed = time.time() - t0
    finish_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    final_row = checkpoint_rows[-1]
    append_status_row(dict(
        learner_seed=learner_seed, status="completed",
        start_time=start_iso, finish_time=finish_iso, duration_seconds=elapsed,
        final_transitions=done,
        final_det_mean_objective=final_row["det_mean_objective"],
        final_stoch_mean_objective=final_row["stoch_mean_objective"],
        final_checkpoint_hash=final_row["policy_hash"],
    ))
    print(f"Seed {learner_seed} complete in {elapsed:.1f}s "
          f"(final det_obj={final_row['det_mean_objective']:.3f})")

    import gc
    del model
    gc.collect()


def main():
    if not (P7.RESULTS_DIR / "phase7_config_manifest.json").exists():
        write_config_manifest()
    else:
        print("Config manifest already exists -- not overwriting (resumed run).")

    t_start = time.time()
    for seed in P7.LEARNER_SEEDS:
        if P7.run_is_complete(seed):
            print(f"SKIP: learner_seed {seed} already complete (verified via hashes).")
            continue
        try:
            run_one_seed(seed)
        except Exception as e:
            append_status_row(dict(
                learner_seed=seed, status="FAILED",
                start_time=time.strftime("%Y-%m-%dT%H:%M:%S"), finish_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                duration_seconds=None, final_transitions=None,
                final_det_mean_objective=None, final_stoch_mean_objective=None, final_checkpoint_hash=None,
            ))
            print(f"FAILED: learner_seed {seed}: {e}", file=sys.stderr)
            raise

    total_elapsed = time.time() - t_start
    print(f"\nAll seeds processed in {total_elapsed:.1f}s")

    all_rows = []
    for seed in P7.LEARNER_SEEDS:
        p = P7.checkpoint_validation_path(seed)
        if p.exists():
            all_rows.append(pd.read_csv(p))
    if all_rows:
        summary_df = pd.concat(all_rows, ignore_index=True)
        summary_df.to_csv(P7.RESULTS_DIR / "phase7_checkpoint_summary.csv", index=False)
        print(f"Saved phase7_checkpoint_summary.csv ({len(summary_df)} rows)")

    all_diag_rows = []
    for seed in P7.LEARNER_SEEDS:
        p = P7.training_diagnostics_path(seed)
        if p.exists():
            df = pd.read_csv(p)
            df.insert(0, "learner_seed", seed)
            all_diag_rows.append(df)
    if all_diag_rows:
        diag_df = pd.concat(all_diag_rows, ignore_index=True)
        diag_df.to_csv(P7.RESULTS_DIR / "phase7_training_diagnostics.csv", index=False)
        print(f"Saved phase7_training_diagnostics.csv ({len(diag_df)} rows)")


if __name__ == "__main__":
    main()
