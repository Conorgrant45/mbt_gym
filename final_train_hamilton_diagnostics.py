"""
final_train_hamilton_diagnostics.py
--------------------------------------------------
Diagnostic-only supplementary run: trains hamilton_ppo FRESH (random init,
log_std_init=-1.5, identical PPO hyperparameters/env/seeds to Phase 7 and to
this experiment's own return_mlp_ppo/return_lstm_ppo runs) using
FinalInstrumentedPPO, purely to capture the per-PPO-update training log that
Phase 7's original Hamilton run never saved (Hamilton was reused read-only
from Phase 7 in final_run_training.py, so it has no equivalent to
training_log_long.csv).

IMPORTANT -- protocol note (this is the SECOND version of this script): an
earlier version of this run used a single uninterrupted model.learn(1_000_000)
call per seed. That produced a policy that underperformed Phase 7's official
Hamilton by ~25-30% for every seed, root-caused (and empirically confirmed
via a controlled A/B test, see conversation history) to
evaluate_checkpoint_and_log's stochastic companion pass calling
torch.manual_seed() once per validation path -- 1,050 times per run (50
validation paths x 21 checkpoints) in the OFFICIAL protocol used by
final_run_training.py for return_mlp_ppo/return_lstm_ppo (and by
phase7_run_training.py originally for hamilton_ppo). Since PPO's rollout
collection samples actions from that same global torch RNG, those resets are
*part of* what "training with seed=N" deterministically produces in this
codebase -- a single clean learn() call is a DIFFERENT (not wrong, just
different) protocol and does not reproduce it.

This version replicates the OFFICIAL interleaved protocol exactly: for each
of the 21 checkpoint timesteps, model.learn(increment) then a full 50-path
validation pass (deterministic + stochastic, via final_episode_runner's
evaluate_episode_pair) -- same call pattern as
final_run_training.process_new_architecture, just writing to diagnostics-only
files so nothing official is touched.

NOT part of the final results: checkpoints/models -> a clearly separate
directory (models/.../hamilton_diagnostics_only/), diagnostic log ->
results/.../hamilton_training_log_long.csv, validation rows ->
results/.../hamilton_diagnostics_only_validation_path_level.csv. None of
these touch checkpoint_manifest.csv, training_log_long.csv (MLP/LSTM),
validation_path_level.csv, or any final_holdout_* file. Every existing
final-results file for hamilton_ppo continues to come from Phase 7's
original reused checkpoints, exactly as before.

Run from repo root:
    python final_train_hamilton_diagnostics.py
"""
import json
import time
from pathlib import Path

import pandas as pd

import final_common as FC
import final_episode_runner as ER
from final_instrumented_ppo import FinalInstrumentedPPO
from train_agents import get_dependency_versions

DIAG_MODELS_DIR = FC.MODELS_DIR / "hamilton_diagnostics_only"
DIAG_LOG_PATH = FC.RESULTS_DIR / "hamilton_training_log_long.csv"
DIAG_VALIDATION_PATH = FC.RESULTS_DIR / "hamilton_diagnostics_only_validation_path_level.csv"
DIAG_SANITY_PATH = FC.RESULTS_DIR / "hamilton_training_diagnostics_sanity_check.csv"

STOCHASTIC_TORCH_SEED_BASE = 900_000  # identical convention to final_run_training.py

PHASE7_FINAL_OBJECTIVE = {
    0: 47.610950, 1: 48.785594, 2: 43.019647, 3: 45.999673, 4: 46.462383,
}  # from results/phase7_groupB_1m_convergence/phase7_run_status.csv, final_det_mean_objective column


def append_csv_row(path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def evaluate_checkpoint_and_log(model, learner_seed: int, timestep: int, wall_clock: float):
    det_vals = []
    for path_seed in FC.VALIDATION_SEEDS:
        row = ER.evaluate_episode_pair(model, "hamilton_ppo", path_seed,
                                        stochastic_torch_seed=STOCHASTIC_TORCH_SEED_BASE + timestep + path_seed)
        row.update(dict(architecture="hamilton_ppo", learner_seed=learner_seed, checkpoint_transition=timestep,
                         validation_path_seed=row.pop("evaluation_seed"), wall_clock_elapsed_seconds=wall_clock))
        append_csv_row(DIAG_VALIDATION_PATH, row)
        det_vals.append(row["full_objective"])
    return sum(det_vals) / len(det_vals)


def main():
    DIAG_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    sanity_rows = []

    for seed in FC.LEARNER_SEEDS:
        seed_log_marker = DIAG_MODELS_DIR / f"seed{seed}_done.json"
        if seed_log_marker.exists():
            print(f"SKIP: hamilton_ppo diagnostics seed {seed} already complete.")
            continue

        t0 = time.time()
        print(f"\n{'='*70}\nTraining hamilton_ppo (diagnostics-only, INTERLEAVED protocol), "
              f"learner_seed={seed}, target={FC.TOTAL_TRANSITIONS} transitions\n{'='*70}")

        model, _ = FC.build_model("hamilton_ppo", seed, model_cls=FinalInstrumentedPPO)

        done = 0
        last_det_mean = None
        for target_t in FC.CHECKPOINT_TIMESTEPS:
            increment = target_t - done
            model.learn(total_timesteps=increment, reset_num_timesteps=False)
            done = model.num_timesteps
            assert done == target_t, f"checkpoint timestep mismatch: expected {target_t}, got {done}"

            cp_path = FC.run_dir("hamilton_ppo", seed).parent.parent / "hamilton_diagnostics_only" / \
                f"hamilton_diag_seed{seed}_t{done:09d}"
            cp_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(cp_path))

            last_det_mean = evaluate_checkpoint_and_log(model, seed, done, time.time() - t0)
            print(f"  t={done:>8}: checkpoint+validation done [{time.time()-t0:.0f}s elapsed] "
                  f"(det_mean_objective={last_det_mean:.3f})")

        train_elapsed = time.time() - t0
        print(f"  All checkpoints done in {train_elapsed:.0f}s "
              f"({len(model.update_records)} PPO updates recorded)")

        diag_df = pd.DataFrame(model.update_records)
        diag_df.insert(0, "learner_seed", seed)
        diag_df.insert(0, "architecture", "hamilton_ppo")
        if DIAG_LOG_PATH.exists():
            existing_header = pd.read_csv(DIAG_LOG_PATH, nrows=0).columns
            diag_df = diag_df.reindex(columns=existing_header)
            diag_df.to_csv(DIAG_LOG_PATH, mode="a", header=False, index=False)
        else:
            diag_df.to_csv(DIAG_LOG_PATH, index=False)
        print(f"  Appended {len(diag_df)} update rows to {DIAG_LOG_PATH}")

        phase7_ref = PHASE7_FINAL_OBJECTIVE[seed]
        print(f"  Sanity check: this run's final det. validation mean = {last_det_mean:.3f} "
              f"(Phase 7's original run: {phase7_ref:.3f}, diff = {last_det_mean - phase7_ref:+.3f})")
        sanity_rows.append(dict(
            learner_seed=seed, this_run_final_det_mean_objective=last_det_mean,
            phase7_final_det_mean_objective=phase7_ref, difference=last_det_mean - phase7_ref,
            total_elapsed_seconds=time.time() - t0,
        ))

        seed_log_marker.write_text(json.dumps(dict(
            learner_seed=seed, final_transitions=model.num_timesteps,
            n_updates_recorded=len(model.update_records), total_elapsed_seconds=time.time() - t0,
            det_mean_objective=last_det_mean, phase7_reference=phase7_ref,
            protocol="interleaved_checkpoint_eval_matching_official",
            last_updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        ), indent=2))

        del model
        import gc
        gc.collect()
        print(f"hamilton_ppo diagnostics seed {seed} complete in {time.time()-t0:.1f}s total")

    if sanity_rows:
        existing = pd.read_csv(DIAG_SANITY_PATH) if DIAG_SANITY_PATH.exists() else pd.DataFrame()
        combined = pd.concat([existing, pd.DataFrame(sanity_rows)], ignore_index=True) if len(existing) else pd.DataFrame(sanity_rows)
        combined.to_csv(DIAG_SANITY_PATH, index=False)
        print(f"\nSaved {DIAG_SANITY_PATH}")

    print(f"\nDependency versions: {get_dependency_versions()}")
    print("All hamilton_ppo diagnostic-only runs processed (interleaved protocol).")


if __name__ == "__main__":
    main()
