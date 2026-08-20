"""
ips_run_training.py
----------------------------
Orchestrator for the inventory-penalty sensitivity experiment's PPO
training batch: all THREE architectures (hamilton_ppo, return_mlp_ppo,
return_lstm_ppo) x 5 learner seeds, trained FRESH under the HIGH-PENALTY
reward calibration (phi=0.10, alpha=0.010) -- unlike final_run_training.py,
Hamilton is NOT reused from any prior run here, since the reward itself has
changed (the whole point of this experiment is to test whether the reward
change alters learned behaviour, so the policy MUST actually be retrained
under it).

Every PPO/RecurrentPPO hyperparameter, the network architecture,
log_std_init=-1.5, the training-environment seed (70,000), the learner
seeds (0-4), and the transition budget (1,000,000) are IDENTICAL to
final_common.py ("the original experiment") -- imported via ips_common.py,
never retyped. The ONLY experimental change relative to the original
experiment's training is the reward's phi/alpha, injected purely through
the environment construction (see ips_common.build_model).

Per the brief ("Evaluate the checkpoint obtained after exactly 1,000,000
transitions. Do not select or tune checkpoints using the new holdout
results."), there is no periodic-validation / offline-checkpoint-selection
procedure here (unlike final_run_training.py / Phases 6-7) -- training runs
straight through to 1,000,000 transitions in one `model.learn()` call, and
only the INITIAL (pre-training) and FINAL (1,000,000-transition) checkpoints
are saved. This is a deliberate simplification, not an oversight: the task
requires only the fixed final checkpoint be evaluated.

Resumable at (architecture, learner_seed)-run granularity, matching the
established convention from Phases 6/7/final_run_training.py: a run already
verified complete (final checkpoint exists and hash-matches its manifest
entry) is skipped; anything else is (re)trained FROM SCRATCH. Exact
environment-RNG-state mid-run resume is not attempted (same documented
reason as every prior phase -- the event-driven env's internal
np.random.Generator state is not serialized by SB3's model.save()).

Run from repo root:
    python ips_run_training.py
Monitor progress:
    Get-Content results/inventory_penalty_sensitivity/run_status.csv -Tail 5 -Wait   (PowerShell)
    tail -f results/inventory_penalty_sensitivity/run_status.csv                     (bash)
"""
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

import ips_common as IC
from final_instrumented_ppo import FinalInstrumentedPPO, FinalInstrumentedRecurrentPPO

CALIBRATION = "high_penalty"
PHI, ALPHA = IC.PHI_HIGH, IC.ALPHA_HIGH


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=IC.REPO_ROOT).decode().strip()
    except Exception as e:
        return f"UNKNOWN ({e})"


def append_csv_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def append_status_row(row: dict):
    append_csv_row(IC.RESULTS_DIR / "run_status.csv", row)


def training_log_path() -> Path:
    return IC.RESULTS_DIR / "training_log_long.csv"


def run_is_complete(architecture: str, learner_seed: int) -> bool:
    rm_path = IC.run_manifest_path(architecture, learner_seed)
    if not rm_path.exists():
        return False
    try:
        manifest = json.loads(rm_path.read_text())
    except json.JSONDecodeError:
        return False
    if manifest.get("final_transitions_reached") != IC.TOTAL_TRANSITIONS:
        return False
    if manifest.get("penalty_calibration") != CALIBRATION:
        return False
    if manifest.get("phi") != PHI or manifest.get("alpha") != ALPHA:
        return False
    for cand in manifest.get("checkpoints", []):
        p = Path(cand["path"])
        if not p.exists() or IC.hash_file_bytes(p) != cand["policy_hash"]:
            return False
    fcp = IC.final_checkpoint_path(architecture, learner_seed)
    return fcp.exists()


def process_architecture(architecture: str):
    recurrent = IC.is_recurrent(architecture)
    model_cls = FinalInstrumentedRecurrentPPO if recurrent else FinalInstrumentedPPO

    for seed in IC.LEARNER_SEEDS:
        if run_is_complete(architecture, seed):
            print(f"SKIP: {architecture} seed {seed} already trained and complete (high-penalty calibration).")
            continue

        t0 = time.time()
        run_dir = IC.run_dir(architecture, seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*70}\nTraining {architecture} [HIGH-PENALTY: phi={PHI}, alpha={ALPHA}] "
              f"(log_std_init={IC.LOG_STD_INIT}), learner_seed={seed}, "
              f"target={IC.TOTAL_TRANSITIONS} transitions\n{'='*70}")

        model, _ = IC.build_model(architecture, seed, PHI, ALPHA, model_cls=model_cls)

        initial_path = IC.initial_checkpoint_path(architecture, seed)
        model.save(str(initial_path))
        print(f"  Saved initial untrained model: {initial_path}.zip")

        model.learn(total_timesteps=IC.TOTAL_TRANSITIONS, reset_num_timesteps=False)
        done = model.num_timesteps
        assert done == IC.TOTAL_TRANSITIONS, f"transition-count mismatch: expected {IC.TOTAL_TRANSITIONS}, got {done}"

        final_path = IC.final_checkpoint_path(architecture, seed)
        model.save(str(final_path).replace(".zip", ""))
        policy_hash = IC.hash_file_bytes(final_path)
        print(f"  Saved final checkpoint @ t={done:,}: {final_path}")

        diag_df = pd.DataFrame(model.update_records)
        diag_df.insert(0, "learner_seed", seed)
        diag_df.insert(0, "architecture", architecture)
        if training_log_path().exists():
            existing_header = pd.read_csv(training_log_path(), nrows=0).columns
            diag_df = diag_df.reindex(columns=existing_header)
            diag_df.to_csv(training_log_path(), mode="a", header=False, index=False)
        else:
            diag_df.to_csv(training_log_path(), index=False)

        run_manifest = dict(
            architecture=architecture, learner_seed=seed, status="trained_this_experiment",
            penalty_calibration=CALIBRATION, phi=PHI, alpha=ALPHA,
            training_env_seed=IC.TRAIN_ENV_SEED, log_std_init=IC.LOG_STD_INIT,
            initial_model_saved=True, initial_model_path=str(initial_path) + ".zip",
            checkpoints=[dict(timestep=done, path=str(final_path), policy_hash=policy_hash,
                               source="trained_this_experiment")],
            final_transitions_reached=done,
            last_updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        IC.run_manifest_path(architecture, seed).write_text(json.dumps(run_manifest, indent=2, default=str))

        elapsed = time.time() - t0
        append_status_row(dict(
            architecture=architecture, learner_seed=seed, penalty_calibration=CALIBRATION, status="completed",
            start_time=time.strftime("%Y-%m-%dT%H:%M:%S"), finish_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
            duration_seconds=elapsed, final_transitions=done,
        ))
        print(f"{architecture} seed {seed} complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

        del model
        gc.collect()


def write_experiment_manifest():
    manifest = dict(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git_commit=git_commit(),
        git_branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=IC.REPO_ROOT).decode().strip(),
        dependency_versions=IC.get_dependency_versions(),
        experiment_name=IC.EXPERIMENT_NAME,
        penalty_calibration=CALIBRATION, phi=PHI, alpha=ALPHA,
        phi_original=IC.PHI_ORIGINAL, alpha_original=IC.ALPHA_ORIGINAL,
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
        holdout_note="IDENTICAL holdout seed range to final_common.HOLDOUT_SEEDS -- reused verbatim from "
                     "the original experiment, not re-derived.",
        checkpoint_note="Only the initial (t=0) and final (t=1,000,000) checkpoints are saved -- no periodic "
                         "validation/offline checkpoint selection is performed for this experiment (only the "
                         "fixed final checkpoint is ever evaluated on holdout).",
    )
    IC.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = IC.RESULTS_DIR / "training_config_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Saved {manifest_path}")
    return manifest


def main():
    write_experiment_manifest()

    t_start = time.time()
    try:
        for architecture in IC.ARCHITECTURES:
            process_architecture(architecture)
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        raise

    total_elapsed = time.time() - t_start
    print(f"\nAll architectures/seeds processed in {total_elapsed:.1f}s ({total_elapsed/3600:.2f} h)")

    rows = []
    for architecture in IC.ARCHITECTURES:
        for seed in IC.LEARNER_SEEDS:
            p = IC.run_manifest_path(architecture, seed)
            if not p.exists():
                continue
            m = json.loads(p.read_text())
            for cand in m.get("checkpoints", []):
                rows.append(dict(architecture=architecture, learner_seed=seed, timestep=cand["timestep"],
                                  path=cand["path"], policy_hash=cand["policy_hash"], source=cand.get("source")))
    pd.DataFrame(rows).to_csv(IC.RESULTS_DIR / "checkpoint_manifest.csv", index=False)
    print(f"Saved {IC.RESULTS_DIR / 'checkpoint_manifest.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
