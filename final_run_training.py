"""
final_run_training.py
----------------------------
Orchestrator for the final architecture-comparison experiment. For each of
the three architectures x 5 learner seeds:

  - hamilton_ppo: REUSED from Phase 7 (verified byte-for-byte compatible by
    final_common.check_hamilton_reuse_compatibility()) -- no training, no
    checkpoint files copied or modified. Every one of Phase 7's 21
    checkpoints is loaded read-only and RE-EVALUATED on this experiment's
    own fresh validation seeds with the richer episode schema Phase 7's
    original evaluation did not capture (final_episode_runner.py).

  - return_mlp_ppo / return_lstm_ppo: trained FRESH (random initialisation,
    log_std_init=-1.5, identical PPO/RecurrentPPO hyperparameters), with the
    SAME 21-point checkpoint cadence as Phase 7's Hamilton runs. The
    INITIAL untrained model is saved before any learning step. Every
    checkpoint's full model (including optimizer state) is saved, and its
    deterministic+stochastic validation performance is recorded
    immediately, using the SAME fixed validation seeds as Hamilton's
    re-evaluation.

Resumable at (architecture, learner_seed)-run granularity: a run already
verified complete (every checkpoint hash + validation row count correct)
is skipped; anything else is (re)trained from scratch -- see
final_common.py's docstring for why an exact environment-state resume is
not attempted (same documented convention as Phases 6-7). Never silently
reports a partially-completed run as complete.

Run from repo root:
    python final_run_training.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

import final_common as FC
import final_episode_runner as ER
import phase4_common as P4
from final_instrumented_ppo import FinalInstrumentedPPO, FinalInstrumentedRecurrentPPO
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from train_agents import get_dependency_versions

STOCHASTIC_TORCH_SEED_BASE = 900_000


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=FC.REPO_ROOT).decode().strip()
    except Exception as e:
        return f"UNKNOWN ({e})"


def append_csv_row(path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def append_status_row(row: dict):
    append_csv_row(FC.RESULTS_DIR / "run_status.csv", row)


def validation_path_level_path() -> "Path":
    return FC.RESULTS_DIR / "validation_path_level.csv"


def training_log_path() -> "Path":
    return FC.RESULTS_DIR / "training_log_long.csv"


def checkpoint_manifest_path() -> "Path":
    return FC.RESULTS_DIR / "checkpoint_manifest.csv"


def run_is_complete(architecture: str, learner_seed: int) -> bool:
    rm_path = FC.run_manifest_path(architecture, learner_seed)
    if not rm_path.exists():
        return False
    try:
        manifest = json.loads(rm_path.read_text())
    except json.JSONDecodeError:
        return False
    if manifest.get("final_transitions_reached") != FC.TOTAL_TRANSITIONS:
        return False
    for cand in manifest.get("checkpoints", []):
        p = Path(cand["path"])
        if not p.exists() or FC.hash_file_bytes(p) != cand["policy_hash"]:
            return False
    val = pd.read_csv(validation_path_level_path()) if validation_path_level_path().exists() else pd.DataFrame()
    if len(val):
        n_rows = len(val[(val["architecture"] == architecture) & (val["learner_seed"] == learner_seed)])
        expected = len(FC.CHECKPOINT_TIMESTEPS) * len(FC.VALIDATION_SEEDS)
        if n_rows != expected:
            return False
    else:
        return False
    return True


def evaluate_checkpoint_and_log(model, architecture: str, learner_seed: int, timestep: int, wall_clock: float):
    for path_seed in FC.VALIDATION_SEEDS:
        row = ER.evaluate_episode_pair(model, architecture, path_seed,
                                        stochastic_torch_seed=STOCHASTIC_TORCH_SEED_BASE + timestep + path_seed)
        row.update(dict(architecture=architecture, learner_seed=learner_seed, checkpoint_transition=timestep,
                         validation_path_seed=row.pop("evaluation_seed"), wall_clock_elapsed_seconds=wall_clock))
        append_csv_row(validation_path_level_path(), row)


def process_hamilton_reuse():
    compat = FC.check_hamilton_reuse_compatibility()
    print(f"Hamilton reuse compatibility check: compatible={compat['compatible']}")
    if not compat["compatible"]:
        raise RuntimeError(
            f"Phase 7 Hamilton checkpoints are NOT compatible with this experiment's configuration "
            f"({compat.get('reason')}) -- Hamilton must be trained fresh instead. Field checks: "
            f"{compat.get('field_checks')}"
        )

    architecture = "hamilton_ppo"
    for seed in FC.LEARNER_SEEDS:
        if run_is_complete(architecture, seed):
            print(f"SKIP: {architecture} seed {seed} already re-evaluated and complete.")
            continue

        t0 = time.time()
        print(f"\n{'='*70}\nRe-evaluating REUSED Hamilton PPO seed {seed} (Phase 7 checkpoints, read-only)\n{'='*70}")
        checkpoint_entries = []
        for timestep in FC.CHECKPOINT_TIMESTEPS:
            src_path = FC.hamilton_reused_checkpoint_path(seed, timestep)
            assert src_path.exists(), f"Reused Hamilton checkpoint missing: {src_path}"
            model = PPO.load(str(src_path))
            evaluate_checkpoint_and_log(model, architecture, seed, timestep, time.time() - t0)
            checkpoint_entries.append(dict(timestep=timestep, path=str(src_path),
                                            policy_hash=FC.hash_file_bytes(src_path), source="phase7_reused"))
            print(f"  t={timestep:>8}: re-evaluated on {len(FC.VALIDATION_SEEDS)} fresh validation paths "
                  f"[{time.time()-t0:.0f}s elapsed]")
            del model

        run_manifest = dict(
            architecture=architecture, learner_seed=seed, status="reused_from_phase7",
            source_experiment="phase7_groupB_1m_convergence",
            source_config_manifest=str(FC.P7.RESULTS_DIR / "phase7_config_manifest.json"),
            reuse_compatibility_check=compat,
            checkpoints=checkpoint_entries,
            final_transitions_reached=FC.TOTAL_TRANSITIONS,
            initial_model_saved=False,
            initial_model_note="Not applicable for reused runs -- Phase 7 already recorded its own "
                                "initial-model provenance via clone-verification/log_std assertions at "
                                "construction time; no separate t=0 checkpoint file exists in either "
                                "Phase 7 or this experiment for Hamilton.",
            last_updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        FC.run_manifest_path(architecture, seed).parent.mkdir(parents=True, exist_ok=True)
        FC.run_manifest_path(architecture, seed).write_text(json.dumps(run_manifest, indent=2, default=str))

        elapsed = time.time() - t0
        append_status_row(dict(
            architecture=architecture, learner_seed=seed, status="reused_from_phase7",
            start_time=time.strftime("%Y-%m-%dT%H:%M:%S"), finish_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
            duration_seconds=elapsed, final_transitions=FC.TOTAL_TRANSITIONS,
        ))
        print(f"Hamilton seed {seed} re-evaluation complete in {elapsed:.1f}s")


def process_new_architecture(architecture: str):
    assert architecture in ("return_mlp_ppo", "return_lstm_ppo")
    recurrent = FC.is_recurrent(architecture)
    model_cls = FinalInstrumentedRecurrentPPO if recurrent else FinalInstrumentedPPO

    for seed in FC.LEARNER_SEEDS:
        if run_is_complete(architecture, seed):
            print(f"SKIP: {architecture} seed {seed} already trained and complete.")
            continue

        t0 = time.time()
        run_dir = FC.run_dir(architecture, seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*70}\nTraining {architecture} (log_std_init={FC.LOG_STD_INIT}), "
              f"learner_seed={seed}, target={FC.TOTAL_TRANSITIONS} transitions\n{'='*70}")

        model, _ = FC.build_model(architecture, seed, model_cls=model_cls)

        # --- Save the INITIAL untrained model before any learning step ---
        initial_path = FC.initial_checkpoint_path(architecture, seed)
        model.save(str(initial_path))
        print(f"  Saved initial untrained model: {initial_path}.zip")

        checkpoint_entries = []
        done = 0
        for target_t in FC.CHECKPOINT_TIMESTEPS:
            increment = target_t - done
            model.learn(total_timesteps=increment, reset_num_timesteps=False)
            done = model.num_timesteps
            assert done == target_t, f"checkpoint timestep mismatch: expected {target_t}, got {done}"

            cp_path = FC.checkpoint_path(architecture, seed, done)
            model.save(str(cp_path).replace(".zip", ""))
            policy_hash = FC.hash_file_bytes(cp_path)
            checkpoint_entries.append(dict(timestep=done, path=str(cp_path), policy_hash=policy_hash,
                                            source="trained_this_experiment"))

            evaluate_checkpoint_and_log(model, architecture, seed, done, time.time() - t0)
            print(f"  t={done:>8}: checkpoint+validation done [{time.time()-t0:.0f}s elapsed]")

        # Write the FULL per-update training log for this run (every PPO
        # update across the entire 1,000,000-transition run).
        diag_df = pd.DataFrame(model.update_records)
        diag_df.insert(0, "learner_seed", seed)
        diag_df.insert(0, "architecture", architecture)
        if training_log_path().exists():
            # FinalInstrumentedPPO and FinalInstrumentedRecurrentPPO build their
            # per-update dict with DIFFERENT key orders (confirmed: this caused a
            # real column-misalignment bug for every return_lstm_ppo row, fixed
            # post-hoc by scripts/fix_training_log_lstm_column_bug.py) -- reindex
            # to the file's own header before an append so column NAMES, not
            # position, determine alignment, regardless of processing order.
            existing_header = pd.read_csv(training_log_path(), nrows=0).columns
            diag_df = diag_df.reindex(columns=existing_header)
            diag_df.to_csv(training_log_path(), mode="a", header=False, index=False)
        else:
            diag_df.to_csv(training_log_path(), index=False)

        run_manifest = dict(
            architecture=architecture, learner_seed=seed, status="trained_this_experiment",
            training_env_seed=FC.TRAIN_ENV_SEED, log_std_init=FC.LOG_STD_INIT,
            initial_model_saved=True, initial_model_path=str(initial_path) + ".zip",
            checkpoints=checkpoint_entries,
            final_transitions_reached=done,
            last_updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        FC.run_manifest_path(architecture, seed).write_text(json.dumps(run_manifest, indent=2, default=str))

        elapsed = time.time() - t0
        final_row_query = pd.read_csv(validation_path_level_path())
        final_row_query = final_row_query[
            (final_row_query["architecture"] == architecture) & (final_row_query["learner_seed"] == seed) &
            (final_row_query["checkpoint_transition"] == FC.TOTAL_TRANSITIONS)
        ]
        final_det_obj = float(final_row_query[final_row_query["deterministic"]]["full_objective"].mean())
        append_status_row(dict(
            architecture=architecture, learner_seed=seed, status="completed",
            start_time=time.strftime("%Y-%m-%dT%H:%M:%S"), finish_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
            duration_seconds=elapsed, final_transitions=done,
        ))
        print(f"{architecture} seed {seed} complete in {elapsed:.1f}s "
              f"(final mean det objective on validation set: {final_det_obj:.3f})")

        import gc
        del model
        gc.collect()


def write_experiment_manifest():
    compat = FC.check_hamilton_reuse_compatibility()
    manifest = dict(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git_commit=git_commit(),
        git_branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=FC.REPO_ROOT).decode().strip(),
        dependency_versions=get_dependency_versions(),
        architectures=list(FC.ARCHITECTURES),
        recurrent_architectures=list(FC.RECURRENT_ARCHITECTURES),
        environment_type=FC.ENVIRONMENT_TYPE,
        learner_seeds=list(FC.LEARNER_SEEDS),
        training_env_seed=FC.TRAIN_ENV_SEED,
        total_transitions=FC.TOTAL_TRANSITIONS,
        log_std_init=FC.LOG_STD_INIT,
        ppo_kwargs=FC.PPO_KWARGS,
        net_arch=FC.NET_ARCH,
        lstm_hidden_size=FC.LSTM_HIDDEN_SIZE, n_lstm_layers=FC.N_LSTM_LAYERS,
        inventory_scale=FC.INVENTORY_SCALE, return_scale=FC.RETURN_SCALE,
        checkpoint_timesteps=FC.CHECKPOINT_TIMESTEPS,
        fixed_200k_1m_timesteps=FC.FIXED_200K_1M_TIMESTEPS,
        validation_seed_start=FC.VALIDATION_SEEDS[0], validation_seed_count=len(FC.VALIDATION_SEEDS),
        holdout_seed_start=FC.HOLDOUT_SEEDS[0], holdout_seed_count=len(FC.HOLDOUT_SEEDS),
        hamilton_reuse_compatibility_check=compat,
        observation_definitions=dict(
            hamilton_ppo="[tanh(q/inventory_scale), tau, hamilton_belief] -- shape (3,)",
            return_mlp_ppo="[tanh(q/inventory_scale), tau, tanh(r/return_scale), tanh(delta_tau/elapsed_time_scale)] -- shape (4,)",
            return_lstm_ppo="identical observation to return_mlp_ppo, shape (4,), consumed by an MlpLstmPolicy "
                            "with its own recurrent state (not part of the raw observation vector)",
        ),
        action_bounds_and_transform="action in [-1,1]^2 (bid, ask); physical depth = (action+1)/2*MAX_DEPTH, "
                                     "MAX_DEPTH = -log(0.01)/kappa; unchanged from Phases 1-7.",
    )
    FC.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = FC.RESULTS_DIR / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Saved {manifest_path}")
    write_experiment_manifest_markdown(manifest)
    return manifest


def write_experiment_manifest_markdown(manifest: dict):
    compat = manifest["hamilton_reuse_compatibility_check"]
    field_checks = compat.get("field_checks", {})
    disjoint_val = FC.verify_seed_range_disjoint(FC.VALIDATION_SEEDS, "validation")
    disjoint_hold = FC.verify_seed_range_disjoint(FC.HOLDOUT_SEEDS, "holdout")
    val_hold_disjoint = FC.verify_validation_holdout_disjoint_from_each_other()

    lines = [
        f"# Experiment Manifest: {FC.EXPERIMENT_NAME}",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Git branch: `{manifest['git_branch']}`  |  Git commit: `{manifest['git_commit']}`",
        "",
        "## Architectures",
        "",
        "| Architecture | Observation | Recurrent |",
        "|---|---|---|",
    ]
    for arch in manifest["architectures"]:
        obs = manifest["observation_definitions"][arch]
        lines.append(f"| `{arch}` | {obs} | {'yes' if arch in manifest['recurrent_architectures'] else 'no'} |")

    lines += [
        "",
        "## Training configuration (identical across all three architectures)",
        "",
        f"- Environment type: `{manifest['environment_type']}`",
        f"- Training env seed: `{manifest['training_env_seed']}`",
        f"- Total transitions per run: `{manifest['total_transitions']:,}`",
        f"- Learner seeds: `{manifest['learner_seeds']}`",
        f"- log_std_init (reduced exploration, all 3 architectures): `{manifest['log_std_init']}`",
        f"- Net arch: `{manifest['net_arch']}`",
        f"- LSTM hidden size / layers (recurrent only): `{manifest['lstm_hidden_size']}` / `{manifest['n_lstm_layers']}`",
        f"- Inventory scale: `{manifest['inventory_scale']}`  |  Return scale: `{manifest['return_scale']}`",
        f"- Action bounds/transform: {manifest['action_bounds_and_transform']}",
        "",
        "### PPO / RecurrentPPO hyperparameters",
        "",
        "| Key | Value |",
        "|---|---|",
    ]
    for k, v in manifest["ppo_kwargs"].items():
        lines.append(f"| `{k}` | `{v}` |")

    lines += [
        "",
        f"### Checkpoint cadence ({len(manifest['checkpoint_timesteps'])} checkpoints)",
        "",
        f"`{manifest['checkpoint_timesteps']}`",
        "",
        f"Fixed primary-analysis checkpoints: `{manifest['fixed_200k_1m_timesteps']}` (200k for the 200k-vs-1m "
        "comparison, 1,000,000 as the PRIMARY final result -- never selected by validation performance).",
        "",
        "## Seed ranges",
        "",
        f"- Validation: `{manifest['validation_seed_start']}` .. "
        f"`{manifest['validation_seed_start'] + manifest['validation_seed_count'] - 1}` "
        f"({manifest['validation_seed_count']} paths)",
        f"- Final unseen holdout: `{manifest['holdout_seed_start']}` .. "
        f"`{manifest['holdout_seed_start'] + manifest['holdout_seed_count'] - 1}` "
        f"({manifest['holdout_seed_count']} paths)",
        "",
        "### Disjointness proof",
        "",
        f"- Validation range vs. every prior project seed range: disjoint = **{disjoint_val['disjoint']}** "
        f"(checked against {len(FC.PRIOR_SEED_RANGES)} prior ranges)",
        f"- Holdout range vs. every prior project seed range: disjoint = **{disjoint_hold['disjoint']}**",
        f"- Validation vs. holdout (this experiment, mutually): disjoint = **{val_hold_disjoint}**",
        "",
        "## Hamilton / Phase 7 reuse-compatibility check",
        "",
        f"**Compatible: {compat.get('compatible')}**"
        + ("" if compat.get("compatible") else f" -- reason: {compat.get('reason')}"),
        "",
        "| Field | Match |",
        "|---|---|",
    ]
    for k, v in field_checks.items():
        lines.append(f"| `{k}` | {v} |")
    lines += [
        "",
        "If compatible, Hamilton's 5 existing Phase 7 checkpoints (all 21 per seed) are reused READ-ONLY "
        "(never copied/modified/retrained) and RE-EVALUATED on this experiment's own fresh validation/holdout "
        "seeds with the richer per-episode schema this experiment requires. If not compatible, hamilton_ppo is "
        "trained fresh identically to the other two architectures.",
        "",
        "## Dependency versions",
        "",
        "```",
        json.dumps(manifest["dependency_versions"], indent=2),
        "```",
    ]
    (FC.RESULTS_DIR / "experiment_manifest.md").write_text("\n".join(lines))
    print(f"Saved {FC.RESULTS_DIR / 'experiment_manifest.md'}")


def main():
    if not (FC.RESULTS_DIR / "experiment_manifest.json").exists():
        write_experiment_manifest()
    else:
        print("experiment_manifest.json already exists -- not overwriting (resumed run).")
        write_experiment_manifest_markdown(json.loads((FC.RESULTS_DIR / "experiment_manifest.json").read_text()))

    t_start = time.time()
    try:
        process_hamilton_reuse()
        process_new_architecture("return_mlp_ppo")
        process_new_architecture("return_lstm_ppo")
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        raise

    total_elapsed = time.time() - t_start
    print(f"\nAll architectures/seeds processed in {total_elapsed:.1f}s")

    # --- Consolidate checkpoint_manifest.csv from every run_manifest.json ---
    rows = []
    for architecture in FC.ARCHITECTURES:
        for seed in FC.LEARNER_SEEDS:
            p = FC.run_manifest_path(architecture, seed)
            if not p.exists():
                continue
            m = json.loads(p.read_text())
            for cand in m.get("checkpoints", []):
                rows.append(dict(architecture=architecture, learner_seed=seed, timestep=cand["timestep"],
                                  path=cand["path"], policy_hash=cand["policy_hash"], source=cand.get("source")))
    pd.DataFrame(rows).to_csv(checkpoint_manifest_path(), index=False)
    print(f"Saved {checkpoint_manifest_path()} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
