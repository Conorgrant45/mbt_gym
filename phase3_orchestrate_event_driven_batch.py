"""
phase3_orchestrate_event_driven_batch.py
--------------------------------------------
Resumable orchestration for Phase 3's 15 event-driven training runs
(3 architectures x 5 learner seeds), each consisting of:

    1. train_agents.py --environment-type event --total-timesteps 200000
       --save-all-checkpoints (candidate checkpoints every --eval-freq
       transitions, final checkpoint saved explicitly by train_agents.py
       itself);
    2. select_checkpoint_offline.py (50 validation episodes per candidate)
       -> ppo_<agent>_<tag>_offline_best.zip, hash-verified.

Run sequentially (this machine has 8GB RAM -- see prior sessions'
MemoryError history). Safe to interrupt (Ctrl-C, crash, session end) and
re-run: before starting or redoing any run, checks whether that run's
expected artefacts already exist on disk AND hash-verify correctly against
their own recorded metadata; only (re)does missing/incomplete/inconsistent
ones. Fails immediately (does not attempt subsequent runs) on any non-zero
subprocess exit code, so a genuine bug never gets silently skipped over.

Does not modify train_agents.py's or select_checkpoint_offline.py's own
model/log path conventions -- event-driven artefacts land exactly where
those scripts already put them (models/<agent_type>_event/,
logs/<agent_type>_event/); this script only adds a dedicated
logs/phase3_event_vs_fixed/ + results/phase3_event_vs_fixed/ namespace for
EXPERIMENT-LEVEL bookkeeping (status, manifest, per-run subprocess logs).

Run from repo root:
    python phase3_orchestrate_event_driven_batch.py --pilot-only
    python phase3_orchestrate_event_driven_batch.py
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PHASE3_LOGS_DIR = REPO_ROOT / "logs" / "phase3_event_vs_fixed"
PHASE3_RESULTS_DIR = REPO_ROOT / "results" / "phase3_event_vs_fixed"

AGENT_TYPES = ("hamilton_ppo", "return_mlp_ppo", "return_lstm_ppo")
LEARNER_SEEDS = (0, 1, 2, 3, 4)
TRAINING_ENV_SEED = 70000
TOTAL_TIMESTEPS = 200_000
EVAL_FREQ = 16_000

# Fresh, non-overlapping validation range: inspected all prior logs/results
# before choosing this. Prior ranges in use anywhere in this project:
# training seeds 0-4 (learner); dev/monitor eval seeds 90001-90005;
# fixed-step offline-selection validation 91001-91050;
# hamilton_ppo_eval_lib HOLDOUT_SEEDS 100000-100199, BENCHMARK_SEEDS
# 100000-100049, MULTISEED_HOLDOUT_SEEDS 110000-110099;
# evaluate_agents_common.py DEFAULT_HOLDOUT_SEEDS 120000-120099;
# evaluate_agents_event_driven.py DEFAULT_HOLDOUT_SEEDS 130000-130099 (module
# default, not yet consumed by an actual run); Phase 1 equivalence-diagnostic
# seeds 900000-900499/950000-950499; Phase 2 smoke/probe seeds 91401-91408,
# 160000-160799ish, 170000-170029. None of these overlap 195000s/200000s.
VALIDATION_SEEDS_START = 195_001
VALIDATION_SEEDS_COUNT = 50

# Holdout: frozen here, in the experiment manifest, BEFORE any holdout
# evaluation is run (Section 6's discipline requirement) -- never used for
# training, monitoring, validation or checkpoint selection anywhere above.
HOLDOUT_SEEDS_START = 200_000
HOLDOUT_SEEDS_COUNT = 200


def run_tag(seed: int) -> str:
    return f"phase3_event_env{TRAINING_ENV_SEED}_learner_{seed}"


def pilot_tag(agent_type: str) -> str:
    return f"phase3_pilot_{agent_type}"


def model_dir(agent_type: str) -> Path:
    return REPO_ROOT / "models" / f"{agent_type}_event"


def log_dir(agent_type: str) -> Path:
    return REPO_ROOT / "logs" / f"{agent_type}_event"


def expected_artifacts(agent_type: str, tag: str) -> dict:
    return dict(
        final_model=model_dir(agent_type) / f"ppo_{agent_type}_{tag}.zip",
        checkpoint_manifest=log_dir(agent_type) / f"checkpoint_manifest_{tag}.json",
        run_config=log_dir(agent_type) / f"run_config_{tag}.json",
        run_summary=log_dir(agent_type) / f"run_summary_{tag}.json",
        offline_best_model=model_dir(agent_type) / f"ppo_{agent_type}_{tag}_offline_best.zip",
        offline_selection=log_dir(agent_type) / f"offline_selection_{tag}.json",
    )


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_is_complete(agent_type: str, seed: int) -> tuple:
    tag = run_tag(seed)
    artifacts = expected_artifacts(agent_type, tag)
    missing = [name for name, p in artifacts.items() if not p.exists()]
    if missing:
        return False, f"missing artefact(s): {missing}"

    sel = json.load(open(artifacts["offline_selection"]))
    recorded_hash = sel.get("copied_policy_hash")
    actual_hash = sha256_file(artifacts["offline_best_model"])
    if recorded_hash != actual_hash:
        return False, f"offline_best hash mismatch (recorded={recorded_hash}, actual={actual_hash})"
    if sel.get("hash_verified") is not True:
        return False, "offline_selection.json hash_verified is not True"

    cfg = json.load(open(artifacts["run_config"]))
    if cfg.get("environment_type") != "event":
        return False, f"run_config environment_type={cfg.get('environment_type')!r}, expected 'event'"
    if cfg.get("training_env_seed") != TRAINING_ENV_SEED:
        return False, f"run_config training_env_seed={cfg.get('training_env_seed')}, expected {TRAINING_ENV_SEED}"
    if cfg.get("learner_seed") != seed:
        return False, f"run_config learner_seed={cfg.get('learner_seed')}, expected {seed}"
    if cfg["cli_args"].get("total_timesteps") != TOTAL_TIMESTEPS:
        return False, f"run_config total_timesteps={cfg['cli_args'].get('total_timesteps')}, expected {TOTAL_TIMESTEPS}"
    if sel.get("learner_seed") != seed or sel.get("training_env_seed") != TRAINING_ENV_SEED:
        return False, "offline_selection.json learner_seed/training_env_seed mismatch"
    if sel.get("validation_seed_start") != VALIDATION_SEEDS_START or sel.get("validation_seed_count") != VALIDATION_SEEDS_COUNT:
        return False, "offline_selection.json validation seed range does not match this experiment's range"

    return True, "OK"


def get_git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def get_package_versions() -> dict:
    import importlib
    versions = {}
    for pkg in ("torch", "stable_baselines3", "sb3_contrib", "gymnasium", "gym", "numpy", "pandas", "scipy"):
        try:
            versions[pkg] = importlib.import_module(pkg).__version__
        except Exception as e:
            versions[pkg] = f"ERROR: {e}"
    versions["python"] = sys.version
    return versions


def run_subprocess_logged(cmd: list, log_path: Path) -> tuple:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(log_path, "w") as f:
        f.write(f"COMMAND: {' '.join(cmd)}\nSTART: {start_iso}\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    end_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"\nEND: {end_iso}\nELAPSED_SECONDS: {elapsed:.1f}\nRETURN_CODE: {proc.returncode}\n")
    return proc.returncode, elapsed, start_iso, end_iso


def do_pilot_run(agent_type: str, status: dict) -> bool:
    tag = pilot_tag(agent_type)
    train_log = PHASE3_LOGS_DIR / f"pilot_train_{agent_type}.log"
    cmd = [
        sys.executable, "train_agents.py",
        "--agent-type", agent_type, "--environment-type", "event",
        "--seed", "0", "--env-seed", str(TRAINING_ENV_SEED), "--learner-seed", "0",
        "--total-timesteps", "16000", "--eval-freq", "8000",
        "--save-all-checkpoints", "--overwrite-checkpoints", "--run-tag", tag,
    ]
    print(f"[PILOT TRAIN] {' '.join(cmd)}")
    rc, elapsed, start_iso, end_iso = run_subprocess_logged(cmd, train_log)
    ok = rc == 0
    status[f"pilot_{agent_type}"] = dict(
        agent_type=agent_type, kind="pilot", status="complete" if ok else "failed",
        return_code=rc, start=start_iso, end=end_iso, elapsed_seconds=elapsed, log=str(train_log),
    )
    if not ok:
        print(f"[FAIL] pilot failed for {agent_type} (rc={rc}) -- see {train_log}")
    return ok


def do_one_run(agent_type: str, seed: int, status: dict) -> bool:
    tag = run_tag(seed)
    complete, reason = run_is_complete(agent_type, seed)
    if complete:
        print(f"[SKIP] {agent_type}/{tag} already complete and verified.")
        status[f"{agent_type}_{tag}"] = dict(agent_type=agent_type, learner_seed=seed, tag=tag, status="skipped_already_complete")
        return True
    else:
        print(f"[RUN] {agent_type}/{tag} not complete ({reason}) -- (re)running.")

    train_log = PHASE3_LOGS_DIR / f"train_{agent_type}_{tag}.log"
    train_cmd = [
        sys.executable, "train_agents.py",
        "--agent-type", agent_type, "--environment-type", "event",
        "--seed", str(seed), "--env-seed", str(TRAINING_ENV_SEED), "--learner-seed", str(seed),
        "--total-timesteps", str(TOTAL_TIMESTEPS), "--eval-freq", str(EVAL_FREQ),
        "--save-all-checkpoints", "--overwrite-checkpoints", "--run-tag", tag,
    ]
    print(f"[TRAIN] {' '.join(train_cmd)}")
    rc, elapsed, start_iso, end_iso = run_subprocess_logged(train_cmd, train_log)
    if rc != 0:
        print(f"[FAIL] training failed for {agent_type}/{tag} (rc={rc}) -- see {train_log}")
        status[f"{agent_type}_{tag}"] = dict(agent_type=agent_type, learner_seed=seed, tag=tag, status="training_failed",
                            return_code=rc, start=start_iso, end=end_iso, elapsed_seconds=elapsed, log=str(train_log))
        return False

    select_log = PHASE3_LOGS_DIR / f"select_{agent_type}_{tag}.log"
    select_cmd = [
        sys.executable, "select_checkpoint_offline.py",
        "--agent-type", agent_type, "--environment-type", "event", "--run-tag", tag,
        "--learner-seed", str(seed), "--training-env-seed", str(TRAINING_ENV_SEED),
        "--validation-seeds-start", str(VALIDATION_SEEDS_START),
        "--validation-seeds-count", str(VALIDATION_SEEDS_COUNT),
        "--overwrite-selection",
    ]
    print(f"[SELECT] {' '.join(select_cmd)}")
    rc2, elapsed2, start2, end2 = run_subprocess_logged(select_cmd, select_log)
    if rc2 != 0:
        print(f"[FAIL] checkpoint selection failed for {agent_type}/{tag} (rc={rc2}) -- see {select_log}")
        status[f"{agent_type}_{tag}"] = dict(agent_type=agent_type, learner_seed=seed, tag=tag, status="selection_failed",
                            return_code=rc2, start=start_iso, end=end2, elapsed_seconds=elapsed + elapsed2,
                            train_log=str(train_log), select_log=str(select_log))
        return False

    complete, reason = run_is_complete(agent_type, seed)
    status[f"{agent_type}_{tag}"] = dict(
        agent_type=agent_type, learner_seed=seed, tag=tag,
        status="complete" if complete else f"post_selection_verification_failed:{reason}",
        return_code=0, start=start_iso, end=end2, elapsed_seconds=elapsed + elapsed2,
        train_log=str(train_log), select_log=str(select_log),
    )
    if not complete:
        print(f"[FAIL] post-selection verification failed for {agent_type}/{tag}: {reason}")
    return complete


def save_status(status: dict):
    PHASE3_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE3_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PHASE3_LOGS_DIR / "orchestration_status.json", "w") as f:
        json.dump(status, f, indent=2, default=str)
    import pandas as pd
    pd.DataFrame(list(status.values())).to_csv(PHASE3_RESULTS_DIR / "phase3_run_status.csv", index=False)


def write_manifest() -> dict:
    PHASE3_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PHASE3_RESULTS_DIR / "phase3_experiment_manifest.json"
    if manifest_path.exists():
        # Freeze once -- never silently overwrite an already-frozen manifest
        # with a different git commit/seed range on a resumed run.
        return json.load(open(manifest_path))
    manifest = dict(
        git_commit=get_git_commit(),
        package_versions=get_package_versions(),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        training_env_seed=TRAINING_ENV_SEED,
        learner_seeds=list(LEARNER_SEEDS),
        agent_types=list(AGENT_TYPES),
        total_timesteps=TOTAL_TIMESTEPS,
        eval_freq=EVAL_FREQ,
        validation_seeds_start=VALIDATION_SEEDS_START,
        validation_seeds_count=VALIDATION_SEEDS_COUNT,
        holdout_seeds_start=HOLDOUT_SEEDS_START,
        holdout_seeds_count=HOLDOUT_SEEDS_COUNT,
        fixed_step_run_tag_pattern="offlinecv_env70000_learner_{seed}",
        event_driven_run_tag_pattern=run_tag("{seed}"),
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot-only", action="store_true",
                    help="Run only a 16,000-transition pilot for each of the 3 agent types "
                         "(learner_seed=0, env-seed=70000), skip checkpoint selection and the full batch.")
    args = p.parse_args()

    PHASE3_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE3_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest()
    print(f"Experiment manifest: {json.dumps(manifest, indent=2)}")

    status_path = PHASE3_LOGS_DIR / "orchestration_status.json"
    status = json.load(open(status_path)) if status_path.exists() else {}

    if args.pilot_only:
        print("\n=== PILOT MODE: 16,000-transition pilot per agent type (learner_seed=0, env-seed=70000) ===")
        all_ok = True
        for agent_type in AGENT_TYPES:
            ok = do_pilot_run(agent_type, status)
            save_status(status)
            all_ok = all_ok and ok
        print("ALL PILOTS OK" if all_ok else "SOME PILOTS FAILED -- see logs/phase3_event_vs_fixed/pilot_train_*.log")
        return 0 if all_ok else 1

    print("\n=== FULL BATCH: 15 event-driven training + checkpoint-selection runs (sequential) ===")
    for agent_type in AGENT_TYPES:
        for seed in LEARNER_SEEDS:
            t_run_start = time.time()
            ok = do_one_run(agent_type, seed, status)
            save_status(status)
            if not ok:
                print(f"\nSTOPPING: run {agent_type}/learner_{seed} failed or failed verification.")
                print(f"Fix the underlying issue, then re-run this script to resume "
                      f"(completed runs will be skipped).")
                return 1
            print(f"[{agent_type}/learner_{seed}] done in {time.time() - t_run_start:.1f}s "
                  f"(cumulative status saved to {status_path})")
    print("\nALL 15 EVENT-DRIVEN RUNS COMPLETE AND VERIFIED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
