"""
ips_evaluate_holdout.py
--------------------------------
Evaluates, on the SAME 500 holdout paths/seeds as the original experiment
(ips_common.HOLDOUT_SEEDS == final_common.HOLDOUT_SEEDS, reused verbatim),
BOTH penalty calibrations:

  "original"     (phi=0.01,  alpha=0.001)  -- the 3 PPO architectures' EXISTING
                 fixed 1,000,000-transition checkpoints (read-only, never
                 retrained -- from final_reduced_exploration_architecture_
                 comparison / phase7_groupB_1m_convergence), the EXISTING
                 frozen clone (phase4_policy_diagnostic), and the analytical
                 oracle/belief_weighted policies RECOMPUTED at (0.01, 0.001)
                 (cheap, deterministic, and guarantees this experiment's
                 numbers are byte-identical in derivation to a fresh
                 evaluation, not merely assumed equal to a previously-saved
                 copy).

  "high_penalty" (phi=0.10, alpha=0.010)  -- the 3 PPO architectures'
                 FRESHLY-trained fixed 1,000,000-transition checkpoints
                 (from ips_run_training.py), the freshly-trained high-
                 penalty frozen clone (ips_train_clone.py), and the
                 analytical oracle/belief_weighted policies recomputed at
                 (0.10, 0.010).

For each (calibration, policy) this evaluation runs exactly ONE deterministic
episode per holdout path (task instruction #6: "Use deterministic policy
means during final evaluation" -- no stochastic companion pass, unlike
final_evaluate_holdout.py's validation-time diagnostic use of one).

Writes, per (calibration, policy[, learner_seed]) -- so this script is
CHEAPLY resumable/re-runnable (skips straight to concatenation, no
re-simulation) as soon as each high-penalty architecture's training
completes:
  - event_level/<calibration>/<tag>.parquet            -- one row per event
  - event_level/<calibration>/<tag>_episode_summary.parquet -- one row per
    holdout path (500 rows) for this one policy; concatenated across every
    (calibration, policy) at the end into episode_level.csv.

Run from repo root:
    python ips_evaluate_holdout.py                       # both calibrations, whatever checkpoints exist
    python ips_evaluate_holdout.py --calibrations original
    python ips_evaluate_holdout.py --calibrations high_penalty
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

import ips_common as IC
import ips_episode_runner as ER
import phase5_common as P5
from phase4_supervised_clone import SupervisedCloneAgent

REFERENCE_POLICIES = ("oracle", "belief_weighted", "frozen_clone")


def policy_tag(policy: str, learner_seed) -> str:
    return policy if learner_seed is None else f"{policy}_seed{learner_seed}"


def event_level_path(calibration: str, policy: str, learner_seed) -> Path:
    d = IC.EVENT_LEVEL_DIR / calibration
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{policy_tag(policy, learner_seed)}.parquet"


def episode_summary_path(calibration: str, policy: str, learner_seed) -> Path:
    d = IC.EVENT_LEVEL_DIR / calibration
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{policy_tag(policy, learner_seed)}_episode_summary.parquet"


def load_ppo_checkpoint(architecture: str, calibration: str, learner_seed: int):
    if calibration == "original":
        path = IC.original_checkpoint_path(architecture, learner_seed)
    else:
        path = IC.final_checkpoint_path(architecture, learner_seed)
    if not path.exists():
        return None, path
    cls = RecurrentPPO if IC.is_recurrent(architecture) else PPO
    return cls.load(str(path)), path


def load_clone_agent(calibration: str):
    path = IC.original_clone_path() if calibration == "original" else IC.clone_checkpoint_path()
    if not path.exists():
        return None, path
    net = P5.build_supervised_clone_net()
    net.load_state_dict(torch.load(path))
    net.eval()
    return SupervisedCloneAgent(net), path


def already_done(calibration: str, policy: str, learner_seed) -> bool:
    ev_path = event_level_path(calibration, policy, learner_seed)
    sm_path = episode_summary_path(calibration, policy, learner_seed)
    if not (ev_path.exists() and sm_path.exists()):
        return False
    try:
        n_events = pd.read_parquet(ev_path, columns=["holdout_episode_seed"])["holdout_episode_seed"].nunique()
        n_summary = len(pd.read_parquet(sm_path))
    except Exception:
        return False
    return n_events == len(IC.HOLDOUT_SEEDS) and n_summary == len(IC.HOLDOUT_SEEDS)


def evaluate_and_persist(policy: str, calibration: str, phi: float, alpha: float, *,
                          model=None, controls=None, learner_seed=None) -> pd.DataFrame:
    """Runs all 500 holdout episodes for ONE policy (unless already fully
    persisted, in which case it loads the cached episode-summary parquet
    instead of re-simulating anything), writes both parquet files, and
    returns the episode-summary DataFrame for this one policy."""
    tag = policy_tag(policy, learner_seed)
    if already_done(calibration, policy, learner_seed):
        print(f"SKIP: {tag} [{calibration}] already fully evaluated on {len(IC.HOLDOUT_SEEDS)} holdout paths.")
        return pd.read_parquet(episode_summary_path(calibration, policy, learner_seed))

    print(f"Evaluating {tag} [{calibration}] on {len(IC.HOLDOUT_SEEDS)} holdout paths...")
    t0 = time.time()
    episode_rows, event_rows = [], []
    for i, seed in enumerate(IC.HOLDOUT_SEEDS):
        summary, ev_rows = ER.evaluate_one(policy, calibration, seed, phi, alpha,
                                            model=model, controls=controls, learner_seed=learner_seed)
        episode_rows.append(summary)
        event_rows.extend(ev_rows)
        if (i + 1) % 100 == 0:
            print(f"    {tag} [{calibration}]: {i+1}/{len(IC.HOLDOUT_SEEDS)} paths done [{time.time()-t0:.0f}s elapsed]")

    ep_df = pd.DataFrame(episode_rows)
    ev_df = pd.DataFrame(event_rows)
    ev_df.to_parquet(event_level_path(calibration, policy, learner_seed), index=False)
    ep_df.to_parquet(episode_summary_path(calibration, policy, learner_seed), index=False)
    print(f"  Saved {len(ev_df)} event rows + {len(ep_df)} episode-summary rows for {tag} [{calibration}] "
          f"[{time.time()-t0:.0f}s]")
    return ep_df


def run_calibration(calibration: str) -> pd.DataFrame:
    cfg = IC.CALIBRATIONS[calibration]
    phi, alpha = cfg["phi"], cfg["alpha"]
    print(f"\n{'='*70}\nCalibration: {calibration}  (phi={phi}, alpha={alpha})\n{'='*70}")

    print(f"Building analytical controls at (phi={phi}, alpha={alpha})...")
    controls = IC.build_analytical_controls(phi, alpha)

    all_ep_dfs = []

    for policy in ("oracle", "belief_weighted"):
        all_ep_dfs.append(evaluate_and_persist(policy, calibration, phi, alpha, controls=controls))

    clone_agent, clone_path = load_clone_agent(calibration)
    if clone_agent is None:
        print(f"WARNING: frozen clone checkpoint not found for calibration={calibration} at {clone_path} -- "
              f"skipping frozen_clone for this calibration "
              f"(run ips_train_clone.py first if calibration=='high_penalty').")
    else:
        all_ep_dfs.append(evaluate_and_persist("frozen_clone", calibration, phi, alpha, model=clone_agent))

    for architecture in IC.ARCHITECTURES:
        for seed in IC.LEARNER_SEEDS:
            model, ckpt_path = load_ppo_checkpoint(architecture, calibration, seed)
            if model is None:
                print(f"WARNING: checkpoint not found for {architecture} seed {seed} calibration={calibration} "
                      f"at {ckpt_path} -- skipping (training not yet complete for this run).")
                continue
            all_ep_dfs.append(evaluate_and_persist(architecture, calibration, phi, alpha,
                                                     model=model, learner_seed=seed))
            del model

    return pd.concat(all_ep_dfs, ignore_index=True) if all_ep_dfs else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibrations", type=str, default="original,high_penalty",
                         help="Comma-separated subset of {original, high_penalty}")
    args = parser.parse_args()
    calibrations = [c.strip() for c in args.calibrations.split(",") if c.strip()]
    for c in calibrations:
        assert c in IC.CALIBRATION_NAMES, f"Unknown calibration {c!r}, expected one of {IC.CALIBRATION_NAMES}"

    t0 = time.time()
    IC.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Holdout range {IC.HOLDOUT_SEEDS[0]}-{IC.HOLDOUT_SEEDS[-1]} ({len(IC.HOLDOUT_SEEDS)} paths), "
          f"identical to the original experiment's own holdout range (asserted at ips_common import time).")

    calibration_frames = {}
    for calibration in calibrations:
        calibration_frames[calibration] = run_calibration(calibration)

    # Rebuild the combined episode_level.csv from EVERY per-policy episode-summary
    # parquet file on disk (not just the calibrations touched this invocation), so
    # partial runs (e.g. "original" today, "high_penalty" once training finishes)
    # always produce one complete, consistent combined file.
    all_frames = []
    for calibration in IC.CALIBRATION_NAMES:
        d = IC.EVENT_LEVEL_DIR / calibration
        if not d.exists():
            continue
        for p in sorted(d.glob("*_episode_summary.parquet")):
            all_frames.append(pd.read_parquet(p))
    combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    episode_level_path = IC.RESULTS_DIR / "episode_level.csv"
    combined.to_csv(episode_level_path, index=False)
    print(f"\nSaved {episode_level_path} ({len(combined)} rows, spanning "
          f"{sorted(combined['penalty_calibration'].unique()) if len(combined) else []})")

    max_recon = combined["reward_reconciliation_error"].max() if len(combined) else float("nan")
    elapsed = time.time() - t0
    manifest_note = dict(
        holdout_seed_start=IC.HOLDOUT_SEEDS[0], holdout_seed_count=len(IC.HOLDOUT_SEEDS),
        calibrations_requested_this_run=calibrations,
        n_episode_rows_total=len(combined), max_reward_reconciliation_error=float(max_recon),
        elapsed_seconds=elapsed, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    (IC.RESULTS_DIR / "holdout_evaluation_manifest.json").write_text(json.dumps(manifest_note, indent=2, default=str))
    print(f"\nHoldout evaluation pass complete in {elapsed:.1f}s ({elapsed/60:.1f} min). "
          f"Max reward-reconciliation error: {max_recon:.3e}")


if __name__ == "__main__":
    main()
