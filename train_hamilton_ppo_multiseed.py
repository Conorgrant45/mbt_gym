"""
train_hamilton_ppo_multiseed.py
-----------------------------------
Orchestrates 5 independent Hamilton PPO training runs (seeds 0-4), each
using the EXACT accepted v1 configuration (train_hamilton_ppo.py's
current defaults: n_steps=4000, batch_size=400, n_epochs=10, gamma=1.0,
gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
max_grad_norm=0.5, net_arch=[64,64], total_timesteps=200000).

Does NOT modify train_hamilton_ppo.py -- each seed is a fresh subprocess
invocation of the unmodified, already-accepted script, varying only
--seed, --run-tag and --model-path. Each subprocess gets a fresh Python
interpreter (fresh torch/numpy/global RNG state), so there is no risk of
state leaking between seeds -- stronger isolation than an in-process loop
would give.

Runs SEQUENTIALLY, not in parallel (explicit requirement -- this
experiment diagnoses seed variability, and concurrent runs competing for
CPU could introduce timing-dependent nondeterminism into an otherwise
deterministic-per-seed training process).

Does NOT continue from the existing ppo_hamilton_v1.zip checkpoint --
each seed starts from a freshly initialised PPO model and freshly
constructed training environment (train_hamilton_ppo.py's PPO(seed=...)
construction and DummyVecEnv construction always start fresh; there is no
checkpoint-loading code path in it at all).

Does NOT overwrite the accepted v1 artifacts -- v1's own model/log/result
files use run_tag="v1" (train_hamilton_ppo.py's default); this script
always passes an explicit --run-tag seed_N and --model-path
models/hamilton_ppo/ppo_hamilton_seed_N, which are disjoint paths.

Run from repo root:
    python train_hamilton_ppo_multiseed.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LOGS_DIR = REPO_ROOT / "logs" / "hamilton_ppo"
TRAIN_SCRIPT = REPO_ROOT / "train_hamilton_ppo.py"

SEEDS = [0, 1, 2, 3, 4]


def run_one_seed(seed: int) -> dict:
    run_tag = f"seed_{seed}"
    model_path = f"models/hamilton_ppo/ppo_hamilton_seed_{seed}"
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--seed", str(seed),
        "--run-tag", run_tag,
        "--model-path", model_path,
        # every other flag is left at train_hamilton_ppo.py's own default,
        # which already matches the accepted v1 configuration exactly
        # (n_steps=4000, batch_size=400, n_epochs=10, gamma=1.0,
        # gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
        # max_grad_norm=0.5, net_arch=[64,64], total_timesteps=200000,
        # eval_seeds=[90001..90005], eval_freq=40000, inventory_scale=10.0)
    ]
    print("=" * 78)
    print(f"Training seed {seed}: {' '.join(cmd)}")
    print("=" * 78)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0
    print(f"\nSeed {seed} subprocess finished in {elapsed:.1f}s with return code {result.returncode}")

    if result.returncode != 0:
        raise RuntimeError(f"Training for seed {seed} failed (return code {result.returncode})")

    summary_path = LOGS_DIR / f"run_summary_{run_tag}.json"
    with open(summary_path) as f:
        summary = json.load(f)

    assert summary["numerical_instability"] == [], (
        f"Seed {seed}: numerical instability detected: {summary['numerical_instability']}"
    )
    assert summary["save_reload_equivalence_ok"] is True, (
        f"Seed {seed}: save/reload equivalence FAILED "
        f"(max action diff={summary['max_action_diff_save_reload']}, "
        f"max objective diff={summary['max_objective_diff_save_reload']})"
    )
    print(f"Seed {seed}: no NaNs/infs, save/reload equivalence confirmed. "
          f"baseline={summary['baseline_summary']['mean_objective']:.4f}  "
          f"final={summary['final_summary']['mean_objective']:.4f}  "
          f"improvement={summary['improvement_over_baseline_mean_objective']:.4f}")

    return dict(seed=seed, elapsed_seconds=elapsed, summary=summary)


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    results = []
    for seed in SEEDS:
        results.append(run_one_seed(seed))

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print(f"ALL {len(SEEDS)} SEEDS TRAINED SUCCESSFULLY in {total_elapsed:.1f}s "
          f"({total_elapsed/60:.1f} min)")
    print("=" * 78)
    for r in results:
        s = r["summary"]
        print(f"  seed {r['seed']}: elapsed={r['elapsed_seconds']:.1f}s  "
              f"baseline_obj={s['baseline_summary']['mean_objective']:.4f}  "
              f"final_obj={s['final_summary']['mean_objective']:.4f}  "
              f"terminal|inv|={s['final_summary']['mean_terminal_abs_inventory']:.3f}  "
              f"frac_near_bound={s['final_summary']['frac_near_bound_any']:.4f}")

    overview_path = LOGS_DIR / "multiseed_training_overview.json"
    with open(overview_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nMultiseed training overview saved to {overview_path}")


if __name__ == "__main__":
    main()
