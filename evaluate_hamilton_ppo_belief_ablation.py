"""
evaluate_hamilton_ppo_belief_ablation.py
--------------------------------------------
Belief-input ablation (item 3) for the saved Hamilton PPO model. Does NOT
retrain the network -- only the belief component of the observation fed
to model.predict() is changed; the environment, reward, and everything
else about the observation (q_scaled, tau) are unchanged.

Four treatments, evaluated on the SAME 50 holdout-range seeds
(hamilton_ppo_eval_lib.BENCHMARK_SEEDS) with common random numbers (the
underlying environment path for a given seed is identical across all four
treatments -- see hamilton_ppo_eval_lib's module docstring: the midprice/
regime path is provably action-independent, so it is also independent of
which belief value is fed to the policy):

    1. correct     -- the true Hamilton filter belief (baseline)
    2. stationary  -- belief fixed at the stationary prior for every step
    3. shuffled    -- the true episode's belief SEQUENCE, permuted within
                      that episode (same set of realised values, wrong order)
    4. complement  -- 1 - b_t

Run from repo root:
    python evaluate_hamilton_ppo_belief_ablation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

import hamilton_ppo_eval_lib as lib

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
MODEL_PATH = REPO_ROOT / "models" / "hamilton_ppo" / "ppo_hamilton_v1.zip"
INVENTORY_SCALE = 10.0
TREATMENTS = ["correct", "stationary", "shuffled", "complement"]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = PPO.load(str(MODEL_PATH))
    print(f"Loaded model from {MODEL_PATH}")
    print(f"Belief ablation on {len(lib.BENCHMARK_SEEDS)} seeds: "
          f"{lib.BENCHMARK_SEEDS[0]}..{lib.BENCHMARK_SEEDS[-1]}, treatments={TREATMENTS}")

    rows = []
    for i, seed in enumerate(lib.BENCHMARK_SEEDS):
        true_beliefs, _ = lib.compute_reference_trajectory(seed, INVENTORY_SCALE)
        for kind in TREATMENTS:
            override = lib.make_belief_override(kind, true_beliefs, seed)
            r = lib.run_eval_episode_detailed(
                model, seed, INVENTORY_SCALE, belief_override=override, deterministic=True,
            )
            r["treatment"] = kind
            rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(lib.BENCHMARK_SEEDS)} seeds done")

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "hamilton_ppo_belief_ablation_v1.csv"
    df.to_csv(out_path, index=False)
    print(f"\nPer-episode ablation results saved to {out_path}")

    # Sanity check: "correct" treatment here must exactly match an
    # un-ablated run (belief_override=true_beliefs is literally the
    # identity transform).
    correct_check = df[df["treatment"] == "correct"]["cumulative_objective"].to_numpy()
    plain = np.array([
        lib.run_eval_episode_detailed(model, s, INVENTORY_SCALE, belief_override=None)["cumulative_objective"]
        for s in lib.BENCHMARK_SEEDS[:5]
    ])
    match = np.allclose(correct_check[:5], plain, atol=1e-9)
    print(f"\nSanity check: 'correct' treatment matches un-ablated evaluation on first 5 seeds: {match}")
    if not match:
        print(f"  correct_check[:5] = {correct_check[:5]}")
        print(f"  plain             = {plain}")

    # --- Paired differences vs the correct-belief treatment ---
    print("\n" + "=" * 100)
    print("PAIRED OBJECTIVE DIFFERENCES: correct-belief minus each ablation (positive => correct belief helps)")
    print("=" * 100)
    piv = df.pivot(index="seed", columns="treatment", values="cumulative_objective")
    summary_rows = []
    for kind in ("stationary", "shuffled", "complement"):
        D = piv["correct"] - piv[kind]
        n = len(D)
        mean = float(D.mean())
        se = float(D.std(ddof=1) / np.sqrt(n))
        ci_lo, ci_hi = mean - 1.96 * se, mean + 1.96 * se
        median = float(D.median())
        win_rate = float((D > 0).mean())
        summary_rows.append(dict(
            comparison=f"correct_minus_{kind}", n=n, mean_diff=mean, se=se,
            ci_lo=ci_lo, ci_hi=ci_hi, median_diff=median, win_rate=win_rate,
        ))
        print(f"\ncorrect vs {kind}:")
        print(f"  mean diff = {mean:.4f}  (SE={se:.4f}, 95% CI=[{ci_lo:.4f}, {ci_hi:.4f}])")
        print(f"  median diff = {median:.4f}   win rate (correct > ablation) = {win_rate:.3f}")

    ablation_summary_df = pd.DataFrame(summary_rows)
    ablation_summary_path = RESULTS_DIR / "hamilton_ppo_belief_ablation_summary_v1.csv"
    ablation_summary_df.to_csv(ablation_summary_path, index=False)
    print(f"\nAblation paired-difference summary saved to {ablation_summary_path}")

    # --- Also compare action statistics across treatments (does the actor's behaviour even change?) ---
    print("\n" + "=" * 100)
    print("ACTION STATISTICS BY TREATMENT (mean over all episodes)")
    print("=" * 100)
    action_summary = df.groupby("treatment")[[
        "bid_action_mean", "ask_action_mean", "mean_signed_inventory",
        "mean_abs_inventory", "terminal_signed_inventory", "cumulative_objective",
    ]].mean()
    print(action_summary.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nDone.")


if __name__ == "__main__":
    main()
