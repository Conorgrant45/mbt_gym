"""
evaluate_hamilton_ppo_multiseed_benchmarks.py
--------------------------------------------------
Paired benchmark comparison: naive / Hamilton belief-weighted / oracle
analytic policies vs. each of the 5 independently-trained Hamilton PPO
models, on the SAME 100 holdout seeds (110000-110099) used for the
common holdout evaluation.

Reuses the repository's validated paired-path machinery: oracle/belief/
naive episodes are run via compare_four_policies_paired.run_episode()
UNCHANGED; PPO episodes reuse evaluate_hamilton_ppo_benchmarks.py's
run_episode_ppo() (from the previous milestone) unchanged. The analytic
policies are only run ONCE on the 100 seeds (their results don't depend
on which PPO training seed they're being compared against); each of the
5 PPO models is then run once on the same 100 seeds and paired against
the shared analytic results.

Run from repo root:
    python evaluate_hamilton_ppo_multiseed_benchmarks.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

import simulate_belief_weighted as SBW  # READ ONLY
import compare_four_policies_paired as C4P  # READ ONLY
from evaluate_hamilton_ppo_benchmarks import run_episode_ppo  # READ ONLY, from the accepted previous milestone
import hamilton_ppo_eval_lib as lib

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models" / "hamilton_ppo"

ANALYTIC_POLICIES = ["oracle", "belief", "naive"]
TRAINING_SEEDS = [0, 1, 2, 3, 4]
N_VERIFY = 10


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = lib.MULTISEED_HOLDOUT_SEEDS
    print(f"Multiseed paired benchmark comparison on {len(seeds)} seeds: {seeds[0]}..{seeds[-1]}")

    print("Building oracle optimal control tables (unchanged existing solver)...")
    controls = {}
    for regime, params in SBW.REGIME_PARAMS.items():
        da, db, qag, qbg = SBW.build_optimal_control(**params)
        controls[regime] = {"delta_ask": da, "delta_bid": db, "q_ask": qag, "q_bid": qbg}

    # --- Run the 3 analytic policies ONCE on all 100 seeds ---
    analytic_rows = []
    analytic_traces = {}  # seed -> {policy: trace}, for path verification
    for i, seed in enumerate(seeds):
        full_trace = i < N_VERIFY
        traces = {}
        for policy in ANALYTIC_POLICIES:
            m, tr = C4P.run_episode(policy, controls, seed, full_trace)
            assert m["steps_taken"] == 4000
            assert m["reconciliation_abs_diff"] < 1e-6
            row = dict(seed=seed, policy=policy)
            row.update(m)
            analytic_rows.append(row)
            if full_trace:
                traces[policy] = tr
        if full_trace:
            analytic_traces[seed] = traces
        if (i + 1) % 25 == 0:
            print(f"  analytic policies: {i+1}/{len(seeds)} seeds done")

    analytic_df = pd.DataFrame(analytic_rows)
    print("Analytic policies (oracle/belief/naive) complete.")

    # --- Run each PPO model once on all 100 seeds ---
    all_rows = list(analytic_rows)
    n_path_mismatches = 0
    n_verified = 0
    for ts in TRAINING_SEEDS:
        model_path = MODELS_DIR / f"ppo_hamilton_seed_{ts}.zip"
        print(f"\ntraining_seed={ts}: loading {model_path}")
        model = PPO.load(str(model_path))
        for i, seed in enumerate(seeds):
            full_trace = i < N_VERIFY
            m, tr = run_episode_ppo(model, seed, full_trace)
            assert m["steps_taken"] == 4000
            assert m["reconciliation_abs_diff"] < 1e-6, (
                f"seed {ts}/{seed}: reconciliation failed (diff={m['reconciliation_abs_diff']:.3e})"
            )
            row = dict(seed=seed, policy="ppo", training_seed=ts)
            row.update(m)
            all_rows.append(row)

            if full_trace:
                n_verified += 1
                ref_trace = analytic_traces[seed]["oracle"]
                same = all(np.array_equal(tr[k], ref_trace[k]) for k in ref_trace)
                if not same:
                    n_path_mismatches += 1
                    for k in ref_trace:
                        if not np.array_equal(tr[k], ref_trace[k]):
                            n_diff = np.sum(tr[k] != ref_trace[k])
                            print(f"  [seed {ts}/{seed}] MISMATCH ppo vs oracle in '{k}': {n_diff}/4000 steps differ")
                assert same, f"seed {ts}/{seed}: exogenous path mismatch between ppo (seed {ts}) and analytic policies"

            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(seeds)} seeds done")

    print(f"\nExogenous-path verification: {n_verified} (training_seed, seed) pairs traced, "
          f"{n_path_mismatches} mismatches -- {'PASSED' if n_path_mismatches == 0 else 'FAILED'}")

    df = pd.DataFrame(all_rows)
    max_recon = df["reconciliation_abs_diff"].max()
    print(f"Objective reconciliation: max |discrepancy| = {max_recon:.3e} -- "
          f"{'PASSED' if max_recon < 1e-6 else 'FAILED'}")

    out_path = RESULTS_DIR / "hamilton_ppo_multiseed_benchmark_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull per-episode benchmark comparison saved to {out_path}")

    # --- Paired comparisons: for each PPO training seed, vs each analytic policy ---
    print("\n" + "=" * 110)
    print("PAIRED COMPARISONS: analytic policy minus PPO (per training seed)")
    print("=" * 110)
    comparison_rows = []
    piv_obj = analytic_df.pivot(index="seed", columns="policy", values="full_objective")
    piv_pnl = analytic_df.pivot(index="seed", columns="policy", values="raw_pnl")
    piv_inv = analytic_df.pivot(index="seed", columns="policy", values="mean_absolute_inventory")
    piv_terminv = analytic_df.pivot(index="seed", columns="policy", values="terminal_absolute_inventory")
    piv_asl = analytic_df.pivot(index="seed", columns="policy", values="adverse_selection_loss")
    piv_fills = analytic_df.pivot(index="seed", columns="policy", values="total_fills")

    for ts in TRAINING_SEEDS:
        ppo_sub = df[(df["policy"] == "ppo") & (df["training_seed"] == ts)].set_index("seed")
        for policy in ANALYTIC_POLICIES:
            D_obj = piv_obj[policy] - ppo_sub["full_objective"]
            n = len(D_obj)
            mean = float(D_obj.mean())
            se = float(D_obj.std(ddof=1) / np.sqrt(n))
            ci_lo, ci_hi = mean - 1.96 * se, mean + 1.96 * se
            median = float(D_obj.median())
            ppo_win_rate = float((D_obj < 0).mean())  # ppo beats analytic policy

            D_pnl = float((piv_pnl[policy] - ppo_sub["raw_pnl"]).mean())
            D_inv = float((piv_inv[policy] - ppo_sub["mean_absolute_inventory"]).mean())
            D_terminv = float((piv_terminv[policy] - ppo_sub["terminal_absolute_inventory"]).mean())
            D_asl = float((piv_asl[policy] - ppo_sub["adverse_selection_loss"]).mean())
            D_fills = float((piv_fills[policy] - ppo_sub["total_fills"]).mean())

            comparison_rows.append(dict(
                training_seed=ts, analytic_policy=policy, n=n,
                mean_objective_diff=mean, se_objective_diff=se, ci_lo=ci_lo, ci_hi=ci_hi,
                median_objective_diff=median, ppo_win_rate=ppo_win_rate,
                mean_raw_pnl_diff=D_pnl, mean_abs_inventory_diff=D_inv,
                mean_terminal_abs_inventory_diff=D_terminv,
                mean_adverse_selection_loss_diff=D_asl, mean_fills_diff=D_fills,
            ))

        print(f"\ntraining_seed={ts}:")
        for row in comparison_rows[-3:]:
            print(f"  {row['analytic_policy']:>10} - ppo: mean_diff={row['mean_objective_diff']:.4f}  "
                  f"SE={row['se_objective_diff']:.4f}  CI=[{row['ci_lo']:.4f},{row['ci_hi']:.4f}]  "
                  f"median={row['median_objective_diff']:.4f}  ppo_win_rate={row['ppo_win_rate']:.3f}")

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = RESULTS_DIR / "hamilton_ppo_multiseed_benchmark_paired_summary.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nPaired-comparison summary saved to {comparison_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
