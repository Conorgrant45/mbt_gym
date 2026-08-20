"""
final/final_evaluate_validation_benchmarks.py
--------------------------------------------
Evaluates the 3 analytical/clone benchmarks (oracle, belief_weighted,
frozen_clone) on this experiment's own FC.VALIDATION_SEEDS (the SAME 50
fixed paths used for every architecture/seed/checkpoint's validation-curve
evaluation in final_run_training.py). Benchmarks have no training and no
checkpoint concept, so this only needs to run ONCE per experiment (not per
checkpoint) -- unlike the PPO validation curve, which is written
incrementally during training.

This is an addition beyond the raw-file list named explicitly in the
experiment brief, added because the required learning-curve plots
("benchmark lines") need benchmark performance measured on the VALIDATION
seed set specifically -- mixing in the (different) holdout-seed benchmark
numbers already in final_holdout_episode_level.csv would not be a like-for-
like reference line on a validation-indexed x-axis. Mirrors Phase 7's own
precedent (phase7_benchmark_on_own_validation_seeds.csv).

Writes:
    validation_benchmark_episode_level.csv  -- one row per (policy,
                                                validation_path_seed)
    validation_benchmark_summary.csv        -- one row per policy, mean/sd/se
                                                of full_objective (+ other
                                                MEAN_FIELDS) over the 50
                                                validation paths

Run from repo root:
    python final/final_evaluate_validation_benchmarks.py
"""
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

import numpy as np
import pandas as pd

import final.final_common as FC
import final.final_episode_runner as ER
import shared.phase4_common as P4
import shared.phase5_common as P5
from shared.phase4_supervised_clone import SupervisedCloneAgent
from final.final_aggregate_validation import MEAN_FIELDS

BENCHMARKS = ("oracle", "belief_weighted", "frozen_clone")


def main():
    FC.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    controls = P4.build_analytical_controls()
    clone_agent = SupervisedCloneAgent(P5.load_supervised_clone_net())

    rows = []
    for policy_name in ("oracle", "belief_weighted"):
        for seed in FC.VALIDATION_SEEDS:
            row = ER.evaluate_benchmark_row(policy_name, controls, seed)
            row.update(policy=policy_name, validation_path_seed=row.pop("evaluation_seed"))
            rows.append(row)
    for seed in FC.VALIDATION_SEEDS:
        row = ER.evaluate_clone_row(clone_agent, seed)
        row.update(policy="frozen_clone", validation_path_seed=row.pop("evaluation_seed"))
        rows.append(row)

    episode_df = pd.DataFrame(rows)
    episode_path = FC.RESULTS_DIR / "validation_benchmark_episode_level.csv"
    episode_df.to_csv(episode_path, index=False)
    print(f"Saved {episode_path} ({len(episode_df)} rows)")

    summary_rows = []
    for policy in BENCHMARKS:
        sub = episode_df[episode_df["policy"] == policy]
        row = dict(policy=policy, n_paths=len(sub))
        for field in MEAN_FIELDS:
            row[f"mean_{field}"] = float(sub[field].mean())
        row["sd_full_objective"] = float(sub["full_objective"].std(ddof=1))
        row["se_full_objective"] = row["sd_full_objective"] / np.sqrt(len(sub))
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = FC.RESULTS_DIR / "validation_benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path} ({len(summary_df)} rows)")


if __name__ == "__main__":
    main()
