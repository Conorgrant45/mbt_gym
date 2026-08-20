"""
final_aggregate_holdout.py
--------------------------------
Reads the raw final_holdout_episode_level.csv (written by
final_evaluate_holdout.py -- one row per (policy, learner_seed,
checkpoint_transition, path_seed), 500 fresh unseen holdout paths, IDENTICAL
across every policy) and produces:

    final_holdout_seed_summary.csv        -- one row per (policy,
                                              learner_seed,
                                              checkpoint_transition),
                                              averaged over the 500 holdout
                                              paths. For the 3 analytical/
                                              clone benchmarks (no learner
                                              seed / checkpoint concept),
                                              learner_seed and
                                              checkpoint_transition are left
                                              blank -- this row IS both the
                                              seed-level and
                                              architecture-level summary for
                                              those policies.
    final_holdout_architecture_summary.csv -- one row per (architecture,
                                              checkpoint_transition) for the
                                              3 PPO architectures only,
                                              averaged over the 5 learner
                                              seeds' OWN seed-level means
                                              (cross-seed mean/SD/SE -- NOT
                                              a flat pool of 2,500
                                              path-level observations, to
                                              keep learner-seed uncertainty
                                              and path-level uncertainty
                                              separate).

Never evaluates a model or reads a checkpoint file -- purely a read-only
aggregation of already-produced CSV rows, reproducible at any time.

Run from repo root:
    python final_aggregate_holdout.py
"""
import numpy as np
import pandas as pd

import final_common as FC
from final_aggregate_validation import MEAN_FIELDS


def main():
    path = FC.RESULTS_DIR / "final_holdout_episode_level.csv"
    assert path.exists(), f"final_holdout_episode_level.csv not found at {path} -- run final_evaluate_holdout.py first"
    df = pd.read_csv(path)
    assert df["deterministic"].all(), "final_holdout_episode_level.csv rows must all be the deterministic-primary row"

    seed_rows = []
    for (policy, seed, t), sub in df.groupby(["policy", "learner_seed", "checkpoint_transition"], dropna=False):
        row = dict(policy=policy, learner_seed=seed, checkpoint_transition=t, n_paths=len(sub))
        for field in MEAN_FIELDS:
            row[f"mean_{field}"] = float(sub[field].mean())
        row["sd_full_objective"] = float(sub["full_objective"].std(ddof=1))
        row["se_full_objective"] = row["sd_full_objective"] / np.sqrt(len(sub))
        seed_rows.append(row)
    seed_df = pd.DataFrame(seed_rows).sort_values(["policy", "learner_seed", "checkpoint_transition"],
                                                    na_position="first")
    seed_path = FC.RESULTS_DIR / "final_holdout_seed_summary.csv"
    seed_df.to_csv(seed_path, index=False)
    print(f"Saved {seed_path} ({len(seed_df)} rows)")

    arch_rows = []
    ppo_seed_df = seed_df[seed_df["policy"].isin(FC.ARCHITECTURES)]
    for (arch, t), sub in ppo_seed_df.groupby(["policy", "checkpoint_transition"]):
        row = dict(architecture=arch, checkpoint_transition=t, n_seeds=len(sub))
        row["cross_seed_mean_full_objective"] = float(sub["mean_full_objective"].mean())
        row["cross_seed_sd_full_objective"] = float(sub["mean_full_objective"].std(ddof=1)) if len(sub) > 1 else 0.0
        row["cross_seed_se_full_objective"] = row["cross_seed_sd_full_objective"] / np.sqrt(len(sub))
        for field in MEAN_FIELDS:
            row[f"cross_seed_mean_{field}"] = float(sub[f"mean_{field}"].mean())
        arch_rows.append(row)
    arch_df = pd.DataFrame(arch_rows).sort_values(["architecture", "checkpoint_transition"])
    arch_path = FC.RESULTS_DIR / "final_holdout_architecture_summary.csv"
    arch_df.to_csv(arch_path, index=False)
    print(f"Saved {arch_path} ({len(arch_df)} rows)")


if __name__ == "__main__":
    main()
