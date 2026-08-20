"""
ips/ips_aggregate_summary.py
--------------------------------
Reads episode_level.csv (written by ips_evaluate_holdout.py -- one row per
(penalty_calibration, policy, learner_seed, holdout_episode_seed), the SAME
500 holdout paths for every policy AND both calibrations) and produces:

  seed_summary.csv     -- one row per (penalty_calibration, policy,
                           learner_seed), averaged over the 500 holdout
                           paths. Reference policies (oracle, belief_weighted,
                           frozen_clone) have learner_seed left blank -- this
                           row IS both the seed-level and policy-level
                           summary for them.
  policy_summary.csv   -- one row per (penalty_calibration, policy): for the
                           3 PPO architectures, cross-seed mean/SD/SE over
                           the 5 learner seeds' OWN seed-level means (the
                           SAME aggregation convention as
                           final_aggregate_holdout.py -- never a flat pool
                           of 2,500 path-level observations, keeping
                           learner-seed uncertainty and path-level
                           uncertainty separate); for the 3 reference
                           policies, identical to their seed_summary row.
  comparison_original_vs_high_penalty.csv
                        -- one row per (policy, learner_seed) x metric:
                           PATH-LEVEL PAIRED contrasts (high_penalty minus
                           original, same 500 holdout paths for both --
                           reusing final_analyze_contrasts.py's own paired-
                           bootstrap procedure verbatim) for every inventory-
                           behaviour, monetary-penalty, pre-penalty-wealth,
                           and penalised-objective metric, kept in
                           SEPARATE ROWS per metric so no comparison
                           conflates a behavioural change with a monetary
                           one.

Never evaluates a model or reads a checkpoint file -- purely a read-only
aggregation/statistics pass over the already-produced episode-level CSV,
reproducible at any time.

Run from repo root:
    python ips/ips_aggregate_summary.py
"""
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

import numpy as np
import pandas as pd

import ips.ips_common as IC
from final.final_analyze_contrasts import bootstrap_ci_of_mean, ci_contains_zero, Z_95, BOOTSTRAP_SEED

REFERENCE_POLICIES = ("oracle", "belief_weighted", "frozen_clone")
ALL_POLICIES = tuple(list(IC.ARCHITECTURES) + list(REFERENCE_POLICIES))

# Every field the brief's "Summary statistics" section asks for, plus a
# handful of existing-schema fields retained for continuity with the
# original experiment's own tables.
MEAN_FIELDS = [
    "time_avg_signed_inventory", "time_avg_abs_inventory", "integrated_squared_inventory", "rms_inventory",
    "max_abs_inventory", "terminal_inventory", "abs_terminal_inventory", "squared_terminal_inventory",
    "n_inventory_boundary_contacts",
    "total_bid_fills", "total_ask_fills", "total_fills",
    "terminal_mtm_wealth_pre_penalty", "running_penalty_contribution", "terminal_penalty_contribution",
    "final_realised_objective",
    # existing (event-weighted) thesis definitions, retained for continuity
    "mean_signed_inventory_event_weighted", "mean_abs_inventory_event_weighted",
    "fills", "raw_pnl", "full_objective", "running_penalty", "terminal_penalty",
]

# Metrics carried into the paired original-vs-high-penalty comparison,
# grouped so the LaTeX/report layer can keep behavioural / monetary /
# objective columns visually and analytically distinct (brief: "Do not
# infer that behavioural changes occurred solely from changes in the
# monetary penalty values").
COMPARISON_METRIC_GROUPS = dict(
    inventory_behaviour=[
        "time_avg_signed_inventory", "time_avg_abs_inventory", "integrated_squared_inventory",
        "rms_inventory", "max_abs_inventory", "terminal_inventory", "abs_terminal_inventory",
        "squared_terminal_inventory", "total_fills",
    ],
    monetary_penalty=["running_penalty_contribution", "terminal_penalty_contribution"],
    pre_penalty_wealth=["terminal_mtm_wealth_pre_penalty"],
    penalised_objective=["final_realised_objective"],
)
COMPARISON_METRICS = [m for group in COMPARISON_METRIC_GROUPS.values() for m in group]
METRIC_TO_GROUP = {m: g for g, ms in COMPARISON_METRIC_GROUPS.items() for m in ms}


def build_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (calibration, policy, seed), sub in df.groupby(["penalty_calibration", "policy", "learner_seed"],
                                                          dropna=False):
        row = dict(penalty_calibration=calibration, policy=policy, learner_seed=seed, n_paths=len(sub))
        for field in MEAN_FIELDS:
            row[f"mean_{field}"] = float(sub[field].mean())
        row["sd_final_realised_objective"] = float(sub["final_realised_objective"].std(ddof=1))
        row["se_final_realised_objective"] = row["sd_final_realised_objective"] / np.sqrt(len(sub))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["penalty_calibration", "policy", "learner_seed"], na_position="first")


def build_policy_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for calibration in IC.CALIBRATION_NAMES:
        cal_df = seed_df[seed_df["penalty_calibration"] == calibration]
        ppo_df = cal_df[cal_df["policy"].isin(IC.ARCHITECTURES)]
        for policy, sub in ppo_df.groupby("policy"):
            row = dict(penalty_calibration=calibration, policy=policy, n_seeds=len(sub))
            row["cross_seed_mean_final_realised_objective"] = float(sub["mean_final_realised_objective"].mean())
            row["cross_seed_sd_final_realised_objective"] = (
                float(sub["mean_final_realised_objective"].std(ddof=1)) if len(sub) > 1 else 0.0)
            row["cross_seed_se_final_realised_objective"] = (
                row["cross_seed_sd_final_realised_objective"] / np.sqrt(len(sub)))
            for field in MEAN_FIELDS:
                row[f"cross_seed_mean_{field}"] = float(sub[f"mean_{field}"].mean())
                row[f"cross_seed_sd_{field}"] = float(sub[f"mean_{field}"].std(ddof=1)) if len(sub) > 1 else 0.0
            rows.append(row)

        ref_df = cal_df[cal_df["policy"].isin(REFERENCE_POLICIES)]
        for _, r in ref_df.iterrows():
            row = dict(penalty_calibration=calibration, policy=r["policy"], n_seeds=1)
            row["cross_seed_mean_final_realised_objective"] = r["mean_final_realised_objective"]
            row["cross_seed_sd_final_realised_objective"] = 0.0
            row["cross_seed_se_final_realised_objective"] = r["se_final_realised_objective"]
            for field in MEAN_FIELDS:
                row[f"cross_seed_mean_{field}"] = r[f"mean_{field}"]
                row[f"cross_seed_sd_{field}"] = 0.0
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["penalty_calibration", "policy"])


def build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """PATH-LEVEL paired contrasts (high_penalty - original), reusing
    final_analyze_contrasts.bootstrap_ci_of_mean verbatim -- same
    normal-theory + bootstrap CI construction as the original experiment's
    own benchmark/200k-vs-1m contrast tables."""
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    identities = []
    for arch in IC.ARCHITECTURES:
        for seed in IC.LEARNER_SEEDS:
            identities.append((arch, seed))
    for ref in REFERENCE_POLICIES:
        identities.append((ref, None))

    for policy, seed in identities:
        if seed is None:
            sub_o = df[(df["penalty_calibration"] == "original") & (df["policy"] == policy)]
            sub_h = df[(df["penalty_calibration"] == "high_penalty") & (df["policy"] == policy)]
        else:
            sub_o = df[(df["penalty_calibration"] == "original") & (df["policy"] == policy)
                       & (df["learner_seed"] == seed)]
            sub_h = df[(df["penalty_calibration"] == "high_penalty") & (df["policy"] == policy)
                       & (df["learner_seed"] == seed)]
        if len(sub_o) == 0 or len(sub_h) == 0:
            continue  # this calibration/policy/seed combination hasn't been evaluated yet

        sub_o = sub_o.set_index("holdout_episode_seed").sort_index()
        sub_h = sub_h.set_index("holdout_episode_seed").sort_index()
        common_idx = sub_o.index.intersection(sub_h.index)
        assert len(common_idx) == len(IC.HOLDOUT_SEEDS), (
            f"{policy} seed {seed}: expected {len(IC.HOLDOUT_SEEDS)} paired holdout paths, "
            f"got {len(common_idx)} (original={len(sub_o)}, high_penalty={len(sub_h)})"
        )
        sub_o, sub_h = sub_o.loc[common_idx], sub_h.loc[common_idx]

        for metric in COMPARISON_METRICS:
            diffs = (sub_h[metric] - sub_o[metric]).to_numpy(dtype=float)
            sd = float(diffs.std(ddof=1))
            se = sd / np.sqrt(len(diffs))
            mean_diff = float(diffs.mean())
            normal_lo, normal_hi = mean_diff - Z_95 * se, mean_diff + Z_95 * se
            boot_lo, boot_hi = bootstrap_ci_of_mean(diffs, rng=rng)
            rows.append(dict(
                policy=policy, learner_seed=seed, metric_group=METRIC_TO_GROUP[metric], metric=metric,
                n_paths=len(diffs),
                mean_original=float(sub_o[metric].mean()), mean_high_penalty=float(sub_h[metric].mean()),
                mean_diff_high_minus_original=mean_diff, sd_diff=sd, se_diff=se,
                normal_ci_lo=normal_lo, normal_ci_hi=normal_hi,
                normal_ci_contains_zero=ci_contains_zero(normal_lo, normal_hi),
                bootstrap_ci_lo=boot_lo, bootstrap_ci_hi=boot_hi,
                bootstrap_ci_contains_zero=ci_contains_zero(boot_lo, boot_hi),
                win_rate_high_penalty_larger=float((diffs > 0).mean()),
            ))
    return pd.DataFrame(rows)


def main():
    path = IC.RESULTS_DIR / "episode_level.csv"
    assert path.exists(), f"episode_level.csv not found at {path} -- run ips_evaluate_holdout.py first"
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} episode rows, calibrations present: {sorted(df['penalty_calibration'].unique())}")

    seed_df = build_seed_summary(df)
    seed_path = IC.RESULTS_DIR / "seed_summary.csv"
    seed_df.to_csv(seed_path, index=False)
    print(f"Saved {seed_path} ({len(seed_df)} rows)")

    policy_df = build_policy_summary(seed_df)
    policy_path = IC.RESULTS_DIR / "policy_summary.csv"
    policy_df.to_csv(policy_path, index=False)
    print(f"Saved {policy_path} ({len(policy_df)} rows)")

    if set(["original", "high_penalty"]).issubset(set(df["penalty_calibration"].unique())):
        comparison_df = build_comparison_table(df)
        comparison_path = IC.RESULTS_DIR / "comparison_original_vs_high_penalty.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Saved {comparison_path} ({len(comparison_df)} rows)")
    else:
        print("Both calibrations not yet present in episode_level.csv -- skipping comparison table "
              "(re-run once ips_evaluate_holdout.py has evaluated both).")


if __name__ == "__main__":
    main()
