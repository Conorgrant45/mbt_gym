"""
fig01_learning_curves.py
--------------------------
Validation-set learning curves across all 21 checkpoints (16,000-1,000,000
transitions), for all three architectures including hamilton_ppo (Phase 7
checkpoints re-evaluated on this experiment's own 50-path validation set --
see DATA_INVENTORY.md §4). Monitoring-set data only, never used for
checkpoint selection or as a final result.

Source (read-only): validation_seed_summary.csv, validation_benchmark_summary.csv.

Relationship to the pre-existing 01_training_budget_slope.py: that script
plots only the two fixed HOLDOUT checkpoints (200k, 1M) from the 500-path
holdout set. This figure is different in both data (validation set, all 21
checkpoints) and purpose (a genuine learning curve, not the pre-registered
200k-vs-1M holdout contrast) -- it is not a duplicate.

CONFIGURATION block below is safe to edit.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import fig_style as FS
import stats_helpers as SH  # noqa: F401 (imported for house-style consistency; no stats computed here)

# ======================================================================
# CONFIGURATION
# ======================================================================
FIG_NAME = "fig01_learning_curves"
Y_LABEL = "Validation objective (mean, 50 paths)"
X_LABEL = "Training transitions"
CHECKPOINT_MARKERS = [200_000, 1_000_000]

INPUT_SEED_SUMMARY = FS.RESULTS_DIR / "validation_seed_summary.csv"
INPUT_BENCHMARK_SUMMARY = FS.RESULTS_DIR / "validation_benchmark_summary.csv"

EXPECTED_N_CHECKPOINTS = 21
EXPECTED_N_SEEDS = 5


def _validate(df: pd.DataFrame, bench_df: pd.DataFrame) -> None:
    FS.verify_policy_set(set(df["architecture"].unique()) | set(bench_df["policy"].unique()),
                          context="fig01 inputs")
    for arch in FS.RL_POLICIES:
        arch_df = df[df["architecture"] == arch]
        n_ckpts = arch_df["checkpoint_transition"].nunique()
        if n_ckpts != EXPECTED_N_CHECKPOINTS:
            raise ValueError(f"{arch!r}: expected {EXPECTED_N_CHECKPOINTS} checkpoints, found {n_ckpts}")
        for seed in range(EXPECTED_N_SEEDS):
            seed_df = arch_df[arch_df["learner_seed"] == seed]
            if seed_df["checkpoint_transition"].nunique() != EXPECTED_N_CHECKPOINTS:
                raise ValueError(f"{arch!r} seed {seed}: missing checkpoints -- "
                                  f"found {sorted(seed_df['checkpoint_transition'].unique())}")
    # Hamilton presence is the specific thing worth asserting per the brief
    # (its rows are re-evaluated Phase 7 checkpoints, not freshly trained --
    # confirm they made it into this file at all).
    assert "hamilton_ppo" in df["architecture"].unique(), \
        "hamilton_ppo rows missing from validation_seed_summary.csv"
    for bench in FS.BENCHMARK_POLICIES:
        if bench not in bench_df["policy"].unique():
            raise ValueError(f"Benchmark {bench!r} missing from {INPUT_BENCHMARK_SUMMARY}")


def main():
    df = pd.read_csv(INPUT_SEED_SUMMARY)
    bench_df = pd.read_csv(INPUT_BENCHMARK_SUMMARY)
    _validate(df, bench_df)

    fig, ax = FS.new_figure(width="full", height_in=3.6)
    stats = {"figure": FIG_NAME, "source_files": [str(INPUT_SEED_SUMMARY), str(INPUT_BENCHMARK_SUMMARY)],
              "architectures": {}, "benchmarks": {}}

    for arch in FS.RL_POLICIES:
        arch_df = df[df["architecture"] == arch].sort_values("checkpoint_transition")
        ckpts = sorted(arch_df["checkpoint_transition"].unique())
        cross_seed_mean = arch_df.groupby("checkpoint_transition")["mean_full_objective"].mean().reindex(ckpts)
        cross_seed_min = arch_df.groupby("checkpoint_transition")["mean_full_objective"].min().reindex(ckpts)
        cross_seed_max = arch_df.groupby("checkpoint_transition")["mean_full_objective"].max().reindex(ckpts)

        ax.fill_between(ckpts, cross_seed_min.values, cross_seed_max.values,
                         color=FS.COLORS[arch], alpha=0.15, linewidth=0, zorder=2)
        ax.plot(ckpts, cross_seed_mean.values, color=FS.COLORS[arch], linestyle=FS.LINESTYLES[arch],
                 linewidth=FS.LINEWIDTHS[arch], marker=FS.MARKERS[arch], markersize=3.5, zorder=3,
                 label=FS.display(arch))

        stats["architectures"][arch] = {
            "checkpoints": [int(c) for c in ckpts],
            "cross_seed_mean": [float(v) for v in cross_seed_mean.values],
            "cross_seed_min": [float(v) for v in cross_seed_min.values],
            "cross_seed_max": [float(v) for v in cross_seed_max.values],
        }

    for bench in FS.BENCHMARK_POLICIES:
        row = bench_df[bench_df["policy"] == bench]
        value = float(row["mean_full_objective"].iloc[0])
        ax.axhline(value, color=FS.COLORS[bench], linestyle=FS.LINESTYLES[bench],
                    linewidth=FS.LINEWIDTHS[bench], zorder=1, label=FS.display(bench))
        stats["benchmarks"][bench] = value

    for marker_t in CHECKPOINT_MARKERS:
        ax.axvline(marker_t, color="0.5", linestyle=":", linewidth=0.8, zorder=0)

    ax.set_xscale("log")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_xticks([16_000, 200_000, 1_000_000])
    ax.set_xticklabels(["16k", "200k", "1M"])
    ax.legend(loc="lower right", ncol=2, frameon=True)
    FS.panel_letter(ax, "a")

    FS.save_figure(fig, FIG_NAME, stats)


if __name__ == "__main__":
    main()
