"""
fig02_holdout_ecdf.py
------------------------
Empirical CDF of episode-level full_objective at the fixed 1,000,000-
transition holdout checkpoint (500 paths), deterministic evaluation only.

Per architecture: one ECDF per learner seed (500 episodes each) -> pointwise
MEDIAN across the 5 seed-ECDFs, plus a pointwise min-max envelope. Benchmarks
(no seed dimension): a single 500-episode ECDF each. Two panels sharing the
probability axis: (a) full range, (b) lower tail F<=0.1 with expanded x, to
make loss-tail behaviour legible (see the outlier-path investigation in
FIGURES_REPORT.md -- a handful of extreme-tail episodes on specific holdout
paths otherwise stretch the full-range x-axis to the point of hiding the
central mass, as the earlier 04_objective_ecdf.py's caveat about this same
data already documents).

Source (read-only): final_holdout_episode_level.csv.

Relationship to 04_objective_ecdf.py: that script uses the pointwise MEAN
across seeds with a configurable band type. This figure uses the MEDIAN with
a min-max envelope (per this deliverable's spec) and adds the explicit
lower-tail panel -- a distinct, complementary figure, not a duplicate.

CONFIGURATION block below is safe to edit.
"""
import numpy as np
import pandas as pd

import fig_style as FS
import stats_helpers as SH  # noqa: F401

# ======================================================================
# CONFIGURATION
# ======================================================================
FIG_NAME = "fig02_holdout_ecdf"
FIXED_CHECKPOINT = 1_000_000
N_GRID_POINTS = 800
LOWER_TAIL_F = 0.10

INPUT_CSV = FS.RESULTS_DIR / "final_holdout_episode_level.csv"

EXPECTED_PATHS_PER_CURVE = 500


def _ecdf_at_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    sorted_vals = np.sort(values)
    return np.searchsorted(sorted_vals, grid, side="right") / len(sorted_vals)


def main():
    df = pd.read_csv(INPUT_CSV)
    FS.verify_policy_set(set(df["policy"].unique()), context="fig02 input")

    fixed = df[(df["checkpoint_transition"] == FIXED_CHECKPOINT) & (df["deterministic"] == True)]  # noqa: E712
    bench = df[df["policy"].isin(FS.BENCHMARK_POLICIES) & df["checkpoint_transition"].isna()
               & (df["deterministic"] == True)]  # noqa: E712

    # --- validation: 500 paths per curve ---
    for arch in FS.RL_POLICIES:
        for seed in range(5):
            n = len(fixed[(fixed["policy"] == arch) & (fixed["learner_seed"] == seed)])
            if n != EXPECTED_PATHS_PER_CURVE:
                raise ValueError(f"{arch!r} seed {seed}: expected {EXPECTED_PATHS_PER_CURVE} holdout "
                                  f"paths at checkpoint {FIXED_CHECKPOINT}, found {n}")
    for b in FS.BENCHMARK_POLICIES:
        n = len(bench[bench["policy"] == b])
        if n != EXPECTED_PATHS_PER_CURVE:
            raise ValueError(f"{b!r}: expected {EXPECTED_PATHS_PER_CURVE} holdout paths, found {n}")

    all_vals = pd.concat([fixed["full_objective"], bench["full_objective"]])
    grid_full = np.linspace(all_vals.min(), all_vals.max(), N_GRID_POINTS)

    fig, (ax_full, ax_tail) = FS.new_figure(width="full", height_in=3.4, ncols=2, nrows=1)
    stats = {"figure": FIG_NAME, "source_file": str(INPUT_CSV), "checkpoint": FIXED_CHECKPOINT,
              "architectures": {}, "benchmarks": {}, "envelope_contains_median_check": {}}

    for ax, is_tail in [(ax_full, False), (ax_tail, True)]:
        grid = grid_full
        for arch in FS.RL_POLICIES:
            seed_ecdfs = np.stack([
                _ecdf_at_grid(fixed[(fixed["policy"] == arch) & (fixed["learner_seed"] == s)]
                               ["full_objective"].to_numpy(), grid)
                for s in range(5)
            ])
            median_ecdf = np.median(seed_ecdfs, axis=0)
            env_lo = seed_ecdfs.min(axis=0)
            env_hi = seed_ecdfs.max(axis=0)

            if not is_tail:
                stats["architectures"][arch] = dict(
                    median_at_zero=float(np.interp(0.0, grid, median_ecdf)),
                )
                stats["envelope_contains_median_check"][arch] = bool(
                    np.all(env_lo <= median_ecdf + 1e-12) and np.all(median_ecdf <= env_hi + 1e-12)
                )

            ax.fill_between(grid, env_lo, env_hi, color=FS.COLORS[arch], alpha=0.15, linewidth=0, zorder=2)
            ax.plot(grid, median_ecdf, color=FS.COLORS[arch], linestyle=FS.LINESTYLES[arch],
                     linewidth=FS.LINEWIDTHS[arch], zorder=3,
                     label=FS.display(arch) if not is_tail else None)

        for b in FS.BENCHMARK_POLICIES:
            values = bench[bench["policy"] == b]["full_objective"].to_numpy()
            ecdf = _ecdf_at_grid(values, grid)
            ax.plot(grid, ecdf, color=FS.COLORS[b], linestyle=FS.LINESTYLES[b], linewidth=FS.LINEWIDTHS[b],
                     zorder=4, label=FS.display(b) if not is_tail else None)

        ax.axvline(0.0, color="firebrick", linewidth=0.8, alpha=0.7, zorder=1)
        ax.set_xlabel("Episode full objective")
        if is_tail:
            ax.set_ylim(0.0, LOWER_TAIL_F)
            # expand x to the region actually spanned by the lower tail
            lo_x = min(
                np.percentile(fixed[fixed["policy"] == arch]["full_objective"], 100 * LOWER_TAIL_F * 0.3)
                for arch in FS.RL_POLICIES
            )
            hi_x = max(
                np.percentile(fixed[fixed["policy"] == arch]["full_objective"], 100 * LOWER_TAIL_F * 1.5)
                for arch in FS.RL_POLICIES
            )
            ax.set_xlim(lo_x, hi_x)
            FS.panel_letter(ax, "b")
        else:
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Empirical cumulative probability")
            FS.panel_letter(ax, "a")

    ax_full.legend(loc="upper left", bbox_to_anchor=(0.0, 0.97), fontsize=6.5, ncol=1, frameon=True)

    fig.suptitle("Holdout objective distribution: full range and lower-tail zoom",
                 fontsize=9.5, fontweight="bold", y=1.06)

    FS.save_figure(fig, FIG_NAME, stats, bbox_inches="tight")


if __name__ == "__main__":
    main()
