"""
fig05_risk_return.py
------------------------
Risk-return scatter at the fixed 1,000,000-transition holdout checkpoint
(500 paths, deterministic): one point per (architecture, learner seed) --
y = mean per-path full_objective, x = CVaR_10 (mean of the worst 10% of
per-path objectives) within that (policy, seed). Benchmarks: one point each
(no seed dimension). Both axes in objective units (directly comparable --
CVaR_10 is itself an objective-unit quantity, not a separate risk metric on
a different scale, so this is NOT a dual-axis plot).

Source (read-only): final_holdout_episode_level.csv.
"""
import numpy as np
import pandas as pd

import fig_style as FS
import stats_helpers as SH

# ======================================================================
# CONFIGURATION
# ======================================================================
FIG_NAME = "fig05_risk_return"
FIXED_CHECKPOINT = 1_000_000
CVAR_ALPHA = 0.10

INPUT_CSV = FS.RESULTS_DIR / "final_holdout_episode_level.csv"


def main():
    df = pd.read_csv(INPUT_CSV)
    FS.verify_policy_set(set(df["policy"].unique()), context="fig05 input")

    fixed = df[(df["checkpoint_transition"] == FIXED_CHECKPOINT) & (df["deterministic"] == True)]  # noqa: E712
    bench = df[df["policy"].isin(FS.BENCHMARK_POLICIES) & df["checkpoint_transition"].isna()
               & (df["deterministic"] == True)]  # noqa: E712

    fig, ax = FS.new_figure(width="full", height_in=3.6)
    stats = {"figure": FIG_NAME, "checkpoint": FIXED_CHECKPOINT, "cvar_alpha": CVAR_ALPHA,
              "architectures": {}, "benchmarks": {}}

    for arch in FS.RL_POLICIES:
        xs, ys = [], []
        for seed in range(5):
            vals = fixed[(fixed["policy"] == arch) & (fixed["learner_seed"] == seed)]["full_objective"].to_numpy()
            if len(vals) != 500:
                raise ValueError(f"{arch!r} seed {seed}: expected 500 paths, found {len(vals)}")
            x = SH.cvar(vals, alpha=CVAR_ALPHA)
            y = float(vals.mean())
            xs.append(x)
            ys.append(y)
        ax.scatter(xs, ys, s=55, color=FS.COLORS[arch], marker=FS.MARKERS[arch], edgecolors="black",
                    linewidths=0.5, alpha=0.9, zorder=3, label=FS.display(arch))
        stats["architectures"][arch] = dict(cvar_by_seed=xs, mean_by_seed=ys)

    for b in FS.BENCHMARK_POLICIES:
        vals = bench[bench["policy"] == b]["full_objective"].to_numpy()
        x = SH.cvar(vals, alpha=CVAR_ALPHA)
        y = float(vals.mean())
        ax.scatter([x], [y], s=90, color=FS.COLORS[b], marker=FS.MARKERS[b], edgecolors="black",
                    linewidths=0.7, zorder=4, label=FS.display(b))
        stats["benchmarks"][b] = dict(cvar=x, mean=y)

    ax.set_xlabel(f"CVaR$_{{{int(CVAR_ALPHA*100)}}}$ (mean of worst {int(CVAR_ALPHA*100)}% paths, objective units)")
    ax.set_ylabel("Mean full objective (500 paths)")
    ax.legend(loc="lower right", fontsize=7, ncol=2, frameon=True)
    FS.panel_letter(ax, "a")

    FS.save_figure(fig, FIG_NAME, stats)


if __name__ == "__main__":
    main()
