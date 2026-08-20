"""
phase7_post_training/phase7_plot_validation_learning_curve.py
------------------------------------------------
Phase 7 post-training analysis, Part A: plot deterministic validation
objective against cumulative training transitions for all 5 Group B
learner seeds, using ONLY the already-completed Phase 7 checkpoint
validation records (phase7_checkpoint_summary.csv) and the already-computed
analytical/clone benchmark evaluations on the SAME validation seeds
(phase7_benchmark_on_own_validation_seeds.csv). No retraining, no new
environment evaluation, no change to checkpoint selection -- this script
only reads existing Phase 7 CSVs and writes new files under
post_training_analysis/.

Run from repo root:
    python phase7_post_training/phase7_plot_validation_learning_curve.py
"""
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shared.phase7_common as P7
import phase7_post_training.phase7_post_training_common as PC

Z_975 = 1.959963984540054
SEED_COLORS = {0: "tab:red", 1: "tab:orange", 2: "tab:green", 3: "tab:blue", 4: "tab:purple"}
BENCHMARK_STYLES = {
    "oracle": dict(color="black", ls="-", lw=1.6, label="analytical oracle"),
    "belief_weighted": dict(color="dimgray", ls="--", lw=1.6, label="belief-weighted analytical"),
    "frozen_clone": dict(color="saddlebrown", ls=":", lw=1.8, label="frozen supervised clone"),
}


def main():
    PC.POST_DIR.mkdir(parents=True, exist_ok=True)

    cp = pd.read_csv(P7.RESULTS_DIR / "phase7_checkpoint_summary.csv")
    bench = pd.read_csv(P7.RESULTS_DIR / "phase7_benchmark_on_own_validation_seeds.csv")

    benchmark_means = {}
    for policy in ("oracle", "belief_weighted", "frozen_clone"):
        vals = bench[bench["policy"] == policy]["full_objective"]
        benchmark_means[policy] = float(vals.mean())
    print("Benchmark means on Phase 7 validation seeds:", benchmark_means)

    selections = {seed: PC.select_checkpoint_for_seed(seed, cp) for seed in P7.LEARNER_SEEDS}
    for seed, sel in selections.items():
        print(f"Seed {seed}: selected checkpoint t={sel['timestep']} "
              f"(det_mean_objective={sel['det_mean_objective']:.3f})")

    # --- Cross-seed mean/CI at every common checkpoint (all 5 seeds share
    # the same 21 checkpoint timesteps by construction) ---
    cross_seed_rows = []
    for t in P7.CHECKPOINT_TIMESTEPS:
        sub = cp[cp["timestep"] == t]
        assert len(sub) == len(P7.LEARNER_SEEDS), f"t={t}: expected {len(P7.LEARNER_SEEDS)} seeds, got {len(sub)}"
        vals = sub["det_mean_objective"].to_numpy()
        mean = float(vals.mean())
        se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
        cross_seed_rows.append(dict(timestep=t, cross_seed_mean=mean, cross_seed_se=se))
    cross_seed_df = pd.DataFrame(cross_seed_rows).sort_values("timestep")

    # --- Output CSV (exact schema requested) ---
    out_rows = []
    for _, row in cp.iterrows():
        seed = int(row["learner_seed"])
        is_selected = int(row["timestep"]) == int(selections[seed]["timestep"])
        out_rows.append(dict(
            learner_seed=seed,
            timestep=int(row["timestep"]),
            elapsed_training_seconds=float(row["wall_clock_elapsed_seconds"]),
            deterministic_validation_mean=float(row["det_mean_objective"]),
            deterministic_validation_se=float(row["det_se_objective"]),
            selected_checkpoint=bool(is_selected),
            oracle_mean=benchmark_means["oracle"],
            belief_weighted_mean=benchmark_means["belief_weighted"],
            frozen_clone_mean=benchmark_means["frozen_clone"],
        ))
    out_df = pd.DataFrame(out_rows).sort_values(["learner_seed", "timestep"])
    csv_path = PC.POST_DIR / "phase7_validation_learning_curve_data.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} ({len(out_df)} rows)")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(11, 7))

    # Raw per-seed observations (preserved, not smoothed).
    for seed in P7.LEARNER_SEEDS:
        sub = cp[cp["learner_seed"] == seed].sort_values("timestep")
        ax.plot(sub["timestep"], sub["det_mean_objective"], marker="o", markersize=4,
                color=SEED_COLORS[seed], alpha=0.85, lw=1.3, label=f"seed {seed}")
        sel_t = selections[seed]["timestep"]
        sel_row = sub[sub["timestep"] == sel_t].iloc[0]
        ax.scatter([sel_t], [sel_row["det_mean_objective"]], marker="*", s=260,
                   color=SEED_COLORS[seed], edgecolor="black", zorder=5,
                   label=f"seed {seed} selected (t={sel_t:,})")

    # Cross-seed mean + 95% CI band.
    ax.plot(cross_seed_df["timestep"], cross_seed_df["cross_seed_mean"], color="black", lw=2.4,
             label="cross-seed mean")
    ax.fill_between(cross_seed_df["timestep"],
                     cross_seed_df["cross_seed_mean"] - Z_975 * cross_seed_df["cross_seed_se"],
                     cross_seed_df["cross_seed_mean"] + Z_975 * cross_seed_df["cross_seed_se"],
                     color="black", alpha=0.12, label="cross-seed 95% CI (mean +/- 1.96*SE)")

    # Vertical reference at 200,000 transitions.
    ax.axvline(200_000, color="gray", ls="-.", lw=1.2, label="200,000 transitions (Phase 6 budget)")

    # Benchmark horizontal lines.
    for policy, style in BENCHMARK_STYLES.items():
        ax.axhline(benchmark_means[policy], **style)

    ax.set_xlabel("Cumulative training transitions")
    ax.set_ylabel("Deterministic validation objective")
    ax.set_title("Phase 7 Group B: deterministic validation objective vs. training transitions\n"
                 "(raw per-seed checkpoints preserved; stars = each seed's selected checkpoint)")
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    png_path = PC.POST_DIR / "phase7_validation_learning_curve.png"
    pdf_path = PC.POST_DIR / "phase7_validation_learning_curve.pdf"
    fig.savefig(png_path, dpi=140)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")

    # --- Summary markdown ---
    write_summary(cp, cross_seed_df, benchmark_means, selections)


def write_summary(cp: pd.DataFrame, cross_seed_df: pd.DataFrame, benchmark_means: dict, selections: dict):
    lines = []
    lines.append("# Phase 7 validation learning curve: summary\n")
    lines.append(
        "Source data: `phase7_checkpoint_summary.csv` (21 checkpoints x 5 seeds, deterministic "
        "validation objective on Phase 7's own fixed 50-seed validation set, seeds 260000-260049) "
        "and `phase7_benchmark_on_own_validation_seeds.csv` (oracle/belief-weighted/frozen-clone "
        "evaluated on the SAME 50 validation seeds). No retraining or new evaluation was performed "
        "to produce this plot/summary.\n"
    )
    lines.append(f"Benchmark means on these validation seeds: oracle={benchmark_means['oracle']:.3f}, "
                 f"belief_weighted={benchmark_means['belief_weighted']:.3f}, "
                 f"frozen_clone={benchmark_means['frozen_clone']:.3f}.\n")

    lines.append("## Selected checkpoint per seed (validation-only selection, Phase 6's rule reused)\n")
    for seed, sel in selections.items():
        lines.append(f"- Seed {seed}: t={sel['timestep']:,}, det_mean_objective={sel['det_mean_objective']:.3f}, "
                     f"det_se={sel['det_se_objective']:.3f}")

    lines.append("\n## Descriptive evidence: still improving beyond 200,000 transitions?\n")
    at_200k = cross_seed_df[cross_seed_df["timestep"] == 200_000]["cross_seed_mean"].iloc[0]
    after_200k = cross_seed_df[cross_seed_df["timestep"] > 200_000]
    best_after = float(after_200k["cross_seed_mean"].max())
    best_after_t = int(after_200k.loc[after_200k["cross_seed_mean"].idxmax(), "timestep"])
    final_1m = float(cross_seed_df[cross_seed_df["timestep"] == P7.TOTAL_TRANSITIONS]["cross_seed_mean"].iloc[0])
    lines.append(
        f"**Descriptive observation**: the cross-seed mean deterministic validation objective at "
        f"200,000 transitions was {at_200k:.3f}; it reaches a maximum of {best_after:.3f} at "
        f"t={best_after_t:,} transitions, and its final (1,000,000-transition) value is {final_1m:.3f} "
        f"-- both clearly above the 200,000-transition value. This is descriptive evidence that mean "
        f"validation performance continued to rise materially beyond 200,000 transitions in this run. "
        f"It is NOT a claim that the policy has converged or that this trend would continue "
        f"indefinitely -- no asymptotic/convergence test is performed here (see "
        f"phase7_groupB_1m_convergence_report.md for the slope analysis over the final 200,000 "
        f"transitions, which found the rate of improvement had become close to flat by the end of "
        f"the run)."
    )

    lines.append("\n## Earliest benchmark-level checkpoint per seed\n")
    lines.append(
        "Using the belief-weighted benchmark and the same statistical-indistinguishability "
        "criterion as `phase7_time_to_benchmark.csv` (PPO mean >= benchmark mean - "
        "1.96*sqrt(PPO_SE^2 + benchmark_SE^2), confirmed by >=2 of the next 3 checkpoints): every "
        "one of the 5 seeds first qualifies at the earliest evaluated checkpoint, 16,000 "
        "transitions (see that file for the full per-seed detail and the important caveat that "
        "this reflects the belief-weighted benchmark's own wide per-episode uncertainty, not "
        "convergence to its point estimate)."
    )
    lines.append(
        "\n**Distinguishing descriptive evidence from convergence claims**: the statement 'mean "
        "validation objective is higher at 1,000,000 transitions than at 200,000 transitions' is "
        "descriptive and directly measured. The statement 'this policy has converged' is an "
        "inferential claim this analysis does NOT make -- the near-flat slope in the final 200,000 "
        "transitions (established in the main Phase 7 report) is consistent with either an "
        "approaching plateau or a temporary slow patch, and this plot alone cannot distinguish "
        "those without further training."
    )

    summary_path = PC.POST_DIR / "phase7_validation_learning_curve_summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
