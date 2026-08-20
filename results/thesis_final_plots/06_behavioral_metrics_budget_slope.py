"""
06_behavioral_metrics_budget_slope.py
----------------------------------------------
Companion figure to 01_training_budget_slope.py: while the full_objective
slope plot shows training budget (200k -> 1M transitions) buys essentially
no statistically detectable NET improvement, the policies are not static in
that range -- every behavioural metric below shifts by a large, consistent
amount across all 5 seeds of every architecture (see
fixed_200k_vs_1m_contrasts.csv: CI excludes zero for 100% of seeds on every
metric here, vs. 0-20% for full_objective). This figure shows that ongoing
behavioural change directly, metric by metric, using the same paired-slope
convention as 01_training_budget_slope.py.

Data source (read-only, never modified):
    results/final_reduced_exploration_architecture_comparison/final_holdout_seed_summary.csv

Same file/convention as 01_training_budget_slope.py -- one row per (policy,
learner_seed, checkpoint_transition), 500 shared final-holdout paths
(seeds 290000-290499), both 200k and 1M fixed checkpoints, plus one
learner_seed=NaN row per analytical benchmark.

For each metric, one thin line per learner seed connects that seed's paired
(200k, 1M) holdout mean (same trained model, two checkpoints, same 500
holdout paths) -- a thicker line shows the cross-seed mean at each
checkpoint. Small multiples, one panel per metric, shared legend.

Run:
    python 06_behavioral_metrics_budget_slope.py

Everything under the "CONFIGURATION" heading below is safe to edit.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ======================================================================
# CONFIGURATION -- edit freely
# ======================================================================
FIGSIZE = (14.0, 8.5)
FONT_SIZE_PANEL_TITLE = 10.5
FONT_SIZE_AXIS_LABEL = 9.5
FONT_SIZE_TICK = 8.5
FONT_SIZE_LEGEND = 9

THIN_LINEWIDTH = 1.0
THIN_ALPHA = 0.5
MEAN_LINEWIDTH = 2.6
MARKER_SIZE = 6.0
THIN_MARKER_SIZE = 3.5
BENCHMARK_LINEWIDTH = 1.2

LEGEND_LOC = "lower center"
OUTPUT_FILENAME = "06_behavioral_metrics_budget_slope.png"
OUTPUT_DPI = 300
SHOW_FIGURE = False

CHECKPOINTS = [200_000, 1_000_000]

# Metric column (in final_holdout_seed_summary.csv, "mean_" prefix already
# included) -> (panel title, show analytical benchmark reference lines).
METRICS = [
    ("mean_fills", "Fills per episode", True),
    ("mean_mean_quoted_spread", "Mean quoted spread", True),
    ("mean_spread_revenue", "Spread revenue", True),
    ("mean_adverse_selection_loss", "Adverse-selection loss", True),
    ("mean_running_penalty", "Running inventory penalty", True),
    ("mean_mean_abs_inventory", "Mean |inventory|", True),
]
N_COLS = 3

BENCHMARKS_TO_SHOW = ["oracle", "belief_weighted", "frozen_clone"]

# Colour-blind-friendly (Okabe-Ito) palette, consistent across all scripts in
# this directory.
ARCHITECTURE_COLORS = {
    "hamilton_ppo": "#0072B2",     # blue
    "return_mlp_ppo": "#E69F00",   # orange
    "return_lstm_ppo": "#009E73",  # bluish green
}
ARCHITECTURE_MARKERS = {
    "hamilton_ppo": "o",
    "return_mlp_ppo": "s",
    "return_lstm_ppo": "^",
}
ARCHITECTURE_LABELS = {
    "hamilton_ppo": "Belief-state PPO",
    "return_mlp_ppo": "Raw-return MLP PPO",
    "return_lstm_ppo": "Raw-return LSTM PPO",
}
# Greyscale, per the thesis-wide reference-policy colour scheme.
BENCHMARK_COLORS = {
    "oracle": "#222222",
    "belief_weighted": "#707070",
    "frozen_clone": "#A6A6A6",
}
BENCHMARK_LINESTYLES = {
    "oracle": "-",
    "belief_weighted": "--",
    "frozen_clone": ":",
}
BENCHMARK_LABELS = {
    "oracle": "Regime-conditioned benchmark",
    "belief_weighted": "Belief-weighted analytical policy",
    "frozen_clone": "Frozen supervised clone",
}

# Panels needing extra y-axis padding so reference-policy lines sitting near
# an edge are clearly visible (not flush against the frame). metric_col -> pad fraction.
YLIM_PAD_PANELS = {
    "mean_mean_quoted_spread": 0.065,
    "mean_spread_revenue": 0.065,
}

GRID_ALPHA = 0.3

# ======================================================================
# Paths (do not edit unless the repo layout changes)
# ======================================================================
PROJECT_ROOT = Path(
    r"C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\mbt_gym"
)
RESULTS_DIR = PROJECT_ROOT / "results" / "final_reduced_exploration_architecture_comparison"
INPUT_CSV = RESULTS_DIR / "final_holdout_seed_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "thesis_final_plots" / "figures"
OUTPUT_PATH = OUTPUT_DIR / OUTPUT_FILENAME

ARCHITECTURES = ["hamilton_ppo", "return_mlp_ppo", "return_lstm_ppo"]
REQUIRED_COLUMNS = ["policy", "learner_seed", "checkpoint_transition"] + [m for m, _, _ in METRICS]


def _validate_inputs() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Required input file not found: {INPUT_CSV}\n"
            "This script only reads already-saved final-holdout results; it never "
            "retrains or re-evaluates a model."
        )
    df = pd.read_csv(INPUT_CSV)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) {missing} in {INPUT_CSV}.\nAvailable columns: {list(df.columns)}"
        )
    return df


def _validate_architecture_coverage(df: pd.DataFrame) -> None:
    arch_rows = df[df["policy"].isin(ARCHITECTURES)]
    for arch in ARCHITECTURES:
        for ckpt in CHECKPOINTS:
            sub = arch_rows[(arch_rows["policy"] == arch) & (arch_rows["checkpoint_transition"] == ckpt)]
            if sub.empty:
                raise ValueError(
                    f"No rows found for architecture={arch!r} at checkpoint_transition={ckpt} in {INPUT_CSV}."
                )


def main() -> None:
    import matplotlib
    if not SHOW_FIGURE:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _validate_inputs()
    _validate_architecture_coverage(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_rows = int(np.ceil(len(METRICS) / N_COLS))
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=FIGSIZE)
    axes_flat = np.array(axes).reshape(-1)

    for ax, (metric_col, panel_title, show_benchmarks) in zip(axes_flat, METRICS):
        for arch in ARCHITECTURES:
            arch_df = df[(df["policy"] == arch) & df["learner_seed"].notna()
                         & df["checkpoint_transition"].isin(CHECKPOINTS)].copy()
            seeds = sorted(arch_df["learner_seed"].unique())

            for seed in seeds:
                seed_df = arch_df[arch_df["learner_seed"] == seed].sort_values("checkpoint_transition")
                if len(seed_df) != len(CHECKPOINTS):
                    raise ValueError(
                        f"{arch!r} learner_seed={seed!r} missing a paired observation for {metric_col!r} "
                        f"at every checkpoint in {CHECKPOINTS}."
                    )
                ax.plot(
                    seed_df["checkpoint_transition"], seed_df[metric_col],
                    color=ARCHITECTURE_COLORS[arch], linewidth=THIN_LINEWIDTH, alpha=THIN_ALPHA,
                    marker=ARCHITECTURE_MARKERS[arch], markersize=THIN_MARKER_SIZE, zorder=2,
                )

            mean_by_ckpt = arch_df.groupby("checkpoint_transition")[metric_col].mean().reindex(CHECKPOINTS)
            ax.plot(
                CHECKPOINTS, mean_by_ckpt.values, color=ARCHITECTURE_COLORS[arch], linewidth=MEAN_LINEWIDTH,
                marker=ARCHITECTURE_MARKERS[arch], markersize=MARKER_SIZE, zorder=4,
                label=ARCHITECTURE_LABELS[arch],
            )

        if show_benchmarks:
            bench_df = df[df["policy"].isin(BENCHMARKS_TO_SHOW) & df["learner_seed"].isna()]
            for bench in BENCHMARKS_TO_SHOW:
                row = bench_df[bench_df["policy"] == bench]
                if not row.empty:
                    value = float(row[metric_col].iloc[0])
                    ax.axhline(value, color=BENCHMARK_COLORS[bench], linestyle=BENCHMARK_LINESTYLES[bench],
                               linewidth=BENCHMARK_LINEWIDTH, alpha=0.85, zorder=1,
                               label=BENCHMARK_LABELS[bench])

        if metric_col in YLIM_PAD_PANELS:
            # Expand the y-axis around the FULL plotted range (seed lines, mean
            # lines, AND the benchmark axhlines) so reference-policy lines
            # sitting near an edge get clear separation from the frame,
            # without arbitrarily compressing the learned-policy variation.
            all_y = np.concatenate([line.get_ydata() for line in ax.get_lines()])
            y_min, y_max = float(np.nanmin(all_y)), float(np.nanmax(all_y))
            pad = YLIM_PAD_PANELS[metric_col] * (y_max - y_min)
            ax.set_ylim(y_min - pad, y_max + pad)

        ax.set_xticks(CHECKPOINTS)
        ax.set_xticklabels([f"{c:,}" for c in CHECKPOINTS], fontsize=FONT_SIZE_TICK)
        x_pad = 0.08 * (CHECKPOINTS[-1] - CHECKPOINTS[0])
        ax.set_xlim(CHECKPOINTS[0] - x_pad, CHECKPOINTS[-1] + x_pad)
        ax.set_title(panel_title, fontsize=FONT_SIZE_PANEL_TITLE)
        ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
        ax.grid(True, alpha=GRID_ALPHA, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_flat[len(METRICS):]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Explicit order (architectures, then reference policies). Interleaved so
    # matplotlib's column-major ncol=3 fill renders it as two natural rows:
    #   row 0: Belief-state PPO | Raw-return MLP PPO | Raw-return LSTM PPO
    #   row 1: Regime-conditioned benchmark | Belief-weighted analytical policy | Frozen supervised clone
    legend_order = [
        "Belief-state PPO", "Regime-conditioned benchmark",
        "Raw-return MLP PPO", "Belief-weighted analytical policy",
        "Raw-return LSTM PPO", "Frozen supervised clone",
    ]
    missing = [lbl for lbl in legend_order if lbl not in by_label]
    if missing:
        raise ValueError(f"Legend label(s) {missing} not found among plotted handles: {list(by_label)}")
    ordered_handles = [by_label[lbl] for lbl in legend_order]
    fig.legend(ordered_handles, legend_order, loc=LEGEND_LOC, fontsize=FONT_SIZE_LEGEND, frameon=True,
               ncol=3, bbox_to_anchor=(0.5, -0.02))

    # No internal title/subtitle -- the LaTeX caption provides them.
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.98))

    fig.savefig(OUTPUT_PATH, dpi=OUTPUT_DPI, bbox_inches="tight")
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)

    print(f"Input file used : {INPUT_CSV}")
    print(f"Output figure   : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
