"""
scripts/plot_final_architecture_results.py
--------------------------------------------------
Standalone, READ-ONLY plotting script for the final reduced-exploration
architecture-comparison experiment. Reads already-produced CSVs from
--results-dir; never trains or evaluates a model; never hard-codes a
numerical result (every number plotted comes from a CSV cell).

Run from repo root (PowerShell), e.g.:

    python scripts/plot_final_architecture_results.py `
        --results-dir results/final_reduced_exploration_architecture_comparison `
        --output-dir results/final_reduced_exploration_architecture_comparison/plots `
        --format both --dpi 150

See `python scripts/plot_final_architecture_results.py --help` for the full
CLI surface (confidence-band type, optional visual-only smoothing, seed/
benchmark visibility toggles, metric selection, x-axis range, legend
position, output filename prefix, figure size, font size).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Z_95 = 1.959963984540054

ARCHITECTURES = ("hamilton_ppo", "return_mlp_ppo", "return_lstm_ppo")
ARCHITECTURE_LABELS = {"hamilton_ppo": "Hamilton belief-state MLP", "return_mlp_ppo": "Raw-return MLP",
                        "return_lstm_ppo": "Raw-return LSTM"}
ARCHITECTURE_COLORS = {"hamilton_ppo": "tab:blue", "return_mlp_ppo": "tab:orange", "return_lstm_ppo": "tab:green"}
SEED_COLORS = {0: "tab:red", 1: "tab:orange", 2: "tab:green", 3: "tab:blue", 4: "tab:purple"}
BENCHMARK_STYLES = {
    "oracle": dict(color="black", ls="-", lw=1.6, label="analytical oracle"),
    "belief_weighted": dict(color="dimgray", ls="--", lw=1.6, label="belief-weighted analytical"),
    "frozen_clone": dict(color="saddlebrown", ls=":", lw=1.8, label="frozen supervised clone"),
}

BEHAVIOURAL_METRICS = [
    ("mean_fills", "Fills per episode"),
    ("mean_fill_to_arrival_ratio", "Fill-to-arrival ratio"),
    ("mean_mean_quoted_spread", "Mean quoted spread"),
    ("mean_spread_revenue", "Spread revenue"),
    ("mean_adverse_selection_loss", "Adverse-selection loss"),
    ("mean_mean_abs_inventory", "Mean |inventory|"),
    ("mean_running_penalty", "Running inventory penalty"),
    ("mean_action_std_bid", "Learned action std (bid)"),
    ("mean_near_bound_rate_stochastic", "Stochastic near-bound rate"),
]

DECOMPOSITION_METRICS = [
    ("mean_spread_revenue", "Spread revenue"),
    ("mean_adverse_selection_loss", "Adverse-selection loss"),
    ("mean_running_penalty", "Running inventory penalty"),
    ("mean_terminal_penalty", "Terminal inventory penalty"),
    ("mean_raw_pnl", "Raw PnL"),
    ("mean_full_objective", "Full objective"),
]


# ======================================================================
# CLI
# ======================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True, type=Path, help="Directory containing the experiment's CSVs.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory to write plots to (default: <results-dir>/plots).")
    p.add_argument("--format", choices=["png", "pdf", "both"], default="png", help="Output image format(s).")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI (png only).")
    p.add_argument("--fig-width", type=float, default=11.0, help="Figure width in inches.")
    p.add_argument("--fig-height", type=float, default=7.0, help="Figure height in inches.")
    p.add_argument("--font-size", type=float, default=10.0, help="Base font size.")
    p.add_argument("--ci-band", choices=["se", "se95"], default="se95",
                   help="Confidence-band type: 'se' = +/-1 SE, 'se95' = +/-1.96 SE (default).")
    p.add_argument("--smooth", action="store_true",
                   help="Apply VISUAL-ONLY rolling smoothing to learning curves (default OFF, unsmoothed). "
                        "Does not alter any underlying data or statistic -- smoothed curves are clearly "
                        "labelled 'rolling mean, visual only' in the legend.")
    p.add_argument("--smooth-window", type=int, default=3, help="Rolling-smoothing window (# checkpoints).")
    p.add_argument("--hide-seeds", action="store_true", help="Hide individual per-seed learning-curve lines.")
    p.add_argument("--hide-benchmarks", action="store_true", help="Hide analytical/clone benchmark reference lines.")
    p.add_argument("--metrics", nargs="+", default=None,
                   help="Restrict the combined behavioural-metric plot (plot 5) to this subset of metric "
                        "column names (see BEHAVIOURAL_METRICS in this script for valid names).")
    p.add_argument("--xmin", type=float, default=None, help="Learning-curve x-axis minimum (transitions).")
    p.add_argument("--xmax", type=float, default=None, help="Learning-curve x-axis maximum (transitions).")
    p.add_argument("--legend-loc", type=str, default="lower right", help="Matplotlib legend location string.")
    p.add_argument("--prefix", type=str, default="final_", help="Output filename prefix.")
    p.add_argument("--checkpoint-markers", nargs="+", type=float, default=[200_000, 1_000_000],
                   help="Vertical reference lines on learning-curve plots, in transitions "
                        "(default: 200,000 and 1,000,000, this experiment's two fixed checkpoints).")
    return p


# ======================================================================
# Data loading (read-only)
# ======================================================================
def load_data(results_dir: Path) -> dict:
    def _read(name, required=True):
        p = results_dir / name
        if not p.exists():
            if required:
                raise FileNotFoundError(f"Required file not found: {p}")
            return None
        return pd.read_csv(p)

    return dict(
        validation_seed=_read("validation_seed_summary.csv"),
        validation_arch=_read("validation_architecture_summary.csv"),
        validation_benchmark=_read("validation_benchmark_summary.csv", required=False),
        holdout_seed=_read("final_holdout_seed_summary.csv", required=False),
        holdout_arch=_read("final_holdout_architecture_summary.csv"),
        fixed_200k_1m=_read("fixed_200k_vs_1m_contrasts.csv"),
        benchmark_paired=_read("benchmark_paired_contrasts.csv"),
        architecture_contrasts=_read("architecture_contrasts.csv"),
    )


def rolling_smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def savefig(fig, output_dir: Path, filename_stem: str, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.format in ("png", "both"):
        fig.savefig(output_dir / f"{filename_stem}.png", dpi=args.dpi)
    if args.format in ("pdf", "both"):
        fig.savefig(output_dir / f"{filename_stem}.pdf")
    plt.close(fig)
    print(f"Saved {filename_stem} ({args.format})")


# ======================================================================
# Plots 1-3: per-architecture learning curves
# ======================================================================
def plot_architecture_learning_curve(architecture: str, data: dict, args, output_dir: Path):
    seed_df = data["validation_seed"]
    arch_df = data["validation_arch"]
    sub_seed = seed_df[seed_df["architecture"] == architecture].sort_values(["learner_seed", "checkpoint_transition"])
    sub_arch = arch_df[arch_df["architecture"] == architecture].sort_values("checkpoint_transition")

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    plt.rcParams.update({"font.size": args.font_size})

    if not args.hide_seeds:
        for seed in sorted(sub_seed["learner_seed"].unique()):
            s = sub_seed[sub_seed["learner_seed"] == seed]
            y = s["mean_full_objective"]
            label = f"seed {seed}"
            if args.smooth:
                y = rolling_smooth(y, args.smooth_window)
                label += " (rolling mean, visual only)"
            ax.plot(s["checkpoint_transition"], y, marker="o", markersize=3, lw=1.1, alpha=0.8,
                    color=SEED_COLORS.get(seed, None), label=label)

    y_mean = sub_arch["cross_seed_mean_full_objective"]
    se_col = "cross_seed_se_full_objective"
    band_mult = Z_95 if args.ci_band == "se95" else 1.0
    if args.smooth:
        y_mean_plot = rolling_smooth(y_mean, args.smooth_window)
        mean_label = "cross-seed mean (rolling mean, visual only)"
    else:
        y_mean_plot = y_mean
        mean_label = "cross-seed mean"
    ax.plot(sub_arch["checkpoint_transition"], y_mean_plot, color="black", lw=2.2, label=mean_label)
    ax.fill_between(sub_arch["checkpoint_transition"],
                     y_mean - band_mult * sub_arch[se_col], y_mean + band_mult * sub_arch[se_col],
                     color="black", alpha=0.15,
                     label=f"cross-seed {'95% CI' if args.ci_band == 'se95' else '+/-1 SE'}")

    if not args.hide_benchmarks and data["validation_benchmark"] is not None:
        vb = data["validation_benchmark"]
        for policy, style in BENCHMARK_STYLES.items():
            row = vb[vb["policy"] == policy]
            if len(row):
                ax.axhline(float(row["mean_full_objective"].iloc[0]), **style)

    for i, marker_t in enumerate(args.checkpoint_markers):
        ax.axvline(marker_t, color="gray", ls="-." if i == 0 else ":", lw=1.2, label=f"{marker_t:,.0f} transitions")

    if args.xmin is not None or args.xmax is not None:
        ax.set_xlim(left=args.xmin, right=args.xmax)

    ax.set_xlabel("Cumulative training transitions")
    ax.set_ylabel("Deterministic validation objective")
    ax.set_title(f"{ARCHITECTURE_LABELS[architecture]}: validation objective vs. training transitions")
    ax.legend(fontsize=max(args.font_size - 2.5, 6), ncol=2, loc=args.legend_loc)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, output_dir, f"{args.prefix}learning_curve_{architecture}", args)


# ======================================================================
# Plot 4: combined objective learning curve, all 3 architectures
# ======================================================================
def plot_combined_objective_curve(data: dict, args, output_dir: Path):
    arch_df = data["validation_arch"]
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    plt.rcParams.update({"font.size": args.font_size})

    band_mult = Z_95 if args.ci_band == "se95" else 1.0
    for architecture in ARCHITECTURES:
        sub = arch_df[arch_df["architecture"] == architecture].sort_values("checkpoint_transition")
        y = sub["cross_seed_mean_full_objective"]
        se = sub["cross_seed_se_full_objective"]
        label = ARCHITECTURE_LABELS[architecture]
        if args.smooth:
            y_plot = rolling_smooth(y, args.smooth_window)
            label += " (rolling mean, visual only)"
        else:
            y_plot = y
        color = ARCHITECTURE_COLORS[architecture]
        ax.plot(sub["checkpoint_transition"], y_plot, color=color, lw=2.2, marker="o", markersize=3, label=label)
        ax.fill_between(sub["checkpoint_transition"], y - band_mult * se, y + band_mult * se, color=color, alpha=0.15)

    if not args.hide_benchmarks and data["validation_benchmark"] is not None:
        vb = data["validation_benchmark"]
        for policy, style in BENCHMARK_STYLES.items():
            row = vb[vb["policy"] == policy]
            if len(row):
                ax.axhline(float(row["mean_full_objective"].iloc[0]), **style)

    for i, marker_t in enumerate(args.checkpoint_markers):
        ax.axvline(marker_t, color="gray", ls="-." if i == 0 else ":", lw=1.2, label=f"{marker_t:,.0f} transitions")
    if args.xmin is not None or args.xmax is not None:
        ax.set_xlim(left=args.xmin, right=args.xmax)

    ax.set_xlabel("Cumulative training transitions")
    ax.set_ylabel("Deterministic validation objective (cross-seed mean)")
    ax.set_title("Combined validation-objective learning curve, all 3 architectures")
    ax.legend(fontsize=max(args.font_size - 2, 7), ncol=2, loc=args.legend_loc)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, output_dir, f"{args.prefix}learning_curve_combined_objective", args)


# ======================================================================
# Plot 5: combined behavioural-metric learning curves
# ======================================================================
def plot_combined_behavioural_metrics(data: dict, args, output_dir: Path):
    arch_df = data["validation_arch"]
    metrics = BEHAVIOURAL_METRICS
    if args.metrics:
        metrics = [(col, label) for col, label in BEHAVIOURAL_METRICS if col in args.metrics]
        if not metrics:
            print(f"WARNING: none of --metrics {args.metrics} matched a known behavioural metric column; skipping plot 5.")
            return

    n = len(metrics)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(args.fig_width * ncols / 2, args.fig_height * nrows / 2))
    plt.rcParams.update({"font.size": args.font_size})
    axes = np.atleast_1d(axes).flatten()

    for i, (col, label) in enumerate(metrics):
        ax = axes[i]
        for architecture in ARCHITECTURES:
            sub = arch_df[arch_df["architecture"] == architecture].sort_values("checkpoint_transition")
            if col not in sub.columns:
                continue
            y = sub[col]
            if args.smooth:
                y = rolling_smooth(y, args.smooth_window)
            ax.plot(sub["checkpoint_transition"], y, color=ARCHITECTURE_COLORS[architecture], lw=1.8, marker="o",
                    markersize=2.5, label=ARCHITECTURE_LABELS[architecture])
        for i, marker_t in enumerate(args.checkpoint_markers):
            ax.axvline(marker_t, color="gray", ls="-." if i == 0 else ":", lw=1.0)
        if args.xmin is not None or args.xmax is not None:
            ax.set_xlim(left=args.xmin, right=args.xmax)
        ax.set_title(label, fontsize=args.font_size)
        ax.grid(alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=max(args.font_size - 1, 7))
    fig.suptitle("Combined behavioural-metric learning curves (cross-seed means)"
                 + (" -- rolling mean, visual only" if args.smooth else ""))
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    savefig(fig, output_dir, f"{args.prefix}learning_curve_behavioural_metrics", args)


# ======================================================================
# Plot 6: final fixed-1m architecture comparison
# ======================================================================
def plot_final_architecture_comparison(data: dict, args, output_dir: Path):
    arch_df = data["holdout_arch"]
    fixed_checkpoint = arch_df["checkpoint_transition"].max()
    sub = arch_df[arch_df["checkpoint_transition"] == fixed_checkpoint].set_index("architecture").reindex(ARCHITECTURES)

    metrics = [
        ("cross_seed_mean_full_objective", "cross_seed_sd_full_objective", "Full objective"),
        ("cross_seed_mean_mean_fills", None, "Fills"),
        ("cross_seed_mean_mean_quoted_spread", None, "Mean quoted spread"),
        ("cross_seed_mean_spread_revenue", None, "Spread revenue"),
        ("cross_seed_mean_adverse_selection_loss", None, "Adverse-selection loss"),
        ("cross_seed_mean_mean_abs_inventory", None, "Mean |inventory|"),
    ]
    metrics = [(c, e, l) for c, e, l in metrics if c in sub.columns]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(args.fig_width * n / 3, args.fig_height * 0.7))
    plt.rcParams.update({"font.size": args.font_size})
    axes = np.atleast_1d(axes)

    x = np.arange(len(ARCHITECTURES))
    for ax, (col, err_col, label) in zip(axes, metrics):
        vals = sub[col].to_numpy()
        errs = sub[err_col].to_numpy() if err_col and err_col in sub.columns else None
        colors = [ARCHITECTURE_COLORS[a] for a in ARCHITECTURES]
        ax.bar(x, vals, yerr=errs, capsize=4, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([ARCHITECTURE_LABELS[a].replace(" ", "\n") for a in ARCHITECTURES], fontsize=args.font_size - 2)
        ax.set_title(label, fontsize=args.font_size)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Final fixed-{fixed_checkpoint:,.0f}-transition architecture comparison (bars = cross-seed mean; "
                 "error bars = cross-seed SD where available)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig(fig, output_dir, f"{args.prefix}final_architecture_comparison", args)


# ======================================================================
# Plot 7: fixed 200k vs 1m comparison
# ======================================================================
def plot_fixed_200k_vs_1m(data: dict, args, output_dir: Path):
    df = data["fixed_200k_1m"]
    metrics = df["metric"].unique().tolist()
    n = len(metrics)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(args.fig_width * ncols / 2, args.fig_height * nrows / 2))
    plt.rcParams.update({"font.size": args.font_size})
    axes = np.atleast_1d(axes).flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sub = df[df["metric"] == metric]
        x = np.arange(len(ARCHITECTURES))
        means, los, his = [], [], []
        for architecture in ARCHITECTURES:
            arch_rows = sub[sub["architecture"] == architecture]
            m = float(arch_rows["mean_change_200k_to_1m"].mean())
            lo = float(arch_rows["bootstrap_ci_lo"].mean())
            hi = float(arch_rows["bootstrap_ci_hi"].mean())
            means.append(m); los.append(m - lo); his.append(hi - m)
        colors = [ARCHITECTURE_COLORS[a] for a in ARCHITECTURES]
        ax.bar(x, means, yerr=[los, his], capsize=4, color=colors)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([ARCHITECTURE_LABELS[a].replace(" ", "\n") for a in ARCHITECTURES], fontsize=args.font_size - 2)
        ax.set_title(metric, fontsize=args.font_size)
        ax.grid(alpha=0.3, axis="y")

    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Change from fixed 200,000-transition to fixed 1,000,000-transition checkpoint\n"
                 "(mean paired change across learner seeds' own path-paired means; error bars = mean bootstrap 95% CI)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig(fig, output_dir, f"{args.prefix}fixed_200k_vs_1m_comparison", args)


# ======================================================================
# Plot 8: PPO-vs-benchmark paired-difference plots with CIs
# ======================================================================
def plot_benchmark_paired_differences(data: dict, args, output_dir: Path):
    df = data["benchmark_paired"]
    benchmarks = df["benchmark"].unique().tolist()
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(args.fig_width * len(benchmarks) / 2, args.fig_height * 0.75))
    plt.rcParams.update({"font.size": args.font_size})
    axes = np.atleast_1d(axes)

    for ax, benchmark in zip(axes, benchmarks):
        sub = df[df["benchmark"] == benchmark]
        xpos = 0
        xticks, xlabels = [], []
        for architecture in ARCHITECTURES:
            arch_sub = sub[sub["architecture"] == architecture].sort_values("learner_seed")
            for _, row in arch_sub.iterrows():
                lo, hi = row["bootstrap_ci_lo"], row["bootstrap_ci_hi"]
                ax.errorbar([xpos], [row["paired_mean_diff"]],
                            yerr=[[row["paired_mean_diff"] - lo], [hi - row["paired_mean_diff"]]],
                            fmt="o", color=ARCHITECTURE_COLORS[architecture], capsize=3)
                xpos += 1
            xticks.append(xpos - len(arch_sub) / 2 - 0.5)
            xlabels.append(ARCHITECTURE_LABELS[architecture])
            xpos += 1
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=args.font_size - 2)
        ax.set_title(f"vs. {benchmark}", fontsize=args.font_size)
        ax.set_ylabel("Paired mean objective diff (PPO - benchmark)")
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("PPO vs. benchmark paired path-level differences, fixed 1,000,000-transition checkpoint\n"
                 "(points = per-seed paired mean diff over 500 holdout paths; bars = bootstrap 95% CI)")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    savefig(fig, output_dir, f"{args.prefix}benchmark_paired_differences", args)


# ======================================================================
# Plot 9: final behavioural decomposition
# ======================================================================
def plot_behavioural_decomposition(data: dict, args, output_dir: Path):
    arch_df = data["holdout_arch"]
    fixed_checkpoint = arch_df["checkpoint_transition"].max()
    sub = arch_df[arch_df["checkpoint_transition"] == fixed_checkpoint].set_index("architecture").reindex(ARCHITECTURES)
    metrics = [(c, l) for c, l in DECOMPOSITION_METRICS if c in sub.columns]

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    plt.rcParams.update({"font.size": args.font_size})
    x = np.arange(len(metrics))
    width = 0.8 / len(ARCHITECTURES)
    for i, architecture in enumerate(ARCHITECTURES):
        vals = [sub.loc[architecture, c] for c, _ in metrics]
        ax.bar(x + i * width, vals, width=width, color=ARCHITECTURE_COLORS[architecture],
               label=ARCHITECTURE_LABELS[architecture])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([l for _, l in metrics], rotation=20, ha="right")
    ax.set_ylabel("Value (holdout cross-seed mean)")
    ax.set_title(f"Final behavioural decomposition, fixed {fixed_checkpoint:,.0f}-transition checkpoint")
    ax.legend(fontsize=max(args.font_size - 1, 7), loc=args.legend_loc)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    savefig(fig, output_dir, f"{args.prefix}behavioural_decomposition", args)


PLOT_FUNCS = [
    ("per_architecture_learning_curves", lambda data, args, out: [
        plot_architecture_learning_curve(a, data, args, out) for a in ARCHITECTURES
        if data["validation_seed"] is not None and (data["validation_seed"]["architecture"] == a).any()
    ]),
    ("combined_objective_curve", plot_combined_objective_curve),
    ("combined_behavioural_metrics", plot_combined_behavioural_metrics),
    ("final_architecture_comparison", plot_final_architecture_comparison),
    ("fixed_200k_vs_1m_comparison", plot_fixed_200k_vs_1m),
    ("benchmark_paired_differences", plot_benchmark_paired_differences),
    ("behavioural_decomposition", plot_behavioural_decomposition),
]


def main():
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.results_dir / "plots"
    data = load_data(args.results_dir)

    for name, fn in PLOT_FUNCS:
        try:
            fn(data, args, output_dir)
        except Exception as e:
            print(f"WARNING: plot group '{name}' failed ({e}); skipping.")

    print(f"\nAll available plots written to {output_dir}")


if __name__ == "__main__":
    main()
