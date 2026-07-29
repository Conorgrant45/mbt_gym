"""
phase6_generate_plots.py
----------------------------
Phase 6, Section 11: summary plots for the 2x2 optimisation experiment --
group-level holdout objective comparison, per-seed robustness, validation
training curves, and the four primary factor effects with bootstrap CIs.

Run from repo root (after phase6_evaluate_holdout.py and
phase6_analyze_factor_effects.py):
    python phase6_generate_plots.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import phase6_common as P6

GROUP_COLORS = {"A": "tab:red", "B": "tab:orange", "C": "tab:blue", "D": "tab:green"}
GROUP_LABELS = {
    "A": "A: random init, log_std=0.0",
    "B": "B: random init, log_std=-1.5",
    "C": "C: clone init, log_std=0.0",
    "D": "D: clone init, log_std=-1.5",
}


def plot_group_holdout_objective(seed_df: pd.DataFrame, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = {"A": 0, "B": 1, "C": 2, "D": 3}
    for group, pos in positions.items():
        sub = seed_df[seed_df["group"] == group]
        ax.scatter([pos] * len(sub), sub["mean_full_objective"], color=GROUP_COLORS[group], s=50, zorder=3)
        ax.errorbar([pos], [sub["mean_full_objective"].mean()], yerr=[sub["mean_full_objective"].std(ddof=1)],
                    fmt="D", color="black", markersize=8, capsize=5, zorder=4)
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([GROUP_LABELS[g] for g in positions], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Holdout mean full objective (per learner seed)")
    ax.set_title("Group comparison: 5 learner-seed holdout means (dots) + across-seed mean/SD (diamond)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_checkpoint_training_curves(checkpoint_df: pd.DataFrame, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title in zip(axes, ["det_mean_objective", "stoch_mean_objective"],
                               ["Deterministic validation objective", "Stochastic validation objective"]):
        for group, sub in checkpoint_df.groupby("group"):
            for seed, sub_s in sub.groupby("learner_seed"):
                sub_s = sub_s.sort_values("timestep")
                ax.plot(sub_s["timestep"], sub_s[col], color=GROUP_COLORS[group], alpha=0.5, lw=1.2)
        ax.set_xlabel("PPO transitions")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    handles = [plt.Line2D([0], [0], color=c, label=GROUP_LABELS[g]) for g, c in GROUP_COLORS.items()]
    axes[0].legend(handles=handles, fontsize=8)
    axes[0].set_ylabel("Full objective")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_log_std_trajectory(checkpoint_df: pd.DataFrame, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    checkpoint_df = checkpoint_df.copy()
    checkpoint_df["log_std_mean"] = (checkpoint_df["log_std_bid"] + checkpoint_df["log_std_ask"]) / 2.0
    for group, sub in checkpoint_df.groupby("group"):
        for seed, sub_s in sub.groupby("learner_seed"):
            sub_s = sub_s.sort_values("timestep")
            ax.plot(sub_s["timestep"], sub_s["log_std_mean"], color=GROUP_COLORS[group], alpha=0.6, lw=1.2)
    ax.set_xlabel("PPO transitions")
    ax.set_ylabel("Mean log_std")
    ax.set_title("Learned exploration (log_std) across training")
    handles = [plt.Line2D([0], [0], color=c, label=GROUP_LABELS[g]) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_factor_effects(effects_df: pd.DataFrame, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    seed_cols = [f"seed{i}_diff" for i in range(5)]
    for i, row in effects_df.iterrows():
        ax.scatter([row[c] for c in seed_cols], [i] * 5, color="tab:gray", s=30, zorder=2, alpha=0.7)
        ax.plot([row["bootstrap_ci_lo"], row["bootstrap_ci_hi"]], [i, i], color="black", lw=2, zorder=3)
        ax.scatter([row["mean_diff"]], [i], color="tab:red", s=80, zorder=4, marker="D")
    ax.axvline(0.0, color="black", lw=0.8, ls="--")
    ax.set_yticks(range(len(effects_df)))
    ax.set_yticklabels(effects_df["contrast"], fontsize=8)
    ax.set_xlabel("Effect on holdout mean full objective")
    ax.set_title("Factor effects: seed-level differences (grey), mean (red diamond), bootstrap 95% CI (black, n=5 -- weak)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    P6.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_df = pd.read_csv(P6.RESULTS_DIR / "phase6_seed_summary.csv")
    checkpoint_df = pd.read_csv(P6.RESULTS_DIR / "phase6_checkpoint_summary.csv")
    effects_df = pd.read_csv(P6.RESULTS_DIR / "phase6_factor_effects.csv")

    plot_group_holdout_objective(seed_df, P6.PLOTS_DIR / "group_holdout_objective.png")
    plot_checkpoint_training_curves(checkpoint_df, P6.PLOTS_DIR / "checkpoint_training_curves.png")
    plot_log_std_trajectory(checkpoint_df, P6.PLOTS_DIR / "log_std_trajectory.png")
    plot_factor_effects(effects_df, P6.PLOTS_DIR / "factor_effects.png")
    print(f"Plots saved under {P6.PLOTS_DIR}")


if __name__ == "__main__":
    main()
