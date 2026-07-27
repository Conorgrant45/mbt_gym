"""
plot_hamilton_ppo_multiseed.py
----------------------------------
Generates the 8 required diagnostic plots for the 5-seed Hamilton PPO
multiseed experiment, from the CSVs already produced by:
    evaluate_hamilton_ppo_multiseed_holdout.py
    evaluate_hamilton_ppo_multiseed_policy_grid.py
    evaluate_hamilton_ppo_multiseed_benchmarks.py

Does not re-run any evaluation -- pure plotting from saved results.

Run from repo root:
    python plot_hamilton_ppo_multiseed.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
IMAGES_DIR = Path(r"C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images")

TRAINING_SEEDS = [0, 1, 2, 3, 4]

sns.set_theme(style="whitegrid", context="notebook")
_PALETTE = sns.color_palette("deep")
SEED_COLOR = {ts: _PALETTE[i] for i, ts in enumerate(TRAINING_SEEDS)}


def load_data():
    episodes = pd.read_csv(RESULTS_DIR / "hamilton_ppo_multiseed_holdout_episodes.csv")
    grid = pd.read_csv(RESULTS_DIR / "hamilton_ppo_multiseed_policy_grid.csv")
    symmetry = pd.read_csv(RESULTS_DIR / "hamilton_ppo_multiseed_symmetry.csv")
    paired = pd.read_csv(RESULTS_DIR / "hamilton_ppo_multiseed_benchmark_paired_summary.csv")
    return episodes, grid, symmetry, paired


def plot_1_objective_by_seed(episodes: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = [episodes[episodes["training_seed"] == ts]["cumulative_objective"].values for ts in TRAINING_SEEDS]
    bp = ax.boxplot(data, positions=TRAINING_SEEDS, widths=0.6, patch_artist=True, showfliers=True)
    for patch, ts in zip(bp["boxes"], TRAINING_SEEDS):
        patch.set_facecolor(SEED_COLOR[ts])
        patch.set_alpha(0.6)
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Cumulative objective (100 holdout episodes)")
    ax.set_title("1. Objective distribution by training seed")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "hamilton_ppo_multiseed_1_objective_by_seed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_2_terminal_inventory_by_seed(episodes: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = [episodes[episodes["training_seed"] == ts]["terminal_signed_inventory"].values for ts in TRAINING_SEEDS]
    bp = ax.boxplot(data, positions=TRAINING_SEEDS, widths=0.6, patch_artist=True, showfliers=True)
    for patch, ts in zip(bp["boxes"], TRAINING_SEEDS):
        patch.set_facecolor(SEED_COLOR[ts])
        patch.set_alpha(0.6)
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8, label="flat (q=0)")
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Terminal signed inventory (shares)")
    ax.set_title("2. Terminal signed inventory distribution by training seed")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "hamilton_ppo_multiseed_2_terminal_inventory_by_seed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_3_action_means_by_seed(episodes: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.35
    x = np.arange(len(TRAINING_SEEDS))
    bid_means = [episodes[episodes["training_seed"] == ts]["bid_action_mean"].mean() for ts in TRAINING_SEEDS]
    bid_sems = [episodes[episodes["training_seed"] == ts]["bid_action_mean"].sem() for ts in TRAINING_SEEDS]
    ask_means = [episodes[episodes["training_seed"] == ts]["ask_action_mean"].mean() for ts in TRAINING_SEEDS]
    ask_sems = [episodes[episodes["training_seed"] == ts]["ask_action_mean"].sem() for ts in TRAINING_SEEDS]
    ax.bar(x - width / 2, bid_means, width, yerr=bid_sems, capsize=4, color=_PALETTE[0], label="bid action mean")
    ax.bar(x + width / 2, ask_means, width, yerr=ask_sems, capsize=4, color=_PALETTE[3], label="ask action mean")
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(ts) for ts in TRAINING_SEEDS])
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Mean normalised action")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("3. Bid and ask action means by training seed (error bars: SEM over episode means)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "hamilton_ppo_multiseed_3_action_means_by_seed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_4_saturation_by_side_seed(episodes: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.2
    x = np.arange(len(TRAINING_SEEDS))
    cols = ["frac_bid_near_low", "frac_bid_near_high", "frac_ask_near_low", "frac_ask_near_high"]
    labels = ["bid near low (-1)", "bid near high (+1)", "ask near low (-1)", "ask near high (+1)"]
    colors = [_PALETTE[0], _PALETTE[9], _PALETTE[3], _PALETTE[1]]
    for i, (col, label, color) in enumerate(zip(cols, labels, colors)):
        means = [episodes[episodes["training_seed"] == ts][col].mean() for ts in TRAINING_SEEDS]
        ax.bar(x + (i - 1.5) * width, means, width, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(ts) for ts in TRAINING_SEEDS])
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Fraction of steps near bound")
    ax.set_ylim(0, 1.0)
    ax.set_title("4. Action saturation by side and training seed")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "hamilton_ppo_multiseed_4_saturation_by_seed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_5_action_vs_inventory(grid: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    tau_fixed, b_fixed = 1.0, 0.7
    for ax, col, title in zip(axes, ["bid_action", "ask_action"], ["Bid action vs inventory", "Ask action vs inventory"]):
        for ts in TRAINING_SEEDS:
            sub = grid[(grid["training_seed"] == ts) & (grid["tau"] == tau_fixed) & (grid["b"] == b_fixed)].sort_values("q")
            ax.plot(sub["q"], sub[col], marker="o", color=SEED_COLOR[ts], label=f"seed {ts}")
        ax.set_xlabel("Inventory q (shares)")
        ax.set_ylabel(col.replace("_", " "))
        ax.set_title(f"{title} (tau={tau_fixed}, b={b_fixed})")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=8, title="training seed")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "hamilton_ppo_multiseed_5_action_vs_inventory.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_6_action_vs_tau(grid: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    q_fixed, b_fixed = 20, 0.7
    for ax, col, title in zip(axes, ["bid_action", "ask_action"], ["Bid action vs time remaining", "Ask action vs time remaining"]):
        for ts in TRAINING_SEEDS:
            sub = grid[(grid["training_seed"] == ts) & (grid["q"] == q_fixed) & (grid["b"] == b_fixed)].sort_values("tau", ascending=False)
            ax.plot(sub["tau"], sub[col], marker="o", color=SEED_COLOR[ts], label=f"seed {ts}")
        ax.invert_xaxis()
        ax.set_xlabel("tau (1=start, 0=terminal)")
        ax.set_ylabel(col.replace("_", " "))
        ax.set_title(f"{title} (q={q_fixed}, b={b_fixed})")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=8, title="training seed")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "hamilton_ppo_multiseed_6_action_vs_tau.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_7_symmetry_error_by_seed(symmetry: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(TRAINING_SEEDS))
    width = 0.35
    mean_err = [symmetry[symmetry["training_seed"] == ts]["mean_symmetry_error"].iloc[0] for ts in TRAINING_SEEDS]
    max_err = [symmetry[symmetry["training_seed"] == ts]["max_symmetry_error"].iloc[0] for ts in TRAINING_SEEDS]
    ax.bar(x - width / 2, mean_err, width, color=_PALETTE[2], label="mean |symmetry error|")
    ax.bar(x + width / 2, max_err, width, color=_PALETTE[4], label="max |symmetry error|")
    ax.set_xticks(x)
    ax.set_xticklabels([str(ts) for ts in TRAINING_SEEDS])
    ax.set_xlabel("Training seed")
    ax.set_ylabel("|a_bid(q,tau,b) - a_ask(-q,tau,b)|")
    ax.set_ylim(0, 2.05)
    ax.set_title("7. Policy symmetry error by training seed")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "hamilton_ppo_multiseed_7_symmetry_error_by_seed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_8_paired_diff_vs_naive(paired: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sub = paired[paired["analytic_policy"] == "naive"].set_index("training_seed").reindex(TRAINING_SEEDS)
    means = sub["mean_objective_diff"].values
    ci_lo = sub["ci_lo"].values
    ci_hi = sub["ci_hi"].values
    yerr = np.vstack([means - ci_lo, ci_hi - means])
    colors = [SEED_COLOR[ts] for ts in TRAINING_SEEDS]
    ax.bar(TRAINING_SEEDS, means, yerr=yerr, capsize=5, color=colors)
    ax.axhline(0, color="black", linestyle="-", linewidth=1.0)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("naive - ppo mean objective diff (95% CI)")
    ax.set_title("8. Paired objective difference vs naive benchmark, by training seed")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "hamilton_ppo_multiseed_8_paired_diff_vs_naive.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    episodes, grid, symmetry, paired = load_data()

    plot_1_objective_by_seed(episodes)
    plot_2_terminal_inventory_by_seed(episodes)
    plot_3_action_means_by_seed(episodes)
    plot_4_saturation_by_side_seed(episodes)
    plot_5_action_vs_inventory(grid)
    plot_6_action_vs_tau(grid)
    plot_7_symmetry_error_by_seed(symmetry)
    plot_8_paired_diff_vs_naive(paired)

    print(f"All 8 plots saved to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
