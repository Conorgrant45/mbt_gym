"""
plot_final_seed_diagnostics.py
------------------------------------
Stage 5, item 13 ("Training-seed failure-case diagnostics") for the final
five-seed dissertation experiment. Pure plotting from already-computed
CSVs (results/agent_comparison_final_all_seeds_episodes.csv and
results/final_symmetry_direction_diagnostics.csv) -- no re-evaluation.
Reuses the same colour convention (sns "deep" palette) as
plot_agent_comparison.py / plot_hamilton_ppo_multiseed.py.

Run from repo root:
    python plot_final_seed_diagnostics.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
IMAGES_DIR = Path(r"C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images")

RL_AGENT_TYPES = ("hamilton_ppo", "return_mlp_ppo", "return_lstm_ppo")
TRAINING_SEEDS = [0, 1, 2, 3, 4]

sns.set_theme(style="whitegrid", context="notebook")
_PALETTE = sns.color_palette("deep")
AGENT_COLOR = {a: _PALETTE[i] for i, a in enumerate(
    ("oracle", "belief_weighted", "randomised", "naive") + RL_AGENT_TYPES
)}


def _savefig(fig, name):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def main():
    episodes = pd.read_csv(RESULTS_DIR / "agent_comparison_final_all_seeds_episodes.csv")
    rl = episodes[episodes["agent_type"].isin(RL_AGENT_TYPES)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) full_objective std per training seed -- the objective-variance symptom
    ax = axes[0, 0]
    for agent_type in RL_AGENT_TYPES:
        sub = rl[rl["agent_type"] == agent_type]
        stds = sub.groupby("training_seed")["full_objective"].std()
        ax.plot(stds.index, stds.values, marker="o", color=AGENT_COLOR[agent_type], label=agent_type)
    ax.set_xlabel("training seed")
    ax.set_ylabel("std(full_objective) across 100 holdout episodes")
    ax.set_title("Objective-variance symptom by training seed")
    ax.legend()

    # (b) terminal signed inventory mean +/- std per training seed
    ax = axes[0, 1]
    width = 0.25
    for i, agent_type in enumerate(RL_AGENT_TYPES):
        sub = rl[rl["agent_type"] == agent_type]
        means = sub.groupby("training_seed")["terminal_signed_inventory"].mean()
        stds = sub.groupby("training_seed")["terminal_signed_inventory"].std()
        x = np.array(TRAINING_SEEDS) + (i - 1) * width
        ax.bar(x, means.values, width, yerr=stds.values, color=AGENT_COLOR[agent_type],
               label=agent_type, capsize=3)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("training seed")
    ax.set_ylabel("terminal_signed_inventory (mean +/- std)")
    ax.set_title("Terminal inventory bias by training seed")
    ax.legend()

    # (c) action std (bid/ask mean, averaged over holdout episodes) per training seed
    ax = axes[1, 0]
    for agent_type in RL_AGENT_TYPES:
        sub = rl[rl["agent_type"] == agent_type]
        combined_std = sub.groupby("training_seed").apply(
            lambda g: (g["bid_action_std"].mean() + g["ask_action_std"].mean()) / 2, include_groups=False
        )
        ax.plot(combined_std.index, combined_std.values, marker="s", color=AGENT_COLOR[agent_type], label=agent_type)
    ax.set_xlabel("training seed")
    ax.set_ylabel("mean(bid_action_std, ask_action_std)")
    ax.set_title("Action-std collapse symptom by training seed")
    ax.legend()

    # (d) symmetry error per training seed (from the policy-grid diagnostic)
    ax = axes[1, 1]
    sym = pd.read_csv(RESULTS_DIR / "final_symmetry_direction_diagnostics.csv")
    for agent_type in RL_AGENT_TYPES:
        sub = sym[sym["agent_type"] == agent_type].sort_values("training_seed")
        ax.plot(sub["training_seed"], sub["symmetry_mean_abs_err"], marker="^",
                color=AGENT_COLOR[agent_type], label=agent_type)
    ax.set_xlabel("training seed")
    ax.set_ylabel("mean|bid(q,tau) - ask(-q,tau)|")
    ax.set_title("Bid/ask symmetry error by training seed")
    ax.legend()

    fig.suptitle("Training-seed failure-case diagnostics (final 5-seed experiment)", fontsize=13)
    fig.tight_layout()
    _savefig(fig, "agent_comparison_14_training_seed_failure_diagnostics.png")


if __name__ == "__main__":
    main()
