"""
phase5_generate_plots.py
----------------------------
Phase 5, Section 8/11: plots showing how objective performance and the
action surface change across PPO updates, for groups A (random init),
B (clone init), and C (frozen clone control).

Run from repo root (after phase5_run_training.py has produced its CSVs):
    python phase5_generate_plots.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import phase5_common as P5

GROUP_COLORS = {"A_random_init": "tab:red", "B_clone_init": "tab:blue", "C_frozen_clone": "tab:green"}
GROUP_LABELS = {"A_random_init": "A: random init", "B_clone_init": "B: clone init", "C_frozen_clone": "C: frozen clone"}


def plot_objective_vs_timestep(df: pd.DataFrame, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title in zip(axes, ["det_mean_objective", "stoch_mean_objective"],
                               ["Deterministic mean objective", "Stochastic mean objective"]):
        for group, sub in df.groupby("group"):
            for seed, sub_s in sub.groupby("learner_seed"):
                sub_s = sub_s.sort_values("timestep")
                ax.plot(sub_s["timestep"], sub_s[col], color=GROUP_COLORS[group], alpha=0.7,
                        marker="o", markersize=3, label=f"{GROUP_LABELS[group]} (seed {seed})" if seed == df["learner_seed"].min() else None)
        ax.set_xlabel("PPO transitions")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    handles = [plt.Line2D([0], [0], color=c, marker="o", label=GROUP_LABELS[g]) for g, c in GROUP_COLORS.items()]
    axes[0].legend(handles=handles, fontsize=9)
    axes[0].set_ylabel("Full objective")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_actor_distance_vs_timestep(df: pd.DataFrame, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for group, sub in df.groupby("group"):
        for seed, sub_s in sub.groupby("learner_seed"):
            sub_s = sub_s.sort_values("timestep")
            ax.plot(sub_s["timestep"], sub_s["actor_distance_from_clone"], color=GROUP_COLORS[group],
                    alpha=0.7, marker="o", markersize=3)
    ax.set_xlabel("PPO transitions")
    ax.set_ylabel("Actor L2 distance from supervised clone")
    ax.set_title("Actor drift from the supervised clone")
    handles = [plt.Line2D([0], [0], color=c, marker="o", label=GROUP_LABELS[g]) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_action_surface_mse_vs_timestep(df: pd.DataFrame, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title in zip(axes, ["bid_action_depth_mse", "ask_action_depth_mse"],
                               ["Bid depth MSE vs analytical", "Ask depth MSE vs analytical"]):
        for group, sub in df.groupby("group"):
            for seed, sub_s in sub.groupby("learner_seed"):
                sub_s = sub_s.sort_values("timestep")
                ax.plot(sub_s["timestep"], sub_s[col], color=GROUP_COLORS[group], alpha=0.7,
                        marker="o", markersize=3)
        ax.set_xlabel("PPO transitions")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    handles = [plt.Line2D([0], [0], color=c, marker="o", label=GROUP_LABELS[g]) for g, c in GROUP_COLORS.items()]
    axes[0].legend(handles=handles, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_update_level_diagnostics(update_df: pd.DataFrame, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    cols = ["policy_loss", "value_loss", "approx_kl", "clip_fraction", "actor_grad_norm_mean", "critic_grad_norm_mean"]
    for ax, col in zip(axes.flat, cols):
        for group, sub in update_df.groupby("group"):
            for seed, sub_s in sub.groupby("learner_seed"):
                sub_s = sub_s.sort_values("n_updates")
                ax.plot(sub_s["n_updates"], sub_s[col], color=GROUP_COLORS[group], alpha=0.7, marker=".", markersize=4)
        ax.set_xlabel("PPO update index")
        ax.set_title(col)
        ax.grid(alpha=0.3)
    handles = [plt.Line2D([0], [0], color=c, marker=".", label=GROUP_LABELS[g]) for g, c in GROUP_COLORS.items() if g != "C_frozen_clone"]
    axes.flat[0].legend(handles=handles, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_drift_deterministic_objective(drift_df: pd.DataFrame, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for seed, sub in drift_df.groupby("learner_seed"):
        sub = sub.sort_values("num_timesteps")
        ax.plot(sub["num_timesteps"], sub["deterministic_objective_change_from_t0"], marker="o", markersize=3,
                label=f"seed {seed}")
    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("PPO transitions")
    ax.set_ylabel("Deterministic objective change from t=0")
    ax.set_title("Group B: per-update deterministic-objective drift from the clone's own t=0 performance")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    P5.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_df = pd.read_csv(P5.PHASE5_RESULTS_DIR / "phase5_checkpoint_metrics.csv")
    update_df = pd.read_csv(P5.PHASE5_RESULTS_DIR / "phase5_update_metrics.csv")
    drift_df = pd.read_csv(P5.PHASE5_RESULTS_DIR / "phase5_action_surface_drift.csv")

    plot_objective_vs_timestep(checkpoint_df, P5.PLOTS_DIR / "objective_vs_timestep.png")
    plot_actor_distance_vs_timestep(checkpoint_df, P5.PLOTS_DIR / "actor_distance_vs_timestep.png")
    plot_action_surface_mse_vs_timestep(checkpoint_df, P5.PLOTS_DIR / "action_surface_mse_vs_timestep.png")
    plot_update_level_diagnostics(update_df, P5.PLOTS_DIR / "update_level_diagnostics.png")
    plot_drift_deterministic_objective(drift_df, P5.PLOTS_DIR / "group_b_deterministic_objective_drift.png")

    print(f"Plots saved under {P5.PLOTS_DIR}")


if __name__ == "__main__":
    main()
