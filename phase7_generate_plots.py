"""
phase7_generate_plots.py
----------------------------
Phase 7: per-seed and aggregate learning-curve plots, log_std trajectory,
explained-variance, and PPO clip-fraction plots.

Run from repo root (after phase7_run_training.py and
phase7_analyze_convergence.py have completed):
    python phase7_generate_plots.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import phase7_common as P7

SEED_COLORS = {0: "tab:red", 1: "tab:orange", 2: "tab:green", 3: "tab:blue", 4: "tab:purple"}


def plot_per_seed_learning_curves(cp: pd.DataFrame):
    for seed in P7.LEARNER_SEEDS:
        sub = cp[cp["learner_seed"] == seed].sort_values("timestep")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, col, title in zip(axes, ["det_mean_objective", "stoch_mean_objective"],
                                   ["Deterministic validation objective", "Stochastic validation objective"]):
            ax.plot(sub["timestep"], sub[col], marker="o", color=SEED_COLORS[seed])
            ax.axvline(200_000, color="black", ls="--", lw=0.8, label="Phase 6 budget (200k)")
            ax.set_xlabel("PPO transitions")
            ax.set_title(title)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[0].set_ylabel("Full objective")
        fig.suptitle(f"Group B, learner seed {seed}: learning curve to 1,000,000 transitions")
        fig.tight_layout()
        fig.savefig(P7.PLOTS_DIR / f"learning_curve_seed{seed}.png", dpi=130)
        plt.close(fig)


def plot_aggregate_learning_curve(cp: pd.DataFrame, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title in zip(axes, ["det_mean_objective", "stoch_mean_objective"],
                               ["Deterministic validation objective", "Stochastic validation objective"]):
        for seed in P7.LEARNER_SEEDS:
            sub = cp[cp["learner_seed"] == seed].sort_values("timestep")
            ax.plot(sub["timestep"], sub[col], marker="o", markersize=3, color=SEED_COLORS[seed],
                     label=f"seed {seed}", alpha=0.85)
        ax.axvline(200_000, color="black", ls="--", lw=0.8)
        ax.set_xlabel("PPO transitions")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("Full objective")
    fig.suptitle("Group B (random init, log_std_init=-1.5): all 5 seeds, 1,000,000 transitions "
                 "(dashed line = Phase 6's 200,000-transition budget)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_log_std(cp: pd.DataFrame, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    cp = cp.copy()
    cp["log_std_mean"] = (cp["log_std_bid"] + cp["log_std_ask"]) / 2.0
    for seed in P7.LEARNER_SEEDS:
        sub = cp[cp["learner_seed"] == seed].sort_values("timestep")
        ax.plot(sub["timestep"], sub["log_std_mean"], marker="o", markersize=3, color=SEED_COLORS[seed],
                label=f"seed {seed}")
    ax.axvline(200_000, color="black", ls="--", lw=0.8)
    ax.set_xlabel("PPO transitions")
    ax.set_ylabel("Mean log_std")
    ax.set_title("Group B: learned exploration (log_std) to 1,000,000 transitions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_explained_variance(diag: pd.DataFrame, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for seed in P7.LEARNER_SEEDS:
        sub = diag[diag["learner_seed"] == seed].sort_values("num_timesteps")
        ax.plot(sub["num_timesteps"], sub["explained_variance"], color=SEED_COLORS[seed], alpha=0.7, lw=1.0)
    ax.axvline(200_000, color="black", ls="--", lw=0.8)
    ax.axhline(0.0, color="gray", lw=0.6)
    ax.set_xlabel("PPO transitions")
    ax.set_ylabel("Explained variance")
    ax.set_title("Group B: critic explained variance per PPO update")
    handles = [plt.Line2D([0], [0], color=c, label=f"seed {s}") for s, c in SEED_COLORS.items()]
    ax.legend(handles=handles, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_clip_fraction(diag: pd.DataFrame, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title in zip(axes, ["clip_fraction", "approx_kl"], ["PPO clip fraction", "Approximate KL"]):
        for seed in P7.LEARNER_SEEDS:
            sub = diag[diag["learner_seed"] == seed].sort_values("num_timesteps")
            ax.plot(sub["num_timesteps"], sub[col], color=SEED_COLORS[seed], alpha=0.7, lw=1.0)
        ax.axvline(200_000, color="black", ls="--", lw=0.8)
        ax.set_xlabel("PPO transitions")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    handles = [plt.Line2D([0], [0], color=c, label=f"seed {s}") for s, c in SEED_COLORS.items()]
    axes[0].legend(handles=handles, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    P7.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cp = pd.read_csv(P7.RESULTS_DIR / "phase7_checkpoint_summary.csv")
    diag = pd.read_csv(P7.RESULTS_DIR / "phase7_training_diagnostics.csv")

    plot_per_seed_learning_curves(cp)
    plot_aggregate_learning_curve(cp, P7.PLOTS_DIR / "aggregate_learning_curve.png")
    plot_log_std(cp, P7.PLOTS_DIR / "log_std_trajectory.png")
    plot_explained_variance(diag, P7.PLOTS_DIR / "explained_variance.png")
    plot_clip_fraction(diag, P7.PLOTS_DIR / "clip_fraction_and_kl.png")
    print(f"Plots saved under {P7.PLOTS_DIR}")


if __name__ == "__main__":
    main()
