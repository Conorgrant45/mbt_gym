"""
phase4_generate_plots.py
----------------------------
Phase 4, Section 3/11: action-surface plots (heatmaps/line slices) comparing
the analytical belief-weighted policy against the mean (+/- 1 std across the
5 learner seeds) of fixed-step and event-driven Hamilton PPO, read from the
already-computed phase4_action_surface_grid.csv (no new rollouts).

Run from repo root:
    python phase4_generate_plots.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import phase4_common as P4

FIXED_SEED_POLICIES = [f"fixed_seed{i}" for i in range(5)]
EVENT_SEED_POLICIES = [f"event_seed{i}" for i in range(5)]


def load_grid() -> pd.DataFrame:
    return pd.read_csv(P4.PHASE4_RESULTS_DIR / "phase4_action_surface_grid.csv")


def seed_mean_std(df: pd.DataFrame, policies: list, value_col: str) -> pd.DataFrame:
    sub = df[df["policy"].isin(policies)]
    g = sub.groupby(["q", "tau", "belief"])[value_col].agg(["mean", "std"]).reset_index()
    return g


def plot_depth_vs_inventory(df: pd.DataFrame, tau: float, belief: float, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, depth_col, title in zip(axes, ["bid_depth", "ask_depth"], ["Bid depth", "Ask depth"]):
        analytical = df[(df["policy"] == "analytical") & (df["tau"] == tau) & (df["belief"] == belief)].sort_values("q")
        ax.plot(analytical["q"], analytical[depth_col], "k-", lw=2, label="Analytical belief-weighted")

        for policies, color, label in ((FIXED_SEED_POLICIES, "tab:blue", "Fixed-step Hamilton PPO"),
                                        (EVENT_SEED_POLICIES, "tab:orange", "Event-driven Hamilton PPO")):
            sub = df[df["policy"].isin(policies) & (df["tau"] == tau) & (df["belief"] == belief)]
            g = sub.groupby("q")[depth_col].agg(["mean", "std"]).reset_index().sort_values("q")
            ax.plot(g["q"], g["mean"], color=color, lw=2, label=f"{label} (mean of 5 seeds)")
            ax.fill_between(g["q"], g["mean"] - g["std"], g["mean"] + g["std"], color=color, alpha=0.2)

        ax.set_xlabel("Inventory q")
        ax.set_title(f"{title} vs inventory (tau={tau}, belief={belief})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Depth")
    axes[0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_depth_vs_tau(df: pd.DataFrame, q: float, belief: float, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, depth_col, title in zip(axes, ["bid_depth", "ask_depth"], ["Bid depth", "Ask depth"]):
        analytical = df[(df["policy"] == "analytical") & (df["q"] == q) & (df["belief"] == belief)].sort_values("tau")
        ax.plot(analytical["tau"], analytical[depth_col], "k-", lw=2, label="Analytical belief-weighted")
        for policies, color, label in ((FIXED_SEED_POLICIES, "tab:blue", "Fixed-step Hamilton PPO"),
                                        (EVENT_SEED_POLICIES, "tab:orange", "Event-driven Hamilton PPO")):
            sub = df[df["policy"].isin(policies) & (df["q"] == q) & (df["belief"] == belief)]
            g = sub.groupby("tau")[depth_col].agg(["mean", "std"]).reset_index().sort_values("tau")
            ax.plot(g["tau"], g["mean"], color=color, lw=2, label=f"{label} (mean of 5 seeds)")
            ax.fill_between(g["tau"], g["mean"] - g["std"], g["mean"] + g["std"], color=color, alpha=0.2)
        ax.set_xlabel("Remaining time (tau)")
        ax.set_title(f"{title} vs remaining time (q={q}, belief={belief})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Depth")
    axes[0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_depth_vs_belief(df: pd.DataFrame, q: float, tau: float, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, depth_col, title in zip(axes, ["bid_depth", "ask_depth"], ["Bid depth", "Ask depth"]):
        analytical = df[(df["policy"] == "analytical") & (df["q"] == q) & (df["tau"] == tau)].sort_values("belief")
        ax.plot(analytical["belief"], analytical[depth_col], "k-", lw=2, label="Analytical belief-weighted")
        for policies, color, label in ((FIXED_SEED_POLICIES, "tab:blue", "Fixed-step Hamilton PPO"),
                                        (EVENT_SEED_POLICIES, "tab:orange", "Event-driven Hamilton PPO")):
            sub = df[df["policy"].isin(policies) & (df["q"] == q) & (df["tau"] == tau)]
            g = sub.groupby("belief")[depth_col].agg(["mean", "std"]).reset_index().sort_values("belief")
            ax.plot(g["belief"], g["mean"], color=color, lw=2, label=f"{label} (mean of 5 seeds)")
            ax.fill_between(g["belief"], g["mean"] - g["std"], g["mean"] + g["std"], color=color, alpha=0.2)
        ax.set_xlabel("Belief b (P(regime=1))")
        ax.set_title(f"{title} vs belief (q={q}, tau={tau})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Depth")
    axes[0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_skew_heatmap(df: pd.DataFrame, tau: float, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, sharex=True)
    panels = [("analytical", [], "Analytical belief-weighted"),
              (None, FIXED_SEED_POLICIES, "Fixed-step Hamilton PPO (mean of 5 seeds)"),
              (None, EVENT_SEED_POLICIES, "Event-driven Hamilton PPO (mean of 5 seeds)")]
    vmax = 0.0
    grids = []
    for single_policy, policies, title in panels:
        if single_policy:
            sub = df[(df["policy"] == single_policy) & (df["tau"] == tau)]
            piv = sub.pivot_table(index="belief", columns="q", values="ask_depth") - \
                  sub.pivot_table(index="belief", columns="q", values="bid_depth")
        else:
            sub = df[df["policy"].isin(policies) & (df["tau"] == tau)]
            skew = sub.groupby(["q", "belief"]).apply(
                lambda g: (g["ask_depth"] - g["bid_depth"]).mean()
            ).reset_index(name="skew")
            piv = skew.pivot_table(index="belief", columns="q", values="skew")
        grids.append((piv, title))
        vmax = max(vmax, float(np.nanmax(np.abs(piv.values))))

    for ax, (piv, title) in zip(axes, grids):
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Inventory q")
    axes[0].set_ylabel("Belief b")
    fig.colorbar(im, ax=axes, label="Quote skew (ask_depth - bid_depth)", shrink=0.85)
    fig.suptitle(f"Inventory-skew surface at tau={tau} (positive = discourages further buying)")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    P4.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_grid()

    plot_depth_vs_inventory(df, tau=0.5, belief=0.5, out_path=P4.PLOTS_DIR / "depth_vs_inventory_tau0.5_belief0.5.png")
    plot_depth_vs_inventory(df, tau=0.9, belief=0.5, out_path=P4.PLOTS_DIR / "depth_vs_inventory_tau0.9_belief0.5.png")
    plot_depth_vs_inventory(df, tau=0.1, belief=0.5, out_path=P4.PLOTS_DIR / "depth_vs_inventory_tau0.1_belief0.5.png")

    plot_depth_vs_tau(df, q=0.0, belief=0.5, out_path=P4.PLOTS_DIR / "depth_vs_tau_q0_belief0.5.png")
    plot_depth_vs_tau(df, q=20.0, belief=0.5, out_path=P4.PLOTS_DIR / "depth_vs_tau_q20_belief0.5.png")

    plot_depth_vs_belief(df, q=0.0, tau=0.5, out_path=P4.PLOTS_DIR / "depth_vs_belief_q0_tau0.5.png")
    plot_depth_vs_belief(df, q=20.0, tau=0.5, out_path=P4.PLOTS_DIR / "depth_vs_belief_q20_tau0.5.png")

    plot_skew_heatmap(df, tau=0.5, out_path=P4.PLOTS_DIR / "skew_heatmap_tau0.5.png")
    plot_skew_heatmap(df, tau=0.1, out_path=P4.PLOTS_DIR / "skew_heatmap_tau0.1.png")

    print(f"Plots saved under {P4.PLOTS_DIR}")


if __name__ == "__main__":
    main()
