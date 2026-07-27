"""
evaluate_hamilton_ppo_policy_grid.py
---------------------------------------
Policy-surface diagnostics (item 4) and symmetry/economic-direction
checks (item 5) for the saved Hamilton PPO model
(models/hamilton_ppo/ppo_hamilton_v1.zip). Pure forward-pass evaluation
of the deterministic actor on a synthetic grid of valid observations --
no environment stepping, no retraining.

Grid:
    q   in {-20,-15,...,20}          (9 values, raw inventory, shares)
    tau in {1.0,0.75,0.5,0.25,0.05}  (5 values, fraction of episode remaining)
    b   in {0,0.1,...,1.0}           (11 values, Hamilton belief)
q is transformed via the SAME tanh(q/inventory_scale) used in
envs/hamilton_ppo_wrapper.py -- not re-derived.

Symmetry check (item 5): the market is intended to be symmetric across
bid/ask, so a^bid(q,tau,b) is compared against a^ask(-q,tau,b) at every
grid point.

Does not modify the market dynamics, Hamilton filter, reward, or the
trained model.

Run from repo root:
    python evaluate_hamilton_ppo_policy_grid.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO

from envs.make_envs import KAPPA

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
MODEL_PATH = REPO_ROOT / "models" / "hamilton_ppo" / "ppo_hamilton_v1.zip"
IMAGES_DIR = Path(r"C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images")

INVENTORY_SCALE = 10.0
MAX_DEPTH = -np.log(0.01) / KAPPA  # same formula as simulate_belief_weighted.py's MAX_DEPTH

Q_GRID = np.arange(-20, 21, 5)
TAU_GRID = np.array([1.0, 0.75, 0.5, 0.25, 0.05])
B_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)

sns.set_theme(style="whitegrid", context="notebook")
_PALETTE = sns.color_palette("deep")


def denormalise_depth(action_value: float) -> float:
    """Inverse of HamiltonPPOWrapper's pass-through normalised action ->
    TradingEnvironment.normalise_action(inverse=True): depth = (a+1)/2 * MAX_DEPTH."""
    return (action_value + 1.0) / 2.0 * MAX_DEPTH


def build_grid_df(model) -> pd.DataFrame:
    rows = []
    for q in Q_GRID:
        q_scaled = float(np.tanh(q / INVENTORY_SCALE))
        for tau in TAU_GRID:
            for b in B_GRID:
                obs = np.array([q_scaled, tau, b], dtype=np.float32)
                action, _ = model.predict(obs, deterministic=True)
                bid_action, ask_action = float(action[0]), float(action[1])
                rows.append(dict(
                    q=int(q), tau=float(tau), b=float(b), q_scaled=q_scaled,
                    bid_action=bid_action, ask_action=ask_action,
                    bid_depth=denormalise_depth(bid_action),
                    ask_depth=denormalise_depth(ask_action),
                ))
    return pd.DataFrame(rows)


def symmetry_check(df: pd.DataFrame) -> pd.DataFrame:
    """a^bid(q,tau,b) vs a^ask(-q,tau,b) at every grid point."""
    idx = df.set_index(["q", "tau", "b"])
    rows = []
    for q in Q_GRID:
        for tau in TAU_GRID:
            for b in B_GRID:
                bid_here = idx.loc[(int(q), float(tau), float(b)), "bid_action"]
                ask_mirror = idx.loc[(int(-q), float(tau), float(b)), "ask_action"]
                rows.append(dict(q=int(q), tau=float(tau), b=float(b),
                                  bid_action=bid_here, ask_action_mirror=ask_mirror,
                                  symmetry_error=bid_here - ask_mirror))
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. bid/ask action vs belief, for selected (q, tau) states ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    selected_q = [-20, -10, 0, 10, 20]
    selected_tau = 1.0
    for ax, action_col, title in zip(axes, ["bid_action", "ask_action"], ["Bid action vs belief", "Ask action vs belief"]):
        for i, q in enumerate(selected_q):
            sub = df[(df["q"] == q) & (df["tau"] == selected_tau)].sort_values("b")
            ax.plot(sub["b"], sub[action_col], marker="o", color=_PALETTE[i % len(_PALETTE)], label=f"q={q}")
        ax.set_xlabel("Belief b")
        ax.set_ylabel(action_col.replace("_", " "))
        ax.set_title(f"{title} (tau={selected_tau})")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=8, title="inventory")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "hamilton_ppo_action_vs_belief_v1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- 2. bid/ask action vs inventory, for selected (tau, b) states ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    selected_b = [0.0, 0.3, 0.5, 0.7, 1.0]
    selected_tau2 = 1.0
    for ax, action_col, title in zip(axes, ["bid_action", "ask_action"], ["Bid action vs inventory", "Ask action vs inventory"]):
        for i, b in enumerate(selected_b):
            sub = df[(df["b"] == b) & (df["tau"] == selected_tau2)].sort_values("q")
            ax.plot(sub["q"], sub[action_col], marker="o", color=_PALETTE[i % len(_PALETTE)], label=f"b={b}")
        ax.set_xlabel("Inventory q (shares)")
        ax.set_ylabel(action_col.replace("_", " "))
        ax.set_title(f"{title} (tau={selected_tau2})")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=8, title="belief")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "hamilton_ppo_action_vs_inventory_v1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- 3. Heatmaps over (inventory, belief) at tau=1.0 and tau=0.05 ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for row, tau in enumerate([1.0, 0.05]):
        for col, action_col in enumerate(["bid_action", "ask_action"]):
            ax = axes[row, col]
            sub = df[df["tau"] == tau]
            pivot = sub.pivot(index="b", columns="q", values=action_col).sort_index(ascending=False)
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                            extent=[Q_GRID.min(), Q_GRID.max(), B_GRID.min(), B_GRID.max()])
            ax.set_xlabel("Inventory q")
            ax.set_ylabel("Belief b")
            ax.set_title(f"{action_col.replace('_',' ')} heatmap (tau={tau})")
            plt.colorbar(im, ax=ax, label="normalised action")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "hamilton_ppo_action_heatmaps_v1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- 4. Behaviour near the terminal horizon: action vs tau for a few inventories, b=stationary-ish ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    selected_q3 = [-20, -10, 0, 10, 20]
    selected_b3 = 0.7  # close to the stationary prior (~0.714)
    for ax, action_col, title in zip(axes, ["bid_action", "ask_action"], ["Bid action vs tau", "Ask action vs tau"]):
        for i, q in enumerate(selected_q3):
            sub = df[(df["q"] == q) & (df["b"] == selected_b3)].sort_values("tau", ascending=False)
            ax.plot(sub["tau"], sub[action_col], marker="o", color=_PALETTE[i % len(_PALETTE)], label=f"q={q}")
        ax.invert_xaxis()
        ax.set_xlabel("tau (1=start, 0=terminal)")
        ax.set_ylabel(action_col.replace("_", " "))
        ax.set_title(f"{title} (b={selected_b3})")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=8, title="inventory")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "hamilton_ppo_action_vs_tau_v1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Plots saved to {IMAGES_DIR}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = PPO.load(str(MODEL_PATH))
    print(f"Loaded model from {MODEL_PATH}")
    print(f"Grid: |q|={len(Q_GRID)} x |tau|={len(TAU_GRID)} x |b|={len(B_GRID)} = "
          f"{len(Q_GRID)*len(TAU_GRID)*len(B_GRID)} points")

    df = build_grid_df(model)
    grid_path = RESULTS_DIR / "hamilton_ppo_policy_grid_v1.csv"
    df.to_csv(grid_path, index=False)
    print(f"Policy grid saved to {grid_path}")

    sym_df = symmetry_check(df)
    sym_path = RESULTS_DIR / "hamilton_ppo_policy_grid_symmetry_v1.csv"
    sym_df.to_csv(sym_path, index=False)
    print(f"Symmetry-check table saved to {sym_path}")

    print("\n" + "=" * 78)
    print("SYMMETRY CHECK: a_bid(q,tau,b) vs a_ask(-q,tau,b)")
    print("=" * 78)
    print(f"  mean |symmetry error| = {sym_df['symmetry_error'].abs().mean():.4f}")
    print(f"  max  |symmetry error| = {sym_df['symmetry_error'].abs().max():.4f}")
    worst = sym_df.reindex(sym_df["symmetry_error"].abs().sort_values(ascending=False).index).head(5)
    print("\n  5 worst symmetry violations:")
    print(worst.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Economic-direction checks ---
    print("\n" + "=" * 78)
    print("ECONOMIC-DIRECTION CHECKS")
    print("=" * 78)

    # Does bid action decrease (tighter -> less aggressive buying) as q increases (more long)?
    corr_bid_q = df.groupby("q")["bid_action"].mean()
    print("\nMean bid_action by inventory q (expect: decreasing as q increases, i.e. less "
          "aggressive buying when already long):")
    print(corr_bid_q.to_string(float_format=lambda x: f"{x:.4f}"))

    corr_ask_q = df.groupby("q")["ask_action"].mean()
    print("\nMean ask_action by inventory q (expect: increasing as q increases, i.e. more "
          "aggressive/wider selling when already long):")
    print(corr_ask_q.to_string(float_format=lambda x: f"{x:.4f}"))

    corr_bid_b = df.groupby("b")["bid_action"].mean()
    corr_ask_b = df.groupby("b")["ask_action"].mean()
    print("\nMean bid_action by belief b:")
    print(corr_bid_b.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nMean ask_action by belief b:")
    print(corr_ask_b.to_string(float_format=lambda x: f"{x:.4f}"))

    # Terminal liquidation pressure: does |action skew| grow as tau->0 for q!=0?
    print("\nMean |bid-ask skew| = |bid_action - (-ask_action)|-style asymmetry by tau, at q=20 and q=-20:")
    for q in (20, -20):
        sub = df[df["q"] == q].groupby("tau")[["bid_action", "ask_action"]].mean()
        print(f"  q={q}:")
        print(sub.to_string(float_format=lambda x: f"{x:.4f}"))

    make_plots(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
