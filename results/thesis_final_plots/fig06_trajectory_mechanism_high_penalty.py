"""
fig06_trajectory_mechanism_high_penalty.py
---------------------------------------------
Companion to fig06_trajectory_mechanism.py: the SAME representative episode
(path_seed=290000, learner_seed=0, checkpoint 1,000,000) and the same two
stacked panels, but under the higher-penalty calibration (phi=0.10,
alpha=0.010) from the inventory-penalty-sensitivity experiment, instead of
the original calibration (phi=0.01, alpha=0.001). Does NOT modify or
replace fig06_trajectory_mechanism.py -- run both to compare directly,
since both use the identical exogenous path (same holdout seed, same
regime/arrival draw).

  (a) inventory_after (step function) for belief-state PPO, raw-return MLP
      PPO, and the belief-weighted analytical benchmark.
  (b) bid/ask quoted depth (step functions) for the belief-state policy, with
      fill events marked as tick markers on the corresponding side's line.

Source (read-only): results/inventory_penalty_sensitivity/event_level/
high_penalty/*.parquet -- per-(policy[, learner_seed]) event-level export,
500 holdout episodes each (seeds 290000-290499, the same holdout set
final_reduced_exploration_architecture_comparison uses). Column names differ
from trajectory_subset.csv (event_index/event_time/latent_region vs.
step_index/time/hidden_regime) but the content needed here is identical;
mapped below rather than renamed in the source files.
"""
import numpy as np
import pandas as pd

import fig_style as FS

# ======================================================================
# CONFIGURATION
# ======================================================================
FIG_NAME = "fig06_trajectory_mechanism_high_penalty"
PATH_SEED = 290_000
LEARNER_SEED = 0
CHECKPOINT = 1_000_000
PENALTY_CALIBRATION = "high_penalty"
PHI, ALPHA = 0.10, 0.010

EVENT_LEVEL_DIR = FS.PROJECT_ROOT / "results" / "inventory_penalty_sensitivity" / "event_level" / PENALTY_CALIBRATION

INVENTORY_POLICIES = ["hamilton_ppo", "return_mlp_ppo", "belief_weighted"]
DEPTH_POLICY = "hamilton_ppo"

COLUMN_MAP = {
    "event_index": "step_index",
    "event_time": "time",
    "latent_regime": "hidden_regime",
    "holdout_episode_seed": "path_seed",
}


def _load(stem: str) -> pd.DataFrame:
    path = EVENT_LEVEL_DIR / f"{stem}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} -- rerun the inventory-penalty-sensitivity holdout export first.")
    return pd.read_parquet(path).rename(columns=COLUMN_MAP)


def _episode(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["path_seed"] == PATH_SEED]
    if sub.empty:
        raise ValueError(f"path_seed={PATH_SEED} not found (available: {sorted(df['path_seed'].unique())[:5]}...)")
    return sub.sort_values("step_index")


def main():
    ham = _episode(_load("hamilton_ppo_seed0"))
    mlp = _episode(_load("return_mlp_ppo_seed0"))
    bw = _episode(_load("belief_weighted"))

    for name, ep in [("hamilton_ppo_seed0", ham), ("return_mlp_ppo_seed0", mlp), ("belief_weighted", bw)]:
        if name != "hamilton_ppo_seed0" and not np.array_equal(ep["hidden_regime"].to_numpy(), ham["hidden_regime"].to_numpy()):
            raise ValueError(f"hidden_regime path differs between hamilton_ppo and {name} for "
                              f"path_seed={PATH_SEED} -- expected an identical exogenous path.")

    fig, axes = FS.new_figure(width="full", height_in=3.6, nrows=2, ncols=1, sharex=True)
    ax_inv, ax_depth = axes

    # ------------------------------------------------------------------
    # Panel (a): inventory
    # ------------------------------------------------------------------
    for policy, ep in [("hamilton_ppo", ham), ("return_mlp_ppo", mlp), ("belief_weighted", bw)]:
        ax_inv.step(ep["time"].to_numpy(), ep["inventory_after"].to_numpy(), where="post",
                     color=FS.COLORS[policy], linestyle=FS.LINESTYLES[policy], linewidth=FS.LINEWIDTHS[policy],
                     zorder=2, label=FS.display(policy))
    ax_inv.axhline(0.0, color="0.7", linewidth=0.6, zorder=0)
    ax_inv.set_ylabel("Inventory")
    ax_inv.legend(loc="upper left", fontsize=6.5, ncol=1, frameon=True)
    FS.panel_letter(ax_inv, "a")

    # ------------------------------------------------------------------
    # Panel (b): quoted depths + fills, belief-state policy only
    # ------------------------------------------------------------------
    tt = ham["time"].to_numpy()
    ax_depth.step(tt, ham["bid_depth"].to_numpy(), where="post", color=FS.COLORS["hamilton_ppo"],
                   linestyle="-", linewidth=1.1, zorder=2, label="Bid depth")
    ax_depth.step(tt, ham["ask_depth"].to_numpy(), where="post", color=FS.COLORS["hamilton_ppo"],
                   linestyle="--", linewidth=1.1, zorder=2, label="Ask depth")
    fill_bid_mask = ham["fill_bid"].to_numpy() == 1
    fill_ask_mask = ham["fill_ask"].to_numpy() == 1
    ax_depth.scatter(tt[fill_bid_mask], ham["bid_depth"].to_numpy()[fill_bid_mask], marker="|", s=90,
                       color="firebrick", zorder=3, label="Bid fill")
    ax_depth.scatter(tt[fill_ask_mask], ham["ask_depth"].to_numpy()[fill_ask_mask], marker="|", s=90,
                       color="black", zorder=3, label="Ask fill")
    ax_depth.set_ylabel("Quoted depth\n(Belief-state PPO)")
    ax_depth.set_xlabel("Time")
    ax_depth.legend(loc="upper right", fontsize=6.5, ncol=2, frameon=True)
    FS.panel_letter(ax_depth, "b")

    n_regime_switches = int(np.sum(np.diff(ham["hidden_regime"].to_numpy()) != 0))
    stats = {
        "figure": FIG_NAME, "path_seed": PATH_SEED, "learner_seed": LEARNER_SEED, "checkpoint": CHECKPOINT,
        "penalty_calibration": PENALTY_CALIBRATION, "phi": PHI, "alpha": ALPHA,
        "n_events": int(len(ham)),
        "n_regime_switches": n_regime_switches,
        "n_bid_fills": int(fill_bid_mask.sum()), "n_ask_fills": int(fill_ask_mask.sum()),
        "note": "Illustrative single-episode mechanism figure (n=1 path) under the higher-penalty "
                "calibration, same exogenous path (path_seed=290000) as fig06_trajectory_mechanism.py "
                "for direct comparison. Not a statistically powered result.",
    }
    FS.save_figure(fig, FIG_NAME, stats)


if __name__ == "__main__":
    main()
