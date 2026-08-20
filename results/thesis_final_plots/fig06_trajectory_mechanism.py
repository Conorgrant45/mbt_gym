"""
fig06_trajectory_mechanism.py
---------------------------------
Single representative episode (path_seed=290000, learner_seed=0, checkpoint
1,000,000), two stacked panels sharing a time axis, event-driven (NO
smoothing/interpolation -- every series is plotted as its genuine piecewise-
constant step function between observed events, matching the environment):

  (a) inventory_after (step function) for belief-state PPO, raw-return MLP
      PPO, and the belief-weighted analytical benchmark.
  (b) bid/ask quoted depth (step functions) for the belief-state policy, with
      fill events marked as tick markers on the corresponding side's line.

(An earlier version had a third panel -- regime shading + the belief-state
policy's Hamilton-filter posterior -- removed on request; the regime/belief
tracking story is already covered by figA1_filter_calibration.py.)

Source (read-only): trajectory_subset.csv -- see 05_inventory_time_profile.py's
docstring for this file's known small-subset scope (5 episodes/policy, only
learner_seed=0, only checkpoint 1,000,000, only belief_weighted among the
3 benchmarks). This figure only needs ONE of those 5 episodes and exactly
the policies present, so it is fully covered by the existing export --
no data gap for this figure specifically.
"""
import numpy as np
import pandas as pd

import fig_style as FS

# ======================================================================
# CONFIGURATION
# ======================================================================
FIG_NAME = "fig06_trajectory_mechanism"
PATH_SEED = 290_000
LEARNER_SEED = 0
CHECKPOINT = 1_000_000

INPUT_CSV = FS.RESULTS_DIR / "trajectory_subset.csv"

INVENTORY_POLICIES = ["hamilton_ppo", "return_mlp_ppo", "belief_weighted"]
DEPTH_POLICY = "hamilton_ppo"


def _episode(df, policy, learner_seed=None):
    if learner_seed is None:
        sub = df[(df["policy"] == policy) & df["learner_seed"].isna() & df["path_seed"] == PATH_SEED]
        sub = df[(df["policy"] == policy) & (df["path_seed"] == PATH_SEED) & df["learner_seed"].isna()]
    else:
        sub = df[(df["policy"] == policy) & (df["learner_seed"] == learner_seed)
                 & (df["checkpoint_transition"] == CHECKPOINT) & (df["path_seed"] == PATH_SEED)]
    return sub.sort_values("step_index")


def main():
    df = pd.read_csv(INPUT_CSV)
    available = set(df["policy"].unique())
    needed = set(INVENTORY_POLICIES) | {DEPTH_POLICY, "hamilton_ppo"}
    missing = needed - available
    if missing:
        raise ValueError(f"trajectory_subset.csv is missing policies {missing} needed for fig06 "
                          f"(available: {available})")
    if PATH_SEED not in df["path_seed"].unique():
        raise ValueError(f"path_seed={PATH_SEED} not found in {INPUT_CSV}; available: "
                          f"{sorted(df['path_seed'].unique())}")

    ham = _episode(df, "hamilton_ppo", LEARNER_SEED)
    mlp = _episode(df, "return_mlp_ppo", LEARNER_SEED)
    bw = _episode(df, "belief_weighted", None)

    # regime path must be identical across policies for this path_seed (same exogenous draw)
    for name, ep in [("return_mlp_ppo", mlp), ("belief_weighted", bw)]:
        if not np.array_equal(ep["hidden_regime"].to_numpy(), ham["hidden_regime"].to_numpy()):
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
        "n_events": int(len(ham)),
        "n_regime_switches": n_regime_switches,
        "n_bid_fills": int(fill_bid_mask.sum()), "n_ask_fills": int(fill_ask_mask.sum()),
        "note": "Illustrative single-episode mechanism figure (n=1 path), not a statistically powered result.",
    }
    FS.save_figure(fig, FIG_NAME, stats)


if __name__ == "__main__":
    main()
