"""
plot_agent_comparison.py
------------------------------
Diagnostic and comparison figures for the three-agent comparison
(hamilton_ppo, return_mlp_ppo, return_lstm_ppo vs the four analytic
benchmarks). Pure plotting / lightweight policy-surface querying -- does
not re-run full episode evaluation (reads evaluate_agents_common.py's
saved CSV) except for the direct model.predict() grid queries used by the
policy-surface functions (10-13), which need no environment stepping.

Colour convention: reuses the exact repository convention already
established in plot_hamilton_ppo_multiseed.py -- sns.set_theme(style=
"whitegrid", context="notebook") + sns.color_palette("deep"), assigned
by zipping the palette with category labels. No colour is hand-picked.

Run from repo root:
    python plot_agent_comparison.py --episodes results/agent_comparison_episodes.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless/batch: this module only ever saves figures to disk, never plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from envs.hamilton_ppo_wrapper import DEFAULT_INVENTORY_SCALE
from envs.return_ppo_wrapper import DEFAULT_RETURN_SCALE

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
IMAGES_DIR = Path(r"C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images")

RL_AGENT_TYPES = ("hamilton_ppo", "return_mlp_ppo", "return_lstm_ppo")
ALL_AGENT_TYPES = ("oracle", "belief_weighted", "randomised", "naive") + RL_AGENT_TYPES

sns.set_theme(style="whitegrid", context="notebook")
_PALETTE = sns.color_palette("deep")
AGENT_COLOR = {a: _PALETTE[i] for i, a in enumerate(ALL_AGENT_TYPES)}

Q_GRID = np.arange(-20, 21, 5)
TAU_GRID = np.array([1.0, 0.75, 0.5, 0.25, 0.05])
B_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)
R_GRID = np.round(np.arange(-3.0, 3.01, 0.5), 2)  # in units of return_scale (tanh argument)


def _savefig(fig, name: str):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def load_episodes(path=None) -> pd.DataFrame:
    path = Path(path) if path else RESULTS_DIR / "agent_comparison_episodes.csv"
    return pd.read_csv(path)


# ======================================================================
# 1. Objective comparison
# ======================================================================
def plot_1_objective_comparison(episodes: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    present = [a for a in ALL_AGENT_TYPES if a in episodes["agent_type"].unique()]
    data = [episodes[episodes["agent_type"] == a]["full_objective"].values for a in present]
    bp = ax.boxplot(data, positions=range(len(present)), widths=0.6, patch_artist=True, showfliers=True)
    for patch, a in zip(bp["boxes"], present):
        patch.set_facecolor(AGENT_COLOR[a])
        patch.set_alpha(0.7)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=30, ha="right")
    ax.set_ylabel("full_objective")
    ax.set_title("Full objective by agent")
    _savefig(fig, "agent_comparison_1_objective.png")


# ======================================================================
# 2. Raw-PnL comparison
# ======================================================================
def plot_2_raw_pnl_comparison(episodes: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    present = [a for a in ALL_AGENT_TYPES if a in episodes["agent_type"].unique()]
    data = [episodes[episodes["agent_type"] == a]["raw_pnl"].values for a in present]
    bp = ax.boxplot(data, positions=range(len(present)), widths=0.6, patch_artist=True, showfliers=True)
    for patch, a in zip(bp["boxes"], present):
        patch.set_facecolor(AGENT_COLOR[a])
        patch.set_alpha(0.7)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=30, ha="right")
    ax.set_ylabel("raw_pnl")
    ax.set_title("Raw PnL by agent")
    _savefig(fig, "agent_comparison_2_raw_pnl.png")


# ======================================================================
# 3. Training-seed dispersion (RL agents only, requires >=2 training seeds present)
# ======================================================================
def plot_3_training_seed_dispersion(episodes: pd.DataFrame):
    rl = episodes[episodes["agent_type"].isin(RL_AGENT_TYPES)]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, agent_type in enumerate(RL_AGENT_TYPES):
        sub = rl[rl["agent_type"] == agent_type]
        if sub["training_seed"].nunique() < 1:
            continue
        seed_means = sub.groupby("training_seed")["full_objective"].mean()
        ax.scatter([i] * len(seed_means), seed_means.values, color=AGENT_COLOR[agent_type],
                   s=60, label=agent_type if i == 0 else None, zorder=3)
        if len(seed_means) > 1:
            ax.errorbar([i], [seed_means.mean()], yerr=[seed_means.std(ddof=1)],
                        fmt="_", color="black", capsize=6, markersize=20)
    ax.set_xticks(range(len(RL_AGENT_TYPES)))
    ax.set_xticklabels(RL_AGENT_TYPES, rotation=20, ha="right")
    ax.set_ylabel("mean full_objective per training seed")
    ax.set_title("Between-training-seed dispersion")
    _savefig(fig, "agent_comparison_3_training_seed_dispersion.png")


# ======================================================================
# 4. Learning curves (from train_agents.py's saved training_curve CSVs).
#
# total_timesteps in these CSVs is now, per PeriodicEvalCallback's
# _on_rollout_start()-based evaluation (see train_agents.py), the number
# of completed environment timesteps whose associated PPO training
# updates have actually finished -- not merely env-steps observed so far.
# ======================================================================
def plot_4_learning_curves(run_tag: str = "diag"):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    any_data = False
    for agent_type in RL_AGENT_TYPES:
        path = RESULTS_DIR / f"{agent_type}_training_curve_{run_tag}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        ax.plot(df["total_timesteps"], df["mean_cumulative_reward"], marker="o",
                color=AGENT_COLOR[agent_type], label=agent_type)
        # Shaded band: +/- 1 evaluation-episode standard deviation across the
        # fixed held-out eval seeds at that checkpoint -- NOT a confidence
        # interval (no standard-error scaling applied).
        ax.fill_between(df["total_timesteps"],
                         df["mean_cumulative_reward"] - df["std_cumulative_reward"],
                         df["mean_cumulative_reward"] + df["std_cumulative_reward"],
                         color=AGENT_COLOR[agent_type], alpha=0.15)
        any_data = True
    ax.set_xlabel("Completed training timesteps")
    ax.set_ylabel("Mean held-out objective")
    ax.set_title(f"Learning curves (run_tag={run_tag})\n"
                 "Shaded band: ±1 evaluation-episode standard deviation (not a confidence interval)")
    if any_data:
        ax.legend()
    _savefig(fig, f"agent_comparison_4_learning_curves_{run_tag}.png")


# ======================================================================
# 5. Inventory trajectories -- requires per-step traces; approximated here
# from mean/terminal/max summary stats already in the episodes CSV
# (a full per-step trace plot would require re-running instrumented
# episodes, out of scope for a pure-plotting script per this module's
# docstring).
# ======================================================================
def plot_5_inventory_summary_by_agent(episodes: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    present = [a for a in ALL_AGENT_TYPES if a in episodes["agent_type"].unique()]
    means = episodes.groupby("agent_type")[["mean_abs_inventory", "max_abs_inventory", "terminal_abs_inventory"]].mean()
    x = np.arange(len(present))
    width = 0.25
    for i, col in enumerate(["mean_abs_inventory", "max_abs_inventory", "terminal_abs_inventory"]):
        ax.bar(x + (i - 1) * width, [means.loc[a, col] for a in present], width, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=30, ha="right")
    ax.set_ylabel("shares")
    ax.set_title("Inventory summary by agent (mean / max / terminal, |.|)")
    ax.legend()
    _savefig(fig, "agent_comparison_5_inventory_summary.png")


# ======================================================================
# 6. Inventory distributions (mean_signed_inventory)
# ======================================================================
def plot_6_inventory_distributions(episodes: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    present = [a for a in ALL_AGENT_TYPES if a in episodes["agent_type"].unique()]
    data = [episodes[episodes["agent_type"] == a]["mean_signed_inventory"].values for a in present]
    bp = ax.boxplot(data, positions=range(len(present)), widths=0.6, patch_artist=True, showfliers=True)
    for patch, a in zip(bp["boxes"], present):
        patch.set_facecolor(AGENT_COLOR[a])
        patch.set_alpha(0.7)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=30, ha="right")
    ax.set_ylabel("mean_signed_inventory")
    ax.set_title("Signed inventory distribution by agent")
    _savefig(fig, "agent_comparison_6_signed_inventory_distribution.png")


# ======================================================================
# 7. Terminal inventory distributions
# ======================================================================
def plot_7_terminal_inventory_distributions(episodes: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    present = [a for a in ALL_AGENT_TYPES if a in episodes["agent_type"].unique()]
    data = [episodes[episodes["agent_type"] == a]["terminal_signed_inventory"].values for a in present]
    bp = ax.boxplot(data, positions=range(len(present)), widths=0.6, patch_artist=True, showfliers=True)
    for patch, a in zip(bp["boxes"], present):
        patch.set_facecolor(AGENT_COLOR[a])
        patch.set_alpha(0.7)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=30, ha="right")
    ax.set_ylabel("terminal_signed_inventory")
    ax.set_title("Terminal signed inventory distribution by agent")
    _savefig(fig, "agent_comparison_7_terminal_inventory_distribution.png")


# ======================================================================
# 8. Bid and ask action distributions
# ======================================================================
def plot_8_bid_ask_action_distributions(episodes: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    present = [a for a in ALL_AGENT_TYPES if a in episodes["agent_type"].unique()]
    for ax, col, title in zip(axes, ["bid_action_mean", "ask_action_mean"], ["Bid action mean", "Ask action mean"]):
        data = [episodes[episodes["agent_type"] == a][col].values for a in present]
        bp = ax.boxplot(data, positions=range(len(present)), widths=0.6, patch_artist=True, showfliers=True)
        for patch, a in zip(bp["boxes"], present):
            patch.set_facecolor(AGENT_COLOR[a])
            patch.set_alpha(0.7)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=30, ha="right")
        ax.set_title(title)
    axes[0].set_ylabel("action value [-1, 1]")
    _savefig(fig, "agent_comparison_8_bid_ask_action_distributions.png")


# ======================================================================
# 9. Mean quoted spread
# ======================================================================
def plot_9_mean_quoted_spread(episodes: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    present = [a for a in ALL_AGENT_TYPES if a in episodes["agent_type"].unique()]
    data = [episodes[episodes["agent_type"] == a]["mean_quoted_spread"].values for a in present]
    bp = ax.boxplot(data, positions=range(len(present)), widths=0.6, patch_artist=True, showfliers=True)
    for patch, a in zip(bp["boxes"], present):
        patch.set_facecolor(AGENT_COLOR[a])
        patch.set_alpha(0.7)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=30, ha="right")
    ax.set_ylabel("mean_quoted_spread (bid_depth + ask_depth)")
    ax.set_title("Mean quoted spread by agent")
    _savefig(fig, "agent_comparison_9_mean_quoted_spread.png")


# ======================================================================
# 10. Policy surfaces over inventory and time (all 3 RL agents; belief/
# return held at a fixed reference value so q,tau are isolated)
# ======================================================================
def _load_model(agent_type: str, path):
    return (RecurrentPPO if agent_type == "return_lstm_ppo" else PPO).load(str(path))


def _predict_ff(model, obs):
    action, _ = model.predict(obs, deterministic=True)
    return action


def _predict_lstm_zero_state(model, obs_sequence):
    """Feed obs_sequence (list of (3,) arrays) through the LSTM from a
    ZERO initial state (episode_start=True at the first obs), returning
    the action after the LAST observation. This is the 'zero-state
    response' / 'response after a predefined ... sequence' diagnostic --
    NOT a claim that this is the policy's behaviour at an arbitrary point
    in a real trajectory (see plot_13 docstring)."""
    state = None
    episode_start = np.array([True])
    action = None
    for obs in obs_sequence:
        action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=True)
        episode_start = np.array([False])
    return action


def plot_10_policy_surface_q_tau(agent_type: str, model_path):
    model = _load_model(agent_type, model_path)
    rows = []
    for q in Q_GRID:
        q_scaled = float(np.tanh(q / DEFAULT_INVENTORY_SCALE))
        for tau in TAU_GRID:
            if agent_type == "hamilton_ppo":
                third = 0.5  # fixed reference belief -- isolates (q, tau)
            else:
                third = 0.0  # fixed reference (zero) scaled return -- isolates (q, tau)
            obs = np.array([q_scaled, tau, third], dtype=np.float32)
            if agent_type == "return_lstm_ppo":
                action = _predict_lstm_zero_state(model, [obs])
            else:
                action = _predict_ff(model, obs)
            rows.append(dict(q=q, tau=tau, bid=float(action[0]), ask=float(action[1])))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    for ax, col, title in zip(axes, ["bid", "ask"], ["Bid action", "Ask action"]):
        piv = df.pivot(index="tau", columns="q", values=col)
        im = ax.imshow(piv.values, aspect="auto", origin="lower",
                        extent=[Q_GRID.min(), Q_GRID.max(), TAU_GRID.min(), TAU_GRID.max()], cmap="viridis")
        ax.set_xlabel("inventory q")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel("tau (time remaining)")
    fig.suptitle(f"{agent_type}: policy surface over (q, tau), reference third-observation-component held fixed")
    _savefig(fig, f"agent_comparison_10_policy_surface_q_tau_{agent_type}.png")


# ======================================================================
# 11. Hamilton-PPO policy surface over belief
# ======================================================================
def plot_11_hamilton_policy_surface_belief(model_path):
    model = _load_model("hamilton_ppo", model_path)
    rows = []
    for q in (-15, 0, 15):
        q_scaled = float(np.tanh(q / DEFAULT_INVENTORY_SCALE))
        for tau in (1.0, 0.5, 0.05):
            for b in B_GRID:
                obs = np.array([q_scaled, tau, b], dtype=np.float32)
                action = _predict_ff(model, obs)
                rows.append(dict(q=q, tau=tau, belief=b, bid=float(action[0]), ask=float(action[1])))
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for q in (-15, 0, 15):
        sub = df[(df["q"] == q) & (df["tau"] == 0.5)]
        ax.plot(sub["belief"], sub["ask"], marker="o", label=f"ask, q={q}")
    ax.set_xlabel("belief b")
    ax.set_ylabel("action value")
    ax.set_title("Hamilton PPO: ask action vs belief (tau=0.5)")
    ax.legend()
    _savefig(fig, "agent_comparison_11_hamilton_policy_surface_belief.png")


# ======================================================================
# 12. Raw-return MLP response surface over current return
# ======================================================================
def plot_12_return_mlp_response_surface(model_path):
    model = _load_model("return_mlp_ppo", model_path)
    rows = []
    for q in (-15, 0, 15):
        q_scaled = float(np.tanh(q / DEFAULT_INVENTORY_SCALE))
        for tau in (1.0, 0.5, 0.05):
            for r in R_GRID:
                scaled_r = float(np.tanh(r))  # r already expressed in units of return_scale, so tanh(r) directly
                obs = np.array([q_scaled, tau, scaled_r], dtype=np.float32)
                action = _predict_ff(model, obs)
                rows.append(dict(q=q, tau=tau, r=r, bid=float(action[0]), ask=float(action[1])))
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for q in (-15, 0, 15):
        sub = df[(df["q"] == q) & (df["tau"] == 0.5)]
        ax.plot(sub["r"], sub["ask"], marker="o", label=f"ask, q={q}")
    ax.set_xlabel(f"scaled return r (units of return_scale={DEFAULT_RETURN_SCALE:.4f})")
    ax.set_ylabel("action value")
    ax.set_title("return_mlp_ppo: ask action vs current return (tau=0.5)")
    ax.legend()
    _savefig(fig, "agent_comparison_12_return_mlp_response_surface.png")


# ======================================================================
# 13. Recurrent-PPO action response to CONTROLLED return histories.
#
# Do NOT describe this as "the policy surface" over the current
# observation alone -- the recurrent policy's action depends on the full
# history via its hidden state, which this function explicitly controls
# (zero-state / calm-sequence / jump-sequence), rather than reading state
# off an arbitrary point in a real trajectory. Each condition is labelled.
# ======================================================================
def plot_13_recurrent_response_to_controlled_histories(model_path, history_len: int = 20):
    model = _load_model("return_lstm_ppo", model_path)
    q_scaled = 0.0  # fixed reference inventory
    tau = 0.5       # fixed reference time

    def make_obs(r_scaled):
        return np.array([q_scaled, tau, float(np.tanh(r_scaled))], dtype=np.float32)

    conditions = {}

    # (a) zero-state response: single observation, no history, state=None/episode_start=True
    conditions["zero_state_calm"] = [make_obs(0.0)]

    # (b) predefined CALM sequence: history_len steps of ~zero return, then a query obs
    conditions["calm_history"] = [make_obs(0.0)] * history_len

    # (c) predefined JUMP sequence: a single large positive return in the middle, then calm
    jump_seq = [make_obs(0.0)] * (history_len // 2) + [make_obs(3.0)] + [make_obs(0.0)] * (history_len // 2)
    conditions["jump_then_calm"] = jump_seq

    # (d) sustained jump sequence: repeated large positive returns
    conditions["sustained_jumps"] = [make_obs(3.0)] * history_len

    rows = []
    for label, obs_seq in conditions.items():
        action = _predict_lstm_zero_state(model, obs_seq)
        rows.append(dict(condition=label, bid=float(action[0]), ask=float(action[1])))
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["bid"], width, label="bid")
    ax.bar(x + width / 2, df["ask"], width, label="ask")
    ax.set_xticks(x)
    ax.set_xticklabels(df["condition"], rotation=20, ha="right")
    ax.set_ylabel("action value")
    ax.set_title("return_lstm_ppo: action after CONTROLLED return histories (labelled conditions, q=0, tau=0.5)")
    ax.legend()
    _savefig(fig, "agent_comparison_13_recurrent_controlled_history_response.png")
    return df


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=str, default=str(RESULTS_DIR / "agent_comparison_episodes.csv"))
    p.add_argument("--run-tag", type=str, default="diag")
    p.add_argument("--hamilton-model", type=str,
                    default=str(REPO_ROOT / "models" / "hamilton_ppo" / "ppo_hamilton_ppo_diag.zip"))
    p.add_argument("--return-mlp-model", type=str,
                    default=str(REPO_ROOT / "models" / "return_mlp_ppo" / "ppo_return_mlp_ppo_diag.zip"))
    p.add_argument("--return-lstm-model", type=str,
                    default=str(REPO_ROOT / "models" / "return_lstm_ppo" / "ppo_return_lstm_ppo_diag.zip"))
    return p.parse_args()


def main():
    args = parse_args()
    episodes = load_episodes(args.episodes)

    print("Generating figures 1-9 (from episode-level CSV)...")
    plot_1_objective_comparison(episodes)
    plot_2_raw_pnl_comparison(episodes)
    plot_3_training_seed_dispersion(episodes)
    plot_4_learning_curves(args.run_tag)
    plot_5_inventory_summary_by_agent(episodes)
    plot_6_inventory_distributions(episodes)
    plot_7_terminal_inventory_distributions(episodes)
    plot_8_bid_ask_action_distributions(episodes)
    plot_9_mean_quoted_spread(episodes)

    print("\nGenerating figures 10-13 (direct model.predict() grid queries)...")
    for agent_type, path in (("hamilton_ppo", args.hamilton_model),
                              ("return_mlp_ppo", args.return_mlp_model),
                              ("return_lstm_ppo", args.return_lstm_model)):
        if Path(path).exists():
            plot_10_policy_surface_q_tau(agent_type, path)
    if Path(args.hamilton_model).exists():
        plot_11_hamilton_policy_surface_belief(args.hamilton_model)
    if Path(args.return_mlp_model).exists():
        plot_12_return_mlp_response_surface(args.return_mlp_model)
    if Path(args.return_lstm_model).exists():
        plot_13_recurrent_response_to_controlled_histories(args.return_lstm_model)

    print("\nDone.")


if __name__ == "__main__":
    main()
