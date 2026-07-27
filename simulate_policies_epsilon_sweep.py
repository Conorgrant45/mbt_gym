"""
simulate_policies_epsilon_sweep.py
------------------------------------
Sweeps the regime-1 adverse-selection jump size (epsilon) downward and
runs all four quoting policies (oracle, belief-weighted, randomised,
naive -- see simulate_belief_weighted.py) at each value, to see how each
policy's performance responds as the adverse-selection signal shrinks.

Standalone experiment. Does NOT modify simulate_belief_weighted.py,
compare_four_policies_paired.py, or envs/make_envs.py's existing default
behaviour -- reuses their pure functions (build_optimal_control,
get_control, normalise_depth) completely unchanged, and only supplies
epsilon/volatility as explicit arguments to the already-parametrised
make_regime_envs(). REGIME_PARAMS, NAIVE_DEPTH and the Hamilton filter's
jump_size are rebuilt locally per epsilon value here, since
simulate_belief_weighted.py hardcodes EPSILON from envs/make_envs.py at
import time.

Both regimes are given the SAME diffusion volatility (matching the
equal-volatility ablation in diagnostic_epsilon_sweep.py), so epsilon is
the only regime-distinguishing signal available to either the filter or
the optimal-control depth calculation. This isolates how much of each
regime-aware policy's edge over naive comes specifically from adverse-
selection awareness, as opposed to a diffusion-volatility difference.

Uses the same paired-episode methodology as compare_four_policies_paired.py:
for each episode index, all four policies see bit-for-bit identical
regime/arrival/jump/diffusion/fill draws (env re-built from the same seed
before each policy's rollout), differing only in which depths they quote
-- this reduces variance a lot relative to independent episodes, which
matters since a full sweep already means (n_epsilon x n_episodes x
4 policies) episode rollouts.

Run from repo root:
    python simulate_policies_epsilon_sweep.py
    python simulate_policies_epsilon_sweep.py --episodes 30
"""

import os
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import simulate_belief_weighted as SBW  # existing policy definitions -- READ ONLY
from envs.make_envs import (
    make_regime_envs, N_STEPS, STEP_SIZE, TRANSITION_MATRIX,
    R0_VOLATILITY, LAMBDA, KAPPA,
    PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION, INITIAL_PRICE,
)
from beliefs.hamilton_filter import HamiltonFilter

sns.set_theme(style="whitegrid", context="notebook")
_PALETTE = sns.color_palette("deep")

IMAGES_DIR = r'C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images'

PHI   = PER_STEP_INVENTORY_AVERSION
ALPHA = TERMINAL_INVENTORY_AVERSION
NOTIONAL_CAPITAL = INITIAL_PRICE * 10_000  # matches simulate_belief_weighted.py's convention

POLICIES       = ["oracle", "belief", "randomised", "naive"]
POLICY_LABELS  = {"oracle": "Oracle", "belief": "Belief-weighted",
                   "randomised": "Randomised", "naive": "Naive"}
POLICY_COLOR   = {"oracle": _PALETTE[0], "belief": _PALETTE[2],
                   "randomised": _PALETTE[1], "naive": _PALETTE[3]}

# Same diffusion volatility for both regimes -- matches
# diagnostic_epsilon_sweep.py's ablation, for a directly comparable
# experiment. Uses the calibrated regime-0 (calm) volatility as the
# shared baseline.
EQUAL_VOLATILITY     = R0_VOLATILITY
EQUAL_VOLATILITY_PCT = EQUAL_VOLATILITY / INITIAL_PRICE
EQUAL_SIGMA_STEP     = EQUAL_VOLATILITY_PCT * np.sqrt(STEP_SIZE)

EPSILON_VALUES = np.geomspace(0.5, 0.0005, 6).tolist()
N_EPISODES     = 20


# ======================================================================
# Epsilon-dependent objects (control tables, naive depth, filter)
# ======================================================================
def build_regime0_control():
    """Regime-0 control is epsilon-independent (eps=0 always) -- build once, reuse."""
    da, db, qag, qbg = SBW.build_optimal_control(kappa=KAPPA, lam=LAMBDA, eps=0.0, phi=PHI, alpha=ALPHA)
    return {"delta_ask": da, "delta_bid": db, "q_ask": qag, "q_bid": qbg}


def build_controls_for_epsilon(epsilon, regime0_control):
    da, db, qag, qbg = SBW.build_optimal_control(kappa=KAPPA, lam=LAMBDA, eps=epsilon, phi=PHI, alpha=ALPHA)
    regime1_control = {"delta_ask": da, "delta_bid": db, "q_ask": qag, "q_bid": qbg}
    return {0: regime0_control, 1: regime1_control}


def make_filter_for_epsilon(epsilon):
    epsilon_pct = epsilon / INITIAL_PRICE
    return HamiltonFilter(
        transition_matrix=TRANSITION_MATRIX,
        regime_volatilities=[EQUAL_SIGMA_STEP, EQUAL_SIGMA_STEP],
        r=0,
        jump_size=epsilon_pct,
        jump_intensity=LAMBDA,
        step_size=STEP_SIZE,
    )


# ======================================================================
# Single paired episode rollout for one policy
# ======================================================================
def run_episode(policy, controls, naive_depth, epsilon, episode_seed):
    np.random.seed(episode_seed)  # pairs the GLOBAL-state regime-switching draws
    env = make_regime_envs(
        switch_within_episode=True, seed=episode_seed, epsilon=epsilon,
        r0_volatility=EQUAL_VOLATILITY, r1_volatility=EQUAL_VOLATILITY,
    )  # fresh env, pairs the child RNG streams

    filt = make_filter_for_epsilon(epsilon) if policy in ("belief", "randomised") else None
    # Dedicated, independent RNG for randomised's own regime sample -- must
    # NOT be the global np.random stream (see compare_four_policies_paired.py).
    rng_random_policy = np.random.default_rng(episode_seed + 1000) if policy == "randomised" else None

    obs = env.reset()
    cash_0, inv_0, mid_0 = env.raw_cash, env.raw_inventory, env.raw_midprice
    mid = mid_0

    inv_abs_sum = 0.0
    running_inventory_penalty = 0.0
    obj_accum = 0.0
    total_fills = 0.0

    step = 0
    done = np.array([False])
    while not np.all(done):
        regime = env.current_regime
        obs_flat = np.array(obs).flatten()
        inventory = obs_flat[1]
        t_idx = env.current_step
        inv_sc = inventory / SBW.INV_UNIT

        if policy == "oracle":
            ctrl = controls[regime]
            ask, bid = SBW.get_control(ctrl["delta_ask"], ctrl["delta_bid"], ctrl["q_ask"], ctrl["q_bid"], t_idx, inv_sc)
        elif policy == "belief":
            belief = filt.update(mid)
            c0, c1 = controls[0], controls[1]
            a0, b0 = SBW.get_control(c0["delta_ask"], c0["delta_bid"], c0["q_ask"], c0["q_bid"], t_idx, inv_sc)
            a1, b1 = SBW.get_control(c1["delta_ask"], c1["delta_bid"], c1["q_ask"], c1["q_bid"], t_idx, inv_sc)
            ask = (1.0 - belief) * a0 + belief * a1
            bid = (1.0 - belief) * b0 + belief * b1
        elif policy == "randomised":
            belief = filt.update(mid)
            sampled_regime = int(rng_random_policy.choice(2, p=[1.0 - belief, belief]))
            ctrl = controls[sampled_regime]
            ask, bid = SBW.get_control(ctrl["delta_ask"], ctrl["delta_bid"], ctrl["q_ask"], ctrl["q_bid"], t_idx, inv_sc)
        else:  # naive
            ask = bid = naive_depth

        action = np.array([[SBW.normalise_depth(bid), SBW.normalise_depth(ask)]])
        obs, reward, done, info = env.step(action)
        mid = info["raw_midprice"]
        obj_accum += float(np.sum(reward))

        inv_after = float(info["raw_state"][1])
        inv_abs_sum += abs(inv_after)
        running_inventory_penalty += PHI * (inv_after ** 2) * STEP_SIZE
        step += 1

    steps_taken = step
    cash_T, inv_T, mid_T = info["raw_state"][0], info["raw_state"][1], info["raw_state"][3]
    raw_pnl = (cash_T + inv_T * mid_T) - (cash_0 + inv_0 * mid_0)
    terminal_inventory_penalty = ALPHA * (inv_T ** 2)
    full_objective = raw_pnl - running_inventory_penalty - terminal_inventory_penalty

    return dict(
        raw_pnl=raw_pnl,
        full_objective=full_objective,
        mean_absolute_inventory=inv_abs_sum / steps_taken,
        terminal_absolute_inventory=abs(inv_T),
        steps_taken=steps_taken,
    )


# ======================================================================
# Sweep
# ======================================================================
def run_sweep(epsilon_values, n_episodes, base_seed=12345):
    print("\nBuilding regime-0 optimal control (epsilon-independent, built once)...")
    t0 = time.time()
    regime0_control = build_regime0_control()
    print(f"  built in {time.time()-t0:.1f}s")

    rows = []
    for epsilon in epsilon_values:
        t_eps = time.time()
        print(f"\n--- epsilon={epsilon:.4f} (price units) ---")
        controls = build_controls_for_epsilon(epsilon, regime0_control)
        naive_depth = (1.0 / KAPPA) + (epsilon / 2.0)

        for ep in range(n_episodes):
            seed = base_seed + ep
            for policy in POLICIES:
                m = run_episode(policy, controls, naive_depth, epsilon, seed)
                assert m["steps_taken"] == N_STEPS, f"eps={epsilon} ep={ep} ({policy}): steps_taken != {N_STEPS}"
                row = dict(epsilon=epsilon, episode=ep, policy=policy)
                row.update(m)
                rows.append(row)

        print(f"  {n_episodes} paired episodes x {len(POLICIES)} policies done in {time.time()-t_eps:.1f}s")

    return pd.DataFrame(rows)


# ======================================================================
# Reporting / plotting
# ======================================================================
def summarise(df):
    df = df.copy()
    df["raw_pnl_pct"] = df["raw_pnl"] / NOTIONAL_CAPITAL * 100.0
    df["full_objective_pct"] = df["full_objective"] / NOTIONAL_CAPITAL * 100.0

    summary = df.groupby(["epsilon", "policy"]).agg(
        mean_pnl_pct=("raw_pnl_pct", "mean"),
        std_pnl_pct=("raw_pnl_pct", "std"),
        mean_obj_pct=("full_objective_pct", "mean"),
        std_obj_pct=("full_objective_pct", "std"),
        mean_abs_inv=("mean_absolute_inventory", "mean"),
    ).reset_index()
    return df, summary


def paired_diff_vs_naive(df, n_episodes):
    """For each epsilon and each non-naive policy, mean paired
    (policy - naive) full_objective difference with a 95% CI."""
    records = []
    for epsilon in sorted(df["epsilon"].unique(), reverse=True):
        sub = df[df["epsilon"] == epsilon]
        piv = sub.pivot(index="episode", columns="policy", values="full_objective")
        for policy in ("oracle", "belief", "randomised"):
            D = (piv[policy] - piv["naive"]) / NOTIONAL_CAPITAL * 100.0
            n = len(D)
            mean = D.mean()
            se = D.std(ddof=1) / np.sqrt(n)
            records.append(dict(epsilon=epsilon, policy=policy, mean_diff_pct=mean,
                                 ci_lo=mean - 1.96 * se, ci_hi=mean + 1.96 * se))
    return pd.DataFrame(records)


def print_summary(summary):
    print("\n" + "=" * 100)
    print("SUMMARY: mean objective (%) and mean PnL (%) by epsilon and policy")
    print("=" * 100)
    header = f"{'epsilon':>10}{'policy':>14}{'mean_obj%':>12}{'std_obj%':>12}{'mean_pnl%':>12}{'std_pnl%':>12}{'mean|inv|':>12}"
    print(header)
    for _, row in summary.iterrows():
        print(f"{row['epsilon']:>10.4f}{row['policy']:>14}{row['mean_obj_pct']:>12.4f}"
              f"{row['std_obj_pct']:>12.4f}{row['mean_pnl_pct']:>12.4f}{row['std_pnl_pct']:>12.4f}"
              f"{row['mean_abs_inv']:>12.2f}")


def plot_results(summary, diffs, n_episodes, fname):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for policy in POLICIES:
        sub = summary[summary["policy"] == policy].sort_values("epsilon")
        ax.errorbar(sub["epsilon"], sub["mean_obj_pct"],
                    yerr=sub["std_obj_pct"] / np.sqrt(n_episodes),
                    marker='o', linewidth=1.8, capsize=3,
                    color=POLICY_COLOR[policy], label=POLICY_LABELS[policy])
    ax.set_xlabel('Epsilon (price units)', fontsize=10)
    ax.set_ylabel('Mean objective (% of notional)', fontsize=10)
    ax.set_title('Risk-adjusted objective vs. epsilon', fontsize=10)
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.legend(fontsize=8)

    ax = axes[1]
    for policy in ("oracle", "belief", "randomised"):
        sub = diffs[diffs["policy"] == policy].sort_values("epsilon")
        ax.plot(sub["epsilon"], sub["mean_diff_pct"], marker='o', linewidth=1.8,
                color=POLICY_COLOR[policy], label=POLICY_LABELS[policy])
        ax.fill_between(sub["epsilon"], sub["ci_lo"], sub["ci_hi"],
                         color=POLICY_COLOR[policy], alpha=0.15)
    ax.axhline(0.0, color='grey', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Epsilon (price units)', fontsize=10)
    ax.set_ylabel('Paired advantage over naive (% of notional)', fontsize=10)
    ax.set_title('Regime-awareness edge vs. epsilon (95% CI)', fontsize=10)
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.legend(fontsize=8)

    fig.suptitle(
        'Policy performance vs. adverse-selection jump size epsilon\n'
        f'($\\sigma_0$=$\\sigma_1$={EQUAL_SIGMA_STEP:.6f} per-step, equal; {n_episodes} paired episodes/point)',
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to {fname}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=N_EPISODES)
    p.add_argument("--base-seed", type=int, default=12345)
    p.add_argument("--output", type=str, default="results/policies_epsilon_sweep.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 78)
    print(f"Sigma_0 = Sigma_1 = {EQUAL_SIGMA_STEP:.6f} per-step (equal, fixed across sweep)")
    print(f"epsilon values = {EPSILON_VALUES} (price units)")
    print(f"policies = {POLICIES}, episodes/epsilon = {args.episodes}")
    print("=" * 78)

    t_start = time.time()
    df = run_sweep(EPSILON_VALUES, args.episodes, base_seed=args.base_seed)
    elapsed = time.time() - t_start
    print(f"\nTotal sweep runtime: {elapsed:.1f}s")

    df, summary = summarise(df)
    diffs = paired_diff_vs_naive(df, args.episodes)
    print_summary(summary)

    print("\n" + "=" * 90)
    print("PAIRED ADVANTAGE OVER NAIVE (mean objective diff %, 95% CI)")
    print("=" * 90)
    print(diffs.to_string(index=False))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(script_dir, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nEpisode-level results saved to: {output_path}")

    timestamp = datetime.now().strftime("%H%M%S")
    plot_fname = os.path.join(IMAGES_DIR, f'policies_epsilon_sweep_{timestamp}.png')
    plot_results(summary, diffs, args.episodes, plot_fname)
