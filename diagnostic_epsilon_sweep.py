"""
diagnostic_epsilon_sweep.py
----------------------------
Sweeps the regime-1 adverse-selection jump size (epsilon) downward from
its calibrated value (0.5, price units) and measures how Hamilton filter
tracking quality degrades as the jump signal shrinks.

Both regimes are given the SAME diffusion volatility (EQUAL_VOLATILITY,
set to R0_VOLATILITY=0.01 -- see envs/make_envs.py), so the two regimes
are diffusion-identical and epsilon is the ONLY signal distinguishing
them. An earlier version of this diagnostic kept the calibrated, distinct
volatilities (sigma_1=3*sigma_0) fixed while sweeping epsilon, and found
accuracy/detection-lag completely flat (~0.99 accuracy, ~2-3 step
detection lag) across five decades of epsilon -- because the 3x diffusion
ratio alone already saturates detection, making epsilon's marginal
contribution invisible. Equalising the volatilities removes that
confound so epsilon's own identifying power can actually be seen.

Uses the actual mbt_gym-driven RegimeSwitchingEnv (via make_regime_envs,
now parametrised on epsilon and per-regime volatility) and the production
HamiltonFilter class -- not a standalone reimplementation -- so results
reflect the real system.

Run from repo root:
    python diagnostic_epsilon_sweep.py
"""

import argparse
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from envs.make_envs import (make_regime_envs,
                            TRANSITION_MATRIX,
                            R0_VOLATILITY,
                            STEP_SIZE,
                            LAMBDA,
                            INITIAL_PRICE)
from beliefs.hamilton_filter import HamiltonFilter

sns.set_theme(style="whitegrid", context="notebook")
_PALETTE = sns.color_palette("deep")

IMAGES_DIR = r'C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images'

# Same diffusion volatility for both regimes -- epsilon is the only
# distinguishing signal. Uses the calibrated regime-0 (calm) volatility
# as the shared baseline.
EQUAL_VOLATILITY  = R0_VOLATILITY
EQUAL_VOLATILITY_PCT = EQUAL_VOLATILITY / INITIAL_PRICE
EQUAL_SIGMA_STEP  = EQUAL_VOLATILITY_PCT * np.sqrt(STEP_SIZE)

# Decreasing from the calibrated epsilon=0.5 (price units) down toward
# the pure-diffusion limit. Log-spaced, not linear, so the sweep covers
# the region where eps_pct is comparable to EQUAL_SIGMA_STEP, not just
# the jump-dominated regime.
EPSILON_VALUES = np.geomspace(0.5, 0.0005, 6).tolist()
N_EPISODES     = 20


DETECTION_WINDOW = 50


def detection_lags_to_regime1(beliefs, regimes, max_window=DETECTION_WINDOW):
    """
    For every 0->1 regime switch within one episode, return the number of
    steps from the switch until belief first crosses >=0.5 (censored at
    max_window if it never does within the window).

    Aggregate accuracy over a whole episode is dominated by regime
    persistence (mean dwell time ~400-1000 steps here -- see
    envs/make_envs.py's TRANSITION_MATRIX derivation): once the filter
    locks on, hundreds of correlated steps keep it locked on regardless of
    how weak the per-step signal is, so aggregate accuracy saturates near
    1.0 even for a tiny epsilon (verified: flat ~0.99 across the whole
    sweep). Detection lag right after a fresh switch isolates the
    per-step signal strength instead, since evidence hasn't accumulated
    yet -- this is where epsilon's contribution should actually show up.
    """
    switch_idx = np.where((regimes[1:] == 1) & (regimes[:-1] == 0))[0] + 1
    lags = []
    for s in switch_idx:
        window = beliefs[s:s + max_window]
        crossings = np.where(window >= 0.5)[0]
        lag = int(crossings[0]) if len(crossings) > 0 else max_window
        lags.append(lag)
    return lags


def run_episode(env, filt):
    """Run one episode, return arrays of beliefs and true regimes at each step."""
    env.reset()
    filt.reset()

    beliefs = []
    true_regimes = []
    done = False

    belief = filt.update(env.raw_midprice)

    while not np.all(done):
        action = env.action_space.sample().reshape(1, -1)
        obs, reward, done, info = env.step(action)

        belief = filt.update(info['raw_midprice'])
        true_regimes.append(info['true_regime'])
        beliefs.append(belief)

    return np.array(beliefs), np.array(true_regimes)


def run_sweep(epsilon_values, n_episodes, seed_base=0):
    results = []

    for epsilon in epsilon_values:
        epsilon_pct = epsilon / INITIAL_PRICE

        env = make_regime_envs(
            switch_within_episode=True,
            epsilon=epsilon,
            r0_volatility=EQUAL_VOLATILITY,
            r1_volatility=EQUAL_VOLATILITY,
        )
        filt = HamiltonFilter(
            transition_matrix=TRANSITION_MATRIX,
            regime_volatilities=[EQUAL_SIGMA_STEP, EQUAL_SIGMA_STEP],
            r=0,
            jump_size=epsilon_pct,
            jump_intensity=LAMBDA,
            step_size=STEP_SIZE,
        )

        all_beliefs = []
        all_regimes = []
        all_lags = []
        np.random.seed(seed_base)  # reproducible regime-switching path per epsilon
        for _ in range(n_episodes):
            beliefs, regimes = run_episode(env, filt)
            all_beliefs.append(beliefs)
            all_regimes.append(regimes)
            all_lags.extend(detection_lags_to_regime1(beliefs, regimes))

        beliefs = np.concatenate(all_beliefs)
        regimes = np.concatenate(all_regimes)
        lags = np.array(all_lags)

        mask1 = regimes == 1
        mask0 = regimes == 0
        accuracy = float(np.mean((beliefs >= 0.5).astype(int) == regimes))
        mean_belief_r1 = float(np.mean(beliefs[mask1])) if mask1.sum() > 0 else float('nan')
        mean_belief_r0 = float(np.mean(beliefs[mask0])) if mask0.sum() > 0 else float('nan')
        separation = mean_belief_r1 - mean_belief_r0
        mean_lag = float(np.mean(lags)) if len(lags) > 0 else float('nan')
        detected_within_10 = float(np.mean(lags <= 10)) if len(lags) > 0 else float('nan')

        print(f"epsilon={epsilon:.4f} (eps_pct={epsilon_pct:.6f}): "
              f"accuracy={accuracy:.3f}  belief|r=1={mean_belief_r1:.3f}  "
              f"belief|r=0={mean_belief_r0:.3f}  separation={separation:.3f}  "
              f"mean_lag={mean_lag:.1f} steps  det<=10={detected_within_10:.2f}  "
              f"(n_switches={len(lags)})")

        results.append(dict(
            epsilon=epsilon,
            epsilon_pct=epsilon_pct,
            accuracy=accuracy,
            mean_belief_r1=mean_belief_r1,
            mean_belief_r0=mean_belief_r0,
            separation=separation,
            mean_lag=mean_lag,
            detected_within_10=detected_within_10,
        ))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=N_EPISODES,
                        help=f"Episodes per epsilon value (default: {N_EPISODES}).")
    args = parser.parse_args()

    print(f"Sigma_0 = Sigma_1 = {EQUAL_SIGMA_STEP:.6f} (per-step, equal, fixed across sweep)")
    print(f"Sweeping epsilon = {EPSILON_VALUES} (price units), "
          f"{args.episodes} episodes each...\n")

    results = run_sweep(EPSILON_VALUES, args.episodes)

    epsilons    = [r['epsilon'] for r in results]
    accuracies  = [r['accuracy'] for r in results]
    separations = [r['separation'] for r in results]
    belief_r1   = [r['mean_belief_r1'] for r in results]
    belief_r0   = [r['mean_belief_r0'] for r in results]
    mean_lags   = [r['mean_lag'] for r in results]
    det10       = [r['detected_within_10'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        'Hamilton filter performance vs. adverse-selection jump size epsilon\n'
        f'($\\sigma_0$=$\\sigma_1$={EQUAL_SIGMA_STEP:.6f} per-step, equal -- epsilon is the only signal; '
        f'{args.episodes} episodes/point)',
        fontsize=11
    )

    ax = axes[0]
    ax.plot(epsilons, accuracies, marker='o', color=_PALETTE[0], linewidth=1.8)
    ax.axhline(0.5, color='grey', linestyle=':', linewidth=0.8, label='chance')
    ax.set_xlabel('Epsilon (price units)', fontsize=10)
    ax.set_ylabel('Belief accuracy (P(regime=1)>=0.5 vs true)', fontsize=10)
    ax.set_title('Classification accuracy', fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(epsilons, belief_r1, marker='o', color=_PALETTE[3], linewidth=1.8,
             label='Mean belief | regime=1')
    ax.plot(epsilons, belief_r0, marker='o', color=_PALETTE[0], linewidth=1.8,
             label='Mean belief | regime=0')
    ax.plot(epsilons, separations, marker='s', color=_PALETTE[7],
             linestyle='--', linewidth=1.2, label='Separation (r1 - r0)')
    ax.set_xlabel('Epsilon (price units)', fontsize=10)
    ax.set_ylabel('Belief', fontsize=10)
    ax.set_title('Belief separation between regimes', fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.legend(fontsize=8)

    ax = axes[2]
    ax2 = ax.twinx()
    ax.plot(epsilons, mean_lags, marker='o', color=_PALETTE[1], linewidth=1.8,
             label=f'Mean detection lag (window={DETECTION_WINDOW})')
    ax2.plot(epsilons, det10, marker='^', color=_PALETTE[2], linewidth=1.5,
              linestyle='--', label='Detected within 10 steps (frac)')
    ax.set_xlabel('Epsilon (price units)', fontsize=10)
    ax.set_ylabel('Mean detection lag (steps)', fontsize=10, color=_PALETTE[1])
    ax2.set_ylabel('Fraction detected <=10 steps', fontsize=10, color=_PALETTE[2])
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(False)
    ax.set_title('Post-switch detection speed (0->1 switches)', fontsize=10)
    ax.set_xscale('log')
    ax.invert_xaxis()
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center left')

    plt.tight_layout()
    timestamp = datetime.now().strftime("%H%M%S")
    fname = os.path.join(IMAGES_DIR, f'epsilon_sweep_{timestamp}.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to {fname}")


if __name__ == "__main__":
    main()
