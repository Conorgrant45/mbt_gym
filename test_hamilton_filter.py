"""
test_hamilton_filter.py
-----------------------
Standalone sanity check for the Hamilton filter.

Runs the filter against the RegimeSwitchingEnv for several episodes and
checks that the belief P(regime=1) is higher on average during stressed
episodes than calm ones.

Also saves a plot (hamilton_filter_test.png) showing belief vs true regime
for a single episode so you can visually inspect tracking quality.

Run from the repo root:
    python test_hamilton_filter.py
"""

import numpy as np
import matplotlib.pyplot as plt

from envs.make_envs import (make_regime_envs,
                            TRANSITION_MATRIX,
                            R0_SIGMA_STEP,
                            R1_SIGMA_STEP,
                            STEP_SIZE,
                            LAMBDA,
                            EPSILON_PCT)
from beliefs.hamilton_filter import HamiltonFilter


def run_episode(env, filt):
    """Run one episode, return arrays of beliefs and true regimes at each step."""
    obs = env.reset()
    filt.reset()

    beliefs = []
    true_regimes = []
    done = False

    # First belief from reset -- use the raw (un-normalised) midprice,
    # since obs is normalised to roughly [-1, 1] and index 3 is not a
    # usable price series.
    belief = filt.update(env.raw_midprice)

    while not np.all(done):
        action = env.action_space.sample().reshape(1, -1)
        obs, reward, done, info = env.step(action)

        belief = filt.update(info['raw_midprice'])
        true_regime = info['true_regime']

        beliefs.append(belief)
        true_regimes.append(true_regime)

    return np.array(beliefs), np.array(true_regimes)


def main():
    print("Building environment and filter...")
    env = make_regime_envs(switch_within_episode=True)

    filt = HamiltonFilter(
        transition_matrix=TRANSITION_MATRIX,
        regime_volatilities=[R0_SIGMA_STEP, R1_SIGMA_STEP],
        r=0,  # verified identical marginal belief to any r>0 for this model, at lower cost
        jump_size=EPSILON_PCT,
        jump_intensity=LAMBDA,
        step_size=STEP_SIZE,
    )

    # --- Run multiple episodes and collect mean beliefs per regime ---
    n_episodes = 100
    mean_beliefs_r0 = []
    mean_beliefs_r1 = []

    for ep in range(n_episodes):
        beliefs, true_regimes = run_episode(env, filt)
        if true_regimes[0] == 0:
            mean_beliefs_r0.append(beliefs.mean())
        else:
            mean_beliefs_r1.append(beliefs.mean())

    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Mean belief in regime 0 episodes : {np.mean(mean_beliefs_r0):.3f}  (should be low)")
    print(f"  Mean belief in regime 1 episodes : {np.mean(mean_beliefs_r1):.3f}  (should be high)")
    print(f"  Regime 0 episodes seen : {len(mean_beliefs_r0)}")
    print(f"  Regime 1 episodes seen : {len(mean_beliefs_r1)}")

    separation = np.mean(mean_beliefs_r1) - np.mean(mean_beliefs_r0)
    print(f"\n  Belief separation (r1 - r0) : {separation:.3f}  (larger is better)")

    if separation > 0.1:
        print("  Filter is tracking regimes correctly.")
    else:
        print("  Warning: low separation. Filter may need tuning.")

    # --- Plot a single episode for visual inspection ---
    print("\nPlotting single episode...")
    env_plot = make_regime_envs(switch_within_episode=True)
    beliefs, true_regimes = run_episode(env_plot, filt)

    steps = np.arange(len(beliefs))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(steps, beliefs, color='steelblue', linewidth=1.5, label='P(regime=1 | history)')
    axes[0].axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[0].set_ylabel('Belief P(stressed)')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc='upper right')
    axes[0].set_title('Hamilton Filter: Belief vs True Regime')

    axes[1].fill_between(steps, true_regimes, step='post',
                         alpha=0.6, color='tomato', label='True regime (1=stressed)')
    axes[1].set_ylabel('True regime')
    axes[1].set_xlabel('Step')
    axes[1].set_ylim(-0.1, 1.5)
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('hamilton_filter_test.png', dpi=150)
    print("  Plot saved to hamilton_filter_test.png")

    print("\nHamilton filter test complete.")


if __name__ == "__main__":
    main()
