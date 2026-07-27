"""
plot_belief_vs_regime_1000.py
------------------------------
Belief-vs-regime tracking plot for the adverse-selection regime-switching
environment: Hamilton filter belief P(regime=1 | F_t) vs the true hidden
regime, over a 1000-step episode that starts in regime 0 (calm) and is
guaranteed to contain at least two regime switches.

Regime 1 here is adverse selection (price jumps on fills, via
ArrivalJumpMidpriceModel) -- not just higher volatility, unlike the
original pure-volatility version of this test (see envs/make_envs.py
and beliefs/hamilton_filter.py for the jump-diffusion emission).

Run from the repo root:
    python plot_belief_vs_regime_1000.py

Uses the actual production regime-switching calibration
(envs.make_envs.TRANSITION_MATRIX -- mean regime durations of
~400-1000 steps at this discretisation), so a 1000-step window
comfortably contains a couple of switches without needing a rare seed
draw (unlike the earlier 200-step version). find_seed() still searches
across seeds to guarantee >= 2 switches.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from envs.make_envs import (
    make_regime_envs,
    TRANSITION_MATRIX,
    R0_SIGMA_STEP, R1_SIGMA_STEP,
    STEP_SIZE, LAMBDA, EPSILON_PCT,
)
from beliefs.hamilton_filter import HamiltonFilter

IMAGES_DIR = r'C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images'
N_PLOT_STEPS = 1000
MIN_SWITCHES = 2


def run_episode(filt, seed, n_steps):
    """Run n_steps starting in regime 0; return per-step beliefs and true regimes."""
    env = make_regime_envs(switch_within_episode=True)
    env.initial_regime = 0
    np.random.seed(seed)
    env.reset()
    filt.reset()

    mid = env.raw_midprice
    filt.update(mid)  # prime the filter with t=0 midprice, nothing to record yet

    beliefs = []
    true_regimes = []
    for _ in range(n_steps):
        action = env.action_space.sample().reshape(1, -1)
        _, _, _, info = env.step(action)
        beliefs.append(filt.update(info['raw_midprice']))
        true_regimes.append(info['true_regime'])

    return np.array(beliefs), np.array(true_regimes)


def find_seed(filt, n_steps, min_switches=MIN_SWITCHES, n_tries=300):
    """Search seeds for an episode with at least min_switches regime switches."""
    for seed in range(n_tries):
        beliefs, regimes = run_episode(filt, seed, n_steps)
        switches = int(np.sum(np.abs(np.diff(regimes))))
        if switches >= min_switches:
            print(f"Using seed={seed} ({switches} regime switches)")
            return beliefs, regimes
    raise RuntimeError(f"No seed in range(0, {n_tries}) gave >= {min_switches} switches")


def shade_regime1(ax, steps, regime):
    """Shade the regions where regime == 1, matching diagnostic_filter_case_a.py's style."""
    in_r1 = np.where(regime == 1)[0]
    if len(in_r1) == 0:
        return
    starts = in_r1[np.concatenate(([True], np.diff(in_r1) > 1))]
    ends   = in_r1[np.concatenate((np.diff(in_r1) > 1, [True]))]
    for s, e in zip(starts, ends):
        ax.axvspan(steps[s], steps[min(e + 1, len(steps) - 1)],
                   alpha=0.10, color='tomato', label='_nolegend_')


def main():
    filt = HamiltonFilter(
        transition_matrix=TRANSITION_MATRIX,
        regime_volatilities=[R0_SIGMA_STEP, R1_SIGMA_STEP],
        r=0,  # verified identical marginal belief to any r>0 for this model, at lower cost
        jump_size=EPSILON_PCT,
        jump_intensity=LAMBDA,
        step_size=STEP_SIZE,
    )

    print("Searching for an episode with >= 2 regime switches...")
    beliefs, regimes = find_seed(filt, N_PLOT_STEPS)
    steps = np.arange(len(beliefs))

    acc   = float(np.mean((beliefs >= 0.5).astype(int) == regimes))
    mb_r1 = float(np.mean(beliefs[regimes == 1])) if (regimes == 1).sum() > 0 else float('nan')
    mb_r0 = float(np.mean(beliefs[regimes == 0])) if (regimes == 0).sum() > 0 else float('nan')
    print(f"Accuracy over {len(steps)} steps: {acc:.3f}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    xlim = (steps[0], steps[-1])

    # Row 0: True regime -- same style as diagnostic_filter_case_a.py's row-0 panels
    ax = axes[0]
    ax.step(steps, regimes, where='post', color='black', linewidth=1.5)
    ax.fill_between(steps, regimes, step='post', alpha=0.12, color='tomato')
    ax.set_ylabel('Regime', fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0 (calm)', '1 (adverse)'], fontsize=8)
    ax.set_ylim(-0.1, 1.4)
    ax.set_title('Hamilton Filter: Belief vs True Regime (adverse selection)', fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(xlim)
    ax.tick_params(labelbottom=False)

    # Row 1: Belief -- same style as diagnostic_filter_case_a.py's row-2 panels
    ax = axes[1]
    ax.plot(steps, beliefs, color='steelblue', linewidth=1.5, label=r'$\pi_t$')
    ax.axhline(0.5, color='grey', linestyle=':', linewidth=0.8)
    shade_regime1(ax, steps, regimes)
    ax.set_ylabel(r'Belief $\pi_t$', fontsize=9)
    ax.set_xlabel('Step', fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.text(0.02, 0.06,
            f'Acc: {acc:.2f}   Belief|r=1: {mb_r1:.2f}   Belief|r=0: {mb_r0:.2f}',
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.grid(True, alpha=0.2)
    ax.set_xlim(xlim)

    plt.tight_layout()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    fname = os.path.join(IMAGES_DIR, f'belief_vs_regime_adverse_selection_{timestamp}.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"Plot saved to {fname}")


if __name__ == "__main__":
    main()
