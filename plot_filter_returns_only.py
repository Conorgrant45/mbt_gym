"""
plot_filter_returns_only.py
----------------------------
How predictive is the Hamilton filter using ONLY the per-step return
emission (jump-diffusion mixture for regime 1), with the RV augmentation
disabled (rv_window=0) -- i.e. the statistically "honest" filter, without
the overlapping-window double-counting identified in the RV-augmented
version (see filter_and_control_fixes.tex discussion).

Plots, on the same real episode (post regime_env.py continuity fix):
    Panel 1: true regime (shaded) + midprice, for context
    Panel 2: belief -- return-only (honest) vs RV-augmented (inflated),
             overlaid for direct visual comparison
    Panel 3: per-step return, with jump events marked

Run from repo root:
    python plot_filter_returns_only.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from datetime import datetime

from envs.make_envs import (
    make_regime_envs,
    KAPPA, LAMBDA, EPSILON, EPSILON_PCT,
    TERMINAL_TIME, N_STEPS, STEP_SIZE,
    TRANSITION_MATRIX,
    R0_SIGMA_STEP, R1_SIGMA_STEP,
)
from beliefs.hamilton_filter import HamiltonFilter

IMAGES_DIR = r'C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images'


def make_return_only_filter():
    return HamiltonFilter(
        transition_matrix=TRANSITION_MATRIX,
        regime_volatilities=[R0_SIGMA_STEP, R1_SIGMA_STEP],
        r=0,  # verified identical marginal belief to any r>0 for this model, at lower cost
        jump_size=EPSILON_PCT,
        jump_intensity=LAMBDA,
        step_size=STEP_SIZE,
        rv_window=0,     # RV disabled -- return-only emission
        rv_weight=0.0,
    )


def make_rv_filter():
    return HamiltonFilter(
        transition_matrix=TRANSITION_MATRIX,
        regime_volatilities=[R0_SIGMA_STEP, R1_SIGMA_STEP],
        r=0,  # verified identical marginal belief to any r>0 for this model, at lower cost
        jump_size=EPSILON_PCT,
        jump_intensity=LAMBDA,
        step_size=STEP_SIZE,
        rv_window=10,
        rv_weight=1.0,
    )


def run_episode(filt_ret, filt_rv, seed):
    """Run one episode, updating both filters on the same price path."""
    env = make_regime_envs(switch_within_episode=True)
    np.random.seed(seed)
    obs = env.reset()
    filt_ret.reset()
    filt_rv.reset()
    done = False

    steps, regimes, mids, belief_ret, belief_rv, returns = [], [], [], [], [], []
    prev_mid = None
    step = 0
    mid = env.raw_midprice

    while not np.all(done):
        if prev_mid is not None and abs(prev_mid) > 1e-10:
            ret = (mid - prev_mid) / abs(prev_mid)
        else:
            ret = 0.0

        b_ret = filt_ret.update(mid)
        b_rv = filt_rv.update(mid)
        regime = env.current_regime

        steps.append(step)
        regimes.append(regime)
        mids.append(mid)
        belief_ret.append(b_ret)
        belief_rv.append(b_rv)
        returns.append(ret)

        prev_mid = mid
        action = env.action_space.sample().reshape(1, -1)
        obs, reward, done, info = env.step(action)
        mid = info['raw_midprice']
        step += 1

    return {
        "steps": np.array(steps), "regime": np.array(regimes), "mid": np.array(mids),
        "belief_ret": np.array(belief_ret), "belief_rv": np.array(belief_rv),
        "returns": np.array(returns),
    }


def find_seed(min_switches=4, n_tries=200):
    filt_ret, filt_rv = make_return_only_filter(), make_rv_filter()
    for seed in range(n_tries):
        data = run_episode(filt_ret, filt_rv, seed)
        switches = int(np.sum(np.abs(np.diff(data["regime"]))))
        if switches >= min_switches:
            print(f"Using seed={seed} ({switches} regime switches)")
            return seed
    return 0


def shade_regime1(ax, steps, regime):
    in_r1 = np.where(regime == 1)[0]
    if len(in_r1) == 0:
        return
    starts = in_r1[np.concatenate(([True], np.diff(in_r1) > 1))]
    ends = in_r1[np.concatenate((np.diff(in_r1) > 1, [True]))]
    for s, e in zip(starts, ends):
        ax.axvspan(steps[s], steps[min(e + 1, len(steps) - 1)], alpha=0.10, color='tomato', label='_nolegend_')


def plot(data):
    steps, regime, mid = data["steps"], data["regime"], data["mid"]
    belief_ret, belief_rv, returns = data["belief_ret"], data["belief_rv"], data["returns"]

    jump_mask = (np.abs(returns) > EPSILON_PCT * 0.5) & (regime == 1)

    fig = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 2, 1.3], hspace=0.12)
    xlim = (steps[0], steps[-1])

    # Panel 1: midprice + true regime shading
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(steps, mid, color='black', linewidth=1.1)
    shade_regime1(ax1, steps, regime)
    ax1.set_ylabel('Midprice', fontsize=10)
    ax1.set_title('Return-only filter vs RV-augmented filter: belief tracking on a real episode', fontsize=11)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(xlim)
    ax1.tick_params(labelbottom=False)

    # Panel 2: belief -- both filters overlaid
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(steps, belief_ret, color='steelblue', linewidth=1.6, label='Return-only (honest)')
    ax2.plot(steps, belief_rv, color='darkorange', linewidth=1.2, alpha=0.8, label='RV-augmented (inflated)')
    ax2.axhline(0.5, color='grey', linestyle=':', linewidth=0.8)
    shade_regime1(ax2, steps, regime)
    ax2.scatter(steps[jump_mask], belief_ret[jump_mask], color='tomato', s=18, zorder=5, marker='^', label='Jump event')
    ax2.set_ylabel(r'Belief $\pi_t$', fontsize=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(xlim)
    ax2.tick_params(labelbottom=False)

    # Panel 3: returns
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(steps, returns, color='purple', linewidth=0.8, alpha=0.8)
    shade_regime1(ax3, steps, regime)
    ax3.axhline(0.0, color='grey', linestyle='--', linewidth=0.6)
    ax3.set_ylabel('Return', fontsize=10)
    ax3.set_xlabel('Step', fontsize=10)
    ax3.grid(True, alpha=0.2)
    ax3.set_xlim(xlim)

    timestamp = datetime.now().strftime("%H%M%S")
    fname = os.path.join(IMAGES_DIR, f'filter_returns_only_{timestamp}.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def print_stats(data):
    regime, belief_ret, belief_rv = data["regime"], data["belief_ret"], data["belief_rv"]
    pred_ret = (belief_ret >= 0.5).astype(int)
    pred_rv = (belief_rv >= 0.5).astype(int)
    print("\n--- Accuracy (threshold 0.5) ---")
    print(f"  Return-only : {np.mean(pred_ret == regime):.4f}")
    print(f"  RV-augmented: {np.mean(pred_rv == regime):.4f}")
    print("\n--- Mean belief by TRUE regime ---")
    for r in [0, 1]:
        mask = regime == r
        print(f"  Regime {r}: return-only mean={belief_ret[mask].mean():.4f}  "
              f"RV mean={belief_rv[mask].mean():.4f}  (n={mask.sum()})")
    print("\n--- Belief variance within regime (higher = more natural fluctuation) ---")
    for r in [0, 1]:
        mask = regime == r
        print(f"  Regime {r}: return-only var={belief_ret[mask].var():.5f}  "
              f"RV var={belief_rv[mask].var():.5f}")


def main():
    seed = find_seed()
    filt_ret, filt_rv = make_return_only_filter(), make_rv_filter()
    data = run_episode(filt_ret, filt_rv, seed)
    print_stats(data)
    plot(data)


if __name__ == "__main__":
    main()
