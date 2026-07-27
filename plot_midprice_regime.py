"""
plot_midprice_regime.py
------------------------
Plots the evolution of the (raw, un-normalised) midprice over one episode
alongside the hidden true regime, so regime switches can be visually
matched to changes in the midprice path (jumps/higher volatility in
regime 1).

Run from repo root:
    python plot_midprice_regime.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from datetime import datetime

from envs.make_envs import make_regime_envs

IMAGES_DIR = r'C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images'


def run_episode(seed):
    """Run one episode and return per-step midprice and true regime."""
    env = make_regime_envs(switch_within_episode=True)
    np.random.seed(seed)
    obs = env.reset()
    done = False

    steps = [0]
    mids = [env.raw_midprice]
    regimes = [env.current_regime]

    step = 0
    while not np.all(done):
        action = env.action_space.sample().reshape(1, -1)
        obs, reward, done, info = env.step(action)
        step += 1
        steps.append(step)
        mids.append(info['raw_midprice'])
        regimes.append(info['true_regime'])

    return {
        "steps": np.array(steps),
        "midprice": np.array(mids),
        "regime": np.array(regimes),
    }


def find_seed(min_switches=4, n_tries=200):
    """Find a seed with enough regime switches for an informative plot."""
    for seed in range(n_tries):
        data = run_episode(seed)
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
        ax.axvspan(steps[s], steps[min(e + 1, len(steps) - 1)],
                   alpha=0.12, color='tomato', label='_nolegend_')


def plot_midprice_regime(data, save_name='midprice_regime'):
    steps = data["steps"]
    mids = data["midprice"]
    regime = data["regime"]

    fig = plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
    xlim = (steps[0], steps[-1])

    # Panel 1: midprice, shaded by regime
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(steps, mids, color='black', linewidth=1.2)
    shade_regime1(ax1, steps, regime)
    ax1.set_ylabel('Midprice', fontsize=10)
    ax1.set_title('Midprice evolution across hidden regime switches', fontsize=11)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(xlim)
    ax1.tick_params(labelbottom=False)

    # Legend proxy for shaded region
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(facecolor='tomato', alpha=0.12, label='Regime 1 (adverse selection)')],
               fontsize=8, loc='upper right')

    # Panel 2: true regime
    ax2 = fig.add_subplot(gs[1])
    ax2.step(steps, regime, where='post', color='black', linewidth=1.2)
    ax2.fill_between(steps, regime, step='post', alpha=0.15, color='tomato')
    ax2.set_ylabel('Regime', fontsize=10)
    ax2.set_xlabel('Step', fontsize=10)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['0 (calm)', '1 (adverse)'], fontsize=9)
    ax2.set_ylim(-0.1, 1.3)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(xlim)

    timestamp = datetime.now().strftime("%H%M%S")
    fname = os.path.join(IMAGES_DIR, f'{save_name}_{timestamp}.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def main():
    seed = find_seed()
    data = run_episode(seed)
    plot_midprice_regime(data)


if __name__ == "__main__":
    main()
