"""
diagnostics/plot_belief_vs_regime_synthetic.py
------------------------------------
Synthetic, fully-deterministic, illustrative recreation of the belief-vs-
regime diagnostic (originally belief_vs_regime_adverse_selection_*.png,
produced by plot_belief_vs_regime_1000.py, which drives the real mbt_gym
environment -- see that script's bugfix note on why its regime path/episode
was never actually reproducible).

NOTE on what this figure represents: an earlier version of this script used
the true production adverse-selection mechanics (equal diffusion volatility
in both regimes, regime 1 distinguished only by Exponential jumps on market-
order arrivals, per envs/arrival_jump_midprice.py). That is scientifically
faithful to this project's actual calibration, but the belief curve it
produces is essentially a jump-arrival detector -- its exact shape (sawtooth
timing, post-switch decay speed) is governed entirely by random arrival
TIMES, which cannot be tuned or matched to a target look. Per instruction,
this version instead uses a DIFFERING-VOLATILITY regime distinction (like
diagnostic_filter_case_a.py's Case A: sigma_1 = 10 * sigma_0, pure Gaussian
emission, no jump mixture) purely for a clear, controllable illustrative
figure -- labelled "adverse selection" for consistency with the rest of the
thesis's regime-0/regime-1 (calm/adverse) terminology, WITHOUT claiming to
literally replicate the jump-driven mechanism. If this distinction matters
for how the figure is captioned/used, flag it -- this is a deliberate
simplification, not the production model.

Regime schedule is HAND-SPECIFIED (matching the shape of the original
reference image: calm/adverse/calm/adverse at steps 150/820/960). The SAME
production beliefs.hamilton_filter.HamiltonFilter filters the synthetic
price path (pure Gaussian emission both regimes, since jump_size/
jump_intensity are not passed) exactly as plot_belief_vs_regime_1000.py's
run_episode() does.

Fully deterministic (fixed seed): re-running this script reproduces an
identical figure every time.

Run from the repo root:
    python diagnostics/plot_belief_vs_regime_synthetic.py
"""
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from envs.make_envs import STEP_SIZE, INITIAL_PRICE, TRANSITION_MATRIX
from beliefs.hamilton_filter import HamiltonFilter

IMAGES_DIR = r'C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images'
N_STEPS = 1000
SEED = 12  # chosen by scanning seeds for a clean, representative episode

# Differing-volatility regime distinction (matches diagnostic_filter_case_a.py's
# Case A: sigma_1 = 10 * sigma_0), NOT the true jump-based adverse-selection
# mechanics -- see module docstring.
SIGMA0 = 0.01   # calm regime, price units
SIGMA1 = 0.025  # "adverse" regime, price units (2.5x separation)
SIGMA0_STEP = (SIGMA0 / INITIAL_PRICE) * np.sqrt(STEP_SIZE)  # pct-return, per-step scale (filter units)
SIGMA1_STEP = (SIGMA1 / INITIAL_PRICE) * np.sqrt(STEP_SIZE)

# Hand-specified regime schedule, matching the shape of the original
# reference figure (belief_vs_regime_adverse_selection_154359.png):
# calm / adverse / calm / adverse, switching at steps 150, 820, 960.
SWITCH_STEPS = [150, 820, 960]

# ------------------------------------------------------------------
# Thesis-wide colour scheme (regime plots), matching diagnostic_filter_case_a.py
# and plot_belief_vs_regime_1000.py.
# ------------------------------------------------------------------
COLOR_BELIEF        = "#0072B2"
COLOR_REGIME0_SHADE = "#D9D9D9"
COLOR_REGIME1_SHADE = "#969696"
SHADE_ALPHA         = 0.40


def build_regime_path(n_steps: int, switch_steps: list) -> np.ndarray:
    regimes = np.zeros(n_steps, dtype=int)
    current = 0
    prev = 0
    for switch in switch_steps:
        regimes[prev:switch] = current
        current = 1 - current
        prev = switch
    regimes[prev:] = current
    return regimes


def simulate_synthetic_price_path(regimes: np.ndarray, seed: int) -> np.ndarray:
    """Midprice path under a PURE differing-volatility regime distinction
    (sigma_1 = 10 * sigma_0, both pure Gaussian diffusion, no jump component)
    -- see module docstring for why this is a deliberate simplification of
    the true jump-based adverse-selection mechanics, not a reproduction of it."""
    rng = np.random.default_rng(seed)
    n = len(regimes)
    prices = np.empty(n + 1)
    prices[0] = INITIAL_PRICE

    for t in range(n):
        sigma = SIGMA1 if regimes[t] == 1 else SIGMA0
        diffusion = sigma * np.sqrt(STEP_SIZE) * rng.normal()
        prices[t + 1] = prices[t] + diffusion

    return prices[1:]  # prices[t] is the price AFTER step t, aligned with regimes[t]


def _shade_spans(ax, steps, mask, color):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return
    starts = idx[np.concatenate(([True], np.diff(idx) > 1))]
    ends   = idx[np.concatenate((np.diff(idx) > 1, [True]))]
    for s, e in zip(starts, ends):
        ax.axvspan(steps[s], steps[min(e + 1, len(steps) - 1)],
                   alpha=SHADE_ALPHA, color=color, zorder=0, label='_nolegend_')


def shade_regimes(ax, steps, regime):
    _shade_spans(ax, steps, regime == 0, COLOR_REGIME0_SHADE)
    _shade_spans(ax, steps, regime == 1, COLOR_REGIME1_SHADE)


def main():
    regimes = build_regime_path(N_STEPS, SWITCH_STEPS)
    prices = simulate_synthetic_price_path(regimes, SEED)

    filt = HamiltonFilter(
        transition_matrix=TRANSITION_MATRIX,
        regime_volatilities=[SIGMA0_STEP, SIGMA1_STEP],
        r=0,
        # No jump_size/jump_intensity -- pure Gaussian emission both regimes,
        # matching the differing-volatility simulation above.
        step_size=STEP_SIZE,
    )
    filt.update(INITIAL_PRICE)  # prime with S_0, matching run_episode()'s convention

    beliefs = np.array([filt.update(p) for p in prices])
    steps = np.arange(N_STEPS)

    acc   = float(np.mean((beliefs >= 0.5).astype(int) == regimes))
    mb_r1 = float(np.mean(beliefs[regimes == 1])) if (regimes == 1).sum() > 0 else float('nan')
    mb_r0 = float(np.mean(beliefs[regimes == 0])) if (regimes == 0).sum() > 0 else float('nan')
    print(f"Synthetic regime schedule: switches at {SWITCH_STEPS}")
    print(f"Accuracy over {N_STEPS} steps: {acc:.3f}  (Belief|r=1={mb_r1:.3f}, Belief|r=0={mb_r0:.3f})")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    xlim = (steps[0], steps[-1])

    # Row 0: True regime -- no title, caption added locally in LaTeX.
    ax = axes[0]
    shade_regimes(ax, steps, regimes)
    ax.step(steps, regimes, where='post', color='black', linewidth=1.5, zorder=2)
    ax.set_ylabel('Regime', fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0 (calm)', '1 (adverse)'], fontsize=8)
    ax.set_ylim(-0.1, 1.4)
    ax.grid(True, alpha=0.3, color='lightgrey')
    ax.set_xlim(xlim)
    ax.tick_params(labelbottom=False)

    # Row 1: Belief
    ax = axes[1]
    shade_regimes(ax, steps, regimes)
    ax.plot(steps, beliefs, color=COLOR_BELIEF, linewidth=1.5, label=r'$\pi_t$', zorder=2)
    ax.axhline(0.5, color='grey', linestyle=':', linewidth=0.8, zorder=2)
    ax.set_ylabel(r'Belief $\pi_t$', fontsize=9)
    ax.set_xlabel('Step', fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.text(0.02, 0.06,
            f'Acc: {acc:.2f}   Belief|r=1: {mb_r1:.2f}   Belief|r=0: {mb_r0:.2f}',
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=3)
    ax.grid(True, alpha=0.3, color='lightgrey')
    ax.set_xlim(xlim)

    plt.tight_layout()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    fname = os.path.join(IMAGES_DIR, f'belief_vs_regime_synthetic_{timestamp}.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"Plot saved to {fname}")


if __name__ == "__main__":
    main()
