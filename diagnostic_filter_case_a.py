"""
diagnostic_filter_case_a.py
----------------------------
Case A: Easy artificial diagnostic for the Hamilton filter.

Uses an intentionally simple regime-switching model where the two regimes
are easy to distinguish (sigma_1 = 10 * sigma_0). The purpose is to verify
that the Hamilton filter recursion and likelihood implementation are correct.

Expected behaviour:
    - During calm periods (Z=0): belief pi_t should remain close to zero.
    - During high-volatility periods (Z=1): belief pi_t should rise toward one.
    - After a regime switch: pi_t should adjust after a short delay.

If the filter fails here, the issue is computational, not conceptual.
If the filter passes here but fails at sigma_1=0.03, the return-only
observation model is too weak to identify the adverse regime reliably.

Run from repo root:
    python diagnostic_filter_case_a.py

Uses envs.make_envs.TRANSITION_MATRIX and STEP_SIZE -- the same
regime-switching calibration as the mbt_gym-based plots (e.g.
plot_belief_vs_regime_1000.py) -- rather than an independent, much
faster-switching matrix. That matrix implies mean regime durations of
~400-1000 steps, so a fixed seed is not guaranteed to show a switch in
1000 steps; find_switch_seed() searches for a regime path (shared by
both Case A and Case B, so they are directly comparable) with at least
MIN_SWITCHES switches.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from datetime import datetime
from scipy.stats import norm

from envs.make_envs import TRANSITION_MATRIX, STEP_SIZE

IMAGES_DIR = r'C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images'

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
# Case A: large separation -- filter should work easily
SIGMA_0      = 0.01
SIGMA_1      = 0.10   # 10x sigma_0

# Same transition matrix and step size as the mbt_gym-based plots, for
# a fair comparison -- see envs/make_envs.py.
P  = np.array(TRANSITION_MATRIX)
DT = STEP_SIZE

# Episode parameters
N_STEPS      = 1000
MIN_SWITCHES = 2

# ------------------------------------------------------------------
# Thesis-wide colour scheme (regime plots): midprice/belief line in blue,
# calm-regime shading in light grey, adverse-selection-regime shading in
# darker grey, shading alpha in [0.35, 0.50] and kept behind the data.
# ------------------------------------------------------------------
COLOR_BELIEF        = "#0072B2"
COLOR_REGIME0_SHADE = "#D9D9D9"
COLOR_REGIME1_SHADE = "#969696"
SHADE_ALPHA         = 0.40


def simulate_regime_path(P, n_steps, seed):
    """Sample a regime path via the Markov chain, starting in regime 0."""
    rng = np.random.default_rng(seed)
    regimes = np.zeros(n_steps, dtype=int)
    regimes[0] = 0   # start in calm regime
    for t in range(1, n_steps):
        regimes[t] = rng.choice(2, p=P[regimes[t - 1]])
    return regimes


def find_switch_seed(P, n_steps, min_switches=MIN_SWITCHES, n_tries=2000):
    """Search seeds for a regime path with at least min_switches switches."""
    for seed in range(n_tries):
        regimes = simulate_regime_path(P, n_steps, seed)
        switches = int(np.sum(np.abs(np.diff(regimes))))
        if switches >= min_switches:
            print(f"Using seed={seed} ({switches} regime switches)")
            return seed
    raise RuntimeError(f"No seed in range(0, {n_tries}) gave >= {min_switches} switches")


# ------------------------------------------------------------------
# Simulate a regime-switching Gaussian return series
# (no mbt_gym -- pure synthetic data to isolate filter behaviour)
# ------------------------------------------------------------------
def simulate_episode(sigma0, sigma1, P, n_steps, dt, seed):
    rng = np.random.default_rng(seed)

    # Sample regime path via Markov chain
    regimes  = np.zeros(n_steps, dtype=int)
    regimes[0] = 0   # start in calm regime
    for t in range(1, n_steps):
        regimes[t] = rng.choice(2, p=P[regimes[t-1]])

    # Sample returns conditional on regime
    sigmas  = np.where(regimes == 0, sigma0, sigma1)
    returns = rng.normal(0.0, sigmas * np.sqrt(dt))

    return regimes, returns


# ------------------------------------------------------------------
# Hamilton filter (standalone, no mbt_gym dependency)
# Uses percentage returns directly.
# ------------------------------------------------------------------
class SimpleHamiltonFilter:
    """
    Two-state Hamilton filter with r=1 lagged state.
    Gaussian emission: f(r_t | Z_t=k) = N(0, sigma_k * sqrt(dt)).

    This is a self-contained implementation for diagnostic purposes,
    independent of the main HamiltonFilter class, so any bugs in that
    class do not affect the diagnostic.
    """

    def __init__(self, P, sigma0, sigma1, dt, r=1):
        self.P      = P
        self.sigma  = [sigma0 * np.sqrt(dt), sigma1 * np.sqrt(dt)]
        self.r      = r
        self.n      = 2

        # All paths of length r+1
        from itertools import product
        self.paths       = list(product(range(self.n), repeat=r+1))
        self.path_to_idx = {path: i for i, path in enumerate(self.paths)}
        self.n_paths     = len(self.paths)

        # Initialise at stationary distribution
        A      = (P.T - np.eye(2))
        A[-1]  = 1.0
        b      = np.zeros(2); b[-1] = 1.0
        pi_stat = np.linalg.solve(A, b)

        self.xi = np.zeros(self.n_paths)
        for i, path in enumerate(self.paths):
            prob = 1.0
            for z in path:
                prob *= pi_stat[z]
            self.xi[i] = prob
        self.xi /= self.xi.sum()

    def update(self, ret):
        joint = np.zeros(self.n_paths)
        for i, path in enumerate(self.paths):
            z_t   = path[0]
            z_tm1 = path[1] if self.r >= 1 else None

            if self.r == 0:
                predicted = sum(
                    self.P[z_prev, z_t] * self.xi[self.path_to_idx[(z_prev,)]]
                    for z_prev in range(self.n)
                )
            else:
                predicted = 0.0
                for z_oldest in range(self.n):
                    prev_path = path[1:] + (z_oldest,)
                    if prev_path in self.path_to_idx:
                        predicted += self.P[z_tm1, z_t] * self.xi[self.path_to_idx[prev_path]]

            # Gaussian emission -- check: uses sigma * sqrt(dt) scaling
            likelihood = norm.pdf(ret, loc=0.0, scale=self.sigma[z_t]) + 1e-300
            joint[i]   = likelihood * predicted

        normaliser = joint.sum()
        if normaliser < 1e-300:
            return self._belief()
        self.xi = joint / normaliser
        return self._belief()

    def _belief(self):
        return float(sum(
            self.xi[self.path_to_idx[path]]
            for path in self.paths if path[0] == 1
        ))


# ------------------------------------------------------------------
# Run diagnostic
# ------------------------------------------------------------------
def run_case(sigma0, sigma1, label, seed, r=1):
    regimes, returns = simulate_episode(sigma0, sigma1, P, N_STEPS, DT, seed)

    filt    = SimpleHamiltonFilter(P, sigma0, sigma1, DT, r=r)
    beliefs = []
    for ret in returns:
        pi = filt.update(ret)
        beliefs.append(pi)

    beliefs = np.array(beliefs)
    steps   = np.arange(N_STEPS)

    # Metrics
    mask1 = regimes == 1
    mask0 = regimes == 0
    acc   = float(np.mean((beliefs >= 0.5).astype(int) == regimes))
    mb_r1 = float(np.mean(beliefs[mask1])) if mask1.sum() > 0 else float('nan')
    mb_r0 = float(np.mean(beliefs[mask0])) if mask0.sum() > 0 else float('nan')

    print(f"\n--- {label} ---")
    print(f"  sigma_0={sigma0}, sigma_1={sigma1}, ratio={sigma1/sigma0:.1f}x")
    print(f"  Accuracy:              {acc:.3f}")
    print(f"  Mean belief regime 1:  {mb_r1:.3f}")
    print(f"  Mean belief regime 0:  {mb_r0:.3f}")
    print(f"  Regime 1 steps: {mask1.sum()}, Regime 0 steps: {mask0.sum()}")

    return steps, regimes, beliefs, returns


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
    """Background shading for BOTH regimes (calm = light grey, adverse-
    selection = darker grey), drawn at zorder=0 so it always sits behind
    the data plotted on top of it."""
    _shade_spans(ax, steps, regime == 0, COLOR_REGIME0_SHADE)
    _shade_spans(ax, steps, regime == 1, COLOR_REGIME1_SHADE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=0,
                        help="Number of lagged regimes in the Hamilton filter's joint "
                             "posterior (default: 0 -- verified to give an identical "
                             "marginal belief to any r>0 for this model, at lower cost).")
    args = parser.parse_args()

    print("Searching for a regime path with >= 2 switches...")
    seed = find_switch_seed(P, N_STEPS)

    # Case A: easy (sigma_1 = 10 * sigma_0)
    steps_a, regimes_a, beliefs_a, returns_a = run_case(
        0.01, 0.10, "Case A: Easy (sigma_1=0.10, ratio=10x)", seed, r=args.r)

    # Case B: realistic (sigma_1 = 3 * sigma_0)
    steps_b, regimes_b, beliefs_b, returns_b = run_case(
        0.01, 0.03, "Case B: Realistic (sigma_1=0.03, ratio=3x)", seed, r=args.r)

    # ------------------------------------------------------------------
    # Plot: 2x3 grid, one column per case
    #
    # No figure suptitle -- a caption is added locally in LaTeX. The
    # per-column 'Case A'/'Case B' headers are kept: they are the only
    # place sigma_1 for that column is identified, not a redundant
    # restatement of an overall figure description.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Common y-axis for the Return row across both columns (a common axis
    # is required to make the sigma_1=0.10 vs sigma_1=0.03 volatility
    # difference visually comparable -- independent auto-scaling per
    # column hides it, since each column then just fills its own range).
    return_abs_max = max(np.abs(returns_a).max(), np.abs(returns_b).max())
    return_ylim = (-1.05 * return_abs_max, 1.05 * return_abs_max)

    for col, (steps, regimes, beliefs, returns, label, sigma1) in enumerate([
        (steps_a, regimes_a, beliefs_a, returns_a, 'Case A ($\\sigma_1=0.10$)', 0.10),
        (steps_b, regimes_b, beliefs_b, returns_b, 'Case B ($\\sigma_1=0.03$)', 0.03),
    ]):
        xlim = (0, N_STEPS)

        # Row 0: True regime
        ax = axes[0, col]
        shade_regimes(ax, steps, regimes)
        ax.step(steps, regimes, where='post', color='black', linewidth=1.5, zorder=2)
        ax.set_ylabel('Regime', fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['0', '1'], fontsize=8)
        ax.set_ylim(-0.1, 1.4)
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3, color='lightgrey')
        ax.set_xlim(xlim)
        ax.tick_params(labelbottom=False)

        # Row 1: Returns (common y-axis across columns -- see return_ylim above)
        ax = axes[1, col]
        shade_regimes(ax, steps, regimes)
        ax.plot(steps, returns, color='grey', linewidth=0.6, alpha=0.8, zorder=2)
        ax.axhline(0, color='black', linewidth=0.5, zorder=2)
        ax.set_ylabel('Return $r_t$', fontsize=9)
        ax.set_ylim(return_ylim)
        ax.grid(True, alpha=0.3, color='lightgrey')
        ax.set_xlim(xlim)
        ax.tick_params(labelbottom=False)

        # Row 2: Belief
        ax = axes[2, col]
        shade_regimes(ax, steps, regimes)
        ax.plot(steps, beliefs, color=COLOR_BELIEF, linewidth=1.5,
                label=r'$\pi_t$', zorder=2)
        ax.axhline(0.5, color='grey', linestyle=':', linewidth=0.8, zorder=2)
        ax.set_ylabel(r'Belief $\pi_t$', fontsize=9)
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        acc = float(np.mean((beliefs >= 0.5).astype(int) == regimes))
        mb1 = float(np.mean(beliefs[regimes==1])) if (regimes==1).sum() > 0 else float('nan')
        mb0 = float(np.mean(beliefs[regimes==0])) if (regimes==0).sum() > 0 else float('nan')
        ax.text(0.02, 0.06,
                f'Acc: {acc:.2f}   Belief|r=1: {mb1:.2f}   Belief|r=0: {mb0:.2f}',
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=3)
        ax.grid(True, alpha=0.3, color='lightgrey')
        ax.set_xlim(xlim)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%H%M%S")
    fname = os.path.join(IMAGES_DIR, f'filter_diagnostic_r{args.r}_{timestamp}.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to {fname}")


if __name__ == "__main__":
    main()
