"""
diagnostic_single_switch.py
----------------------------
Simulates a single regime switch at a fixed step and measures how
quickly the Hamilton filter detects it.

Episode structure:
    Steps 0   to switch_step-1 : regime 0 (calm)
    Steps switch_step to N_STEPS: regime 1 (adverse selection)

The filter starts with prior belief at the stationary distribution and
must detect the switch from the price return signal alone.

Key metric: detection lag = number of steps after switch_step until
belief pi_t crosses 0.5 for the first time.

Run from repo root:
    python diagnostic_single_switch.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.stats import norm
from itertools import product

IMAGES_DIR = r'C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images'

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
SIGMA_0     = 0.01
SIGMA_1     = 0.03     # 3x -- realistic
DT          = 0.005
N_STEPS     = 200
SWITCH_STEP = 50       # switch from regime 0 to regime 1 at step 50
N_SEEDS     = 50       # average detection lag over many seeds

P = np.array([[0.95, 0.05],
              [0.10, 0.90]])

# Also test with EPSILON jump signal
EPSILON_PCT = 0.01    # percentage return jump size (1% of price)
LAMBDA      = 140.0


# ------------------------------------------------------------------
# Standalone Hamilton filter (no mbt_gym dependency)
# ------------------------------------------------------------------
class SimpleHamiltonFilter:
    def __init__(self, P, sigma0, sigma1, dt, r=1,
                 jump_size=None, jump_intensity=None):
        self.P          = P
        self.sigma      = [sigma0 * np.sqrt(dt), sigma1 * np.sqrt(dt)]
        self.r          = r
        self.jump_size  = jump_size
        self.jump_prob  = float(jump_intensity * dt) if jump_intensity else None
        self._use_mix   = (jump_size is not None and jump_intensity is not None)
        self.n          = 2

        self.paths       = list(product(range(self.n), repeat=r+1))
        self.path_to_idx = {path: i for i, path in enumerate(self.paths)}
        self.n_paths     = len(self.paths)

        # Stationary init
        A = (P.T - np.eye(2)); A[-1] = 1.0
        b = np.zeros(2); b[-1] = 1.0
        pi_stat = np.linalg.solve(A, b)
        self.xi = np.zeros(self.n_paths)
        for i, path in enumerate(self.paths):
            prob = 1.0
            for z in path: prob *= pi_stat[z]
            self.xi[i] = prob
        self.xi /= self.xi.sum()

    def _emission(self, ret, regime):
        sigma = self.sigma[regime]
        if regime == 0 or not self._use_mix:
            return float(norm.pdf(ret, 0.0, sigma) + 1e-300)
        p   = self.jump_prob
        eps = self.jump_size
        w0  = (1-p)**2 + p**2
        wu  = p*(1-p)
        wd  = p*(1-p)
        return float(w0*norm.pdf(ret,0.0,sigma) +
                     wu*norm.pdf(ret,+eps,sigma) +
                     wd*norm.pdf(ret,-eps,sigma) + 1e-300)

    def update(self, ret):
        joint = np.zeros(self.n_paths)
        for i, path in enumerate(self.paths):
            z_t   = path[0]
            z_tm1 = path[1] if self.r >= 1 else None
            if self.r == 0:
                predicted = sum(self.P[zp, z_t]*self.xi[self.path_to_idx[(zp,)]]
                                for zp in range(self.n))
            else:
                predicted = 0.0
                for zo in range(self.n):
                    pp = path[1:] + (zo,)
                    if pp in self.path_to_idx:
                        predicted += self.P[z_tm1, z_t]*self.xi[self.path_to_idx[pp]]
            joint[i] = self._emission(ret, z_t) * predicted
        s = joint.sum()
        if s < 1e-300: return self._belief()
        self.xi = joint / s
        return self._belief()

    def _belief(self):
        return float(sum(self.xi[self.path_to_idx[p]]
                        for p in self.paths if p[0]==1))


# ------------------------------------------------------------------
# Simulate single-switch episode
# ------------------------------------------------------------------
def simulate_single_switch(sigma0, sigma1, switch_step, n_steps, dt,
                            seed, jump_size=None, jump_intensity=None):
    rng = np.random.default_rng(seed)

    # Fixed regime path: 0 until switch_step, then 1
    regimes = np.zeros(n_steps, dtype=int)
    regimes[switch_step:] = 1

    # Sample returns
    returns = np.zeros(n_steps)
    for t in range(n_steps):
        if regimes[t] == 0:
            returns[t] = rng.normal(0.0, sigma0*np.sqrt(dt))
        else:
            # Jump-diffusion in regime 1
            diff = rng.normal(0.0, sigma1*np.sqrt(dt))
            jump = 0.0
            if jump_intensity is not None and jump_size is not None:
                p = jump_intensity * dt
                if rng.random() < p:
                    jump = jump_size * rng.choice([-1, 1])
            returns[t] = diff + jump

    return regimes, returns


def run_detection_experiment(sigma0, sigma1, switch_step, n_steps, dt,
                              n_seeds, jump_size=None, jump_intensity=None,
                              label=""):
    lags = []
    all_beliefs = []

    for seed in range(n_seeds):
        regimes, returns = simulate_single_switch(
            sigma0, sigma1, switch_step, n_steps, dt, seed,
            jump_size, jump_intensity)

        filt = SimpleHamiltonFilter(
            P, sigma0, sigma1, dt, r=0,  # verified identical marginal belief to any r>0 for this model, at lower cost
            jump_size=jump_size, jump_intensity=jump_intensity)

        beliefs = []
        for ret in returns:
            pi = filt.update(ret)
            beliefs.append(pi)
        beliefs = np.array(beliefs)
        all_beliefs.append(beliefs)

        # Detection lag: first step after switch where belief > 0.5
        post_switch = beliefs[switch_step:]
        crossings   = np.where(post_switch >= 0.5)[0]
        lag = int(crossings[0]) if len(crossings) > 0 else n_steps - switch_step
        lags.append(lag)

    mean_beliefs = np.mean(all_beliefs, axis=0)
    p10_beliefs  = np.percentile(all_beliefs, 10, axis=0)
    p90_beliefs  = np.percentile(all_beliefs, 90, axis=0)

    print(f"\n--- {label} ---")
    print(f"  Mean detection lag:   {np.mean(lags):.1f} steps")
    print(f"  Median detection lag: {np.median(lags):.1f} steps")
    print(f"  Min / Max lag:        {np.min(lags)} / {np.max(lags)} steps")
    print(f"  Detected within 5 steps:  {np.mean(np.array(lags)<=5)*100:.0f}%")
    print(f"  Detected within 10 steps: {np.mean(np.array(lags)<=10)*100:.0f}%")
    print(f"  Mean belief post-switch:  {np.mean(mean_beliefs[switch_step:]):.3f}")
    print(f"  Mean belief pre-switch:   {np.mean(mean_beliefs[:switch_step]):.3f}")

    return mean_beliefs, p10_beliefs, p90_beliefs, lags


def main():
    steps = np.arange(N_STEPS)

    # Configuration 1: Gaussian only, sigma_1=0.03
    b1, lo1, hi1, lags1 = run_detection_experiment(
        SIGMA_0, SIGMA_1, SWITCH_STEP, N_STEPS, DT, N_SEEDS,
        jump_size=None, jump_intensity=None,
        label=f"Gaussian only, sigma_1={SIGMA_1}")

    # Configuration 2: Gaussian + jump-mixture, sigma_1=0.03
    b2, lo2, hi2, lags2 = run_detection_experiment(
        SIGMA_0, SIGMA_1, SWITCH_STEP, N_STEPS, DT, N_SEEDS,
        jump_size=EPSILON_PCT, jump_intensity=LAMBDA,
        label=f"Jump-mixture, sigma_1={SIGMA_1}, eps={EPSILON_PCT}")

    # Configuration 3: Gaussian only, sigma_1=0.10 (easy reference)
    b3, lo3, hi3, lags3 = run_detection_experiment(
        SIGMA_0, 0.10, SWITCH_STEP, N_STEPS, DT, N_SEEDS,
        jump_size=None, jump_intensity=None,
        label="Gaussian only, sigma_1=0.10 (easy reference)")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Single regime switch detection (switch at step {SWITCH_STEP}, '
        f'averaged over {N_SEEDS} seeds)',
        fontsize=11
    )

    # Panel 1: Mean belief trajectories
    ax = axes[0]
    ax.axvline(SWITCH_STEP, color='black', linestyle='--',
               linewidth=1.5, label=f'True switch (step {SWITCH_STEP})')
    ax.axhline(0.5, color='grey', linestyle=':', linewidth=0.8)
    ax.axvspan(SWITCH_STEP, N_STEPS, alpha=0.06, color='tomato')

    ax.plot(steps, b1, color='steelblue', linewidth=1.8,
            label=f'Gaussian ($\\sigma_1$={SIGMA_1})')
    ax.fill_between(steps, lo1, hi1, color='steelblue', alpha=0.15)

    ax.plot(steps, b2, color='green', linewidth=1.8,
            label=f'Jump-mixture ($\\sigma_1$={SIGMA_1}, $\\epsilon$={EPSILON_PCT})')
    ax.fill_between(steps, lo2, hi2, color='green', alpha=0.15)

    ax.plot(steps, b3, color='tomato', linewidth=1.5, linestyle='--',
            label=f'Gaussian ($\\sigma_1$=0.10, easy reference)')
    ax.fill_between(steps, lo3, hi3, color='tomato', alpha=0.10)

    ax.set_xlabel('Step', fontsize=10)
    ax.set_ylabel(r'Belief $\pi_t$', fontsize=10)
    ax.set_title('Mean belief with 10th-90th percentile band', fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, N_STEPS)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Panel 2: Detection lag distributions
    ax = axes[1]
    bins = np.arange(0, 31, 1)
    ax.hist(lags1, bins=bins, alpha=0.6, color='steelblue',
            label=f'Gaussian ($\\sigma_1$={SIGMA_1}), mean={np.mean(lags1):.1f}')
    ax.hist(lags2, bins=bins, alpha=0.6, color='green',
            label=f'Jump-mixture, mean={np.mean(lags2):.1f}')
    ax.hist(lags3, bins=bins, alpha=0.5, color='tomato',
            label=f'Easy ref, mean={np.mean(lags3):.1f}')
    ax.set_xlabel('Detection lag (steps)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Distribution of detection lag across seeds', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%H%M%S")
    fname = os.path.join(IMAGES_DIR, f'detection_lag_{timestamp}.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to {fname}")


if __name__ == "__main__":
    main()
