"""
plot_belief_evolution.py
------------------------
Generates three belief tracking plots on the same episode and random seed,
showing the evolution of the Hamilton filter across three specifications:

    Plot 1: r=0, Gaussian emission, no RV augmentation
            Shows the baseline oscillation problem.

    Plot 2: r=1, jump-mixture emission, no RV augmentation
            Shows partial improvement -- between-jump belief drops remain.

    Plot 3: r=1, jump-mixture emission, RV augmentation (w=8)
            Belief is stable throughout regime 1, but this is NOT adopted
            as the final specification: the stability comes from an
            overlapping-window double-counting artifact in the RV
            emission (each return is re-scored as evidence for up to
            rv_window steps), not genuine identifiability. See
            beliefs/hamilton_filter.py's module docstring and
            filter_and_control_fixes.tex. Kept here only as the
            with-RV side of the ablation; Plot 2 (return-only) is the
            adopted specification project-wide.

Each plot has three panels: true regime, belief, and log-likelihood ratio.
All three use the same episode and random seed for a fair comparison.

Run from repo root:
    python plot_belief_evolution.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from datetime import datetime
from scipy.stats import norm

from envs.make_envs import (
    make_regime_envs,
    KAPPA, LAMBDA, EPSILON,
    TERMINAL_TIME, N_STEPS, STEP_SIZE,
    TRANSITION_MATRIX,
    R0_VOLATILITY, R1_VOLATILITY,
    R0_SIGMA_STEP, R1_SIGMA_STEP, EPSILON_PCT,
)
from beliefs.hamilton_filter import HamiltonFilter

IMAGES_DIR = r'C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images'

INV_UNIT  = 1.0 / 10000.0
MAX_DEPTH = -np.log(0.01) / KAPPA


def normalise_depth(depth):
    return float(np.clip((depth / MAX_DEPTH) * 2.0 - 1.0, -1.0, 1.0))


def run_episode(filt, seed):
    """Run one episode with a given filter and return per-step arrays."""
    env = make_regime_envs(switch_within_episode=True)
    np.random.seed(seed)
    obs  = env.reset()
    filt.reset()
    done = False

    steps        = []
    true_regimes = []
    beliefs      = []
    returns      = []
    prev_mid     = None

    step = 0
    mid  = env.raw_midprice
    while not np.all(done):
        if prev_mid is not None and abs(prev_mid) > 1e-10:
            ret = (mid - prev_mid) / abs(prev_mid)
        else:
            ret = 0.0

        pi = filt.update(mid)
        regime = env.current_regime

        steps.append(step)
        true_regimes.append(regime)
        beliefs.append(pi)
        returns.append(ret)

        prev_mid = mid
        action   = env.action_space.sample().reshape(1, -1)
        obs, reward, done, info = env.step(action)
        mid      = info['raw_midprice']
        step += 1

    return {
        "steps":   np.array(steps),
        "regime":  np.array(true_regimes),
        "belief":  np.array(beliefs),
        "returns": np.array(returns),
    }


def find_seed(min_switches=4, n_tries=200):
    """Find a seed with enough regime switches for an informative plot."""
    filt_test = HamiltonFilter(
        transition_matrix=TRANSITION_MATRIX,
        regime_volatilities=[R0_SIGMA_STEP, R1_SIGMA_STEP],
        r=1,
        jump_size=EPSILON_PCT,
        jump_intensity=LAMBDA,
        step_size=STEP_SIZE,
        rv_window=8,
        rv_weight=1.0,
    )
    for seed in range(n_tries):
        data     = run_episode(filt_test, seed)
        switches = int(np.sum(np.abs(np.diff(data["regime"]))))
        if switches >= min_switches:
            print(f"Using seed={seed} ({switches} regime switches)")
            return seed
    return 0


def compute_llr(ret, sigma0, sigma1, jump_size, jump_prob, eps=1e-10):
    """Log-likelihood ratio log[f(r|Z=1) / f(r|Z=0)]."""
    p  = jump_prob
    w0 = (1 - p)**2 + p**2
    wu = p * (1 - p)
    wd = p * (1 - p)
    f1 = (w0 * norm.pdf(ret, 0.0,         sigma1) +
          wu * norm.pdf(ret, +jump_size,   sigma1) +
          wd * norm.pdf(ret, -jump_size,   sigma1))
    f0 = norm.pdf(ret, 0.0, sigma0)
    return np.log((f1 + eps) / (f0 + eps))


def shade_regime1(ax, steps, regime):
    in_r1 = np.where(regime == 1)[0]
    if len(in_r1) == 0:
        return
    starts = in_r1[np.concatenate(([True], np.diff(in_r1) > 1))]
    ends   = in_r1[np.concatenate((np.diff(in_r1) > 1, [True]))]
    for s, e in zip(starts, ends):
        ax.axvspan(steps[s], steps[min(e+1, len(steps)-1)],
                   alpha=0.08, color='tomato', label='_nolegend_')


def plot_single(data, title, save_name, jump_size_pct):
    """Three-panel plot: regime, belief, LLR."""
    steps   = data["steps"]
    regime  = data["regime"]
    belief  = data["belief"]
    returns = data["returns"]

    llrs = np.array([
        compute_llr(r, R0_SIGMA_STEP, R1_SIGMA_STEP,
                    jump_size_pct, LAMBDA * STEP_SIZE)
        for r in returns
    ])

    jump_mask  = (np.abs(returns) > jump_size_pct * 0.5) & (regime == 1)
    jump_steps = steps[jump_mask]
    jump_llrs  = llrs[jump_mask]
    jump_bels  = belief[jump_mask]

    fig = plt.figure(figsize=(13, 8))
    gs  = gridspec.GridSpec(3, 1, hspace=0.35)
    xlim = (steps[0], steps[-1])

    # Panel 1: True regime
    ax1 = fig.add_subplot(gs[0])
    ax1.step(steps, regime, where='post', color='black', linewidth=1.5)
    ax1.fill_between(steps, regime, step='post', alpha=0.15, color='tomato')
    ax1.set_ylabel('Regime', fontsize=10)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['0 (calm)', '1 (adverse)'], fontsize=9)
    ax1.set_ylim(-0.1, 1.4)
    ax1.set_title(title, fontsize=11)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(xlim)
    ax1.tick_params(labelbottom=False)

    # Panel 2: Belief
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(steps, belief, color='steelblue', linewidth=1.5,
             label=r'$\pi_t = P(\mathrm{regime}=1 \mid \mathcal{F}_t)$')
    ax2.axhline(0.5, color='grey', linestyle=':', linewidth=0.8)
    shade_regime1(ax2, steps, regime)
    ax2.scatter(jump_steps, jump_bels, color='tomato', s=20,
                zorder=5, marker='^', label='Jump event')
    ax2.set_ylabel(r'Belief $\pi_t$', fontsize=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(xlim)
    ax2.tick_params(labelbottom=False)

    # Panel 3: LLR
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(steps, llrs, color='purple', linewidth=1.0, alpha=0.8,
             label=r'$\log[f(r_t|Z=1)/f(r_t|Z=0)]$')
    ax3.axhline(0.0, color='grey', linestyle='--', linewidth=0.8)
    shade_regime1(ax3, steps, regime)
    ax3.scatter(jump_steps, jump_llrs, color='tomato', s=20,
                zorder=5, marker='^', label='Jump event')
    ax3.set_ylabel('Log-LR', fontsize=10)
    ax3.set_xlabel('Step', fontsize=10)
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.2)
    ax3.set_xlim(xlim)

    timestamp = datetime.now().strftime("%H%M%S")
    fname     = os.path.join(IMAGES_DIR, f'{save_name}_{timestamp}.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def main():
    seed = find_seed()

    jump_size_pct = EPSILON_PCT

    # ------------------------------------------------------------------
    # Configuration 1: r=0, Gaussian emission, no RV
    # ------------------------------------------------------------------
    print("Running configuration 1: r=0, Gaussian, no RV...")
    filt1 = HamiltonFilter(
        transition_matrix=TRANSITION_MATRIX,
        regime_volatilities=[R0_SIGMA_STEP, R1_SIGMA_STEP],
        r=0,
        jump_size=None,     # Gaussian only
        jump_intensity=None,
        step_size=STEP_SIZE,
        rv_window=0,        # no RV
        rv_weight=0.0,
    )
    data1 = run_episode(filt1, seed)
    plot_single(
        data1,
        title=r'Specification 1: $r=0$, Gaussian emission, no realised variance',
        save_name='belief_spec1_r0_gaussian',
        jump_size_pct=jump_size_pct,
    )

    # ------------------------------------------------------------------
    # Configuration 2: r=1, jump-mixture emission, no RV
    # ------------------------------------------------------------------
    print("Running configuration 2: r=1, jump-mixture, no RV...")
    filt2 = HamiltonFilter(
        transition_matrix=TRANSITION_MATRIX,
        regime_volatilities=[R0_SIGMA_STEP, R1_SIGMA_STEP],
        r=1,
        jump_size=EPSILON_PCT,
        jump_intensity=LAMBDA,
        step_size=STEP_SIZE,
        rv_window=0,        # no RV
        rv_weight=0.0,
    )
    data2 = run_episode(filt2, seed)
    plot_single(
        data2,
        title=r'Specification 2: $r=1$, jump-diffusion mixture emission, no realised variance',
        save_name='belief_spec2_r1_mixture',
        jump_size_pct=jump_size_pct,
    )

    # ------------------------------------------------------------------
    # Configuration 3: r=1, jump-mixture emission, with RV (w=8)
    # ------------------------------------------------------------------
    print("Running configuration 3: r=1, jump-mixture, RV w=8...")
    filt3 = HamiltonFilter(
        transition_matrix=TRANSITION_MATRIX,
        regime_volatilities=[R0_SIGMA_STEP, R1_SIGMA_STEP],
        r=1,
        jump_size=EPSILON_PCT,
        jump_intensity=LAMBDA,
        step_size=STEP_SIZE,
        rv_window=8,
        rv_weight=1.0,
    )
    data3 = run_episode(filt3, seed)
    plot_single(
        data3,
        title=r'Specification 3: $r=1$, jump-diffusion mixture emission, realised variance $w=8$',
        save_name='belief_spec3_r1_rv',
        jump_size_pct=jump_size_pct,
    )

    # ------------------------------------------------------------------
    # Print summary metrics for all three
    # ------------------------------------------------------------------
    print("\n--- Belief Summary ---")
    print(f"{'Metric':<35} {'Spec 1':>10} {'Spec 2':>10} {'Spec 3':>10}")
    print("-" * 65)

    for label, key_fn in [
        ("Mean belief (regime 1)",
         lambda d: np.mean(d["belief"][d["regime"] == 1])),
        ("Mean belief (regime 0)",
         lambda d: 1 - np.mean(d["belief"][d["regime"] == 0])),
        ("Belief var (regime 1)",
         lambda d: np.var(d["belief"][d["regime"] == 1])),
        ("Accuracy",
         lambda d: np.mean((d["belief"] >= 0.5).astype(int) == d["regime"])),
    ]:
        vals = [key_fn(data1), key_fn(data2), key_fn(data3)]
        print(f"{label:<35} {vals[0]:>10.3f} {vals[1]:>10.3f} {vals[2]:>10.3f}")

    print("\nDone. Three plots saved to images folder.")


if __name__ == "__main__":
    main()
