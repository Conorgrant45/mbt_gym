"""
plot_spread_diagnostic.py
--------------------------
Four-panel diagnostic plot for a single episode:

    Panel 1: True hidden regime (step function)
    Panel 2: Log-likelihood ratio log[f(r_t|Z=1) / f(r_t|Z=0)] with jump markers
    Panel 3: Hamilton filter belief P(regime=1 | F_t)
    Panel 4: Oracle vs belief-weighted ask depths

Run from repo root:
    python plot_spread_diagnostic.py
"""

import numpy as np
from scipy.linalg import expm
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from envs.make_envs import (
    make_regime_envs,
    KAPPA, LAMBDA, EPSILON,
    TERMINAL_TIME, N_STEPS, STEP_SIZE,
    TRANSITION_MATRIX,
    R0_VOLATILITY, R1_VOLATILITY,
    R0_SIGMA_STEP, R1_SIGMA_STEP, EPSILON_PCT,
    PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION,
)
from beliefs.hamilton_filter import HamiltonFilter

IMAGES_DIR = r'C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images'

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
# phi/alpha must match the environment's actual RunningInventoryPenalty.
REGIME_PARAMS = {
    0: {"kappa": KAPPA, "lam": LAMBDA, "eps": 0.0,    "phi": PER_STEP_INVENTORY_AVERSION, "alpha": TERMINAL_INVENTORY_AVERSION},
    1: {"kappa": KAPPA, "lam": LAMBDA, "eps": EPSILON, "phi": PER_STEP_INVENTORY_AVERSION, "alpha": TERMINAL_INVENTORY_AVERSION},
}

Q_MAX     = 50
N_T       = N_STEPS
T         = TERMINAL_TIME
MAX_DEPTH = -np.log(0.01) / KAPPA
INV_UNIT  = 1.0 / 10000.0


def normalise_depth(depth):
    return float(np.clip((depth / MAX_DEPTH) * 2.0 - 1.0, -1.0, 1.0))


def build_optimal_control(kappa, lam, eps, phi, alpha, q_max=Q_MAX, n_t=N_T, terminal_time=T):
    # Solve on a grid padded by 1 inventory level on each side, then discard
    # the outermost cell. The reflecting boundary condition (A[0,1]=0,
    # A[N-1,N-2]=0 below) decouples the boundary row's dynamics, which makes
    # log(W) blow up in the single ask/bid cell that references it directly
    # -- everything else on the grid is smooth. Padding pushes that corrupted
    # cell outside the range we actually return.
    q_max_solve = q_max + 1
    N      = 2 * q_max_solve + 1
    q_grid = np.arange(-q_max_solve, q_max_solve + 1, dtype=float)
    r_plus  = lam * np.exp(-1.0 - kappa * eps)
    r_minus = lam * np.exp(-1.0 - kappa * eps)
    A = np.zeros((N, N))
    for i, q in enumerate(q_grid):
        A[i, i] = kappa * (phi * q**2)
        if i > 0:
            A[i, i - 1] = -r_plus
        if i < N - 1:
            A[i, i + 1] = -r_minus
    A[0, 1]     = 0.0
    A[N-1, N-2] = 0.0
    w_T    = np.exp(-kappa * alpha * q_grid**2)
    # t_grid[n] = n*dt exactly, matching the wrapper's global clock
    # (RegimeSwitchingEnv.current_step * dt) -- linspace(0, terminal_time,
    # n_t) instead spaces points by terminal_time/(n_t-1), a subtle
    # off-by-one mismatch against the actual simulation step size.
    dt     = terminal_time / n_t
    t_grid = np.arange(n_t) * dt
    W      = np.zeros((n_t, N))
    for n, t in enumerate(t_grid):
        W[n] = expm(A * (t - terminal_time)) @ w_T
        W[n] = np.maximum(W[n], 1e-300)
    delta_ask = (1.0/kappa) + eps + (1.0/kappa) * np.log(W[:, 1:] / W[:, :-1])
    delta_bid = (1.0/kappa) + eps + (1.0/kappa) * np.log(W[:, :-1] / W[:, 1:])
    q_ask = q_grid[1:]
    q_bid = q_grid[:-1]
    return delta_ask[:, 1:-1], delta_bid[:, 1:-1], q_ask[1:-1], q_bid[1:-1]


def get_control(delta_ask, delta_bid, q_ask_grid, q_bid_grid, t_idx, inventory_scaled):
    q       = int(np.clip(np.round(inventory_scaled), q_bid_grid[0], q_ask_grid[-1]))
    ask_idx = int(np.clip(np.searchsorted(q_ask_grid, q), 0, len(q_ask_grid) - 1))
    bid_idx = int(np.clip(np.searchsorted(q_bid_grid, q), 0, len(q_bid_grid) - 1))
    return max(float(delta_ask[t_idx, ask_idx]), 0.0), max(float(delta_bid[t_idx, bid_idx]), 0.0)


def jump_mixture_llr(ret, sigma0, sigma1, jump_size, jump_prob, eps=1e-10):
    """
    Log-likelihood ratio log[f(r_t | Z=1) / f(r_t | Z=0)].

    Uses the jump-diffusion mixture for regime 1 and Gaussian for regime 0.
    Large positive values = strong evidence for regime 1 (jump observed).
    Large negative values = strong evidence for regime 0 (tiny return).
    """
    p = jump_prob
    # Regime 1: jump-mixture density
    w_no  = (1 - p)**2 + p**2
    w_up  = p * (1 - p)
    w_dn  = p * (1 - p)
    f1 = (w_no  * norm.pdf(ret, 0.0,        sigma1) +
          w_up  * norm.pdf(ret, +jump_size,  sigma1) +
          w_dn  * norm.pdf(ret, -jump_size,  sigma1))
    # Regime 0: Gaussian density
    f0 = norm.pdf(ret, 0.0, sigma0)
    return np.log((f1 + eps) / (f0 + eps))


def run_single_episode(controls, filt, seed=None):
    env = make_regime_envs(switch_within_episode=True)
    if seed is not None:
        np.random.seed(seed)

    obs  = env.reset()
    filt.reset()
    done = False
    # env.raw_midprice is only safe right after reset()/step(); the regime
    # advances every step (switch_within_episode=True), so re-reading the
    # property at the top of the next iteration can silently switch to a
    # sub-env that's been idle since reset(), frozen at its own reset price.
    #
    # mid_regime is the regime that actually produced `mid` -- what pi
    # should be scored against. env.current_regime has already advanced to
    # the regime for the UPCOMING step, which is right for the oracle's
    # action but the wrong label to pair with pi for accuracy purposes.
    mid        = env.raw_midprice
    mid_regime = env.current_regime

    steps        = []
    true_regimes = []
    beliefs      = []
    oracle_ask   = []
    belief_ask   = []
    returns      = []
    llrs         = []
    prev_mid     = None

    step = 0
    while not np.all(done):
        obs_flat  = np.array(obs).flatten()
        inventory = obs_flat[1]
        # Use the wrapper's authoritative global step counter directly
        # (t = t_idx * dt) rather than reconstructing it from the
        # normalised time observation.
        t_idx = env.current_step
        inv_sc    = inventory / INV_UNIT
        regime    = env.current_regime

        # Compute return and LLR
        if prev_mid is not None and abs(prev_mid) > 1e-10:
            ret = (mid - prev_mid) / abs(prev_mid)
        else:
            ret = 0.0

        llr = jump_mixture_llr(
            ret,
            sigma0=R0_SIGMA_STEP,
            sigma1=R1_SIGMA_STEP,
            jump_size=EPSILON_PCT,
            jump_prob=LAMBDA * STEP_SIZE,
        )

        pi = filt.update(mid)

        ctrl  = controls[regime]
        o_ask, o_bid = get_control(
            ctrl["delta_ask"], ctrl["delta_bid"],
            ctrl["q_ask"],     ctrl["q_bid"],
            t_idx, inv_sc
        )
        ctrl0 = controls[0]
        ctrl1 = controls[1]
        a0, _ = get_control(ctrl0["delta_ask"], ctrl0["delta_bid"], ctrl0["q_ask"], ctrl0["q_bid"], t_idx, inv_sc)
        a1, _ = get_control(ctrl1["delta_ask"], ctrl1["delta_bid"], ctrl1["q_ask"], ctrl1["q_bid"], t_idx, inv_sc)
        bw_ask = (1 - pi) * a0 + pi * a1

        steps.append(step)
        true_regimes.append(mid_regime)
        beliefs.append(pi)
        oracle_ask.append(o_ask)
        belief_ask.append(bw_ask)
        returns.append(ret)
        llrs.append(llr)

        prev_mid = mid
        # mbt_gym action layout is [bid_depth, ask_depth] (BID_INDEX=0, ASK_INDEX=1)
        action = np.array([[normalise_depth(o_bid), normalise_depth(o_ask)]])
        obs, reward, done, info = env.step(action)
        mid        = info['raw_midprice']
        mid_regime = info['true_regime']
        step += 1

    return {
        "steps":   np.array(steps),
        "regime":  np.array(true_regimes),
        "belief":  np.array(beliefs),
        "oracle_ask": np.array(oracle_ask),
        "belief_ask": np.array(belief_ask),
        "returns": np.array(returns),
        "llrs":    np.array(llrs),
    }


def find_informative_episode(controls, filt, n_tries=200):
    for seed in range(n_tries):
        filt.reset()
        data = run_single_episode(controls, filt, seed=seed)
        switches = np.sum(np.abs(np.diff(data["regime"])))
        if switches >= 3:
            print(f"Found informative episode (seed={seed}, switches={int(switches)})")
            return data
    return run_single_episode(controls, filt, seed=0)


def shade_regime1(ax, steps, regime):
    """Shade regime 1 periods on an axes."""
    in_r1 = np.where(regime == 1)[0]
    if len(in_r1) == 0:
        return
    starts = in_r1[np.concatenate(([True], np.diff(in_r1) > 1))]
    ends   = in_r1[np.concatenate((np.diff(in_r1) > 1, [True]))]
    for s, e in zip(starts, ends):
        ax.axvspan(steps[s], steps[min(e+1, len(steps)-1)],
                   alpha=0.08, color='tomato', label='_nolegend_')


def plot_episode(data, save_path):
    steps   = data["steps"]
    regime  = data["regime"]
    belief  = data["belief"]
    llrs    = data["llrs"]
    returns = data["returns"]

    # Identify jump steps: large returns in regime 1 only
    # Jumps only occur in regime 1 by model construction
    jump_threshold = EPSILON_PCT * 0.5   # half the expected jump size in pct
    jump_mask  = (np.abs(returns) > jump_threshold) & (regime == 1)
    jump_steps = steps[jump_mask]
    jump_vals  = belief[jump_mask]

    fig = plt.figure(figsize=(14, 13))
    gs  = gridspec.GridSpec(4, 1, hspace=0.4)

    xlim = (steps[0], steps[-1])

    # --- Panel 1: True regime ---
    ax1 = fig.add_subplot(gs[0])
    ax1.step(steps, regime, where='post', color='black', linewidth=1.5, label='True regime')
    ax1.fill_between(steps, regime, step='post', alpha=0.15, color='tomato')
    ax1.set_ylabel('Regime', fontsize=10)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['0 (calm)', '1 (adverse)'], fontsize=9)
    ax1.set_ylim(-0.1, 1.4)
    ax1.set_title('Regime, log-likelihood ratio, belief, and quoted depths', fontsize=11)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(xlim)

    # --- Panel 2: Log-likelihood ratio ---
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(steps, llrs, color='purple', linewidth=1.0, alpha=0.8,
             label=r'$\log\,[f(r_t|Z=1)/f(r_t|Z=0)]$')
    ax2.axhline(0.0, color='grey', linestyle='--', linewidth=0.8)
    shade_regime1(ax2, steps, regime)

    # Mark jump events (regime 1 only)
    jump_llr_vals = llrs[jump_mask]
    ax2.scatter(jump_steps, jump_llr_vals, color='tomato', s=25, zorder=5,
                label='Jump event', marker='^')

    ax2.set_ylabel('Log-LR', fontsize=10)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim(xlim)

    # --- Panel 3: Hamilton filter belief ---
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(steps, belief, color='steelblue', linewidth=1.5,
             label=r'$\pi_t = P(\mathrm{regime}=1 \mid \mathcal{F}_t)$')
    ax3.axhline(0.5, color='grey', linestyle=':', linewidth=0.8)
    shade_regime1(ax3, steps, regime)

    # Mark jump events on belief panel
    ax3.scatter(jump_steps, jump_vals, color='tomato', s=25, zorder=5,
                label='Jump event', marker='^')

    ax3.set_ylabel(r'Belief $\pi_t$', fontsize=10)
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.25)
    ax3.set_xlim(xlim)

    # --- Panel 4: Quoted ask depths ---
    ax4 = fig.add_subplot(gs[3])
    ax4.step(steps, data["oracle_ask"], where='post',
             color='black', linewidth=1.5, label=r'Oracle $\delta^{+,*}$', zorder=3)
    ax4.plot(steps, data["belief_ask"],
             color='green', linewidth=1.2, alpha=0.85,
             label=r'Belief-weighted $\hat{\delta}^+_t$', zorder=2)
    shade_regime1(ax4, steps, regime)

    ax4.set_ylabel(r'Ask depth $\delta^+$', fontsize=10)
    ax4.set_xlabel('Step', fontsize=10)
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.25)
    ax4.set_xlim(xlim)

    import os
    from datetime import datetime
    timestamp = datetime.now().strftime("%H%M%S")
    fname = os.path.join(save_path, f'spread_diagnostic_{timestamp}.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"Plot saved to {fname}")


def main():
    print("Building optimal controls...")
    controls = {}
    for regime, params in REGIME_PARAMS.items():
        da, db, qag, qbg = build_optimal_control(**params)
        controls[regime] = {"delta_ask": da, "delta_bid": db, "q_ask": qag, "q_bid": qbg}

    filt = HamiltonFilter(
        transition_matrix=TRANSITION_MATRIX,
        regime_volatilities=[R0_SIGMA_STEP, R1_SIGMA_STEP],
        r=0,  # verified identical marginal belief to any r>0 for this model, at lower cost
        jump_size=EPSILON_PCT,
        jump_intensity=LAMBDA,
        step_size=STEP_SIZE,
    )

    print("Searching for informative episode...")
    data = find_informative_episode(controls, filt)
    print(f"Episode length : {len(data['steps'])} steps")
    print(f"Regime switches: {int(np.sum(np.abs(np.diff(data['regime']))))}")
    jump_mask_main = (np.abs(data["returns"]) > EPSILON_PCT * 0.5) & (data["regime"] == 1)
    print(f"Jump events    : {int(jump_mask_main.sum())} (regime 1 only)")

    plot_episode(data, IMAGES_DIR)


if __name__ == "__main__":
    main()
