"""
plot_final_belief.py
--------------------
Four-panel diagnostic plot for a single representative episode:

    Panel 1: True hidden regime (step function)
    Panel 2: Hamilton filter belief P(regime=1 | F_t)
    Panel 3: Ask depths quoted by all four agents vs oracle optimal

This figure directly connects the filter output to policy behaviour,
showing how each agent responds to regime uncertainty.

Run from repo root:
    python plot_final_belief.py
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import expm
from datetime import datetime

from envs.make_envs import (
    make_regime_envs,
    KAPPA, LAMBDA, EPSILON, EPSILON_PCT,
    TERMINAL_TIME, N_STEPS, STEP_SIZE,
    TRANSITION_MATRIX,
    R0_VOLATILITY, R1_VOLATILITY,
    R0_SIGMA_STEP, R1_SIGMA_STEP,
    PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION,
)
from beliefs.hamilton_filter import HamiltonFilter

IMAGES_DIR = r'C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images'

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
NAIVE_DEPTH = (1.0 / KAPPA) + (EPSILON / 2.0)


def normalise_depth(depth):
    return float(np.clip((depth / MAX_DEPTH) * 2.0 - 1.0, -1.0, 1.0))


def build_optimal_control(kappa, lam, eps, phi, alpha,
                          q_max=Q_MAX, n_t=N_T, terminal_time=T):
    # Solve on a grid padded by 1 inventory level on each side, then discard
    # the outermost cell. The reflecting boundary condition (A[0,1]=0,
    # A[N-1,N-2]=0 below) decouples the boundary row's dynamics, which makes
    # log(W) blow up in the single ask/bid cell that references it directly
    # -- everything else on the grid is smooth. Padding pushes that corrupted
    # cell outside the range we actually return.
    q_max_solve = q_max + 1
    N      = 2 * q_max_solve + 1
    q_grid = np.arange(-q_max_solve, q_max_solve + 1, dtype=float)
    r_p    = lam * np.exp(-1.0 - kappa * eps)
    r_m    = lam * np.exp(-1.0 - kappa * eps)
    A      = np.zeros((N, N))
    for i, q in enumerate(q_grid):
        A[i, i] = kappa * (phi * q**2)
        if i > 0:     A[i, i - 1] = -r_p
        if i < N - 1: A[i, i + 1] = -r_m
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
    d_ask = (1.0/kappa) + eps + (1.0/kappa) * np.log(W[:, 1:] / W[:, :-1])
    d_bid = (1.0/kappa) + eps + (1.0/kappa) * np.log(W[:, :-1] / W[:, 1:])
    q_ask = q_grid[1:]
    q_bid = q_grid[:-1]
    return d_ask[:, 1:-1], d_bid[:, 1:-1], q_ask[1:-1], q_bid[1:-1]


def get_control(d_ask, d_bid, q_ask, q_bid, t_idx, inv_sc):
    q       = int(np.clip(np.round(inv_sc), q_bid[0], q_ask[-1]))
    ask_idx = int(np.clip(np.searchsorted(q_ask, q), 0, len(q_ask) - 1))
    bid_idx = int(np.clip(np.searchsorted(q_bid, q), 0, len(q_bid) - 1))
    return max(float(d_ask[t_idx, ask_idx]), 0.0), max(float(d_bid[t_idx, bid_idx]), 0.0)


def shade_regime1(ax, steps, regime):
    in_r1 = np.where(regime == 1)[0]
    if len(in_r1) == 0:
        return
    starts = in_r1[np.concatenate(([True], np.diff(in_r1) > 1))]
    ends   = in_r1[np.concatenate((np.diff(in_r1) > 1, [True]))]
    for s, e in zip(starts, ends):
        ax.axvspan(steps[s], steps[min(e+1, len(steps)-1)],
                   alpha=0.08, color='tomato', label='_nolegend_')


def run_episode(controls, filt, seed):
    env = make_regime_envs(switch_within_episode=True)
    np.random.seed(seed)
    obs  = env.reset()
    filt.reset()
    done = False
    # env.raw_midprice is only safe to read right after reset()/step() -- as
    # soon as the regime advances (every step, since switch_within_episode),
    # the property starts reading whichever sub-env is newly current, which
    # may not have been touched this episode and is frozen at its own reset
    # price. Track the last known-good midprice explicitly instead of
    # re-reading the (possibly stale) property at the top of the loop.
    #
    # mid_regime is the regime that actually produced `mid` (i.e. the regime
    # for the LAST completed step) -- this is what the filter's belief pi
    # should be scored against. env.current_regime, by contrast, has already
    # advanced to the regime governing the UPCOMING step, which is what the
    # oracle needs (it's allowed to know the regime about to apply) but is
    # the wrong label to pair with pi for accuracy/calibration purposes.
    mid        = env.raw_midprice
    mid_regime = env.current_regime

    steps            = []
    true_regimes     = []
    beliefs          = []
    ask_oracle       = []
    ask_belief       = []
    ask_randomised   = []
    ask_naive        = []

    rng_rand = np.random.default_rng(seed + 1000)

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

        pi = filt.update(mid)

        # Oracle
        ctrl = controls[regime]
        o_ask, _ = get_control(ctrl["delta_ask"], ctrl["delta_bid"],
                                ctrl["q_ask"],     ctrl["q_bid"], t_idx, inv_sc)

        # Belief-weighted
        c0, c1 = controls[0], controls[1]
        a0, _ = get_control(c0["delta_ask"], c0["delta_bid"],
                             c0["q_ask"],     c0["q_bid"], t_idx, inv_sc)
        a1, _ = get_control(c1["delta_ask"], c1["delta_bid"],
                             c1["q_ask"],     c1["q_bid"], t_idx, inv_sc)
        bw_ask = (1.0 - pi) * a0 + pi * a1

        # Randomised
        sampled = int(rng_rand.choice(2, p=[1.0 - pi, pi]))
        ctrl_r  = controls[sampled]
        r_ask, _ = get_control(ctrl_r["delta_ask"], ctrl_r["delta_bid"],
                                ctrl_r["q_ask"],     ctrl_r["q_bid"], t_idx, inv_sc)

        steps.append(step)
        true_regimes.append(mid_regime)
        beliefs.append(pi)
        ask_oracle.append(o_ask)
        ask_belief.append(bw_ask)
        ask_randomised.append(r_ask)
        ask_naive.append(NAIVE_DEPTH)

        action = np.array([[normalise_depth(o_ask), normalise_depth(o_ask)]])
        obs, reward, done, info = env.step(action)
        mid        = info['raw_midprice']
        mid_regime = info['true_regime']
        step += 1

    return {
        "steps":          np.array(steps),
        "regime":         np.array(true_regimes),
        "belief":         np.array(beliefs),
        "ask_oracle":     np.array(ask_oracle),
        "ask_belief":     np.array(ask_belief),
        "ask_randomised": np.array(ask_randomised),
        "ask_naive":      np.array(ask_naive),
    }


def find_good_seed(controls, filt, min_switches=4, n_tries=100):
    for seed in range(n_tries):
        data     = run_episode(controls, filt, seed)
        switches = int(np.sum(np.abs(np.diff(data["regime"]))))
        if switches >= min_switches:
            print(f"Using seed={seed} ({switches} switches, {len(data['steps'])} steps)")
            return data
    return run_episode(controls, filt, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=0,
                        help="Number of lagged regimes in the Hamilton filter's joint "
                             "posterior (default: 0 -- verified to give an identical "
                             "marginal belief to any r>0 for this model, at lower cost).")
    parser.add_argument("--window", type=int, default=None,
                        help="If set, plot only this many steps (centered on the first "
                             "regime switch) instead of the full episode -- a shorter, "
                             "more digestible example. Filter metrics are still computed "
                             "over the full episode.")
    args = parser.parse_args()

    print("Building optimal controls...")
    controls = {}
    for regime, params in REGIME_PARAMS.items():
        da, db, qag, qbg = build_optimal_control(**params)
        controls[regime] = {
            "delta_ask": da, "delta_bid": db,
            "q_ask": qag,    "q_bid": qbg,
        }

    # Return-only filter (RV augmentation disabled -- see
    # beliefs/hamilton_filter.py module docstring: overlapping-window
    # double-counting inflated confidence/detection lag).
    filt = HamiltonFilter(
        transition_matrix=TRANSITION_MATRIX,
        regime_volatilities=[R0_SIGMA_STEP, R1_SIGMA_STEP],
        r=args.r,
        jump_size=EPSILON_PCT,
        jump_intensity=LAMBDA,
        step_size=STEP_SIZE,
    )

    print("Finding informative episode...")
    data = find_good_seed(controls, filt)

    steps   = data["steps"]
    regimes = data["regime"]
    beliefs = data["belief"]

    # Filter metrics
    mask1 = regimes == 1
    mask0 = regimes == 0
    acc   = float(np.mean((beliefs >= 0.5).astype(int) == regimes))
    mb_r1 = float(np.mean(beliefs[mask1])) if mask1.sum() > 0 else float('nan')
    mb_r0 = float(np.mean(beliefs[mask0])) if mask0.sum() > 0 else float('nan')

    print(f"\nFilter metrics (computed over the full {len(steps)}-step episode):")
    print(f"  Accuracy             : {acc:.3f}")
    print(f"  Mean belief regime 1 : {mb_r1:.3f}")
    print(f"  Mean belief regime 0 : {mb_r0:.3f}")

    # ------------------------------------------------------------------
    # Optional windowing -- zoom the PLOT to a short, digestible example
    # (metrics above already reflect the full episode).
    # ------------------------------------------------------------------
    if args.window is not None:
        switch_idx = np.where(np.diff(regimes) != 0)[0]
        center = int(switch_idx[0]) + 1 if len(switch_idx) > 0 else len(steps) // 2
        half = args.window // 2
        lo = max(0, center - half)
        hi = min(len(steps), lo + args.window)
        lo = max(0, hi - args.window)
        sl = slice(lo, hi)
        print(f"  Zooming plot to steps [{steps[lo]}, {steps[hi-1]}] "
              f"({hi-lo} steps, centered on the first regime switch at step {steps[center]})")
        steps   = steps[sl]
        regimes = regimes[sl]
        beliefs = beliefs[sl]
        for key in ("ask_oracle", "ask_belief", "ask_randomised", "ask_naive"):
            data[key] = data[key][sl]

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(3, 1, hspace=0.12, height_ratios=[1, 1.2, 1.5])
    xlim = (steps[0], steps[-1])

    # Panel 1: True regime
    ax1 = fig.add_subplot(gs[0])
    ax1.step(steps, regimes, where='post', color='black', linewidth=1.5,
             label='True regime')
    ax1.fill_between(steps, regimes, step='post', alpha=0.12, color='tomato')
    ax1.set_ylabel('Regime', fontsize=10)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['0 (calm)', '1 (adverse)'], fontsize=9)
    ax1.set_ylim(-0.1, 1.4)
    window_suffix = f', {args.window}-step window' if args.window is not None else ''
    ax1.set_title(
        'Regime detection and policy response: '
        r'Hamilton filter belief $\pi_t$ and quoted ask depths'
        f' (r={args.r}{window_suffix})',
        fontsize=11
    )
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(xlim)
    ax1.tick_params(labelbottom=False)

    # Panel 2: Hamilton filter belief
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(steps, beliefs, color='steelblue', linewidth=1.5,
             label=r'$\pi_t = P(Z_t=1 \mid \mathcal{F}_t)$')
    ax2.axhline(0.5, color='grey', linestyle=':', linewidth=0.9,
                label='Threshold (0.5)')
    shade_regime1(ax2, steps, regimes)
    ax2.set_ylabel(r'Belief $\pi_t$', fontsize=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(xlim)
    ax2.tick_params(labelbottom=False)
    ax2.text(0.01, 0.06,
             f'Acc: {acc:.2f}   Belief|r=1: {mb_r1:.2f}   Belief|r=0: {mb_r0:.2f}',
             transform=ax2.transAxes, fontsize=8,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 3: Ask depths
    ax3 = fig.add_subplot(gs[2])
    ax3.step(steps, data["ask_oracle"],     where='post',
             color='black',     linewidth=1.5, label=r'Oracle $\delta^{+,*}$ (true regime)',  zorder=4)
    ax3.plot(steps, data["ask_belief"],
             color='green',     linewidth=1.2, alpha=0.9,
             label=r'Belief-weighted $\hat{\delta}^+_t$', zorder=3)
    ax3.plot(steps, data["ask_randomised"],
             color='orange',    linewidth=1.0, alpha=0.8,
             label=r'Randomised $\delta^+_{I_t}$',        zorder=2)
    ax3.axhline(NAIVE_DEPTH, color='tomato', linewidth=1.2,
                linestyle='--', label=f'Naive (fixed $\\delta^+={NAIVE_DEPTH:.3f}$)', zorder=1)
    shade_regime1(ax3, steps, regimes)
    ax3.set_ylabel(r'Ask depth $\delta^+$', fontsize=10)
    ax3.set_xlabel('Step', fontsize=10)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.2)
    ax3.set_xlim(xlim)

    timestamp = datetime.now().strftime("%H%M%S")
    window_tag = f'_window{args.window}' if args.window is not None else ''
    fname = os.path.join(IMAGES_DIR, f'belief_vs_regime_r{args.r}{window_tag}_{timestamp}.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to {fname}")


if __name__ == "__main__":
    main()
