"""
plot_filter_comparison.py
--------------------------
Compares the Hamilton filter across r = 0, 1, 2, 3 on the same episode.

A 2x2 grid where each cell shows three sub-panels for one value of r:
    (a) True hidden regime
    (b) Hamilton filter belief P(regime=1 | F_t)
    (c) Oracle vs belief-weighted ask depth

Quantitative metrics are printed to the terminal:
    - Mean detection lag (0->1 and 1->0 switches separately)
    - Step-wise classification accuracy
    - Mean belief in correct regime
    - Belief variance within each regime

Run from repo root:
    python plot_filter_comparison.py
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from datetime import datetime

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
R_VALUES  = [0, 1, 2, 3]


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


def run_episode_with_filter(controls, filt, seed):
    """Run one episode and return per-step arrays for plotting and metrics."""
    # seed= must be passed to make_regime_envs() itself, not just
    # np.random.seed() beforehand -- the midprice/arrival/fill models each
    # have their own independent RNG that np.random.seed() does not touch.
    # Without it, every call here (once per r value) gets an unrelated
    # random episode that only happens to share a regime-switching pattern,
    # making the r=0..3 comparison meaningless.
    env = make_regime_envs(switch_within_episode=True, seed=seed)
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

        ctrl = controls[regime]
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

        # mbt_gym action layout is [bid_depth, ask_depth] (BID_INDEX=0, ASK_INDEX=1)
        action = np.array([[normalise_depth(o_bid), normalise_depth(o_ask)]])
        obs, reward, done, info = env.step(action)
        mid        = info['raw_midprice']
        mid_regime = info['true_regime']
        step += 1

    return {
        "steps":      np.array(steps),
        "regime":     np.array(true_regimes),
        "belief":     np.array(beliefs),
        "oracle_ask": np.array(oracle_ask),
        "belief_ask": np.array(belief_ask),
    }


def find_seed(controls, n_tries=500, min_switches=4):
    """Find a seed that produces an episode with enough regime switches."""
    filt_test = HamiltonFilter(
        transition_matrix=TRANSITION_MATRIX,
        regime_volatilities=[R0_SIGMA_STEP, R1_SIGMA_STEP],
        r=1,
        jump_size=EPSILON_PCT,
        jump_intensity=LAMBDA,
        step_size=STEP_SIZE,
    )
    for seed in range(n_tries):
        data = run_episode_with_filter(controls, filt_test, seed=seed)
        switches = int(np.sum(np.abs(np.diff(data["regime"]))))
        if switches >= min_switches:
            print(f"Using seed={seed} ({switches} regime switches)")
            return seed
    print("Warning: using seed=0")
    return 0


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def compute_metrics(data):
    regime = data["regime"]
    belief = data["belief"]
    steps  = data["steps"]

    # Classification accuracy
    predicted = (belief >= 0.5).astype(int)
    accuracy  = float(np.mean(predicted == regime))

    # Mean belief in correct regime
    mask1 = regime == 1
    mask0 = regime == 0
    mean_belief_r1 = float(np.mean(belief[mask1])) if mask1.sum() > 0 else np.nan
    mean_belief_r0 = float(np.mean(1 - belief[mask0])) if mask0.sum() > 0 else np.nan

    # Belief variance within each regime
    var_r1 = float(np.var(belief[mask1])) if mask1.sum() > 0 else np.nan
    var_r0 = float(np.var(belief[mask0])) if mask0.sum() > 0 else np.nan

    # Detection lag at each switch
    switch_points = np.where(np.abs(np.diff(regime)) > 0)[0] + 1
    lags_0to1 = []
    lags_1to0 = []

    for sw in switch_points:
        new_regime = regime[sw]
        threshold  = 0.5
        # Find first step after switch where belief crosses threshold
        lag = None
        for t in range(sw, min(sw + 50, len(belief))):
            if new_regime == 1 and belief[t] >= threshold:
                lag = t - sw
                break
            elif new_regime == 0 and belief[t] < threshold:
                lag = t - sw
                break
        if lag is not None:
            if new_regime == 1:
                lags_0to1.append(lag)
            else:
                lags_1to0.append(lag)

    mean_lag_0to1 = float(np.mean(lags_0to1)) if lags_0to1 else np.nan
    mean_lag_1to0 = float(np.mean(lags_1to0)) if lags_1to0 else np.nan

    return {
        "accuracy":       accuracy,
        "mean_belief_r1": mean_belief_r1,
        "mean_belief_r0": mean_belief_r0,
        "var_r1":         var_r1,
        "var_r0":         var_r0,
        "mean_lag_0to1":  mean_lag_0to1,
        "mean_lag_1to0":  mean_lag_1to0,
    }


def print_metrics_table(all_metrics):
    """Print a clean table of metrics across r values."""
    print("\n" + "="*75)
    print(f"{'Metric':<30} {'r=0':>10} {'r=1':>10} {'r=2':>10} {'r=3':>10}")
    print("="*75)

    metrics_to_print = [
        ("Accuracy",               "accuracy",       ":.3f"),
        ("Mean belief (regime 1)", "mean_belief_r1", ":.3f"),
        ("Mean belief (regime 0)", "mean_belief_r0", ":.3f"),
        ("Belief var (regime 1)",  "var_r1",         ":.4f"),
        ("Belief var (regime 0)",  "var_r0",         ":.4f"),
        ("Lag 0->1 (steps)",       "mean_lag_0to1",  ":.1f"),
        ("Lag 1->0 (steps)",       "mean_lag_1to0",  ":.1f"),
    ]

    for label, key, fmt in metrics_to_print:
        row = f"{label:<30}"
        for r in R_VALUES:
            val = all_metrics[r][key]
            if np.isnan(val):
                row += f"{'N/A':>10}"
            else:
                row += format(val, fmt[1:]).rjust(10)
        print(row)
    print("="*75)


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
def shade_regime1(ax, steps, regime):
    in_r1 = np.where(regime == 1)[0]
    if len(in_r1) == 0:
        return
    starts = in_r1[np.concatenate(([True], np.diff(in_r1) > 1))]
    ends   = in_r1[np.concatenate((np.diff(in_r1) > 1, [True]))]
    for s, e in zip(starts, ends):
        ax.axvspan(steps[s], steps[min(e+1, len(steps)-1)],
                   alpha=0.08, color='tomato', label='_nolegend_')


def plot_grid(all_data, save_path):
    fig = plt.figure(figsize=(18, 16))
    outer_gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.3)

    for idx, r in enumerate(R_VALUES):
        row = idx // 2
        col = idx % 2
        data = all_data[r]

        steps   = data["steps"]
        regime  = data["regime"]
        belief  = data["belief"]
        xlim    = (steps[0], steps[-1])

        # 3 sub-panels per cell
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer_gs[row, col],
            hspace=0.08, height_ratios=[1, 1.2, 1.2]
        )

        # Sub-panel (a): True regime
        ax_r = fig.add_subplot(inner_gs[0])
        ax_r.step(steps, regime, where='post', color='black', linewidth=1.2)
        ax_r.fill_between(steps, regime, step='post', alpha=0.15, color='tomato')
        ax_r.set_ylabel('Regime', fontsize=8)
        ax_r.set_yticks([0, 1])
        ax_r.set_yticklabels(['0', '1'], fontsize=7)
        ax_r.set_ylim(-0.1, 1.4)
        ax_r.set_title(f'$r = {r}$', fontsize=11, fontweight='bold')
        ax_r.grid(True, alpha=0.2)
        ax_r.set_xlim(xlim)
        ax_r.tick_params(labelbottom=False)

        # Sub-panel (b): Belief
        ax_b = fig.add_subplot(inner_gs[1])
        ax_b.plot(steps, belief, color='steelblue', linewidth=1.2,
                  label=r'$\pi_t$')
        ax_b.axhline(0.5, color='grey', linestyle=':', linewidth=0.7)
        shade_regime1(ax_b, steps, regime)
        ax_b.set_ylabel(r'$\pi_t$', fontsize=8)
        ax_b.set_ylim(-0.05, 1.05)
        ax_b.grid(True, alpha=0.2)
        ax_b.set_xlim(xlim)
        ax_b.tick_params(labelbottom=False)

        # Sub-panel (c): Ask depths
        ax_d = fig.add_subplot(inner_gs[2])
        ax_d.step(steps, data["oracle_ask"], where='post',
                  color='black', linewidth=1.2,
                  label=r'Oracle $\delta^{+,*}$')
        ax_d.plot(steps, data["belief_ask"],
                  color='green', linewidth=1.0, alpha=0.85,
                  label=r'Belief $\hat{\delta}^+_t$')
        shade_regime1(ax_d, steps, regime)
        ax_d.set_ylabel(r'$\delta^+$', fontsize=8)
        ax_d.set_xlabel('Step', fontsize=8)
        ax_d.legend(fontsize=7, loc='upper right')
        ax_d.grid(True, alpha=0.2)
        ax_d.set_xlim(xlim)

    fig.suptitle(
        'Hamilton filter comparison across lag orders $r \\in \\{0,1,2,3\\}$\n'
        'Same episode, same random seed',
        fontsize=12
    )

    timestamp = datetime.now().strftime("%H%M%S")
    fname = os.path.join(save_path, f'filter_comparison_{timestamp}.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to {fname}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("Building optimal controls...")
    controls = {}
    for regime, params in REGIME_PARAMS.items():
        da, db, qag, qbg = build_optimal_control(**params)
        controls[regime] = {
            "delta_ask": da, "delta_bid": db,
            "q_ask": qag,    "q_bid": qbg
        }

    # Find a seed with enough switches
    seed = find_seed(controls)

    # Run episode with each filter lag order
    all_data    = {}
    all_metrics = {}

    for r in R_VALUES:
        print(f"Running filter r={r}...")
        filt = HamiltonFilter(
            transition_matrix=TRANSITION_MATRIX,
            regime_volatilities=[R0_SIGMA_STEP, R1_SIGMA_STEP],
            r=r,
            jump_size=EPSILON_PCT,
            jump_intensity=LAMBDA,
            step_size=STEP_SIZE,
        )
        data           = run_episode_with_filter(controls, filt, seed=seed)
        all_data[r]    = data
        all_metrics[r] = compute_metrics(data)
        switches = int(np.sum(np.abs(np.diff(data["regime"]))))
        print(f"  Switches: {switches}, Steps: {len(data['steps'])}")

    print_metrics_table(all_metrics)
    plot_grid(all_data, IMAGES_DIR)


if __name__ == "__main__":
    main()
