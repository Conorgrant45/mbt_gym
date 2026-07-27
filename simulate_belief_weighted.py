"""
simulate_belief_weighted.py
---------------------------
Runs four agents in the RegimeSwitchingEnv and compares performance:

1. Oracle: knows true regime at each step, applies regime-specific
   optimal control delta^{+/-,*}(t, q).

2. Belief-weighted: uses Hamilton filter belief to interpolate controls
   (Section 3.2.2 of the dissertation):
       delta^+_t = pi_t * delta^{+,*}_{r=1}(t,q) + (1-pi_t) * delta^{+,*}_{r=0}(t,q)

3. Randomised: samples a regime according to the belief and applies
   that regime's full optimal control (Section 3.2.3):
       I_t ~ Bernoulli(pi_t), then delta^+_t = delta^{+,*}_{I_t}(t,q)

4. Naive: fixed symmetric spread, no regime awareness, no inventory
   skewing. Lower bound on performance.

Run from repo root:
    python simulate_belief_weighted.py
"""

import os
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from datetime import datetime

from envs.make_envs import (
    make_regime_envs,
    KAPPA, LAMBDA, EPSILON, EPSILON_PCT,
    TERMINAL_TIME, N_STEPS, STEP_SIZE,
    TRANSITION_MATRIX,
    R0_VOLATILITY, R1_VOLATILITY,
    R0_SIGMA_STEP, R1_SIGMA_STEP,
    PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION,
    INITIAL_PRICE,
)
from beliefs.hamilton_filter import HamiltonFilter

IMAGES_DIR = r'C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images'

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
# phi/alpha must match the environment's actual RunningInventoryPenalty
# (PER_STEP_INVENTORY_AVERSION / TERMINAL_INVENTORY_AVERSION), otherwise the
# analytic "oracle" control is optimal for a different reward than the one
# actually being scored.
REGIME_PARAMS = {
    0: {"kappa": KAPPA, "lam": LAMBDA, "eps": 0.0,     "phi": PER_STEP_INVENTORY_AVERSION, "alpha": TERMINAL_INVENTORY_AVERSION},
    1: {"kappa": KAPPA, "lam": LAMBDA, "eps": EPSILON,  "phi": PER_STEP_INVENTORY_AVERSION, "alpha": TERMINAL_INVENTORY_AVERSION},
}

Q_MAX     = 50
N_T       = N_STEPS
T         = TERMINAL_TIME
MAX_DEPTH = -np.log(0.01) / KAPPA
INV_UNIT  = 1.0 / 10000.0


def normalise_depth(depth: float) -> float:
    return float(np.clip((depth / MAX_DEPTH) * 2.0 - 1.0, -1.0, 1.0))


NAIVE_DEPTH  = (1.0 / KAPPA) + (EPSILON / 2.0)
NAIVE_ACTION = np.array([[normalise_depth(NAIVE_DEPTH), normalise_depth(NAIVE_DEPTH)]])


# ------------------------------------------------------------------
# Optimal control
# ------------------------------------------------------------------
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


def make_filter():
    """
    Build the Hamilton filter with jump-mixture emission, return-only
    (RV augmentation disabled -- see beliefs/hamilton_filter.py module
    docstring for why it's not used: overlapping-window double-counting
    inflates within-regime confidence and detection lag).

    r=0: verified (analytically and numerically, to machine precision)
    that the marginal belief P(Z_t|F_t) is identical for any r given this
    model -- the emission conditions only on the current regime z_t, and
    the transition is first-order Markov, so higher r only tracks
    (unused) joint/smoothed path information at strictly higher
    computational cost. r=0 is therefore the efficient default.
    """
    return HamiltonFilter(
        transition_matrix=TRANSITION_MATRIX,
        regime_volatilities=[R0_SIGMA_STEP, R1_SIGMA_STEP],
        r=0,
        jump_size=EPSILON_PCT,
        jump_intensity=LAMBDA,
        step_size=STEP_SIZE,
    )


# ------------------------------------------------------------------
# Agent runner
# ------------------------------------------------------------------
def run_agent(agent_type, n_episodes=2000, controls=None, seed=42):
    """
    Run one agent for n_episodes.

    agent_type : 'oracle' | 'belief' | 'randomised' | 'naive'
    """
    np.random.seed(seed)
    filt = make_filter() if agent_type in ('belief', 'randomised') else None
    env  = make_regime_envs(switch_within_episode=True)

    episode_pnl      = []
    episode_regime   = []
    episode_mean_inv = []
    episode_obj      = []

    for ep in range(n_episodes):
        obs      = env.reset()
        done     = False

        cash_0    = env.raw_cash
        inv_0     = env.raw_inventory
        mid_0     = env.raw_midprice
        # env.raw_midprice is only safe right after reset()/step(); the
        # regime advances every step (switch_within_episode=True), so
        # re-reading the property at the top of the next iteration can
        # silently switch to a sub-env that's been idle since reset(),
        # frozen at its own reset price.
        mid       = mid_0
        regime_ep = env.current_regime
        inventories = []
        obj_accum   = 0.0   # cumulative reward = risk-adjusted CJ objective

        if filt is not None:
            filt.reset()

        while not np.all(done):
            obs_flat  = np.array(obs).flatten()
            inventory = obs_flat[1]
            # Use the wrapper's authoritative global step counter directly
            # (t = t_idx * dt) rather than reconstructing it from the
            # normalised time observation.
            t_idx = env.current_step
            inv_sc    = inventory / INV_UNIT

            if agent_type == 'oracle':
                regime   = env.current_regime
                ctrl     = controls[regime]
                ask, bid = get_control(ctrl["delta_ask"], ctrl["delta_bid"],
                                       ctrl["q_ask"],     ctrl["q_bid"],
                                       t_idx, inv_sc)
                # mbt_gym action layout is [bid_depth, ask_depth] (BID_INDEX=0, ASK_INDEX=1)
                action = np.array([[normalise_depth(bid), normalise_depth(ask)]])

            elif agent_type == 'belief':
                belief = filt.update(mid)
                c0, c1 = controls[0], controls[1]
                a0, b0 = get_control(c0["delta_ask"], c0["delta_bid"],
                                     c0["q_ask"],     c0["q_bid"], t_idx, inv_sc)
                a1, b1 = get_control(c1["delta_ask"], c1["delta_bid"],
                                     c1["q_ask"],     c1["q_bid"], t_idx, inv_sc)
                ask = (1.0 - belief) * a0 + belief * a1
                bid = (1.0 - belief) * b0 + belief * b1
                # mbt_gym action layout is [bid_depth, ask_depth] (BID_INDEX=0, ASK_INDEX=1)
                action = np.array([[normalise_depth(bid), normalise_depth(ask)]])

            elif agent_type == 'randomised':
                belief = filt.update(mid)
                sampled_regime  = int(np.random.choice(2, p=[1.0 - belief, belief]))
                ctrl            = controls[sampled_regime]
                ask, bid        = get_control(ctrl["delta_ask"], ctrl["delta_bid"],
                                              ctrl["q_ask"],     ctrl["q_bid"],
                                              t_idx, inv_sc)
                # mbt_gym action layout is [bid_depth, ask_depth] (BID_INDEX=0, ASK_INDEX=1)
                action = np.array([[normalise_depth(bid), normalise_depth(ask)]])

            else:  # naive
                action = NAIVE_ACTION.copy()

            obs, reward, done, info = env.step(action)
            mid = info['raw_midprice']
            inventories.append(float(info['raw_state'][1]))
            obj_accum += float(np.sum(reward))

        cash_T = info['raw_state'][0]
        inv_T  = info['raw_state'][1]
        mid_T  = info['raw_state'][3]
        alpha  = TERMINAL_INVENTORY_AVERSION
        # Mark-to-market PnL: cash plus terminal inventory value (net of the
        # terminal inventory penalty alpha*q^2), matching simulate_oracle.py.
        pnl    = cash_T + inv_T * (mid_T - alpha * inv_T) - (cash_0 + inv_0 * mid_0)

        episode_pnl.append(float(pnl))
        episode_regime.append(regime_ep)
        episode_mean_inv.append(float(np.mean(np.abs(inventories))))
        episode_obj.append(obj_accum)

    return (np.array(episode_pnl),
            np.array(episode_regime),
            np.array(episode_mean_inv),
            np.array(episode_obj))


def print_stats(name, pnl, regimes, mean_inv):
    print(f"\n--- {name} ---")
    print(f"  Mean PnL          : {pnl.mean():.4f}")
    print(f"  Std PnL           : {pnl.std():.4f}")
    print(f"  Sharpe (approx)   : {pnl.mean() / (pnl.std() + 1e-8):.4f}")
    print(f"  Mean abs inventory: {mean_inv.mean():.6f}")
    for r in [0, 1]:
        mask = regimes == r
        if mask.sum() > 0:
            print(f"  Regime {r} ({mask.sum()} ep): "
                  f"Mean PnL={pnl[mask].mean():.4f}, Std={pnl[mask].std():.4f}")


def main():
    n_episodes = 200
    n_seeds    = 3   # number of independent seeds for averaging
    seeds      = list(range(n_seeds))

    # All PnL/objective figures below are reported as a percentage return on
    # NOTIONAL_CAPITAL -- the dollar value of the largest position
    # (TradingEnvironment's default max_inventory = 10,000 shares) the agent
    # could ever hold -- rather than as raw price-unit dollar amounts.
    NOTIONAL_CAPITAL = INITIAL_PRICE * 10_000

    # Build optimal controls for both regimes
    print("Building optimal controls...")
    controls = {}
    for regime, params in REGIME_PARAMS.items():
        da, db, qag, qbg = build_optimal_control(**params)
        controls[regime] = {
            "delta_ask": da, "delta_bid": db,
            "q_ask": qag,    "q_bid": qbg,
        }
        print(f"  Regime {regime}: min ask={da.min():.3f}, max ask={da.max():.3f}")

    agents = [
        ('oracle',      'Oracle',          'steelblue'),
        ('belief',      'Belief-weighted', 'green'),
        ('randomised',  'Randomised',      'orange'),
        ('naive',       'Naive',           'tomato'),
    ]

    # ------------------------------------------------------------------
    # Run all agents across all seeds
    # pnl_matrix[agent][seed] = array of episode PnLs
    # ------------------------------------------------------------------
    all_pnl = {a: [] for a, _, _ in agents}
    all_inv = {a: [] for a, _, _ in agents}
    all_reg = {a: [] for a, _, _ in agents}
    all_obj = {a: [] for a, _, _ in agents}

    for seed in seeds:
        print(f"\nSeed {seed+1}/{n_seeds}...")
        for agent_type, label, colour in agents:
            pnl, reg, inv, obj = run_agent(agent_type, n_episodes=n_episodes,
                                           controls=controls, seed=seed)
            all_pnl[agent_type].append(pnl)
            all_inv[agent_type].append(inv)
            all_reg[agent_type].append(reg)
            all_obj[agent_type].append(obj)
            pnl_pct = pnl / NOTIONAL_CAPITAL * 100.0
            obj_pct = obj / NOTIONAL_CAPITAL * 100.0
            print(f"  {label}: mean pnl={pnl_pct.mean():.4f}%, sharpe={pnl.mean()/(pnl.std()+1e-8):.3f}, "
                  f"mean obj={obj_pct.mean():.4f}%")

    # ------------------------------------------------------------------
    # Summary table (averaged across seeds)
    #
    # "Objective" is the cumulative per-episode reward returned by the env
    # (RunningInventoryPenalty / CJ criterion): mark-to-market PnL minus the
    # running inventory penalty phi*q_t^2*dt and the terminal penalty
    # alpha*q_T^2. This is the risk-adjusted quantity the optimal control
    # is actually derived to maximise, as opposed to raw PnL -- see the
    # "why doesn't the oracle win on raw PnL" discussion in
    # filter_and_control_fixes.tex.
    # ------------------------------------------------------------------
    print("\n" + "="*130)
    print(f"{'Agent':<20} {'Mean PnL %':>12} {'Std PnL %':>12} {'Sharpe':>10} {'Total PnL %':>13} "
          f"{'Mean Obj %':>12} {'Std Obj %':>12} {'Obj Sharpe':>12} {'Total Obj %':>13}")
    print("="*130)
    for agent_type, label, _ in agents:
        pnl_all     = np.concatenate(all_pnl[agent_type])
        obj_all     = np.concatenate(all_obj[agent_type])
        pnl_all_pct = pnl_all / NOTIONAL_CAPITAL * 100.0
        obj_all_pct = obj_all / NOTIONAL_CAPITAL * 100.0
        # Sharpe is scale-invariant (mean/std), so it's identical whether
        # computed on raw PnL or on the % return -- shown once, unitless.
        sharpe     = pnl_all.mean() / (pnl_all.std() + 1e-8)
        obj_sharpe = obj_all.mean() / (obj_all.std() + 1e-8)
        # Total PnL/Obj %: sum across all episodes and seeds -- the
        # aggregate return on NOTIONAL_CAPITAL if every episode's PnL/Obj
        # were added up (not compounded), i.e. total exposure over the
        # whole n_episodes * n_seeds run, not a per-episode average.
        print(f"{label:<20} {pnl_all_pct.mean():>11.4f}% {pnl_all_pct.std():>11.4f}% {sharpe:>10.4f} "
              f"{pnl_all_pct.sum():>12.4f}% "
              f"{obj_all_pct.mean():>11.4f}% {obj_all_pct.std():>11.4f}% {obj_sharpe:>12.4f} "
              f"{obj_all_pct.sum():>12.4f}%")
    print("="*130)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    # Panel 1: PnL distribution KDE (pooled across seeds), as % of NOTIONAL_CAPITAL
    ax = axes[0]
    pool = np.concatenate([np.concatenate(all_pnl[a]) for a, _, _ in agents]) / NOTIONAL_CAPITAL * 100.0
    x_grid = np.linspace(np.percentile(pool, 1), np.percentile(pool, 99), 300)
    for agent_type, label, colour in agents:
        pnl_pool = np.concatenate(all_pnl[agent_type]) / NOTIONAL_CAPITAL * 100.0
        kde = gaussian_kde(pnl_pool, bw_method=0.3)
        ax.plot(x_grid, kde(x_grid), color=colour, linewidth=2.0,
                label=f'{label} ({pnl_pool.mean():.4f}%)')
        ax.axvline(pnl_pool.mean(), color=colour, linestyle='--',
                   linewidth=1.0, alpha=0.7)
    ax.set_xlabel('Episode PnL (%)')
    ax.set_ylabel('Density')
    ax.set_title(f'PnL Distribution (KDE, {n_seeds} seeds)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Mean cumulative percentage return with 10th-90th percentile band
    ax = axes[1]
    for agent_type, label, colour in agents:
        # Per-episode PnL as a percentage return on NOTIONAL_CAPITAL,
        # cumulatively summed: shape (n_seeds, n_episodes).
        cum_mat = np.stack([np.cumsum(p / NOTIONAL_CAPITAL * 100.0) for p in all_pnl[agent_type]])
        mean_c  = cum_mat.mean(axis=0)
        lo_c    = np.percentile(cum_mat, 10, axis=0)
        hi_c    = np.percentile(cum_mat, 90, axis=0)
        episodes = np.arange(n_episodes)
        ax.plot(episodes, mean_c, color=colour, linewidth=1.8, label=label)
        ax.fill_between(episodes, lo_c, hi_c, color=colour, alpha=0.15)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title(f'Mean Cumulative PnL Return\n(shaded: 10th-90th percentile, {n_seeds} seeds)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Inventory KDE (pooled)
    ax = axes[2]
    pool_inv = np.concatenate([np.concatenate(all_inv[a]) for a, _, _ in agents])
    x_inv = np.linspace(0, np.percentile(pool_inv, 99), 300)
    for agent_type, label, colour in agents:
        inv_pool = np.concatenate(all_inv[agent_type])
        kde = gaussian_kde(inv_pool, bw_method=0.3)
        ax.plot(x_inv, kde(x_inv), color=colour, linewidth=2.0, label=label)
    ax.set_xlabel('Mean absolute inventory')
    ax.set_ylabel('Density')
    ax.set_title(f'Inventory Management (KDE, {n_seeds} seeds)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Mean cumulative objective (risk-adjusted CJ criterion) trajectory,
    # same construction as Panel 2 but for the objective rather than raw PnL.
    ax = axes[3]
    for agent_type, label, colour in agents:
        cum_mat = np.stack([np.cumsum(o / NOTIONAL_CAPITAL * 100.0) for o in all_obj[agent_type]])
        mean_c  = cum_mat.mean(axis=0)
        lo_c    = np.percentile(cum_mat, 10, axis=0)
        hi_c    = np.percentile(cum_mat, 90, axis=0)
        episodes = np.arange(n_episodes)
        ax.plot(episodes, mean_c, color=colour, linewidth=1.8, label=label)
        ax.fill_between(episodes, lo_c, hi_c, color=colour, alpha=0.15)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Objective (%)')
    ax.set_title(f'Mean Cumulative Objective (Risk-Adjusted)\n(shaded: 10th-90th percentile, {n_seeds} seeds)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        'Four-way policy comparison: Oracle vs Belief-weighted vs Randomised vs Naive',
        fontsize=11
    )
    plt.tight_layout()

    timestamp = datetime.now().strftime("%H%M%S")
    fname = os.path.join(IMAGES_DIR, f'policy_comparison_{timestamp}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {fname}")


if __name__ == "__main__":
    main()
