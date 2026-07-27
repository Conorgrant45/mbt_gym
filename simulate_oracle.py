"""
simulate_oracle.py
------------------
Simulates two agents in the RegimeSwitchingEnv and compares their performance:

1. Oracle agent: knows the true regime at each step, applies the semi-analytical
   optimal control derived in Section 3.1.3 of the dissertation.

2. Naive baseline: quotes a fixed symmetric spread at all times, with no
   inventory skewing and no regime awareness. This is the lower bound.

Run from repo root:
    python simulate_oracle.py
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

from envs.make_envs import (
    make_regime_envs,
    KAPPA, LAMBDA, EPSILON,
    TERMINAL_TIME, N_STEPS,
    PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION,
)

# ------------------------------------------------------------------
# Optimal control parameters per regime
# ------------------------------------------------------------------
# phi/alpha must match the environment's actual RunningInventoryPenalty
# (PER_STEP_INVENTORY_AVERSION / TERMINAL_INVENTORY_AVERSION), otherwise the
# analytic "oracle" control is optimal for a different reward than the one
# actually being scored.
REGIME_PARAMS = {
    0: {"kappa": KAPPA, "lam": LAMBDA, "eps": 0.0,    "phi": PER_STEP_INVENTORY_AVERSION, "alpha": TERMINAL_INVENTORY_AVERSION},
    1: {"kappa": KAPPA, "lam": LAMBDA, "eps": EPSILON, "phi": PER_STEP_INVENTORY_AVERSION, "alpha": TERMINAL_INVENTORY_AVERSION},
}

Q_MAX = 5
N_T   = N_STEPS
T     = TERMINAL_TIME

# Action space: mbt_gym normalises depths to [-1, 1]
# max_depth = -log(0.01) / fill_exponent
MAX_DEPTH = -np.log(0.01) / KAPPA


def normalise_depth(depth: float) -> float:
    """Convert raw quote depth to mbt_gym normalised action in [-1, 1]."""
    return float(np.clip((depth / MAX_DEPTH) * 2.0 - 1.0, -1.0, 1.0))


# ------------------------------------------------------------------
# Naive baseline: fixed symmetric depth at q=0, t=0 level
# ------------------------------------------------------------------
# This is 1/kappa + mean(eps across regimes), the oracle's quote at zero inventory
NAIVE_DEPTH = (1.0 / KAPPA) + (EPSILON / 2.0)
NAIVE_ACTION = np.array([[normalise_depth(NAIVE_DEPTH), normalise_depth(NAIVE_DEPTH)]])


def build_optimal_control(kappa, lam, eps, phi, alpha, q_max=Q_MAX, n_t=N_T, terminal_time=T):
    # Solve on a grid padded by 1 inventory level on each side, then discard
    # the outermost cell. The reflecting boundary condition (A[0,1]=0,
    # A[N-1,N-2]=0 below) decouples the boundary row's dynamics, which makes
    # log(W) blow up in the single ask/bid cell that references it directly
    # -- everything else on the grid is smooth. Padding pushes that corrupted
    # cell outside the range we actually return. With Q_MAX as small as it
    # is here, realistic inventory can plausibly reach the boundary, so this
    # matters more than in the Q_MAX=50 scripts.
    q_max_solve = q_max + 1
    N = 2 * q_max_solve + 1
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

    w_T = np.exp(-kappa * alpha * q_grid**2)

    # t_grid[n] = n*dt exactly, matching the wrapper's global clock
    # (RegimeSwitchingEnv.current_step * dt) -- linspace(0, terminal_time,
    # n_t) instead spaces points by terminal_time/(n_t-1), a subtle
    # off-by-one mismatch against the actual simulation step size.
    dt = terminal_time / n_t
    t_grid = np.arange(n_t) * dt
    W = np.zeros((n_t, N))
    for n, t in enumerate(t_grid):
        W[n] = expm(A * (t - terminal_time)) @ w_T
        W[n] = np.maximum(W[n], 1e-300)

    delta_ask = (1.0/kappa) + eps + (1.0/kappa) * np.log(W[:, 1:] / W[:, :-1])
    delta_bid = (1.0/kappa) + eps + (1.0/kappa) * np.log(W[:, :-1] / W[:, 1:])

    q_ask = q_grid[1:]
    q_bid = q_grid[:-1]
    return delta_ask[:, 1:-1], delta_bid[:, 1:-1], q_ask[1:-1], q_bid[1:-1]


def get_control(delta_ask, delta_bid, q_ask_grid, q_bid_grid, t_idx, inventory):
    q = int(np.clip(np.round(inventory), q_bid_grid[0], q_ask_grid[-1]))

    ask_idx = int(np.clip(np.searchsorted(q_ask_grid, q), 0, len(q_ask_grid) - 1))
    bid_idx = int(np.clip(np.searchsorted(q_bid_grid, q), 0, len(q_bid_grid) - 1))

    return max(float(delta_ask[t_idx, ask_idx]), 0.0), max(float(delta_bid[t_idx, bid_idx]), 0.0)


def run_agent(agent_type, n_episodes=500, switch_within_episode=True, controls=None):
    """
    Run one agent for n_episodes.

    agent_type: 'oracle' or 'naive'
    """
    env = make_regime_envs(switch_within_episode=switch_within_episode)

    episode_pnl      = []
    episode_regime   = []
    episode_mean_inv = []

    for ep in range(n_episodes):
        obs      = env.reset()
        done     = False

        cash_0    = env.raw_cash
        inv_0     = env.raw_inventory
        mid_0     = env.raw_midprice
        regime_ep = env.current_regime
        inventories = []

        while not np.all(done):
            obs_flat  = np.array(obs).flatten()
            inventory = env.raw_inventory

            if agent_type == 'oracle':
                # Use the wrapper's authoritative global step counter
                # directly (t = t_idx * dt) rather than reconstructing it
                # from the normalised time observation.
                t_idx  = env.current_step
                regime = env.current_regime
                ctrl   = controls[regime]
                ask, bid = get_control(
                    ctrl["delta_ask"], ctrl["delta_bid"],
                    ctrl["q_ask"],     ctrl["q_bid"],
                    t_idx, inventory
                )
                # mbt_gym action layout is [bid_depth, ask_depth] (BID_INDEX=0, ASK_INDEX=1)
                action = np.array([[normalise_depth(bid), normalise_depth(ask)]])
            else:
                action = NAIVE_ACTION.copy()

            obs, reward, done, info = env.step(action)
            inventories.append(float(info['raw_state'][1]))

        cash_T = info['raw_state'][0]
        inv_T  = info['raw_state'][1]
        mid_T  = info['raw_state'][3]
        alpha  = REGIME_PARAMS[regime_ep]["alpha"]
        pnl    = cash_T + inv_T * (mid_T - alpha * inv_T) - (cash_0 + inv_0 * mid_0)

        episode_pnl.append(float(pnl))
        episode_regime.append(regime_ep)
        episode_mean_inv.append(np.mean(np.abs(inventories)))

    return (np.array(episode_pnl),
            np.array(episode_regime),
            np.array(episode_mean_inv))


def print_stats(name, pnl, regimes, mean_inv):
    print(f"\n--- {name} ---")
    print(f"  Mean PnL          : {pnl.mean():.4f}")
    print(f"  Std PnL           : {pnl.std():.4f}")
    print(f"  Sharpe (approx)   : {pnl.mean() / (pnl.std() + 1e-8):.4f}")
    print(f"  Mean abs inventory: {mean_inv.mean():.4f}")
    for r in [0, 1]:
        mask = regimes == r
        if mask.sum() > 0:
            print(f"  Regime {r} ({mask.sum()} ep): Mean PnL={pnl[mask].mean():.4f}, Std={pnl[mask].std():.4f}")


def main():
    n_episodes = 500

    # Precompute oracle controls
    print("Building optimal controls...")
    controls = {}
    for regime, params in REGIME_PARAMS.items():
        da, db, qag, qbg = build_optimal_control(**params)
        controls[regime] = {"delta_ask": da, "delta_bid": db, "q_ask": qag, "q_bid": qbg}
        print(f"  Regime {regime}: min ask={da.min():.3f}, min bid={db.min():.3f}")

    print(f"\nRunning oracle agent ({n_episodes} episodes)...")
    oracle_pnl, oracle_reg, oracle_inv = run_agent(
        'oracle', n_episodes=n_episodes, controls=controls
    )

    print(f"Running naive baseline ({n_episodes} episodes)...")
    naive_pnl, naive_reg, naive_inv = run_agent(
        'naive', n_episodes=n_episodes
    )

    print_stats("Oracle Agent", oracle_pnl, oracle_reg, oracle_inv)
    print_stats("Naive Baseline", naive_pnl, naive_reg, naive_inv)

    # ------------------------------------------------------------------
    # Comparison plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # PnL distributions
    ax = axes[0]
    bins = np.linspace(
        min(oracle_pnl.min(), naive_pnl.min()),
        max(oracle_pnl.max(), naive_pnl.max()),
        50
    )
    ax.hist(oracle_pnl, bins=bins, alpha=0.6, color='steelblue', label=f'Oracle (mean={oracle_pnl.mean():.3f})')
    ax.hist(naive_pnl,  bins=bins, alpha=0.6, color='tomato',    label=f'Naive  (mean={naive_pnl.mean():.3f})')
    ax.axvline(oracle_pnl.mean(), color='steelblue', linestyle='--', linewidth=1.5)
    ax.axvline(naive_pnl.mean(),  color='tomato',    linestyle='--', linewidth=1.5)
    ax.set_xlabel('Episode PnL')
    ax.set_ylabel('Count')
    ax.set_title('PnL Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cumulative PnL
    ax = axes[1]
    ax.plot(np.cumsum(oracle_pnl), color='steelblue', linewidth=1.5, label='Oracle')
    ax.plot(np.cumsum(naive_pnl),  color='tomato',    linewidth=1.5, label='Naive')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative PnL')
    ax.set_title('Cumulative PnL')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Inventory management
    ax = axes[2]
    bins_inv = np.linspace(0, max(oracle_inv.max(), naive_inv.max()), 40)
    ax.hist(oracle_inv, bins=bins_inv, alpha=0.6, color='steelblue', label='Oracle')
    ax.hist(naive_inv,  bins=bins_inv, alpha=0.6, color='tomato',    label='Naive')
    ax.set_xlabel('Mean absolute inventory')
    ax.set_ylabel('Count')
    ax.set_title('Inventory Management')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Oracle vs Naive Baseline', fontsize=13)
    plt.tight_layout()
    import os as _os; from datetime import datetime as _dt; plt.savefig(_os.path.join(r'C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images', f'oracle_pnl_{_dt.now().strftime("%H%M%S")}.png'), dpi=150)
    print("\nPlot saved to oracle_pnl.png")


if __name__ == "__main__":
    main()
