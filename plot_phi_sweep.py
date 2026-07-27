"""
plot_phi_sweep.py
------------------
Outer 2x2 grid, one cell per value of per-step inventory aversion (phi),
each cell containing the same inner 2x2 grid of diagnostics used in
simulate_belief_weighted.py (PnL KDE, cumulative PnL, inventory KDE,
cumulative objective) -- 16 panels total, to visually compare how
increasing phi separates the four policies (Oracle / Belief-weighted /
Randomised / Naive).

Edit the three values below, then run from the repo root:
    python plot_phi_sweep.py

Keep N_EPISODES/N_SEEDS small while testing (e.g. 50/1) -- the full run
across 4 phi values takes 4x as long as a single simulate_belief_weighted.py
run at the same episode/seed count.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from scipy.stats import gaussian_kde
from datetime import datetime

import envs.make_envs as me
from simulate_belief_weighted import build_optimal_control, run_agent, INITIAL_PRICE

IMAGES_DIR = r'C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images'

# ------------------------------------------------------------------
# Editable parameters
# ------------------------------------------------------------------
PHI_VALUES = [0.01, 0.1, 0.5, 1.0]   # must have exactly 4 entries -- one per outer grid cell
N_EPISODES = 200
N_SEEDS    = 5
# ------------------------------------------------------------------

AGENTS = [
    ('oracle',     'Oracle',          'steelblue'),
    ('belief',     'Belief-weighted', 'green'),
    ('randomised', 'Randomised',      'orange'),
    ('naive',      'Naive',           'tomato'),
]

NOTIONAL_CAPITAL = INITIAL_PRICE * 10_000


def build_controls_for_phi(phi):
    """Analytic optimal control for both regimes at this phi (alpha held
    fixed at whatever TERMINAL_INVENTORY_AVERSION is currently set to in
    envs/make_envs.py)."""
    controls = {}
    for regime, eps in [(0, 0.0), (1, me.EPSILON)]:
        da, db, qag, qbg = build_optimal_control(
            kappa=me.KAPPA, lam=me.LAMBDA, eps=eps,
            phi=phi, alpha=me.TERMINAL_INVENTORY_AVERSION,
        )
        controls[regime] = {"delta_ask": da, "delta_bid": db, "q_ask": qag, "q_bid": qbg}
    return controls


def run_phi_point(phi, n_episodes, n_seeds):
    """
    Patch envs.make_envs.PER_STEP_INVENTORY_AVERSION to `phi` (make_regime_envs
    reads this module-level constant at call time, so mutating it here before
    calling make_regime_envs -- via run_agent, imported from
    simulate_belief_weighted -- is sufficient; no need to edit make_envs.py),
    build the matching analytic control, and run all four agents x n_seeds.
    """
    me.PER_STEP_INVENTORY_AVERSION = phi
    controls = build_controls_for_phi(phi)

    all_pnl = {a: [] for a, _, _ in AGENTS}
    all_obj = {a: [] for a, _, _ in AGENTS}
    all_inv = {a: [] for a, _, _ in AGENTS}
    for seed in range(n_seeds):
        for agent_type, label, colour in AGENTS:
            pnl, reg, inv, obj = run_agent(agent_type, n_episodes=n_episodes,
                                            controls=controls, seed=seed)
            all_pnl[agent_type].append(pnl)
            all_obj[agent_type].append(obj)
            all_inv[agent_type].append(inv)
    return all_pnl, all_obj, all_inv


def plot_phi_cell(fig, outer_spec, phi, all_pnl, all_obj, all_inv, n_episodes):
    inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_spec, hspace=0.55, wspace=0.35)
    episodes = np.arange(n_episodes)

    # (0,0) PnL distribution KDE
    ax = fig.add_subplot(inner[0, 0])
    for agent_type, label, colour in AGENTS:
        pnl_pool = np.concatenate(all_pnl[agent_type]) / NOTIONAL_CAPITAL * 100.0
        kde = gaussian_kde(pnl_pool, bw_method=0.3)
        x_grid = np.linspace(np.percentile(pnl_pool, 1), np.percentile(pnl_pool, 99), 200)
        ax.plot(x_grid, kde(x_grid), color=colour, linewidth=1.3)
    ax.set_title('PnL KDE', fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)

    # (0,1) mean cumulative PnL, 10th-90th percentile band
    ax = fig.add_subplot(inner[0, 1])
    for agent_type, label, colour in AGENTS:
        cum_mat = np.stack([np.cumsum(p / NOTIONAL_CAPITAL * 100.0) for p in all_pnl[agent_type]])
        ax.plot(episodes, cum_mat.mean(axis=0), color=colour, linewidth=1.2)
        ax.fill_between(episodes, np.percentile(cum_mat, 10, axis=0),
                         np.percentile(cum_mat, 90, axis=0), color=colour, alpha=0.12)
    ax.set_title('Cumulative PnL', fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)

    # (1,0) inventory KDE
    ax = fig.add_subplot(inner[1, 0])
    for agent_type, label, colour in AGENTS:
        inv_pool = np.concatenate(all_inv[agent_type])
        kde = gaussian_kde(inv_pool, bw_method=0.3)
        x_inv = np.linspace(0, np.percentile(inv_pool, 99), 200)
        ax.plot(x_inv, kde(x_inv), color=colour, linewidth=1.3)
    ax.set_title('Inventory KDE', fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)

    # (1,1) mean cumulative objective, 10th-90th percentile band
    ax = fig.add_subplot(inner[1, 1])
    for agent_type, label, colour in AGENTS:
        cum_mat = np.stack([np.cumsum(o / NOTIONAL_CAPITAL * 100.0) for o in all_obj[agent_type]])
        ax.plot(episodes, cum_mat.mean(axis=0), color=colour, linewidth=1.2)
        ax.fill_between(episodes, np.percentile(cum_mat, 10, axis=0),
                         np.percentile(cum_mat, 90, axis=0), color=colour, alpha=0.12)
    ax.set_title('Cumulative Objective', fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)

    # phi label centred above this outer cell's 2x2 block
    bbox = outer_spec.get_position(fig)
    fig.text((bbox.x0 + bbox.x1) / 2, bbox.y1 + 0.008, f'phi = {phi}',
              ha='center', va='bottom', fontsize=13, fontweight='bold')


def add_quadrant_dividers(fig, outer):
    """Draw 4 line segments (forming a '+') along the gutters between the
    four outer phi cells, to visually separate the four sections."""
    bbox_tl, bbox_tr = outer[0].get_position(fig), outer[1].get_position(fig)
    bbox_bl, bbox_br = outer[2].get_position(fig), outer[3].get_position(fig)

    x_mid = (bbox_tl.x1 + bbox_tr.x0) / 2
    y_mid = (bbox_tl.y0 + bbox_bl.y1) / 2
    x_left, x_right = min(bbox_tl.x0, bbox_bl.x0), max(bbox_tr.x1, bbox_br.x1)
    y_top, y_bot = max(bbox_tl.y1, bbox_tr.y1) + 0.02, min(bbox_bl.y0, bbox_br.y0) - 0.01

    segments = [
        ((x_mid, x_mid), (y_mid, y_top)),    # vertical, top half
        ((x_mid, x_mid), (y_bot, y_mid)),    # vertical, bottom half
        ((x_left, x_mid), (y_mid, y_mid)),   # horizontal, left half
        ((x_mid, x_right), (y_mid, y_mid)),  # horizontal, right half
    ]
    for xdata, ydata in segments:
        fig.add_artist(mlines.Line2D(xdata, ydata, transform=fig.transFigure,
                                      color='black', linewidth=1.2, alpha=0.6))


def main():
    if len(PHI_VALUES) != 4:
        raise ValueError(f"PHI_VALUES must have exactly 4 entries, got {len(PHI_VALUES)}: {PHI_VALUES}")

    original_phi = me.PER_STEP_INVENTORY_AVERSION
    fig = plt.figure(figsize=(20, 16))
    outer = gridspec.GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)

    try:
        for i, phi in enumerate(PHI_VALUES):
            print(f"[{i+1}/4] Running phi={phi} ({N_EPISODES} episodes x {N_SEEDS} seeds)...")
            all_pnl, all_obj, all_inv = run_phi_point(phi, N_EPISODES, N_SEEDS)
            plot_phi_cell(fig, outer[i], phi, all_pnl, all_obj, all_inv, N_EPISODES)
    finally:
        me.PER_STEP_INVENTORY_AVERSION = original_phi   # don't leak the monkey-patch

    add_quadrant_dividers(fig, outer)

    handles = [mlines.Line2D([], [], color=colour, linewidth=2, label=label)
               for _, label, colour in AGENTS]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=11, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle(
        f'Effect of per-step inventory aversion (phi) on policy comparison\n'
        f'{N_EPISODES} episodes x {N_SEEDS} seeds per phi value',
        fontsize=14,
    )

    timestamp = datetime.now().strftime("%H%M%S")
    fname = os.path.join(IMAGES_DIR, f'phi_sweep_{timestamp}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fname}")


if __name__ == "__main__":
    main()
