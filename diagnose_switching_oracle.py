"""
diagnose_switching_oracle.py
-----------------------------
Standalone sensitivity diagnostic. Does NOT modify any production file --
simulate_belief_weighted.build_optimal_control is imported and called
exactly as production does, purely as a read-only reference ("the existing
oracle"). All new solver code lives in this file only.

Objective
---------
Determine whether accounting for future Markov regime switches materially
changes the optimal quote depths, by solving a coupled switching-aware HJB
directly in h-space (not assuming the linearising w=exp(kappa h) transform
survives the addition of a switching term) and comparing it against the
existing regime-static oracle control.

Derivation
----------
The existing static solver integrates the LINEAR ODE

    dw_q/dt = kappa*phi*q^2*w_q - r_+ w_{q-1} - r_- w_{q+1},   w_q(T) = exp(-kappa*alpha*q^2)

where w_q = exp(kappa*h_q). Substituting w_q = exp(kappa*h_q) recovers the
ORIGINAL nonlinear HJB this was derived from (verified algebraically -- see
below), for a single regime with no switching:

    dh_q/dt = phi*q^2 - (r_+/kappa)*exp(kappa*(h_{q-1}-h_q)) - (r_-/kappa)*exp(kappa*(h_{q+1}-h_q))
    h_q(T) = -alpha*q^2,   r_+ = r_- = lambda*exp(-1-kappa*eps)

(Check: dw_q/dt = kappa*(dh_q/dt)*exp(kappa*h_q) = kappa*phi*q^2*exp(kappa*h_q)
 - r_+*exp(kappa*h_{q-1}) - r_-*exp(kappa*h_{q+1}); dividing by kappa*exp(kappa*h_q)
 gives exactly the h-space ODE above.)

Adding a two-regime Markov switching generator with intensities nu_01, nu_10,
the ansatz V^{(i)}(t,x,q,s) = x + q*s + h_q^{(i)}(t) makes the switching
contribution to the HJB exactly nu_{i,1-i}*(V^{(1-i)} - V^{(i)}) =
nu_{i,1-i}*(h_q^{(1-i)} - h_q^{(i)}) -- the x+q*s terms cancel identically,
since cash/inventory/price do not jump when the regime switches, only the
DYNAMICS change. This term is ADDITIVE and does not interact with the
per-instant sup_delta optimisation (which is regime-local and unchanged in
form), so the quote formula and the r_+/r_- coefficients are UNCHANGED from
the static case -- only h_q^{(i)}(t) itself changes. The coupled system is:

    dh_q^{(i)}/dt = phi*q^2 - (r_+^{(i)}/kappa)*exp(kappa*(h_{q-1}^{(i)}-h_q^{(i)}))
                            - (r_-^{(i)}/kappa)*exp(kappa*(h_{q+1}^{(i)}-h_q^{(i)}))
                            - nu_{i,1-i}*(h_q^{(1-i)}(t) - h_q^{(i)}(t))
    h_q^{(i)}(T) = -alpha*q^2   for both i=0,1

This is NONLINEAR (via the exp(kappa*Delta h) fill terms) even though the
switching term itself is linear in h -- substituting w=exp(kappa*h) does NOT
linearise the switching term (it becomes nu*(1/kappa)*log(w_j/w_i), not
nu*(w_j-w_i)), so per the task instructions this is solved directly in h via
explicit RK4 time-stepping, NOT assumed to reduce to a linear w-space system.

Quote formula (identical form to the static case, since sup_delta is
regime-local and unaffected by the additive switching term):
    delta^{a,*(i)}(t,q) = 1/kappa + eps_i + (h_q^{(i)} - h_{q-1}^{(i)})
    delta^{b,*(i)}(t,q) = 1/kappa + eps_i + (h_q^{(i)} - h_{q+1}^{(i)})

Boundary treatment mirrors the existing solver's padded-grid fix exactly
(Q_MAX_SOLVE = Q_MAX+1, then the outermost level is discarded): the ask
term is masked to zero at q_grid[0] AND q_grid[-1] is naturally absent, the
bid term is masked to zero at q_grid[-1] AND q_grid[0] is naturally absent
-- replicating A[0,1]=0 / A[N-1,N-2]=0 from the existing solver's matrix
construction so the two solvers agree exactly (to numerical-integration
tolerance) when nu_01=nu_10=0.

Run:
    python diagnose_switching_oracle.py
"""

import os
import time
import numpy as np
from scipy.linalg import logm
import matplotlib.pyplot as plt
from datetime import datetime

import simulate_belief_weighted as SBW  # existing static oracle -- READ ONLY
from envs.make_envs import (
    KAPPA, LAMBDA, EPSILON, TERMINAL_TIME, N_STEPS,
    PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION,
    TRANSITION_MATRIX_DT_0_005, make_regime_envs, INITIAL_PRICE,
)

IMAGES_DIR = r'C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images'

# ------------------------------------------------------------------
# Parameters -- reproduced exactly from the existing static solver
# ------------------------------------------------------------------
Q_MAX = SBW.Q_MAX          # identical inventory grid (requirement 3)
N_T   = N_STEPS             # 4000, identical time grid (requirement 3)
T     = TERMINAL_TIME       # 1.0
DT    = T / N_T

KAPPA_ = KAPPA
PHI    = PER_STEP_INVENTORY_AVERSION
ALPHA  = TERMINAL_INVENTORY_AVERSION
LAM    = LAMBDA
EPS = {0: 0.0, 1: EPSILON}   # matches SBW.REGIME_PARAMS exactly

print("=" * 78)
print("Parameters reproduced from the existing static solver / production env")
print("=" * 78)
print(f"KAPPA={KAPPA_}  PHI={PHI}  ALPHA={ALPHA}  LAMBDA={LAM}  EPS={EPS}")
print(f"Q_MAX={Q_MAX}  N_T={N_T}  T={T}  DT={DT}")

# ------------------------------------------------------------------
# Continuous-time switching generator
# ------------------------------------------------------------------
DT_OLD = 0.005
Q_GEN = logm(np.array(TRANSITION_MATRIX_DT_0_005)) / DT_OLD
if np.iscomplexobj(Q_GEN):
    assert np.max(np.abs(Q_GEN.imag)) < 1e-10, "Q has non-negligible imaginary part"
    Q_GEN = Q_GEN.real
NU01 = float(Q_GEN[0, 1])
NU10 = float(Q_GEN[1, 0])

print("\n" + "=" * 78)
print("Continuous-time regime-switching generator")
print("=" * 78)
print(f"Q =\n{Q_GEN}")
print(f"nu_01 (0->1 transition intensity) = {NU01:.6f}")
print(f"nu_10 (1->0 transition intensity) = {NU10:.6f}")
print(f"Implied mean duration regime 0 (1/-Q[0,0]) = {1.0/(-Q_GEN[0,0]):.6f}")
print(f"Implied mean duration regime 1 (1/-Q[1,1]) = {1.0/(-Q_GEN[1,1]):.6f}")


# ======================================================================
# Grid & boundary conventions -- mirrors build_optimal_control exactly
# ======================================================================
def make_grid(q_max):
    q_max_solve = q_max + 1
    N = 2 * q_max_solve + 1
    q_grid = np.arange(-q_max_solve, q_max_solve + 1, dtype=float)
    return q_max_solve, N, q_grid


Q_MAX_SOLVE, N, Q_GRID = make_grid(Q_MAX)


def ask_bid_masks(N):
    idx = np.arange(N)
    ask_mask = (idx > 0).astype(float)
    bid_mask = (idx < N - 1).astype(float)
    ask_mask[N - 1] = 0.0  # mirrors A[N-1, N-2] = 0
    bid_mask[0] = 0.0      # mirrors A[0, 1] = 0
    return ask_mask, bid_mask


ASK_MASK, BID_MASK = ask_bid_masks(N)


def fill_rate(kappa, lam, eps):
    return lam * np.exp(-1.0 - kappa * eps)


# ======================================================================
# Coupled nonlinear h-space RHS and RK4 backward integrator
# ======================================================================
def h_rhs_single(h, q_grid, kappa, phi, r, ask_mask, bid_mask):
    N = len(h)
    h_left = np.empty(N)
    h_left[1:] = h[:-1]
    h_left[0] = h[0]
    h_right = np.empty(N)
    h_right[:-1] = h[1:]
    h_right[-1] = h[-1]
    exp_ask = np.exp(kappa * (h_left - h))
    exp_bid = np.exp(kappa * (h_right - h))
    return phi * q_grid ** 2 - (ask_mask * r / kappa) * exp_ask - (bid_mask * r / kappa) * exp_bid


def coupled_rhs(h0, h1, q_grid, kappa, phi, r0, r1, nu01, nu10, ask_mask, bid_mask):
    rhs0 = h_rhs_single(h0, q_grid, kappa, phi, r0, ask_mask, bid_mask) - nu01 * (h1 - h0)
    rhs1 = h_rhs_single(h1, q_grid, kappa, phi, r1, ask_mask, bid_mask) - nu10 * (h0 - h1)
    return rhs0, rhs1


def solve_switching(kappa, phi, alpha, r0, r1, nu01, nu10, q_grid, ask_mask, bid_mask, n_t, dt):
    """Backward RK4 integration from t=T (h=-alpha q^2) to t=0. Returns H0, H1 of shape (n_t, N)."""
    N = len(q_grid)
    H0 = np.zeros((n_t, N))
    H1 = np.zeros((n_t, N))
    h0 = -alpha * q_grid ** 2
    h1 = -alpha * q_grid ** 2
    H0[n_t - 1] = h0
    H1[n_t - 1] = h1

    def f(a, b):
        return coupled_rhs(a, b, q_grid, kappa, phi, r0, r1, nu01, nu10, ask_mask, bid_mask)

    for n in range(n_t - 2, -1, -1):
        k1_0, k1_1 = f(h0, h1)
        k2_0, k2_1 = f(h0 - 0.5 * dt * k1_0, h1 - 0.5 * dt * k1_1)
        k3_0, k3_1 = f(h0 - 0.5 * dt * k2_0, h1 - 0.5 * dt * k2_1)
        k4_0, k4_1 = f(h0 - dt * k3_0, h1 - dt * k3_1)
        h0 = h0 - (dt / 6.0) * (k1_0 + 2 * k2_0 + 2 * k3_0 + k4_0)
        h1 = h1 - (dt / 6.0) * (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1)
        H0[n] = h0
        H1[n] = h1
    return H0, H1


def quotes_from_h(H, q_grid, kappa, eps):
    """Mirrors build_optimal_control's post-processing exactly, computed directly from h."""
    d_ask = (1.0 / kappa) + eps + (H[:, 1:] - H[:, :-1])
    d_bid = (1.0 / kappa) + eps + (H[:, :-1] - H[:, 1:])
    q_ask = q_grid[1:]
    q_bid = q_grid[:-1]
    return d_ask[:, 1:-1], d_bid[:, 1:-1], q_ask[1:-1], q_bid[1:-1]


# ======================================================================
# Validation A: zero-switching reproduces the existing static controls
# ======================================================================
def run_validation_A():
    print("\n" + "=" * 78)
    print("VALIDATION A: zero-switching (nu_01=nu_10=0) reproduces static controls")
    print("=" * 78)

    results = {}
    all_ok = True
    for regime in (0, 1):
        eps = EPS[regime]
        r = fill_rate(KAPPA_, LAM, eps)

        # Existing static solver, called exactly as production does (read-only import)
        da_static, db_static, qa_static, qb_static = SBW.build_optimal_control(
            kappa=KAPPA_, lam=LAM, eps=eps, phi=PHI, alpha=ALPHA,
            q_max=Q_MAX, n_t=N_T, terminal_time=T,
        )

        # New h-space solver with nu=0 (the OTHER regime's h is irrelevant since
        # its coupling term is multiplied by zero, but the coupled solve still
        # needs a second regime object -- use a duplicate of the same regime,
        # which under nu=0 has literally no effect on this regime's own h).
        H0, H1 = solve_switching(KAPPA_, PHI, ALPHA, r, r, 0.0, 0.0, Q_GRID, ASK_MASK, BID_MASK, N_T, DT)
        da_new, db_new, qa_new, qb_new = quotes_from_h(H0, Q_GRID, KAPPA_, eps)

        assert np.array_equal(qa_static, qa_new) and np.array_equal(qb_static, qb_new), \
            "Inventory grids differ between static and switching-aware solvers"

        ask_diff = np.abs(da_new - da_static)
        bid_diff = np.abs(db_new - db_static)
        max_ask, mean_ask = ask_diff.max(), ask_diff.mean()
        max_bid, mean_bid = bid_diff.max(), bid_diff.mean()

        # The outermost 1-2 returned grid points (q close to +/-Q_MAX) directly
        # reference the discarded, padded boundary level (q=+/-(Q_MAX+1)), which
        # is a fully decoupled, degenerate ODE row (Error 7 in the project's own
        # write-up: "makes log(W) blow up in the single ask/bid cell that
        # references it directly ... everything else remains smooth"). The
        # existing solver computes this via a matrix exponential of w=exp(kappa*h)
        # -- an astronomically large number at this row (w ~ exp(kappa*phi*Q_MAX^2))
        # -- then recovers h via log(w), which loses floating-point precision at
        # that scale. This solver works directly in h and never exponentiates to
        # those extremes, so it does not inherit that precision loss. Report both
        # the full grid and the interior (excluding the outermost 2 points each
        # side, where the effect empirically vanishes) to make this transparent.
        interior = (qa_static > qa_static.min() + 2) & (qa_static < qa_static.max() - 2)
        max_ask_int, mean_ask_int = ask_diff[:, interior].max(), ask_diff[:, interior].mean()
        max_bid_int, mean_bid_int = bid_diff[:, interior].max(), bid_diff[:, interior].mean()

        results[regime] = dict(max_ask=max_ask, mean_ask=mean_ask, max_bid=max_bid, mean_bid=mean_bid)
        print(f"\nRegime {regime} (eps={eps}):")
        print(f"  [full grid, q in [{qa_static.min():.0f},{qa_static.max():.0f}]]")
        print(f"  max_abs_ask_difference  = {max_ask:.3e}")
        print(f"  max_abs_bid_difference  = {max_bid:.3e}")
        print(f"  mean_abs_ask_difference = {mean_ask:.3e}")
        print(f"  mean_abs_bid_difference = {mean_bid:.3e}")
        print(f"  [interior grid, excluding outermost 2 points each side -- realistic")
        print(f"   simulated inventory stays within |q|~2-4, far inside this region]")
        print(f"  max_abs_ask_difference  = {max_ask_int:.3e}")
        print(f"  max_abs_bid_difference  = {max_bid_int:.3e}")
        print(f"  mean_abs_ask_difference = {mean_ask_int:.3e}")
        print(f"  mean_abs_bid_difference = {mean_bid_int:.3e}")

        tol_interior = 1e-3
        ok = max_ask_int < tol_interior and max_bid_int < tol_interior
        all_ok = all_ok and ok
        print(f"  PASS on interior grid (< {tol_interior:.0e})? {ok}")

    return all_ok, results


# ======================================================================
# Validation B: identical-regime parameters -> h_0 == h_1, ask_0 == ask_1, bid_0 == bid_1
# ======================================================================
def run_validation_B():
    print("\n" + "=" * 78)
    print("VALIDATION B: identical-regime parameters -> symmetric solution")
    print("=" * 78)
    eps = EPS[1]  # arbitrary common value for both regimes
    r = fill_rate(KAPPA_, LAM, eps)
    H0, H1 = solve_switching(KAPPA_, PHI, ALPHA, r, r, NU01, NU10, Q_GRID, ASK_MASK, BID_MASK, N_T, DT)
    da0, db0, qa, qb = quotes_from_h(H0, Q_GRID, KAPPA_, eps)
    da1, db1, _, _ = quotes_from_h(H1, Q_GRID, KAPPA_, eps)

    h_diff = np.abs(H0 - H1).max()
    ask_diff = np.abs(da0 - da1).max()
    bid_diff = np.abs(db0 - db1).max()
    print(f"max|h_0 - h_1|     = {h_diff:.3e}")
    print(f"max|ask_0 - ask_1| = {ask_diff:.3e}")
    print(f"max|bid_0 - bid_1| = {bid_diff:.3e}")
    tol = 1e-10
    ok = h_diff < tol and ask_diff < tol and bid_diff < tol
    print(f"PASS (< {tol:.0e})? {ok}")
    return ok


# ======================================================================
# Validation C: terminal condition
# ======================================================================
def run_validation_C():
    print("\n" + "=" * 78)
    print("VALIDATION C: terminal condition h_i(T,q) == -alpha*q^2")
    print("=" * 78)
    r0 = fill_rate(KAPPA_, LAM, EPS[0])
    r1 = fill_rate(KAPPA_, LAM, EPS[1])
    H0, H1 = solve_switching(KAPPA_, PHI, ALPHA, r0, r1, NU01, NU10, Q_GRID, ASK_MASK, BID_MASK, N_T, DT)
    expected_terminal = -ALPHA * Q_GRID ** 2
    diff0 = np.abs(H0[-1] - expected_terminal).max()
    diff1 = np.abs(H1[-1] - expected_terminal).max()
    print(f"max|h_0(T,q) - (-alpha q^2)| = {diff0:.3e}")
    print(f"max|h_1(T,q) - (-alpha q^2)| = {diff1:.3e}")
    ok = diff0 < 1e-12 and diff1 < 1e-12
    print(f"PASS (< 1e-12)? {ok}")
    return ok, H0, H1


# ======================================================================
# Main control comparison
# ======================================================================
Q_POINTS = [-10, -5, 0, 5, 10]
T_FRACS = [0.0, 0.25, 0.5, 0.75, 0.95]


def solve_all_controls():
    static = {}
    for regime in (0, 1):
        eps = EPS[regime]
        da, db, qa, qb = SBW.build_optimal_control(
            kappa=KAPPA_, lam=LAM, eps=eps, phi=PHI, alpha=ALPHA,
            q_max=Q_MAX, n_t=N_T, terminal_time=T,
        )
        static[regime] = dict(ask=da, bid=db, q_ask=qa, q_bid=qb)

    r0 = fill_rate(KAPPA_, LAM, EPS[0])
    r1 = fill_rate(KAPPA_, LAM, EPS[1])
    H0, H1 = solve_switching(KAPPA_, PHI, ALPHA, r0, r1, NU01, NU10, Q_GRID, ASK_MASK, BID_MASK, N_T, DT)
    da0, db0, qa0, qb0 = quotes_from_h(H0, Q_GRID, KAPPA_, EPS[0])
    da1, db1, qa1, qb1 = quotes_from_h(H1, Q_GRID, KAPPA_, EPS[1])
    switching = {0: dict(ask=da0, bid=db0, q_ask=qa0, q_bid=qb0),
                 1: dict(ask=da1, bid=db1, q_ask=qa1, q_bid=qb1)}
    return static, switching


def diff_stats(static_arr, switching_arr, t_idx_grid, q_grid_vals):
    diff = switching_arr - static_arr
    abs_diff = np.abs(diff)
    max_abs = abs_diff.max()
    mean_abs = abs_diff.mean()
    rms = np.sqrt(np.mean(diff ** 2))
    p95 = np.percentile(abs_diff, 95)
    rel = abs_diff / (np.abs(static_arr) + 1e-12)
    max_rel = rel.max()
    loc = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    loc_t = t_idx_grid[loc[0]] * DT
    loc_q = q_grid_vals[loc[1]]
    return dict(max_abs=max_abs, mean_abs=mean_abs, rms=rms, p95=p95, max_rel=max_rel,
                loc_t=loc_t, loc_q=loc_q)


def run_main_comparison(static, switching):
    print("\n" + "=" * 78)
    print("MAIN CONTROL COMPARISON: switching-aware vs static oracle")
    print("=" * 78)

    t_idx_grid = np.arange(N_T)
    stats_all = {}
    fill_prob_diffs = []

    for regime in (0, 1):
        s = static[regime]
        w = switching[regime]
        assert np.array_equal(s["q_ask"], w["q_ask"]) and np.array_equal(s["q_bid"], w["q_bid"])

        ask_stats = diff_stats(s["ask"], w["ask"], t_idx_grid, s["q_ask"])
        bid_stats = diff_stats(s["bid"], w["bid"], t_idx_grid, s["q_bid"])
        stats_all[regime] = dict(ask=ask_stats, bid=bid_stats)

        print(f"\n--- Regime {regime} (eps={EPS[regime]}) ---")
        for side, st in (("ASK", ask_stats), ("BID", bid_stats)):
            print(f"  {side}: max_abs={st['max_abs']:.5f}  mean_abs={st['mean_abs']:.5f}  "
                  f"rms={st['rms']:.5f}  p95_abs={st['p95']:.5f}  max_rel={st['max_rel']:.4f}  "
                  f"loc=(t={st['loc_t']:.4f}, q={st['loc_q']:.0f})")

        # Fill probability change: exp(-kappa*depth)
        p_fill_static_ask = np.exp(-KAPPA_ * s["ask"])
        p_fill_switch_ask = np.exp(-KAPPA_ * w["ask"])
        p_fill_static_bid = np.exp(-KAPPA_ * s["bid"])
        p_fill_switch_bid = np.exp(-KAPPA_ * w["bid"])
        fill_prob_diffs.append(np.abs(p_fill_switch_ask - p_fill_static_ask))
        fill_prob_diffs.append(np.abs(p_fill_switch_bid - p_fill_static_bid))

    fill_prob_diffs = np.concatenate([d.ravel() for d in fill_prob_diffs])
    max_fill_diff = fill_prob_diffs.max()
    mean_fill_diff = fill_prob_diffs.mean()
    print(f"\nFill-probability change (p_fill = exp(-kappa*depth)):")
    print(f"  max_abs_fill_probability_difference  = {max_fill_diff:.5f}")
    print(f"  mean_abs_fill_probability_difference = {mean_fill_diff:.5f}")

    # Compact table at requested (q, t/T) points
    print("\n" + "=" * 100)
    print("Compact table: static vs switching-aware ask/bid depths at requested (t/T, q) points")
    print("=" * 100)
    header = f"{'regime':<7}{'t/T':>6}{'q':>5}{'ask_static':>12}{'ask_switch':>12}{'d_ask':>9}" \
             f"{'bid_static':>12}{'bid_switch':>12}{'d_bid':>9}"
    print(header)
    for regime in (0, 1):
        s, w = static[regime], switching[regime]
        for tf in T_FRACS:
            n = int(round(tf * N_T))
            n = min(n, N_T - 1)
            for q in Q_POINTS:
                ia = np.where(s["q_ask"] == q)[0]
                ib = np.where(s["q_bid"] == q)[0]
                if len(ia) == 0 or len(ib) == 0:
                    continue
                ia, ib = ia[0], ib[0]
                a_s, a_w = s["ask"][n, ia], w["ask"][n, ia]
                b_s, b_w = s["bid"][n, ib], w["bid"][n, ib]
                print(f"{regime:<7}{tf:>6.2f}{q:>5d}{a_s:>12.5f}{a_w:>12.5f}{a_w-a_s:>9.5f}"
                      f"{b_s:>12.5f}{b_w:>12.5f}{b_w-b_s:>9.5f}")

    return stats_all, max_fill_diff, mean_fill_diff


# ======================================================================
# Plots
# ======================================================================
def make_plots(static, switching):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    t_axis = np.arange(N_T) * DT

    # Plot 1: quote-depth difference vs time for q=-5,0,5, regimes 0 & 1
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {-5: "#4C72B0", 0: "#55A868", 5: "#C44E52"}
    for ax, side in zip(axes, ["ask", "bid"]):
        for regime, ls in zip((0, 1), ("-", "--")):
            s, w = static[regime], switching[regime]
            grid = s["q_ask"] if side == "ask" else s["q_bid"]
            for q in (-5, 0, 5):
                idx = np.where(grid == q)[0]
                if len(idx) == 0:
                    continue
                idx = idx[0]
                diff = w[side][:, idx] - s[side][:, idx]
                ax.plot(t_axis, diff, color=colors[q], linestyle=ls, linewidth=1.4,
                        label=f"regime {regime}, q={q}")
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("t")
        ax.set_ylabel(f"{side} depth: switching - static")
        ax.set_title(f"{side.capitalize()} depth difference vs time")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.2)
    plt.suptitle("Switching-aware minus static oracle: quote-depth difference over time", fontsize=12)
    plt.tight_layout()
    fname1 = os.path.join(IMAGES_DIR, f"switching_oracle_diff_vs_time_{timestamp}.png")
    plt.savefig(fname1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: heatmaps of abs(switching - static) over (t, q), per regime, ask & bid
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for row, regime in enumerate((0, 1)):
        s, w = static[regime], switching[regime]
        for col, side in enumerate(["ask", "bid"]):
            ax = axes[row, col]
            grid = s["q_ask"] if side == "ask" else s["q_bid"]
            abs_diff = np.abs(w[side] - s[side])
            im = ax.pcolormesh(t_axis, grid, abs_diff.T, shading="auto", cmap="viridis")
            ax.set_xlabel("t")
            ax.set_ylabel("q")
            ax.set_title(f"Regime {regime}, {side}: |switching - static|")
            fig.colorbar(im, ax=ax)
    plt.suptitle("Heatmap of |switching-aware - static| quote-depth difference", fontsize=12)
    plt.tight_layout()
    fname2 = os.path.join(IMAGES_DIR, f"switching_oracle_heatmap_{timestamp}.png")
    plt.savefig(fname2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return fname1, fname2


# ======================================================================
# Staged simulation (only if thresholds exceeded)
# ======================================================================
def run_paired_simulation(static, switching, n_episodes=200):
    print("\n" + "=" * 78)
    print(f"STAGED SIMULATION: paired comparison, n_episodes={n_episodes}")
    print("=" * 78)

    def controls_dict(d):
        return {r: {"delta_ask": d[r]["ask"], "delta_bid": d[r]["bid"],
                    "q_ask": d[r]["q_ask"], "q_bid": d[r]["q_bid"]} for r in (0, 1)}

    static_controls = controls_dict(static)
    switching_controls = controls_dict(switching)

    NOTIONAL_CAPITAL = INITIAL_PRICE * 10_000
    pnl_static_all, pnl_switch_all, obj_static_all, obj_switch_all = [], [], [], []

    for seed in range(n_episodes):
        for controls, pnl_list, obj_list in (
            (static_controls, pnl_static_all, obj_static_all),
            (switching_controls, pnl_switch_all, obj_switch_all),
        ):
            env = make_regime_envs(switch_within_episode=True, seed=seed)
            obs = env.reset()
            done = np.array([False])
            cash_0, inv_0, mid_0 = env.raw_cash, env.raw_inventory, env.raw_midprice
            obj_accum = 0.0
            while not np.all(done):
                obs_flat = np.array(obs).flatten()
                inventory = obs_flat[1]
                t_idx = env.current_step
                inv_sc = inventory / SBW.INV_UNIT
                regime = env.current_regime
                ctrl = controls[regime]
                ask, bid = SBW.get_control(ctrl["delta_ask"], ctrl["delta_bid"],
                                            ctrl["q_ask"], ctrl["q_bid"], t_idx, inv_sc)
                action = np.array([[SBW.normalise_depth(bid), SBW.normalise_depth(ask)]])
                obs, reward, done, info = env.step(action)
                obj_accum += float(np.sum(reward))
            cash_T, inv_T, mid_T = info["raw_state"][0], info["raw_state"][1], info["raw_state"][3]
            pnl = cash_T + inv_T * (mid_T - ALPHA * inv_T) - (cash_0 + inv_0 * mid_0)
            pnl_list.append(pnl)
            obj_list.append(obj_accum)

    pnl_static_all = np.array(pnl_static_all)
    pnl_switch_all = np.array(pnl_switch_all)
    obj_static_all = np.array(obj_static_all)
    obj_switch_all = np.array(obj_switch_all)

    def paired_report(name, static_vals, switch_vals):
        diff = switch_vals - static_vals
        mean_diff = diff.mean() / NOTIONAL_CAPITAL * 100
        se = diff.std(ddof=1) / np.sqrt(len(diff)) / NOTIONAL_CAPITAL * 100
        ci = (mean_diff - 1.96 * se, mean_diff + 1.96 * se)
        print(f"\n{name} (switching - static), % of notional capital:")
        print(f"  mean paired difference = {mean_diff:.6f}%")
        print(f"  standard error         = {se:.6f}%")
        print(f"  95% CI                 = [{ci[0]:.6f}%, {ci[1]:.6f}%]")
        return mean_diff, se, ci

    paired_report("Raw PnL", pnl_static_all, pnl_switch_all)
    paired_report("Full objective", obj_static_all, obj_switch_all)


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    t_start = time.time()

    ok_A, _ = run_validation_A()
    ok_B = run_validation_B()
    ok_C, H0_full, H1_full = run_validation_C()

    print("\n" + "=" * 78)
    print(f"Validation summary: A={ok_A}  B={ok_B}  C={ok_C}")
    print("=" * 78)

    if not (ok_A and ok_B and ok_C):
        raise SystemExit("Validation failed -- stopping before main comparison, per task instructions.")

    print("\nAll validations passed. Proceeding to main control comparison.")

    static_controls, switching_controls = solve_all_controls()
    stats_all, max_fill_diff, mean_fill_diff = run_main_comparison(static_controls, switching_controls)

    fname1, fname2 = make_plots(static_controls, switching_controls)
    print(f"\nPlot 1 (depth difference vs time) saved to: {fname1}")
    print(f"Plot 2 (heatmaps)                 saved to: {fname2}")

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    max_overall = max(stats_all[r][side]["max_abs"] for r in (0, 1) for side in ("ask", "bid"))

    r0_ask_diff = switching_controls[0]["ask"] - static_controls[0]["ask"]
    r1_ask_diff = switching_controls[1]["ask"] - static_controls[1]["ask"]
    widens_calm = r0_ask_diff.mean() > 0
    narrows_adverse = r1_ask_diff.mean() < 0

    q0_idx_ask = np.where(static_controls[0]["q_ask"] == 0)[0]
    far_idx_ask = np.where(np.abs(static_controls[0]["q_ask"]) >= 10)[0]
    near_zero_diff = np.abs(r0_ask_diff[:, q0_idx_ask]).mean() if len(q0_idx_ask) else float("nan")
    far_diff = np.abs(r0_ask_diff[:, far_idx_ask]).mean() if len(far_idx_ask) else float("nan")

    print(f"Calm-regime (0) mean signed ask difference (switching-static): {r0_ask_diff.mean():+.6f}"
          f"  -> {'WIDENS' if widens_calm else 'does NOT widen'} calm-regime quotes on average")
    print(f"Adverse-regime (1) mean signed ask difference: {r1_ask_diff.mean():+.6f}"
          f"  -> {'NARROWS' if narrows_adverse else 'does NOT narrow'} adverse-regime quotes on average")
    print(f"Mean |difference| near q=0: {near_zero_diff:.6f}   at |q|>=10: {far_diff:.6f}"
          f"  -> {'differs mainly at large inventory' if far_diff > 2*near_zero_diff else 'comparable across inventory'}")
    print(f"Near q=0 difference is {'effectively indistinguishable (<0.001)' if near_zero_diff < 1e-3 else 'non-negligible'}.")

    # ------------------------------------------------------------------
    # Staged simulation rule
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("STAGED SIMULATION RULE")
    print("=" * 78)
    print(f"max_abs_depth_difference        = {max_overall:.6f}   (threshold: > 0.02)")
    print(f"max_abs_fill_probability_diff   = {max_fill_diff:.6f}   (threshold: > 0.01)")

    trigger = (max_overall > 0.02) or (max_fill_diff > 0.01)
    if trigger:
        print("\nAt least one threshold exceeded -- running the paired 200-episode simulation.")
        run_paired_simulation(static_controls, switching_controls, n_episodes=200)
    else:
        print("\nNeither threshold exceeded. Per the staged-simulation rule, stopping here.")
        print("Conclusion: switching anticipation produces control-table differences too small")
        print("to plausibly explain the oracle-vs-naive similarity found earlier -- the effect")
        print("is dominated by other factors (e.g. spread-revenue flatness near the optimum),")
        print("not by the regime-static approximation in the existing oracle.")

    print(f"\nTotal runtime: {time.time()-t_start:.1f}s")
