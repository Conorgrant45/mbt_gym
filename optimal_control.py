"""
optimal_control.py
------------------
Implements the semi-analytical solution for the adverse selection market making
optimal control derived in Section 3.1.3 of the dissertation.

Under symmetric fill decay kappa+ = kappa- = kappa, the nonlinear ODE system
for h_q(t) is linearised via the transformation w_q(t) = exp(kappa * h_q(t)),
yielding the linear system:

    d/dt w(t) = A w(t)

where A is a tridiagonal matrix. The solution is:

    w(t) = exp(A(t - T)) w(T)

The optimal controls are then:

    delta+*(t,q) = 1/kappa + epsilon+ + (1/kappa) * log(w_q(t) / w_{q-1}(t))
    delta-*(t,q) = 1/kappa + epsilon- + (1/kappa) * log(w_q(t) / w_{q+1}(t))

Reference: Section 3.1.3 of dissertation, Hamilton (1989), Cartea et al. (2015).

Run from repo root:
    python optimal_control.py
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
KAPPA   = 2.0          # symmetric fill decay
LAMBDA  = 0.75         # symmetric arrival intensity (lambda+ = lambda-)
EPSILON = 0.3          # symmetric adverse selection jump (epsilon+ = epsilon-)
PHI     = 1e-5         # running inventory penalty
ALPHA   = 0.001        # terminal inventory penalty
T       = 1.0          # terminal time
Q_MAX   = 5            # inventory grid: q in {-Q_MAX, ..., Q_MAX}
N_T     = 200          # number of time steps


# ------------------------------------------------------------------
# Derived constants (from dissertation Section 3.1.3)
# ------------------------------------------------------------------
# d  = lambda+ * epsilon+ - lambda- * epsilon-  (= 0 in symmetric case)
# r+ = lambda+ * exp(-1 - kappa * epsilon+)
# r- = lambda- * exp(-1 - kappa * epsilon-)
d      = LAMBDA * EPSILON - LAMBDA * EPSILON   # = 0 (symmetric)
r_plus  = LAMBDA * np.exp(-1.0 - KAPPA * EPSILON)
r_minus = LAMBDA * np.exp(-1.0 - KAPPA * EPSILON)  # = r_plus (symmetric)

print(f"Parameters: kappa={KAPPA}, lambda={LAMBDA}, epsilon={EPSILON}")
print(f"Derived:    d={d:.6f}, r+={r_plus:.6f}, r-={r_minus:.6f}")
print(f"Grid:       Q_max={Q_MAX}, N_t={N_T}, T={T}")
print()


# ------------------------------------------------------------------
# Build tridiagonal matrix A
#
# Solved on a grid padded by 1 inventory level on each side, then the
# outermost level is discarded below. The reflecting boundary condition
# (A[0,1]=0, A[N-1,N-2]=0) decouples the boundary row's dynamics, which
# makes log(w) blow up in the single ask/bid quote that references that
# row directly -- everything else on the grid is smooth. Padding pushes
# that corrupted level outside the range we actually keep.
# ------------------------------------------------------------------
Q_MAX_SOLVE = Q_MAX + 1
N = 2 * Q_MAX_SOLVE + 1          # number of inventory levels (solve grid)
q_grid = np.arange(-Q_MAX_SOLVE, Q_MAX_SOLVE + 1, dtype=float)

A = np.zeros((N, N))

for i, q in enumerate(q_grid):
    # Diagonal: kappa * (phi * q^2 - q * d)
    A[i, i] = KAPPA * (PHI * q**2 - q * d)

    # Sub-diagonal (q-1 -> q): -r+
    if i > 0:
        A[i, i - 1] = -r_plus

    # Super-diagonal (q+1 -> q): -r-
    if i < N - 1:
        A[i, i + 1] = -r_minus

# Boundary treatment: suppress quotes at inventory limits
# At q = -Q_MAX_SOLVE: no ask quote (agent cannot decrease inventory further)
# At q = +Q_MAX_SOLVE: no bid quote (agent cannot increase inventory further)
A[0, 1]      = 0.0   # suppress r- term at lower boundary
A[N-1, N-2]  = 0.0   # suppress r+ term at upper boundary


# ------------------------------------------------------------------
# Terminal condition: w_q(T) = exp(kappa * h_q(T)) = exp(-kappa*alpha*q^2)
# ------------------------------------------------------------------
w_T = np.exp(-KAPPA * ALPHA * q_grid**2)


# ------------------------------------------------------------------
# Solution via matrix exponential: w(t) = exp(A(t-T)) w(T)
# ------------------------------------------------------------------
t_grid = np.linspace(0, T, N_T)

# Store w and h across time
W = np.zeros((N_T, N))   # w_q(t)
H = np.zeros((N_T, N))   # h_q(t) = (1/kappa) * log(w_q(t))

for n, t in enumerate(t_grid):
    W[n] = expm(A * (t - T)) @ w_T
    # Clamp to positive values before log (numerical safety)
    W[n] = np.maximum(W[n], 1e-300)
    H[n] = (1.0 / KAPPA) * np.log(W[n])

# Discard the boundary-corrupted outermost inventory level on each side,
# back down to the originally-requested Q_MAX range (see note above A[0,1]).
q_grid = q_grid[1:-1]
W      = W[:, 1:-1]
H      = H[:, 1:-1]


# ------------------------------------------------------------------
# Optimal controls
# ------------------------------------------------------------------
# delta+*(t,q) = 1/kappa + epsilon + (1/kappa)*log(w_q / w_{q-1})
# delta-*(t,q) = 1/kappa + epsilon + (1/kappa)*log(w_q / w_{q+1})
# Defined for interior inventory levels only

# Shape: (N_T, 2*Q_MAX) -- ask depth for q in {-Q_MAX+1, ..., Q_MAX}
delta_ask = (1.0/KAPPA) + EPSILON + (1.0/KAPPA) * np.log(W[:, 1:] / W[:, :-1])

# Shape: (N_T, 2*Q_MAX) -- bid depth for q in {-Q_MAX, ..., Q_MAX-1}
delta_bid = (1.0/KAPPA) + EPSILON + (1.0/KAPPA) * np.log(W[:, :-1] / W[:, 1:])

q_ask_grid = q_grid[1:]    # q values for ask quotes
q_bid_grid = q_grid[:-1]   # q values for bid quotes


# ------------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------------
print("Sanity checks:")

# 1. Terminal condition: h_q(T) = -alpha * q^2
h_T_computed = H[-1]
h_T_expected = -ALPHA * q_grid**2
max_err = np.max(np.abs(h_T_computed - h_T_expected))
print(f"  Terminal condition max error: {max_err:.2e}  (should be ~0)")

# 2. Symmetry: delta+*(t, q) = delta-*(t, -q) in symmetric case
# Ask at q=1 should equal bid at q=-1, etc.
t_mid_idx = N_T // 2
ask_at_q1  = delta_ask[t_mid_idx, Q_MAX]      # q=1
bid_at_qm1 = delta_bid[t_mid_idx, Q_MAX - 1]  # q=-1
print(f"  Symmetry check at t=T/2: delta+(q=1)={ask_at_q1:.4f}, delta-(q=-1)={bid_at_qm1:.4f}  (should match)")

# 3. Controls should be positive
min_ask = delta_ask.min()
min_bid = delta_bid.min()
print(f"  Min ask depth: {min_ask:.4f}  (should be >= 0)")
print(f"  Min bid depth: {min_bid:.4f}  (should be >= 0)")
print()


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: h_q(t) across time for selected inventory levels
ax = axes[0]
for q_val in [-3, -1, 0, 1, 3]:
    idx = q_val + Q_MAX
    ax.plot(t_grid, H[:, idx], label=f'q={q_val}')
ax.set_xlabel('Time t')
ax.set_ylabel('h_q(t)')
ax.set_title('Value function h_q(t)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Optimal ask and bid depths at t=0 across inventory
ax = axes[1]
ax.plot(q_ask_grid, delta_ask[0, :], 'b-o', markersize=4, label='Ask depth delta+*')
ax.plot(q_bid_grid, delta_bid[0, :], 'r-o', markersize=4, label='Bid depth delta-*')
ax.axhline(1.0/KAPPA, color='grey', linestyle='--', alpha=0.7, label='1/kappa (base spread)')
ax.set_xlabel('Inventory q')
ax.set_ylabel('Quote depth')
ax.set_title('Optimal controls at t=0')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Evolution of ask depth at q=0 across time
ax = axes[2]
# ask at q=0 corresponds to index Q_MAX in delta_ask (q_ask_grid[Q_MAX-1] = 0)
ask_q0_idx = np.where(q_ask_grid == 0)[0]
bid_q0_idx = np.where(q_bid_grid == 0)[0]
if len(ask_q0_idx) > 0:
    ax.plot(t_grid, delta_ask[:, ask_q0_idx[0]], 'b-', label='Ask depth at q=0')
if len(bid_q0_idx) > 0:
    ax.plot(t_grid, delta_bid[:, bid_q0_idx[0]], 'r-', label='Bid depth at q=0')
ax.set_xlabel('Time t')
ax.set_ylabel('Quote depth')
ax.set_title('Control evolution at q=0')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
import os as _os; from datetime import datetime as _dt; plt.savefig(_os.path.join(r'C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images', f'optimal_control_{_dt.now().strftime("%H%M%S")}.png'), dpi=150)
print("Plot saved to optimal_control.png")


# ------------------------------------------------------------------
# Print control table at t=0
# ------------------------------------------------------------------
print("\nOptimal controls at t=0:")
print(f"{'q':>6} {'delta+*':>12} {'delta-*':>12}")
print("-" * 32)
for i, q_val in enumerate(q_ask_grid):
    q_int = int(q_val)
    ask = delta_ask[0, i]
    bid = delta_bid[0, i]
    print(f"{q_int:>6} {ask:>12.4f} {bid:>12.4f}")
