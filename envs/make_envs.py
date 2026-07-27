"""
make_envs.py
------------
Constructs the two mbt_gym TradingEnvironment instances and returns a
RegimeSwitchingEnv ready for training.

Parameters are aligned with the optimal control derivation in Section 3.1.3
of the dissertation:

    kappa  = 2.0    (fill decay, ExponentialFillFunction fill_exponent)
    lambda = 0.75   (arrival intensity, PoissonArrivalModel)
    epsilon = 0.3   (adverse selection jump size, BrownianMotionJumpMidpriceModel)

Regime 0: calm, no adverse selection
    - BrownianMotionMidpriceModel (no jumps)
    - PoissonArrivalModel
    - ExponentialFillFunction

Regime 1: adverse selection
    - BrownianMotionJumpMidpriceModel (jump_size = epsilon)
    - PoissonArrivalModel
    - ExponentialFillFunction
"""

import numpy as np
from scipy.linalg import logm, expm

from mbt_gym.gym.TradingEnvironment import TradingEnvironment
from mbt_gym.gym.ModelDynamics import LimitOrderModelDynamics
from mbt_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from envs.arrival_jump_midprice import ArrivalJumpMidpriceModel
from mbt_gym.stochastic_processes.arrival_models import PoissonArrivalModel
from mbt_gym.stochastic_processes.fill_probability_models import ExponentialFillFunction
from mbt_gym.rewards.RewardFunctions import RunningInventoryPenalty

from envs.regime_env import RegimeSwitchingEnv

# ------------------------------------------------------------------
# Shared episode structure
#
# n_steps=4000 (dt=0.00025) refines the discretisation so that
# arrival_rate * dt = 140 * 0.00025 = 0.035 is small relative to 1,
# consistent with the continuous-time Poisson/HJB assumption that
# P(exactly one arrival in dt) = O(dt) and P(two-or-more arrivals in dt)
# = o(dt). At the previous dt=0.005, arrival_rate*dt=0.7 was O(1), not
# small -- P(both sides arrive in the same step) = 0.7^2 = 0.49 was the
# same order of magnitude as P(one arrival) = 0.42, badly violating that
# assumption (diagnosed empirically: see arrival-discretisation session
# notes). arrival_rate itself is unchanged; only dt shrinks.
# ------------------------------------------------------------------
TERMINAL_TIME    = 1.0
N_STEPS          = 4000
NUM_TRAJECTORIES = 1
STEP_SIZE        = TERMINAL_TIME / N_STEPS

# ------------------------------------------------------------------
# Model parameters (aligned with optimal control derivation)
# ------------------------------------------------------------------
KAPPA         = 1.5
R0_VOLATILITY = 0.01     # CJR calibrated
# Equal to R0_VOLATILITY (was 0.03, a 3x ratio) -- recalibrated so the two
# regimes are diffusion-identical and EPSILON is the ONLY signal
# distinguishing them. diagnostic_epsilon_sweep.py showed the Hamilton
# filter can learn the regime from the jump signal alone (accuracy
# 0.89-0.95 across a wide epsilon range with equal volatilities), and a
# single varying parameter (the jump) is a more defensible identification
# strategy than conflating it with an independent volatility difference --
# see filter_and_control_fixes.tex Section on the epsilon sweep diagnostics.
R1_VOLATILITY = 0.01
EPSILON       = 0.5      # price units -- permanent impact per market order arrival
LAMBDA        = 140   # active trading

INITIAL_PRICE = 100.0    # matches the midprice models' default initial_price

# Percentage-return-scale versions of the above. The midprice models
# (BrownianMotionMidpriceModel / ArrivalJumpMidpriceModel) treat
# volatility/jump_size as ABSOLUTE price units, but HamiltonFilter
# operates on PERCENTAGE returns (mid_t - mid_{t-1}) / mid_{t-1}.
# Always use these _PCT versions -- not the raw constants above --
# when constructing a HamiltonFilter, or the emission likelihoods
# will be off by a factor of ~INITIAL_PRICE.
R0_VOLATILITY_PCT = R0_VOLATILITY / INITIAL_PRICE
R1_VOLATILITY_PCT = R1_VOLATILITY / INITIAL_PRICE
EPSILON_PCT       = EPSILON / INITIAL_PRICE

# HamiltonFilter's regime_volatilities expects the PER-STEP return std
# (see its docstring), not a volatility coefficient -- R0_VOLATILITY_PCT/
# R1_VOLATILITY_PCT still need the sqrt(dt) diffusion scaling applied
# below. Passing the _PCT constants directly (missing this factor) makes
# the filter's assumed sigma ~1/sqrt(STEP_SIZE) too large for both regimes.
R0_SIGMA_STEP = R0_VOLATILITY_PCT * np.sqrt(STEP_SIZE)
R1_SIGMA_STEP = R1_VOLATILITY_PCT * np.sqrt(STEP_SIZE)
# ------------------------------------------------------------------
# Reward parameters
# ------------------------------------------------------------------
PER_STEP_INVENTORY_AVERSION  = 0.01
TERMINAL_INVENTORY_AVERSION  = 0.001

# ------------------------------------------------------------------
# Markov transition matrix
#
# TRANSITION_MATRIX_DT_0_005 is the original calibration, defined at
# dt_old=0.005. Simply reusing it at the new dt=0.00025 would make regime
# switching 20x faster in continuous-time terms (P[i,j] is a per-step
# probability, and there are 20x more steps per unit time now). Instead,
# recover the underlying continuous-time generator Q implied by the old
# matrix/dt, then re-discretise at the new dt -- this preserves expected
# regime durations (and the stationary distribution) in continuous time
# exactly, up to the residual discrete-vs-continuous-time discretisation
# bias, which shrinks (not grows) at the finer dt.
# ------------------------------------------------------------------
_DT_OLD = 0.005
TRANSITION_MATRIX_DT_0_005 = np.array([
    [0.95, 0.05],
    [0.02, 0.98],
])

_Q = logm(TRANSITION_MATRIX_DT_0_005) / _DT_OLD
if np.iscomplexobj(_Q):
    assert np.max(np.abs(_Q.imag)) < 1e-10, "logm(P_old) has non-negligible imaginary part."
    _Q = _Q.real

# Continuous-time generator matrix (rows sum to zero), exposed publicly so any
# continuous-time/event-driven consumer (e.g. envs/event_driven_regime_env.py)
# uses this EXACT same generator rather than re-deriving it via its own
# logm(TRANSITION_MATRIX_DT_0_005)/_DT_OLD call -- single source of truth,
# same "aliases not redefinitions" convention as PHI/ALPHA/DT elsewhere in
# this project. Does not change any existing behaviour: _Q itself is
# unchanged, this only binds it to a second, public name.
TRANSITION_GENERATOR = _Q.copy()

_P_new = expm(_Q * STEP_SIZE)
if np.iscomplexobj(_P_new):
    assert np.max(np.abs(_P_new.imag)) < 1e-10, "expm(Q*dt_new) has non-negligible imaginary part."
    _P_new = _P_new.real

# Renormalise rows to sum to exactly 1 (guards against residual
# logm/expm floating-point error) and validate before use.
_P_new = _P_new / _P_new.sum(axis=1, keepdims=True)
assert np.all(_P_new >= 0.0) and np.all(_P_new <= 1.0), "P_new has invalid (out-of-range) probabilities."
assert np.allclose(_P_new.sum(axis=1), 1.0), "P_new rows do not sum to 1."

TRANSITION_MATRIX = _P_new.tolist()


def make_regime_envs(
    num_trajectories: int = NUM_TRAJECTORIES,
    switch_within_episode: bool = False,
    seed: int = None,
    epsilon: float = EPSILON,
    r0_volatility: float = R0_VOLATILITY,
    r1_volatility: float = R1_VOLATILITY,
):
    """
    Build the RegimeSwitchingEnv from two TradingEnvironment instances.

    Parameters
    ----------
    num_trajectories : int
        Number of parallel trajectories for vectorised rollouts.
    switch_within_episode : bool
        If True the regime can switch at every step.
        If False (default) the regime is fixed per episode.
    epsilon : float
        Regime-1 adverse-selection jump size, in absolute price units
        (same convention as the module-level EPSILON constant, which is
        the default). Overriding this does NOT change EPSILON_PCT/
        R1_SIGMA_STEP -- callers sweeping epsilon must recompute
        epsilon / INITIAL_PRICE themselves when building a matching
        HamiltonFilter (see diagnostic_epsilon_sweep.py).
    r0_volatility, r1_volatility : float
        Diffusion volatility for each regime, in absolute price units
        (same convention as the module-level R0_VOLATILITY/R1_VOLATILITY
        constants, which are the defaults). Overriding these does NOT
        change R0_SIGMA_STEP/R1_SIGMA_STEP -- callers must recompute the
        matching per-step sigma themselves when building a HamiltonFilter
        (see diagnostic_epsilon_sweep.py).
    seed : int, optional
        If provided, seeds every underlying stochastic process (midprice,
        arrival, fill probability, AND regime-transition draws -- for both
        regimes) so that the same seed reproduces an identical episode
        (same price path, arrivals, fills, regime path), fully
        independently of any other environment instance or evaluation call
        happening elsewhere in the same process.

        RNG-ISOLATION FIX: RegimeSwitchingEnv's regime-transition draws
        used to read from the legacy global np.random API, making them
        (unlike the midprice/arrival/fill models, already isolated
        Generators below) vulnerable to being silently reseeded/consumed
        by anything else in the process that called np.random.seed(...) --
        in particular, PeriodicEvalCallback's evaluation episodes, which
        caused training runs with different eval_freq values to diverge
        despite identical seeds (see envs/regime_env.py's class docstring
        and tests/test_rng_isolation.py for the full audit). Fixed by
        spawning a 7th independent child seed here (below) and passing it
        to RegimeSwitchingEnv as an explicit, already-isolated
        np.random.Generator -- calling np.random.seed(seed) elsewhere in
        the process can no longer affect ANY part of an env built here.
        Any script that reuses one "seed" across multiple
        make_regime_envs() calls and expects the same underlying episode
        (e.g. comparing filter configurations on "the same" episode) needs
        to pass seed= here, or every call gets an unrelated random episode.

    Returns
    -------
    env : RegimeSwitchingEnv
    """
    if seed is not None:
        child_seeds = np.random.SeedSequence(seed).spawn(7)
    else:
        child_seeds = [None] * 7
    # child_seeds[6]: regime-transition draws (RegimeSwitchingEnv's own
    # isolated Generator -- see its class docstring). Spawning 7 children
    # here instead of 6 does NOT change child_seeds[0:6] (verified:
    # SeedSequence.spawn(n)'s child i depends only on the parent and i,
    # not on n) -- every existing midprice/arrival/fill RNG realisation
    # for a given seed is unchanged by this fix.
    regime_rng = np.random.default_rng(child_seeds[6])

    # --- Regime 0: calm, no adverse selection ---
    env_r0 = TradingEnvironment(
        terminal_time=TERMINAL_TIME,
        n_steps=N_STEPS,
        reward_function=RunningInventoryPenalty(
            per_step_inventory_aversion=PER_STEP_INVENTORY_AVERSION,
            terminal_inventory_aversion=TERMINAL_INVENTORY_AVERSION,
        ),
        model_dynamics=LimitOrderModelDynamics(
            midprice_model=BrownianMotionMidpriceModel(
                volatility=r0_volatility,
                step_size=STEP_SIZE,
                num_trajectories=num_trajectories,
                seed=child_seeds[0],
            ),
            arrival_model=PoissonArrivalModel(
                intensity=np.array([[LAMBDA, LAMBDA]]),
                step_size=STEP_SIZE,
                num_trajectories=num_trajectories,
                seed=child_seeds[1],
            ),
            fill_probability_model=ExponentialFillFunction(
                fill_exponent=KAPPA,
                step_size=STEP_SIZE,
                num_trajectories=num_trajectories,
                seed=child_seeds[2],
            ),
            num_trajectories=num_trajectories,
        ),
        num_trajectories=num_trajectories,
    )

    # --- Regime 1: adverse selection, price jumps on fills ---
    env_r1 = TradingEnvironment(
        terminal_time=TERMINAL_TIME,
        n_steps=N_STEPS,
        reward_function=RunningInventoryPenalty(
            per_step_inventory_aversion=PER_STEP_INVENTORY_AVERSION,
            terminal_inventory_aversion=TERMINAL_INVENTORY_AVERSION,
        ),
        model_dynamics=LimitOrderModelDynamics(
            midprice_model=ArrivalJumpMidpriceModel(
                volatility=r1_volatility,
                jump_size=epsilon,
                step_size=STEP_SIZE,
                terminal_time=TERMINAL_TIME,
                num_trajectories=num_trajectories,
                seed=child_seeds[3],
            ),
            arrival_model=PoissonArrivalModel(
                intensity=np.array([[LAMBDA, LAMBDA]]),
                step_size=STEP_SIZE,
                num_trajectories=num_trajectories,
                seed=child_seeds[4],
            ),
            fill_probability_model=ExponentialFillFunction(
                fill_exponent=KAPPA,
                step_size=STEP_SIZE,
                num_trajectories=num_trajectories,
                seed=child_seeds[5],
            ),
            num_trajectories=num_trajectories,
        ),
        num_trajectories=num_trajectories,
    )

    return RegimeSwitchingEnv(
        env_r0,
        env_r1,
        transition_matrix=TRANSITION_MATRIX,
        switch_within_episode=switch_within_episode,
        regime_rng=regime_rng,
    )
