"""
event_time_hamilton_filter.py
--------------------------------
Event-time counterpart of beliefs/hamilton_filter.py's HamiltonFilter,
estimating

    b_n = P(Z_{t_n} = 1 | observable information through event n)

where t_n is the n-th OBSERVABLE event time (a buy or sell market-order
arrival, or the terminal horizon) and Delta-tau_n = t_n - t_{n-1} is the
(variable) elapsed time since the previous observable event. Does not modify
beliefs/hamilton_filter.py -- this is a new, parallel class for the
event-driven environment, exactly as envs/event_driven_regime_env.py is a
new, parallel environment.

======================================================================
Is the waiting time itself regime-informative?
======================================================================
NO, for the production calibration. envs/make_envs.py's make_regime_envs()
constructs BOTH regime sub-environments' PoissonArrivalModel with the
IDENTICAL intensity array `np.array([[LAMBDA, LAMBDA]])` -- i.e. total
arrival intensity (lambda_bid + lambda_ask = 280) does not depend on the
hidden regime at all; only the MIDPRICE model differs between regimes
(BrownianMotionMidpriceModel vs ArrivalJumpMidpriceModel), never the arrival
process. Since the distribution of Delta-tau_n is therefore identical
whether the hidden regime was 0 or 1 throughout the interval, OBSERVING
Delta-tau_n carries zero Bayesian evidence about the regime path -- the CTMC
prediction step is exactly

    b_pred_n = b_{n-1} @ expm(Q * Delta-tau_n)                          (*)

with no additional waiting-time likelihood factor. This is verified, not
assumed: __init__ raises NotImplementedError if constructed with regime-
dependent arrival intensities, since (*) would then be the WRONG prediction
equation (a killed-generator/event-time filtering equation would be needed
instead -- not implemented in this phase; see class docstring).

(*) is itself exact regardless of how many hidden regime transitions
occurred inside Delta-tau_n -- solving the Kolmogorov forward equation for a
CTMC over an interval of length Delta-tau_n integrates over every possible
number and timing of hidden transitions during that interval, by
construction of the matrix exponential.

======================================================================
Emission model
======================================================================
At an OBSERVABLE ARRIVAL (as opposed to the fixed-step filter's per-tick
update, where an arrival is a RARE tick-level event requiring a "no arrival"
branch), an arrival is GUARANTEED -- that is what triggered this observable
event. So there is no "neither side arrived" branch here, unlike
HamiltonFilter._emission_return's w_no term. Instead:

    f(r_n | Z_{t_n}=0, Delta-tau_n)
        = N(r_n; 0, sigma_0^2 * Delta-tau_n)                    [calm]

    f(r_n | Z_{t_n}=1, Delta-tau_n)
        = P(buy|arrival)  * EMG( r_n; sigma_1*sqrt(Delta-tau_n), epsilon)
        + P(sell|arrival) * EMG(-r_n; sigma_1*sqrt(Delta-tau_n), epsilon)

where P(buy|arrival) = lambda_ask / (lambda_ask + lambda_bid) and
P(sell|arrival) = lambda_bid / (lambda_ask + lambda_bid) -- both exactly 0.5
in production (lambda_bid == lambda_ask == LAMBDA).

Preserving the existing (return-only) information set: the environment's
info dict DOES reveal which side actually arrived (info['arrival_side']),
but this is NEVER passed to the filter's likelihood -- the mixture above
MARGINALISES over the arrival side using its a-priori relative intensity,
exactly mirroring how HamiltonFilter._emission_return marginalises over
"no/buy/sell arrival this tick" using p=jump_intensity*step_size, rather
than conditioning on the environment's own internal arrival draws. Feeding
info['arrival_side'] into the likelihood would silently add information the
existing (fixed-step) filter never had -- not done here.

At the TERMINAL event (horizon reached, event_type != 'arrival'), no jump
can ever occur (matching the environment's own "no jump at the horizon"
behaviour), so both hypotheses collapse to the pure calm Gaussian:

    f(r_n | Z_{t_n}=i, Delta-tau_n, terminal) = N(r_n; 0, sigma_i^2 * Delta-tau_n)

======================================================================
Equal-sigma restriction (explicit guard, not a silent approximation)
======================================================================
The diffusion variance Var(Delta S | Z=i, Delta-tau) = sigma_i^2 * Delta-tau
above implicitly assumes the hypothesised regime i held for the WHOLE
interval. If sigma_0 != sigma_1, and one or more hidden regime switches
occurred inside Delta-tau_n (which the filter, by construction, cannot
observe), the TRUE diffusion variance is a path integral over whichever
regimes were actually occupied and for how long -- a harder continuous-time
filtering problem (tractable in principle via an augmented
generator/Kronecker-sum ODE, but NOT implemented in this phase). Because
the current production calibration has R0_VOLATILITY == R1_VOLATILITY
(both 0.01, see envs/make_envs.py), sigma_i^2*Delta-tau is EXACT here
regardless of the hidden switch path (both regimes contribute the identical
per-unit-time variance) -- but __init__ raises NotImplementedError if ever
constructed with unequal regime volatilities, rather than silently reusing
this formula where it would only be approximate. Do NOT implement the
sigma=0 (jump-only) ablation by relaxing this guard -- that is explicitly
deferred to a separate ablation after this integration is validated.

References
----------
Hamilton, J.D. (1989). Econometrica, 57(2), 357-384.
"""

import numpy as np
from scipy.linalg import expm
from scipy.stats import norm, exponnorm
from scipy.special import logsumexp


def stationary_distribution_from_generator(Q: np.ndarray) -> np.ndarray:
    """Same formula as envs.event_driven_regime_env.stationary_distribution_from_generator
    -- duplicated here (not imported) only to keep beliefs/ free of a
    dependency on envs/; the two implementations are tested for numerical
    agreement in tests/test_event_driven_agent_integration.py."""
    n = Q.shape[0]
    A = np.vstack([Q.T, np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    return pi


class EventTimeHamiltonFilter:
    """
    Event-time Hamilton filter. One update() call per observable event
    (arrival or terminal) -- never called from an internal hidden regime
    switch (those are not observable and carry no separate update).

    Parameters
    ----------
    transition_generator : array-like, shape (2, 2)
        Continuous-time generator Q (rows sum to zero). Production value:
        envs.make_envs.TRANSITION_GENERATOR.
    regime_volatilities : array-like, shape (2,)
        Percentage-return diffusion volatility PER UNIT TIME (NOT per-step
        -- unlike HamiltonFilter's regime_volatilities, which bakes in
        sqrt(step_size) once at construction, this filter scales by
        sqrt(delta_tau) at every update() call since delta_tau varies).
        Production value: [R0_VOLATILITY_PCT, R1_VOLATILITY_PCT] from
        envs.make_envs -- NOT R0_SIGMA_STEP/R1_SIGMA_STEP (those are
        fixed-step-only, dt-baked-in quantities).
    jump_size : float
        Mean jump magnitude in PERCENTAGE-return units (epsilon_pct =
        EPSILON / INITIAL_PRICE from envs.make_envs), matching
        HamiltonFilter's own jump_size convention.
    lambda_bid, lambda_ask : float
        Sell-MO / buy-MO arrival intensity for regime 0. Used only to form
        the marginal arrival-side split P(buy|arrival)/P(sell|arrival) --
        NOT scaled by any step_size (event-time arrivals are continuous-time
        Poisson, not a discretised per-tick Bernoulli).
    lambda_bid_regime1, lambda_ask_regime1 : float, optional
        Regime-1 arrival intensities. Default: equal to the regime-0 values
        (the production case). If NOT equal, raises NotImplementedError --
        see module docstring's "is the waiting time itself regime-
        informative?" section.
    initial_belief : float, optional
        Prior P(regime=1) at reset. Defaults to the stationary distribution
        of transition_generator.
    eps : float
        Small constant for numerical stability (matches HamiltonFilter).
    """

    def __init__(
        self,
        transition_generator,
        regime_volatilities,
        jump_size: float,
        lambda_bid: float,
        lambda_ask: float,
        lambda_bid_regime1: float = None,
        lambda_ask_regime1: float = None,
        initial_belief: float = None,
        eps: float = 1e-10,
    ):
        self.Q = np.array(transition_generator, dtype=np.float64)
        if self.Q.shape != (2, 2):
            raise NotImplementedError("EventTimeHamiltonFilter only supports a 2-state chain in this phase.")
        if not np.allclose(self.Q.sum(axis=1), 0.0, atol=1e-8):
            raise ValueError("transition_generator rows must sum to zero.")

        self.sigma = np.array(regime_volatilities, dtype=np.float64)
        if not np.isclose(self.sigma[0], self.sigma[1]):
            raise NotImplementedError(
                "EventTimeHamiltonFilter requires regime_volatilities[0] == regime_volatilities[1] "
                "(production: R0_VOLATILITY == R1_VOLATILITY == 0.01) -- exact filtering for unequal "
                "regime-dependent diffusion over an interval spanning hidden regime switches is not "
                "implemented in this phase (see module docstring's 'Equal-sigma restriction' section). "
                "This is a deliberate guard, not a silent approximation."
            )
        self.jump_size = float(jump_size)

        lambda_bid_regime1 = lambda_bid if lambda_bid_regime1 is None else lambda_bid_regime1
        lambda_ask_regime1 = lambda_ask if lambda_ask_regime1 is None else lambda_ask_regime1
        if not (np.isclose(lambda_bid, lambda_bid_regime1) and np.isclose(lambda_ask, lambda_ask_regime1)):
            raise NotImplementedError(
                "EventTimeHamiltonFilter requires arrival intensities to be IDENTICAL across regimes "
                "(production: lambda_bid == lambda_ask == LAMBDA for both regimes) -- the simplified "
                "CTMC prediction b_pred = b_prev @ expm(Q * delta_tau) is only exact in that case. If "
                "arrival intensity differs by regime, the waiting time itself is informative and a "
                "killed-generator/event-time filtering equation is required instead -- not implemented "
                "in this phase (see module docstring)."
            )
        total_lambda = lambda_bid + lambda_ask
        self.p_buy = lambda_ask / total_lambda
        self.p_sell = lambda_bid / total_lambda

        self.eps = eps
        self._pi_stat = stationary_distribution_from_generator(self.Q)
        self._initial_belief = float(initial_belief) if initial_belief is not None else float(self._pi_stat[1])

        self.belief = self._initial_belief
        self._prev_price = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def reset(self, initial_price: float, initial_belief: float = None):
        """Reset filter at episode start. initial_price is cached as the
        reference price for the first update()'s return calculation -- the
        event-driven wrapper always has a well-defined initial_price (unlike
        the fixed-step filter's reset(), there is no "first call is a
        pass-through" special case needed here, since every subsequent
        update() call has a genuine, well-defined delta_tau and return from
        the very first observable event onward)."""
        self.belief = float(initial_belief) if initial_belief is not None else self._initial_belief
        self._prev_price = float(initial_price)

    def update(self, price: float, delta_tau: float, is_arrival: bool) -> float:
        """
        Ingest the price at the current observable event and the elapsed
        time since the previous one, and return the updated belief
        P(Z_{t_n}=1 | information through event n).

        Parameters
        ----------
        price : float
            Midprice at the current observable event (info['price_after']).
        delta_tau : float
            Elapsed time since the previous observable event
            (info['elapsed_time']). Must be > 0.
        is_arrival : bool
            True for an arrival event (jump possible in regime 1), False for
            the terminal event (no jump possible regardless of regime).

        Returns
        -------
        belief : float
            P(Z_{t_n}=1 | F_{t_n}).
        """
        if delta_tau <= 0:
            raise ValueError(f"delta_tau must be positive, got {delta_tau}")

        ret = (price - self._prev_price) / (abs(self._prev_price) + self.eps)
        self._prev_price = float(price)

        # Step 1: CTMC prediction over the elapsed interval -- exact
        # regardless of hidden switches within it (see module docstring).
        b_vec_prev = np.array([1.0 - self.belief, self.belief])
        P_pred = expm(self.Q * delta_tau)
        b_vec_pred = np.clip(b_vec_prev @ P_pred, 0.0, 1.0)
        b_vec_pred = b_vec_pred / b_vec_pred.sum()

        # Step 2: emission likelihoods, log domain.
        sigma_scaled = self.sigma[0] * np.sqrt(delta_tau)  # equal across regimes -- see guard above
        log_l0 = norm.logpdf(ret, loc=0.0, scale=sigma_scaled)

        if is_arrival:
            K = self.jump_size / sigma_scaled
            log_f_up = exponnorm.logpdf(ret, K, loc=0.0, scale=sigma_scaled)
            log_f_down = exponnorm.logpdf(-ret, K, loc=0.0, scale=sigma_scaled)
            log_l1 = logsumexp([np.log(self.p_buy) + log_f_up, np.log(self.p_sell) + log_f_down])
        else:
            log_l1 = norm.logpdf(ret, loc=0.0, scale=sigma_scaled)

        log_pred = np.log(b_vec_pred + self.eps)
        log_joint = log_pred + np.array([log_l0, log_l1])

        normaliser = logsumexp(log_joint)
        if not np.isfinite(normaliser):
            self.belief = float(np.clip(b_vec_pred[1], 0.0, 1.0))
            return self.belief

        posterior = np.exp(log_joint - normaliser)
        posterior = posterior / posterior.sum()
        self.belief = float(np.clip(posterior[1], 0.0, 1.0))
        return self.belief

    @property
    def stationary_belief(self) -> float:
        return float(self._pi_stat[1])
