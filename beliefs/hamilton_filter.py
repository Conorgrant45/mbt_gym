"""
hamilton_filter.py
------------------
General Hamilton (1989) filter for a two-state hidden Markov model
with r lagged states in the joint posterior, using a per-step return
emission (Gaussian for regime 0, jump-diffusion mixture for regime 1).

The jump magnitude on each market-order arrival is itself random --
Exponential(mean=eps), not a fixed constant -- matching the midprice
model (envs/arrival_jump_midprice.py). "No jump" (neither side arrives,
prob (1-p)^2) stays Gaussian; each one-sided arrival (prob p(1-p))
contributes an Exponentially-Modified-Gaussian (EMG) branch, since the
net return is Normal(0,sigma_1) + Exponential(eps). The "both sides
arrive" case (prob p^2) is not degenerate here (unlike the fixed-jump
model, where +eps-eps cancels exactly): the difference of two iid
Exponential(eps) draws is Laplace(0,eps), which is itself an equal
mixture of +Exponential(eps) and -Exponential(eps) -- so it splits
50/50 into the two EMG branches:

    f(r_t | Z_t=0) = N(0, sigma_0^2)                          [Gaussian]

    f(r_t | Z_t=1) = jump-diffusion mixture                    [Mixture]
        = (1-p)^2        * N(0, sigma_1^2)
          + p(1 - p/2)   * EMG(r_t;  sigma_1, eps)
          + p(1 - p/2)   * EMG(-r_t; sigma_1, eps)

where EMG(x; sigma, eps) is the density of Normal(0,sigma) + Exponential(eps).

An optional rolling realised-variance (RV) emission augmentation is
still implemented (rv_window > 0, rv_weight > 0) but is DISABLED BY
DEFAULT and not recommended: it multiplies in f(RV_t | Z_t) alongside
f(r_t | Z_t) at every step, where RV_t is a mean of the last rv_window
squared returns. Because each individual return then contributes to
its own step's f(r_t|Z_t) AND to the next (rv_window - 1) steps' RV
emissions, the same evidence gets re-scored as if independent up to
rv_window times. Verified empirically: a single isolated jump embedded
in otherwise-zero returns inflates belief persistence from 3 steps
(return-only) to ~(rv_window + 2) steps, and in a sustained-regime-1
test collapses all natural belief fluctuation (min belief 1.0 across
the whole stretch, vs a legitimate dip to 0.24 without RV) -- see
filter_and_control_fixes.tex. Kept in the code only for reference/
ablation studies (e.g. plot_belief_evolution.py); do not enable it in
anything feeding a trading decision or a reported accuracy figure.

References
----------
Hamilton, J.D. (1989). Econometrica, 57(2), 357-384.
"""

import numpy as np
from itertools import product
from scipy.stats import norm, gamma as gamma_dist, exponnorm


class HamiltonFilter:
    """
    Two-state Hamilton filter with r lagged states and optional
    rolling realised variance emission augmentation.

    Parameters
    ----------
    transition_matrix : array-like, shape (2, 2)
        Row-stochastic Markov transition matrix.
    regime_volatilities : array-like, shape (2,)
        Per-step midprice return volatility for each regime.
    r : int
        Number of lagged regimes. r=0 recovers the standard filter.
    jump_size : float, optional
        Jump magnitude epsilon. Enables jump-mixture emission for regime 1.
    jump_intensity : float, optional
        Jump arrival rate lambda per unit time.
    step_size : float
        Time step dt.
    rv_window : int
        Rolling window length w for realised variance. Disabled by
        default (0) -- see module docstring for why. Only enable for
        deliberate ablation/comparison against the return-only filter.
    rv_weight : float
        Weight applied to the log RV likelihood before combining with
        the return likelihood, if rv_window > 0. Disabled by default (0.0).
    initial_belief : float, optional
        Initial P(regime=1). Defaults to stationary distribution.
    eps : float
        Small constant for numerical stability.
    """

    def __init__(
        self,
        transition_matrix,
        regime_volatilities,
        r: int = 1,
        jump_size: float = None,
        jump_intensity: float = None,
        step_size: float = 0.005,
        rv_window: int = 0,
        rv_weight: float = 0.0,
        initial_belief: float = None,
        eps: float = 1e-10,
    ):
        self.P            = np.array(transition_matrix, dtype=np.float64)
        self.sigma        = np.array(regime_volatilities, dtype=np.float64)
        self.r            = r
        self.eps          = eps
        self.n            = 2
        self.jump_size    = jump_size
        self.step_size    = step_size
        self.jump_prob    = float(jump_intensity * step_size) if jump_intensity is not None else None
        self._use_mixture = (jump_size is not None and jump_intensity is not None)
        self.rv_window    = rv_window
        self.rv_weight    = rv_weight

        # Expected per-step return variance under each regime.
        # v_0 = sigma_0^2
        # v_1 = sigma_1^2 + 2*p*(2-p)*epsilon^2   (random-jump-size contribution)
        #
        # Derivation: the net jump this step is J = A*1_ask - B*1_bid, with
        # A,B iid Exponential(eps) independent of the (independent) arrival
        # indicators. E[J]=0, and Var(1_side*X) = p*E[X^2] - p^2*E[X]^2
        # = p*2*eps^2 - p^2*eps^2 = eps^2*p*(2-p) per side (using
        # E[X^2]=2*eps^2 for Exponential(mean=eps)); summing both
        # (independent) sides gives Var(J) = 2*eps^2*p*(2-p). This replaces
        # the deterministic-jump formula 2*p*(1-p)*eps^2, which assumed a
        # fixed jump size (no within-jump variance).
        self._rv_var = np.zeros(2)
        self._rv_var[0] = self.sigma[0]**2
        if self._use_mixture:
            p = self.jump_prob
            self._rv_var[1] = (self.sigma[1]**2
                               + 2.0 * p * (2.0 - p) * self.jump_size**2)
        else:
            self._rv_var[1] = self.sigma[1]**2

        # Precompute regime-1 return-emission mixture weights and the EMG
        # shape parameter (see module docstring for the derivation).
        if self._use_mixture:
            p = self.jump_prob
            self._w_no = (1.0 - p)**2
            self._w_pm = p * (1.0 - p / 2.0)
            self._emg_K = self.jump_size / self.sigma[1]

        # Joint posterior paths
        self.paths       = list(product(range(self.n), repeat=r + 1))
        self.n_paths     = len(self.paths)
        self.path_to_idx = {path: i for i, path in enumerate(self.paths)}

        self._pi_stat = self._stationary_distribution()

        self._initial_belief = (float(initial_belief) if initial_belief is not None
                                else float(self._pi_stat[1]))

        self.xi              = np.zeros(self.n_paths)
        self._prev_midprice  = None
        self._return_buffer  = []   # stores recent percentage returns for RV

        self.reset()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, initial_belief: float = None):
        """Reset filter at episode start."""
        b        = initial_belief if initial_belief is not None else self._initial_belief
        marginal = np.array([1.0 - b, b])

        for i, path in enumerate(self.paths):
            prob = 1.0
            for z in path:
                prob *= marginal[z]
            self.xi[i] = prob

        self.xi             /= self.xi.sum()
        self._prev_midprice  = None
        self._return_buffer  = []

    def update(self, midprice: float) -> float:
        """
        Ingest current midprice and return updated belief P(regime=1 | F_t).

        Uses both the current percentage return and the rolling realised
        variance as emission signals, assuming conditional independence
        given the regime.

        Parameters
        ----------
        midprice : float
            Current midprice from the environment observation.

        Returns
        -------
        belief : float
            P(Z_t=1 | observations up to t).
        """
        if self._prev_midprice is None:
            self._prev_midprice = midprice
            return self._marginal_belief()

        # Percentage return
        ret = (midprice - self._prev_midprice) / (abs(self._prev_midprice) + self.eps)
        self._prev_midprice = midprice

        # Update return buffer for RV
        self._return_buffer.append(ret)
        if len(self._return_buffer) > self.rv_window:
            self._return_buffer.pop(0)

        # Compute rolling realised variance (mean squared return)
        rv = float(np.mean(np.array(self._return_buffer)**2)) if self._return_buffer else None

        # Predict and update
        joint = np.zeros(self.n_paths)

        for i, path in enumerate(self.paths):
            z_t   = path[0]
            z_tm1 = path[1] if self.r >= 1 else None

            # Step 1: Prediction
            if self.r == 0:
                predicted = sum(
                    self.P[z_prev, z_t] * self.xi[self.path_to_idx[(z_prev,)]]
                    for z_prev in range(self.n)
                )
            else:
                predicted = 0.0
                for z_oldest in range(self.n):
                    prev_path = path[1:] + (z_oldest,)
                    if prev_path in self.path_to_idx:
                        predicted += self.P[z_tm1, z_t] * self.xi[self.path_to_idx[prev_path]]

            # Step 2: Joint emission (conditional independence)
            l_ret = self._emission_return(ret, z_t)
            l_rv  = self._emission_rv(rv, z_t) if (rv is not None and self.rv_window > 0) else 1.0

            # Combine in log space, then exponentiate
            # rv_weight controls relative influence of the two signals
            log_likelihood = np.log(l_ret + self.eps) + self.rv_weight * np.log(l_rv + self.eps)
            joint[i] = np.exp(log_likelihood) * predicted

        # Step 3: Normalise
        normaliser = joint.sum()
        if normaliser < self.eps:
            return self._marginal_belief()

        # Step 4: Bayes update
        self.xi = joint / normaliser

        return self._marginal_belief()

    def update_from_obs(self, obs: np.ndarray) -> float:
        """
        Extract midprice from mbt_gym observation (index 3) and call update().

        WARNING: mbt_gym's TradingEnvironment normalises observations to
        roughly [-1, 1] by default (normalise_observation_space=True), in
        which case index 3 is NOT a usable midprice. Only call this with
        an observation taken from an env constructed with
        normalise_observation_space=False. Otherwise, use the raw price
        directly (e.g. RegimeSwitchingEnv.raw_midprice / info['raw_midprice']).
        """
        obs_flat = np.array(obs).flatten()
        midprice = float(obs_flat[3])
        return self.update(midprice)

    # ------------------------------------------------------------------
    # Emission models
    # ------------------------------------------------------------------

    def _emission_return(self, ret: float, regime: int) -> float:
        """
        Single-step return emission f(r_t | Z_t = regime).

        Regime 0: Gaussian N(0, sigma_0).
        Regime 1: Jump-diffusion mixture (if jump parameters provided),
                  otherwise Gaussian N(0, sigma_1). The jump branches use
                  an Exponentially-Modified-Gaussian density, since the
                  jump magnitude is Exponential(mean=eps), not fixed --
                  see module docstring for the derivation.
        """
        sigma = self.sigma[regime]

        if regime == 0 or not self._use_mixture:
            return float(norm.pdf(ret, loc=0.0, scale=sigma) + self.eps)

        f_up   = exponnorm.pdf(ret,  self._emg_K, loc=0.0, scale=sigma)
        f_down = exponnorm.pdf(-ret, self._emg_K, loc=0.0, scale=sigma)

        likelihood = (
            self._w_no * norm.pdf(ret, loc=0.0, scale=sigma)
            + self._w_pm * f_up
            + self._w_pm * f_down
        )
        return float(likelihood + self.eps)

    def _emission_rv(self, rv: float, regime: int) -> float:
        """
        Rolling realised variance emission f(RV_t | Z_t = regime).

        Approximated as Gamma(w/2, 2*v_k/w) where v_k is the expected
        per-step return variance under regime k and w is the window length.

        This is derived from the chi-squared distribution of w independent
        squared Gaussian returns, rescaled to account for the jump
        contribution in regime 1.

        Parameters
        ----------
        rv : float
            Rolling mean squared return over the last w steps.
        regime : int
            Current regime hypothesis (0 or 1).

        Returns
        -------
        likelihood : float
            Gamma density evaluated at rv.
        """
        if rv is None or rv <= 0:
            return 1.0

        w  = len(self._return_buffer)   # actual window (may be < rv_window early in episode)
        vk = self._rv_var[regime]

        if vk < self.eps:
            return 1.0

        # Gamma(alpha, scale) where alpha = w/2, scale = 2*vk/w
        # This models the sum of w squared returns; we observe the mean, so
        # the mean of the Gamma is vk and variance is 2*vk^2/w.
        alpha = w / 2.0
        scale = 2.0 * vk / w

        return float(gamma_dist.pdf(rv, a=alpha, scale=scale) + self.eps)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _marginal_belief(self) -> float:
        """P(Z_t=1 | F_t) by summing xi over all paths with z_t=1."""
        belief = sum(
            self.xi[self.path_to_idx[path]]
            for path in self.paths
            if path[0] == 1
        )
        return float(np.clip(belief, 0.0, 1.0))

    def _stationary_distribution(self) -> np.ndarray:
        """Stationary distribution of the transition matrix."""
        A     = (self.P.T - np.eye(self.n))
        A[-1] = 1.0
        b     = np.zeros(self.n)
        b[-1] = 1.0
        return np.linalg.solve(A, b)

    @property
    def belief(self) -> float:
        """Current P(Z_t=1 | F_t)."""
        return self._marginal_belief()
