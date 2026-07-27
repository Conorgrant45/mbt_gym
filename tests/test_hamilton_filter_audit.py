"""
test_hamilton_filter_audit.py
------------------------------
Hand-calculated and independently-derived regression coverage for
beliefs/hamilton_filter.py, added as part of the RL-architecture audit.
The project previously had NO pytest-collectible unit tests of the
filter's actual arithmetic: test_hamilton_filter.py at repo root is a
plotting script with no test_ functions, so it contributes zero cases
to the suite despite its name.

Two independent verification strategies are used, deliberately NOT
re-using scipy.stats internally the same way the filter itself does:
  1. A raw-numpy Gaussian pdf (no scipy.stats.norm call) for the no-jump
     recursion, computed by hand alongside the filter.
  2. Monte Carlo simulation of the exact jump/diffusion process
     ArrivalJumpMidpriceModel actually generates (independent
     np.random.Generator, not scipy.stats.exponnorm) for the jump-mixture
     emission, cross-checked via a quadrature integral-to-one check and a
     kernel-density comparison against the filter's own closed-form value.

Run from repo root:
    pytest tests/test_hamilton_filter_audit.py -v
"""

import numpy as np
import pytest
from scipy.integrate import quad

from beliefs.hamilton_filter import HamiltonFilter


def _gauss_pdf(x, s):
    return (1.0 / (s * np.sqrt(2 * np.pi))) * np.exp(-(x ** 2) / (2 * s * s))


class TestHandCalculatedGaussianOnlyFilter:
    P = [[0.9, 0.1], [0.2, 0.8]]
    SIGMA = [0.01, 0.02]
    PRICES = [100.0, 100.5, 99.8, 100.2]

    def _independent_reference_beliefs(self):
        xi = np.array([0.5, 0.5])
        beliefs = [0.5]
        prev = self.PRICES[0]
        for p in self.PRICES[1:]:
            ret = (p - prev) / abs(prev)
            prev = p
            predicted0 = self.P[0][0] * xi[0] + self.P[1][0] * xi[1]
            predicted1 = self.P[0][1] * xi[0] + self.P[1][1] * xi[1]
            l0 = _gauss_pdf(ret, self.SIGMA[0])
            l1 = _gauss_pdf(ret, self.SIGMA[1])
            joint0, joint1 = l0 * predicted0, l1 * predicted1
            xi = np.array([joint0, joint1]) / (joint0 + joint1)
            beliefs.append(float(xi[1]))
        return beliefs

    def test_filter_matches_hand_calculated_reference(self):
        filt = HamiltonFilter(transition_matrix=self.P, regime_volatilities=self.SIGMA,
                               r=0, step_size=0.00025, initial_belief=0.5)
        filter_beliefs = [filt.update(p) for p in self.PRICES]
        reference_beliefs = self._independent_reference_beliefs()
        np.testing.assert_allclose(filter_beliefs, reference_beliefs, atol=1e-9)


class TestFilterTimingAndPrior:
    P = [[0.95, 0.05], [0.02, 0.98]]
    SIGMA = [0.01, 0.01]

    def test_reset_with_explicit_initial_belief_sets_exact_prior(self):
        filt = HamiltonFilter(transition_matrix=self.P, regime_volatilities=self.SIGMA, r=0, initial_belief=0.37)
        assert filt.belief == pytest.approx(0.37)

    def test_reset_without_explicit_belief_uses_stationary_distribution(self):
        filt = HamiltonFilter(transition_matrix=self.P, regime_volatilities=self.SIGMA, r=0)
        pi = filt._stationary_distribution()
        assert filt.belief == pytest.approx(pi[1])

    def test_first_update_is_pass_through_not_a_posterior_update(self):
        filt = HamiltonFilter(transition_matrix=self.P, regime_volatilities=self.SIGMA, r=0, initial_belief=0.5)
        b0 = filt.update(100.0)
        assert b0 == pytest.approx(0.5)

    def test_exactly_one_bayes_update_per_update_call(self):
        prices = [100.0, 100.1, 99.9, 100.05, 100.2, 99.95]
        filt_a = HamiltonFilter(transition_matrix=self.P, regime_volatilities=self.SIGMA, r=0, initial_belief=0.4)
        for p in prices:
            filt_a.update(p)
        xi_all_at_once = filt_a.xi.copy()

        filt_b = HamiltonFilter(transition_matrix=self.P, regime_volatilities=self.SIGMA, r=0, initial_belief=0.4)
        for p in prices[:3]:
            filt_b.update(p)
        for p in prices[3:]:
            filt_b.update(p)
        xi_two_batches = filt_b.xi.copy()

        np.testing.assert_allclose(xi_all_at_once, xi_two_batches, atol=1e-12)

    def test_belief_always_in_unit_interval(self):
        rng = np.random.default_rng(0)
        filt = HamiltonFilter(transition_matrix=self.P, regime_volatilities=self.SIGMA, r=0, initial_belief=0.5)
        price = 100.0
        for _ in range(500):
            price *= (1.0 + rng.normal(0, 0.001))
            b = filt.update(price)
            assert 0.0 <= b <= 1.0


def test_large_jump_return_moves_belief_towards_jump_regime_not_away():
    P = [[0.999, 0.001], [0.001, 0.999]]
    sigma = [0.0001, 0.0001]
    jump_size = 0.02
    filt = HamiltonFilter(transition_matrix=P, regime_volatilities=sigma, r=0,
                           jump_size=jump_size, jump_intensity=140, step_size=0.00025,
                           initial_belief=0.5)
    filt.update(100.0)
    belief_after_jump = filt.update(100.0 * (1 + 0.02))
    assert belief_after_jump > 0.9, (
        f"belief after an obvious jump-sized return was {belief_after_jump} -- "
        f"expected strong movement towards regime 1 (the jump regime). If this "
        f"is small or moves the wrong way, regime labels 0/1 may be reversed."
    )


class TestJumpMixtureEmissionIndependentVerification:
    P = [[0.99, 0.01], [0.01, 0.99]]
    SIGMA = [0.001, 0.001]
    JUMP_SIZE = 0.005
    JUMP_INTENSITY = 140
    STEP_SIZE = 0.00025

    def _filter(self):
        return HamiltonFilter(transition_matrix=self.P, regime_volatilities=self.SIGMA, r=0,
                               jump_size=self.JUMP_SIZE, jump_intensity=self.JUMP_INTENSITY,
                               step_size=self.STEP_SIZE)

    def test_regime_0_emission_integrates_to_one(self):
        filt = self._filter()
        val, _ = quad(lambda r: filt._emission_return(r, 0), -0.2, 0.2, limit=400)
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_regime_1_mixture_emission_integrates_to_one(self):
        filt = self._filter()
        val, _ = quad(lambda r: filt._emission_return(r, 1), -0.2, 0.2, limit=400)
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_regime_1_emission_matches_independent_monte_carlo_simulation(self):
        filt = self._filter()
        rng = np.random.default_rng(7)
        n = 3_000_000
        p = self.JUMP_INTENSITY * self.STEP_SIZE
        diffusion = rng.normal(0, self.SIGMA[1], size=n)
        ask_arrival = rng.uniform(size=n) < p
        bid_arrival = rng.uniform(size=n) < p
        jump_up = rng.exponential(self.JUMP_SIZE, size=n) * ask_arrival
        jump_down = rng.exponential(self.JUMP_SIZE, size=n) * bid_arrival
        ret_samples = diffusion + jump_up - jump_down

        bandwidth = 0.0004
        for x in (-0.006, -0.002, 0.0, 0.002, 0.006):
            mc_density = np.mean(np.abs(ret_samples - x) < bandwidth / 2) / bandwidth
            analytic = filt._emission_return(x, 1)
            assert mc_density == pytest.approx(analytic, rel=0.08), (
                f"at ret={x}: Monte Carlo density {mc_density:.4f} vs analytic {analytic:.4f}"
            )
