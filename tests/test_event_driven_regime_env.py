"""
test_event_driven_regime_env.py
----------------------------------
Phase 1 required tests (event-driven regime-switching environment audit
brief, Section 5, items 1-20). Item 20 (existing fixed-step tests continue
to pass) is the full existing suite, run separately, not duplicated here.

Run from repo root:
    pytest tests/test_event_driven_regime_env.py -v
"""
import numpy as np
import pytest

from envs.event_driven_regime_env import (
    EventDrivenRegimeSwitchingEnv,
    MAX_DEPTH,
    stationary_distribution_from_generator,
    denormalise_depth,
)
from envs.make_envs import TRANSITION_GENERATOR, KAPPA, EPSILON, LAMBDA, TERMINAL_TIME
import simulate_belief_weighted as SBW


def make_env(seed=0, **kwargs):
    return EventDrivenRegimeSwitchingEnv(seed=seed, **kwargs)


def run_full_episode(env, action=np.array([0.0, 0.0])):
    """action=[0,0] normalised -> bid_depth=ask_depth=MAX_DEPTH/2 (a moderate,
    non-degenerate, non-saturating quote)."""
    obs = env.reset()
    steps = []
    done = False
    while not done:
        obs, reward, done, info = env.step(action)
        steps.append((obs.copy(), reward, done, info))
    return steps


# ======================================================================
# Item 1: terminates exactly at T
# ======================================================================
def test_terminates_exactly_at_T():
    env = make_env(seed=1)
    steps = run_full_episode(env)
    assert steps[-1][3]["event_type"] == "terminal"
    assert env.time == TERMINAL_TIME
    assert steps[-1][0][2] == TERMINAL_TIME  # TIME_INDEX of the returned obs


def test_terminates_exactly_at_T_many_seeds():
    for seed in range(20):
        env = make_env(seed=seed)
        steps = run_full_episode(env)
        assert env.time == TERMINAL_TIME
        assert steps[-1][3]["event_type"] == "terminal"
        assert sum(1 for s in steps if s[3]["event_type"] == "terminal") == 1


# ======================================================================
# Item 2: hidden regime transitions do not produce agent-visible steps
# ======================================================================
def test_internal_switches_do_not_produce_steps():
    """With a switching rate inflated far above the arrival rate, force many
    hidden switches per RL step, and confirm the number of RL steps returned
    is unaffected (only observable arrivals/terminal produce a step)."""
    fast_switch_Q = np.array([[-5000.0, 5000.0], [5000.0, -5000.0]])
    env_fast = EventDrivenRegimeSwitchingEnv(seed=7, transition_generator=fast_switch_Q)
    steps_fast = run_full_episode(env_fast)
    total_internal_switches = sum(s[3]["number_internal_regime_switches"] for s in steps_fast)
    assert total_internal_switches > 1000  # many hidden switches happened
    # every one of those steps is still exactly one arrival or one terminal event
    for _, _, _, info in steps_fast:
        assert info["event_type"] in ("arrival", "terminal")

    env_slow = EventDrivenRegimeSwitchingEnv(
        seed=7, transition_generator=np.array([[-1e-9, 1e-9], [1e-9, -1e-9]])
    )
    steps_slow = run_full_episode(env_slow)
    # Roughly the same number of OBSERVABLE steps regardless of how many
    # hidden switches happen internally (arrival process is unaffected by
    # regime-switch rate) -- same order of magnitude, not exact (different
    # seeE9d draw consumption shifts the exact arrival realisation).
    assert abs(len(steps_fast) - len(steps_slow)) < 0.3 * len(steps_slow)


# ======================================================================
# Item 3: quotes remain unchanged through internal hidden switches
# ======================================================================
def test_quotes_unchanged_through_internal_switches():
    """The bid/ask depth used for the terminating arrival's fill must be
    exactly the depth passed to THIS step() call, regardless of how many
    hidden switches preceded it within the same step."""
    fast_switch_Q = np.array([[-3000.0, 3000.0], [3000.0, -3000.0]])
    env = EventDrivenRegimeSwitchingEnv(seed=3, transition_generator=fast_switch_Q)
    env.reset()
    bid_action, ask_action = 0.2, -0.4
    obs, reward, done, info = env.step(np.array([bid_action, ask_action]))
    assert info["number_internal_regime_switches"] > 0  # confirm switches actually happened this step
    expected_bid_depth = denormalise_depth(bid_action, MAX_DEPTH)
    expected_ask_depth = denormalise_depth(ask_action, MAX_DEPTH)
    assert info["bid_depth"] == pytest.approx(expected_bid_depth)
    assert info["ask_depth"] == pytest.approx(expected_ask_depth)
    # If a fill happened, reconstruct the execution price and confirm it used
    # exactly this step's depths (see item 16 for the full sign check).
    if info["fill_indicator"][0]:  # bid fill
        price_pre_jump = info["price_before"] + info["brownian_increment"]
        implied_cash_change = -(price_pre_jump - expected_bid_depth)
        assert (info["cash_after"] - info["cash_before"]) == pytest.approx(implied_cash_change)


# ======================================================================
# Item 4: event waiting times have the correct exponential distribution
# ======================================================================
def test_arrival_waiting_times_are_exponential():
    """Disable regime switching (q_out=0 for both regimes) and use a huge
    horizon so essentially every event is an arrival; the combined
    inter-arrival time should be Exponential(lambda_bid + lambda_ask)."""
    no_switch_Q = np.zeros((2, 2))
    env = EventDrivenRegimeSwitchingEnv(
        seed=11, transition_generator=no_switch_Q, terminal_time=50.0, initial_regime=0,
    )
    env.reset()
    gaps = []
    done = False
    while not done and len(gaps) < 20_000:
        obs, reward, done, info = env.step(np.array([0.0, 0.0]))
        if info["event_type"] == "arrival":
            gaps.append(info["elapsed_time"])
    gaps = np.array(gaps)
    theoretical_mean = 1.0 / (2 * LAMBDA)
    se = gaps.std(ddof=1) / np.sqrt(len(gaps))
    assert abs(gaps.mean() - theoretical_mean) < 5 * se
    # Exponential(rate): std == mean
    assert gaps.std(ddof=1) == pytest.approx(theoretical_mean, rel=0.1)


# ======================================================================
# Item 5: buy/sell proportions match relative intensities
# ======================================================================
def test_buy_sell_proportions_match_intensities():
    no_switch_Q = np.zeros((2, 2))
    env = EventDrivenRegimeSwitchingEnv(
        seed=13, transition_generator=no_switch_Q, terminal_time=50.0, initial_regime=0,
        lambda_bid=100.0, lambda_ask=300.0,
    )
    env.reset()
    n_buy = n_sell = 0
    done = False
    while not done and (n_buy + n_sell) < 20_000:
        obs, reward, done, info = env.step(np.array([0.0, 0.0]))
        if info["event_type"] == "arrival":
            if info["arrival_side"] == "buy":
                n_buy += 1
            else:
                n_sell += 1
    frac_buy = n_buy / (n_buy + n_sell)
    expected = 300.0 / (100.0 + 300.0)
    se = np.sqrt(expected * (1 - expected) / (n_buy + n_sell))
    assert abs(frac_buy - expected) < 5 * se


# ======================================================================
# Item 6: mean observable event count matches integrated arrival intensity
# ======================================================================
def test_mean_observable_arrival_count_matches_integrated_intensity():
    n_episodes = 60
    counts = []
    for seed in range(n_episodes):
        env = make_env(seed=100_000 + seed)
        steps = run_full_episode(env)
        counts.append(sum(1 for s in steps if s[3]["event_type"] == "arrival"))
    counts = np.array(counts)
    expected = 2 * LAMBDA * TERMINAL_TIME  # 280
    se = counts.std(ddof=1) / np.sqrt(n_episodes)
    assert abs(counts.mean() - expected) < 4 * se
    assert 200 < counts.mean() < 360  # sanity bound, not a tight refit


# ======================================================================
# Item 7: regime occupation proportions match the CTMC
# ======================================================================
def test_regime_occupation_matches_ctmc_stationary_distribution():
    """At the real calibration, arrivals (rate 280) are far more frequent
    than regime switches (rate ~10.4 / ~4.1), so the vast majority of RL
    steps contain zero internal switches -- attributing each step's
    elapsed_time to its regime_at_event is therefore a good (not exact)
    proxy for true occupation time, with a documented small bias toward
    the post-switch regime on the rare steps that do straddle a switch."""
    pi_theoretical = stationary_distribution_from_generator(TRANSITION_GENERATOR)
    time_in_regime = {0: 0.0, 1: 0.0}
    n_episodes = 40
    for seed in range(n_episodes):
        env = make_env(seed=200_000 + seed)
        steps = run_full_episode(env)
        for _, _, _, info in steps:
            time_in_regime[info["regime_at_event"]] += info["elapsed_time"]
    total_time = time_in_regime[0] + time_in_regime[1]
    occupation_1 = time_in_regime[1] / total_time
    assert abs(occupation_1 - pi_theoretical[1]) < 0.05  # generous MC + attribution-bias tolerance


def test_ctmc_race_formula_matches_theoretical_stationary_distribution():
    """Directly simulate the regime-only competing-event race (holding time
    Exponential(-Q[z,z]), deterministic flip for a 2-state chain) using the
    SAME TRANSITION_GENERATOR the environment uses, independent of the
    arrival/fill machinery, and confirm long-run occupation matches the
    theoretical stationary distribution -- this is the exact sub-process
    embedded in EventDrivenRegimeSwitchingEnv._advance_to_next_event's
    switch-handling branch."""
    rng = np.random.default_rng(999)
    Q = TRANSITION_GENERATOR
    z = 0
    t = 0.0
    T = 200_000.0
    time_in = {0: 0.0, 1: 0.0}
    while t < T:
        q_out = -Q[z, z]
        dt = rng.exponential(1.0 / q_out)
        time_in[z] += dt
        t += dt
        z = 1 - z
    pi_theoretical = stationary_distribution_from_generator(Q)
    occupation_1 = time_in[1] / T
    assert abs(occupation_1 - pi_theoretical[1]) < 0.01


# ======================================================================
# Item 8: Brownian variance matches integrated sigma^2 * dt
# ======================================================================
def test_brownian_increment_variance_matches_integrated_variance():
    no_switch_Q = np.zeros((2, 2))
    env = EventDrivenRegimeSwitchingEnv(
        seed=17, transition_generator=no_switch_Q, terminal_time=200.0, initial_regime=0,
    )
    env.reset()
    standardised = []
    done = False
    while not done and len(standardised) < 20_000:
        obs, reward, done, info = env.step(np.array([0.0, 0.0]))
        if info["integrated_variance"] > 0:
            standardised.append(info["brownian_increment"] / np.sqrt(info["integrated_variance"]))
    standardised = np.array(standardised)
    assert abs(standardised.mean()) < 0.05
    assert abs(standardised.std(ddof=1) - 1.0) < 0.05


# ======================================================================
# Item 9: arrival and fill draws remain independent
# ======================================================================
def test_arrival_and_fill_draws_independent():
    no_switch_Q = np.zeros((2, 2))
    depth = 0.4
    env = EventDrivenRegimeSwitchingEnv(
        seed=19, transition_generator=no_switch_Q, terminal_time=80.0, initial_regime=0,
    )
    env.reset()
    action = np.array([2 * depth / MAX_DEPTH - 1.0, 2 * depth / MAX_DEPTH - 1.0])
    fills_by_side = {"buy": [], "sell": []}
    done = False
    while not done and (len(fills_by_side["buy"]) + len(fills_by_side["sell"])) < 20_000:
        obs, reward, done, info = env.step(action)
        if info["event_type"] == "arrival":
            side = info["arrival_side"]
            filled = info["fill_indicator"][1] if side == "buy" else info["fill_indicator"][0]
            fills_by_side[side].append(filled)
    expected_fill_prob = np.exp(-KAPPA * depth)
    for side in ("buy", "sell"):
        arr = np.array(fills_by_side[side])
        se = np.sqrt(expected_fill_prob * (1 - expected_fill_prob) / len(arr))
        assert abs(arr.mean() - expected_fill_prob) < 5 * se


# ======================================================================
# Item 10: jump occurrence is tied to arrivals, not fills
# ======================================================================
def test_jump_occurs_regardless_of_fill():
    """Regime 1, forced, with an enormous quoted depth so fills are
    (almost) never realised -- jumps must still occur on every arrival."""
    stay_in_1_Q = np.array([[0.0, 0.0], [0.0, 0.0]])
    env = EventDrivenRegimeSwitchingEnv(
        seed=23, transition_generator=stay_in_1_Q, terminal_time=30.0, initial_regime=1,
    )
    env.reset()
    huge_depth_action = np.array([1.0, 1.0])  # depth = MAX_DEPTH -> fill_prob = 0.01
    n_arrivals = 0
    n_jumps = 0
    n_fills = 0
    done = False
    while not done and n_arrivals < 5_000:
        obs, reward, done, info = env.step(huge_depth_action)
        if info["event_type"] == "arrival":
            n_arrivals += 1
            if info["jump_increment"] != 0.0:
                n_jumps += 1
            n_fills += int(info["fill_indicator"].sum())
    assert n_jumps == n_arrivals  # every arrival in regime 1 jumps, fill or not
    assert n_fills < 0.05 * n_arrivals  # fills are rare at this depth


# ======================================================================
# Item 11: jump magnitudes have the correct exponential mean
# ======================================================================
def test_jump_magnitude_matches_exponential_mean():
    stay_in_1_Q = np.array([[0.0, 0.0], [0.0, 0.0]])
    env = EventDrivenRegimeSwitchingEnv(
        seed=29, transition_generator=stay_in_1_Q, terminal_time=40.0, initial_regime=1,
    )
    env.reset()
    magnitudes = []
    done = False
    while not done and len(magnitudes) < 10_000:
        obs, reward, done, info = env.step(np.array([0.0, 0.0]))
        if info["event_type"] == "arrival":
            magnitudes.append(abs(info["jump_increment"]))
    magnitudes = np.array(magnitudes)
    se = magnitudes.std(ddof=1) / np.sqrt(len(magnitudes))
    assert abs(magnitudes.mean() - EPSILON) < 5 * se


# ======================================================================
# Item 12: buy/sell jump signs are correct
# ======================================================================
def test_jump_signs_match_arrival_side():
    stay_in_1_Q = np.array([[0.0, 0.0], [0.0, 0.0]])
    env = EventDrivenRegimeSwitchingEnv(
        seed=31, transition_generator=stay_in_1_Q, terminal_time=30.0, initial_regime=1,
    )
    env.reset()
    done = False
    n_checked = 0
    while not done and n_checked < 5_000:
        obs, reward, done, info = env.step(np.array([0.0, 0.0]))
        if info["event_type"] == "arrival":
            n_checked += 1
            if info["arrival_side"] == "buy":
                assert info["jump_increment"] > 0.0
            else:
                assert info["jump_increment"] < 0.0

    # regime 0: no jump model at all, regardless of arrival side
    stay_in_0_Q = np.array([[0.0, 0.0], [0.0, 0.0]])
    env0 = EventDrivenRegimeSwitchingEnv(
        seed=37, transition_generator=stay_in_0_Q, terminal_time=30.0, initial_regime=0,
    )
    env0.reset()
    done = False
    n_checked = 0
    while not done and n_checked < 5_000:
        obs, reward, done, info = env0.step(np.array([0.0, 0.0]))
        if info["event_type"] == "arrival":
            n_checked += 1
            assert info["jump_increment"] == 0.0


# ======================================================================
# Item 13: running penalties use the actual elapsed time
# ======================================================================
def test_running_penalty_uses_actual_elapsed_time():
    env = make_env(seed=41)
    steps = run_full_episode(env)
    for _, _, _, info in steps:
        expected = env.phi * (info["inventory_before"] ** 2) * info["elapsed_time"]
        assert info["running_penalty_increment"] == pytest.approx(expected, rel=1e-9, abs=1e-12)


# ======================================================================
# Item 14: terminal penalty is applied exactly once
# ======================================================================
def test_terminal_penalty_applied_exactly_once():
    for seed in range(15):
        env = make_env(seed=seed)
        steps = run_full_episode(env)
        nonzero_terminal = [s for s in steps if s[3]["terminal_penalty_increment"] != 0.0]
        assert len(nonzero_terminal) == 1
        assert nonzero_terminal[0][3]["event_type"] == "terminal"
        last_info = steps[-1][3]
        expected = env.alpha * (last_info["inventory_after"] ** 2)
        assert last_info["terminal_penalty_increment"] == pytest.approx(expected)


# ======================================================================
# Item 15: reward decomposition identity
# ======================================================================
def test_reward_decomposition_identity():
    for seed in range(15):
        env = make_env(seed=1000 + seed)
        cash0, inv0, price0 = env.reset()[0], 0.0, env.initial_price
        steps = run_full_episode(env)
        cumulative_reward = sum(s[1] for s in steps)
        running_penalty_total = sum(s[3]["running_penalty_increment"] for s in steps)
        terminal_penalty_total = sum(s[3]["terminal_penalty_increment"] for s in steps)
        last_info = steps[-1][3]
        terminal_mtm_wealth = last_info["cash_after"] + last_info["inventory_after"] * last_info["price_after"]
        initial_mtm_wealth = env.initial_cash + env.initial_inventory * env.initial_price
        reconstructed = (terminal_mtm_wealth - initial_mtm_wealth) - running_penalty_total - terminal_penalty_total
        assert reconstructed == pytest.approx(cumulative_reward, rel=1e-9, abs=1e-6)


# ======================================================================
# Item 16: bid/ask cash/inventory signs match the fixed-step environment
# ======================================================================
def test_bid_ask_fill_signs():
    env = make_env(seed=51)
    env.reset()
    action = np.array([0.3, -0.1])
    n_bid_fills = n_ask_fills = 0
    done = False
    while not done and (n_bid_fills < 5 or n_ask_fills < 5):
        obs, reward, done, info = env.step(action)
        price_pre_jump = info["price_before"] + info["brownian_increment"]
        if info["fill_indicator"][0]:  # bid fill -> agent buys
            n_bid_fills += 1
            assert info["inventory_after"] - info["inventory_before"] == 1
            expected_cash_change = -(price_pre_jump - info["bid_depth"])
            assert (info["cash_after"] - info["cash_before"]) == pytest.approx(expected_cash_change)
        if info["fill_indicator"][1]:  # ask fill -> agent sells
            n_ask_fills += 1
            assert info["inventory_after"] - info["inventory_before"] == -1
            expected_cash_change = price_pre_jump + info["ask_depth"]
            assert (info["cash_after"] - info["cash_before"]) == pytest.approx(expected_cash_change)
        if done:
            break


# ======================================================================
# Item 17/18: seed reproducibility
# ======================================================================
def test_identical_seeds_reproduce_identical_event_paths():
    def trace(seed):
        env = make_env(seed=seed)
        steps = run_full_episode(env)
        return [
            (info["event_type"], info["arrival_side"], info["regime_at_event"],
             round(info["cash_after"], 10), round(info["inventory_after"], 10),
             round(info["price_after"], 10), round(reward, 10))
            for obs, reward, done, info in steps
        ]

    assert trace(123) == trace(123)


def test_different_seeds_produce_different_paths():
    env_a = make_env(seed=1)
    env_b = make_env(seed=2)
    steps_a = run_full_episode(env_a)
    steps_b = run_full_episode(env_b)
    prices_a = [s[3]["price_after"] for s in steps_a]
    prices_b = [s[3]["price_after"] for s in steps_b]
    assert prices_a[: min(len(prices_a), len(prices_b))] != prices_b[: min(len(prices_a), len(prices_b))]


# ======================================================================
# Item 19: true regime does not appear in learned observations
# ======================================================================
def test_true_regime_not_in_observation():
    env = make_env(seed=61)
    obs = env.reset()
    assert obs.shape == (4,)
    steps = run_full_episode(env)
    for obs, reward, done, info in steps:
        assert obs.shape == (4,)
        # obs is exactly [cash, inventory, time, price] -- structurally no
        # regime slot exists for a value to leak into.
        assert obs[0] == pytest.approx(info["cash_after"])
        assert obs[1] == pytest.approx(info["inventory_after"])
        assert obs[3] == pytest.approx(info["price_after"])
        # regime is only ever available via info, never via obs
        assert "true_regime" in info


# ======================================================================
# Supporting parity / sanity checks (not separately numbered, but load-
# bearing for the above: constants must match the fixed-step environment)
# ======================================================================
def test_max_depth_matches_fixed_step_convention():
    assert MAX_DEPTH == SBW.MAX_DEPTH


def test_denormalise_depth_matches_normalise_depth_inverse():
    for raw_action in (-1.0, -0.3, 0.0, 0.5, 1.0):
        depth = denormalise_depth(raw_action, MAX_DEPTH)
        assert SBW.normalise_depth(depth) == pytest.approx(raw_action, abs=1e-9)


def test_stationary_distribution_matches_discrete_chain():
    from envs.make_envs import TRANSITION_MATRIX

    pi_continuous = stationary_distribution_from_generator(TRANSITION_GENERATOR)
    P = np.array(TRANSITION_MATRIX)
    A = (P.T - np.eye(2))
    A[-1] = 1.0
    b = np.zeros(2)
    b[-1] = 1.0
    pi_discrete = np.linalg.solve(A, b)
    assert np.allclose(pi_continuous, pi_discrete, atol=1e-8)


def test_generator_rows_sum_to_zero():
    assert np.allclose(TRANSITION_GENERATOR.sum(axis=1), 0.0, atol=1e-8)


def test_reset_returns_no_regime_leak_and_valid_initial_state():
    env = make_env(seed=71)
    obs = env.reset()
    assert obs[0] == 0.0  # initial cash
    assert obs[1] == 0.0  # initial inventory
    assert obs[2] == 0.0  # initial time
    assert obs[3] == pytest.approx(100.0)  # initial price
    assert env.current_regime in (0, 1)
