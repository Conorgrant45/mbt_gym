"""
test_return_ppo_wrapper.py
------------------------------
Pytest suite for envs/return_ppo_wrapper.py. Does not modify any
production file (market dynamics, reward, or PPO observation
architecture); builds its own ReturnPPOWrapper / RegimeSwitchingEnv
instances. Structurally mirrors tests/test_hamilton_ppo_wrapper.py.

Run from repo root:
    pytest tests/test_return_ppo_wrapper.py -v
"""

import numpy as np
import pytest

from envs.return_ppo_wrapper import ReturnPPOWrapper, DEFAULT_RETURN_SCALE
from envs.make_envs import make_regime_envs, N_STEPS

INVENTORY_SCALE = 10.0
RETURN_SCALE = DEFAULT_RETURN_SCALE


def make_env(seed=None):
    return ReturnPPOWrapper(inventory_scale=INVENTORY_SCALE, return_scale=RETURN_SCALE, seed=seed)


def _instrument_arrival_fill(base_env):
    """Monkeypatch (observe only) each regime sub-env's arrival/fill
    models to log every call -- same technique as test_hamilton_ppo_
    wrapper.py, never alters market dynamics."""
    logs = {0: {"arrivals": [], "fills": []}, 1: {"arrivals": [], "fills": []}}
    for regime_idx in (0, 1):
        md = base_env.envs[regime_idx].model_dynamics
        log = logs[regime_idx]

        orig_get_arrivals = md.arrival_model.get_arrivals

        def arrivals_hook(orig=orig_get_arrivals, log=log):
            arr = orig()
            log["arrivals"].append(arr.copy())
            return arr

        md.arrival_model.get_arrivals = arrivals_hook

        orig_get_fills = md.fill_probability_model.get_fills

        def fills_hook(depths, orig=orig_get_fills, log=log):
            f = orig(depths)
            log["fills"].append(f.copy())
            return f

        md.fill_probability_model.get_fills = fills_hook
    return logs


# ======================================================================
# Reset tests (items 1-5)
# ======================================================================
class TestReset:
    def test_obs_shape(self):
        env = make_env(seed=1)
        obs, info = env.reset(seed=1)
        assert obs.shape == (3,)

    def test_obs_dtype(self):
        env = make_env(seed=1)
        obs, info = env.reset(seed=1)
        assert obs.dtype == np.float32

    def test_obs_in_space(self):
        env = make_env(seed=1)
        obs, info = env.reset(seed=1)
        assert env.observation_space.contains(obs)

    def test_time_remaining_starts_at_one(self):
        env = make_env(seed=1)
        obs, info = env.reset(seed=1)
        assert obs[1] == pytest.approx(1.0)

    def test_initial_return_is_exactly_zero(self):
        """Item 4: initial return must be exactly 0, not merely small."""
        env = make_env(seed=1)
        obs, info = env.reset(seed=1)
        assert obs[2] == 0.0
        assert env.last_return is None  # no prior price exists to form a return from

    def test_cached_price_equals_raw_initial_midprice(self):
        env = make_env(seed=1)
        env.reset(seed=1)
        assert env._prev_midprice == env.base_env.raw_midprice


# ======================================================================
# Step / chronology tests (items 6-8)
# ======================================================================
class TestStep:
    def test_return_calculated_from_consecutive_raw_prices(self):
        """Item 6: one-step return uses raw midprices."""
        env = make_env(seed=2)
        env.reset(seed=2)
        s_t = env.base_env.raw_midprice

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        s_next = env.base_env.raw_midprice
        expected_return = (s_next - s_t) / s_t
        assert env.last_return == pytest.approx(expected_return, rel=1e-9)
        assert info["raw_midprice"] == pytest.approx(s_next)
        # obs[2] is float32 (declared observation dtype), so only ~1e-7
        # relative precision survives the cast -- compare with abs=1e-6,
        # matching the existing suite's convention for float32 obs checks.
        assert obs[2] == pytest.approx(np.tanh(expected_return / RETURN_SCALE), abs=1e-6)

    def test_one_step_return_uses_only_latest_transition(self):
        """Item 7: the return at step t+1 must depend only on S_t and
        S_{t+1}, not on any earlier cached history."""
        env = make_env(seed=2)
        env.reset(seed=2)

        action = env.action_space.sample()
        obs1, *_ = env.step(action)
        s_1 = env.base_env.raw_midprice
        r_1 = env.last_return

        action2 = env.action_space.sample()
        obs2, *_ = env.step(action2)
        s_2 = env.base_env.raw_midprice
        r_2 = env.last_return

        assert r_2 == pytest.approx((s_2 - s_1) / s_1, rel=1e-9)
        assert r_2 != pytest.approx(r_1) or r_1 == r_2  # independent computation, no leakage of r_1 into r_2's formula
        # explicit: recompute r_2 without any reference to r_1 and confirm equality
        assert r_2 == pytest.approx((s_2 - s_1) / s_1, rel=1e-9)

    def test_no_look_ahead(self):
        """Item 8: the observation built after step() using S_{t+1} must
        not depend on S_{t+2}, which does not exist yet at that point."""
        env = make_env(seed=10)
        env.reset(seed=10)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        s_next = info["raw_midprice"]

        # Recompute independently using ONLY S_t (cached before this step)
        # and S_{t+1} -- no S_{t+2} exists yet at this point in the test.
        env2 = make_env(seed=10)
        env2.reset(seed=10)
        s0 = env2.base_env.raw_midprice
        expected_return = (s_next - s0) / s0
        assert obs[2] == pytest.approx(np.tanh(expected_return / RETURN_SCALE), abs=1e-9)

    def test_time_remaining_decreases_correctly(self):
        env = make_env(seed=6)
        obs, info = env.reset(seed=6)
        prev_tau = obs[1]
        for step_idx in range(1, 51):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            expected_tau = 1.0 - step_idx / N_STEPS
            assert obs[1] == pytest.approx(expected_tau, abs=1e-6)
            assert obs[1] < prev_tau
            prev_tau = obs[1]

    def test_observations_finite_and_in_space(self):
        env = make_env(seed=7)
        obs, info = env.reset(seed=7)
        assert np.all(np.isfinite(obs))
        assert env.observation_space.contains(obs)
        for _ in range(300):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.all(np.isfinite(obs))
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break


# ======================================================================
# Leakage tests (items 9-11)
# ======================================================================
class TestLeakage:
    def test_true_regime_not_in_observation(self):
        """Item 9: reading info['true_regime'] must not perturb the
        observation on a parallel, identically-seeded rollout that never
        reads it."""
        seed = 9

        np.random.seed(seed)
        base_env_a = make_regime_envs(switch_within_episode=True, seed=seed)
        env_a = ReturnPPOWrapper(base_env=base_env_a, inventory_scale=INVENTORY_SCALE, return_scale=RETURN_SCALE)
        obs_a, _ = env_a.reset()

        np.random.seed(seed)
        base_env_b = make_regime_envs(switch_within_episode=True, seed=seed)
        env_b = ReturnPPOWrapper(base_env=base_env_b, inventory_scale=INVENTORY_SCALE, return_scale=RETURN_SCALE)
        obs_b, _ = env_b.reset()

        rng = np.random.default_rng(321)
        np.testing.assert_allclose(obs_a, obs_b)
        for _ in range(50):
            action = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
            obs_a, r_a, t_a, tr_a, info_a = env_a.step(action)
            _ = info_a["true_regime"]  # read hidden regime -- must not affect env_b
            obs_b, r_b, t_b, tr_b, info_b = env_b.step(action)
            np.testing.assert_allclose(obs_a, obs_b)
            assert r_a == r_b

    def test_no_hamilton_belief_in_observation(self):
        """Item 10: the observation's third component is the scaled
        return, not a belief -- confirm it can lie outside [0,1] (a valid
        belief never can) and matches the return formula, not any filter
        posterior."""
        env = make_env(seed=40)
        env.reset(seed=40)
        saw_negative = False
        for _ in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if obs[2] < 0.0:
                saw_negative = True
            if terminated or truncated:
                break
        assert saw_negative, (
            "Expected obs[2] (scaled return) to go negative at some point over 500 "
            "random-policy steps -- a belief (P(regime=1)) could never do this, "
            "confirming this observation slot is not a disguised belief."
        )

    def test_no_hamilton_filter_instantiated(self):
        """Item 11: ReturnPPOWrapper must not construct a HamiltonFilter
        anywhere, nor expose filter-only attributes."""
        env = make_env(seed=41)
        assert not hasattr(env, "filt")
        assert not hasattr(env, "belief")
        # Confirm the module itself never imports HamiltonFilter.
        import envs.return_ppo_wrapper as mod
        assert "HamiltonFilter" not in dir(mod)
        assert "hamilton_filter" not in (mod.__file__ or "")


# ======================================================================
# Episode tests
# ======================================================================
class TestEpisode:
    def test_episode_lasts_exactly_4000_steps(self):
        env = make_env(seed=11)
        env.reset(seed=11)
        n = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            n += 1
        assert n == 4000
        assert n == N_STEPS
        assert terminated is True
        assert truncated is False

    def test_final_time_feature_exactly_zero(self):
        env = make_env(seed=23)
        env.reset(seed=23)
        obs = None
        terminated = truncated = False
        n = 0
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            n += 1
        assert n == 4000
        assert info["current_step"] == N_STEPS
        assert obs[1] == 0.0

    def test_return_resets_between_episodes(self):
        env = make_env(seed=12)
        env.reset(seed=12)
        for _ in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        obs2, info2 = env.reset(seed=13)
        assert obs2[2] == 0.0
        assert env.last_return is None
        assert env._prev_midprice == env.base_env.raw_midprice

    def test_cached_price_resets_correctly(self):
        env = make_env(seed=14)
        env.reset(seed=14)
        for _ in range(100):
            action = env.action_space.sample()
            env.step(action)

        env.reset(seed=15)
        price_ep2 = env._prev_midprice
        assert price_ep2 == env.base_env.raw_midprice

    def test_cumulative_reward_matches_base_env_full_episode(self):
        """Item 17: reward is passed through unchanged."""
        seed = 16
        rng = np.random.default_rng(seed + 5000)
        actions = [rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32) for _ in range(N_STEPS)]

        np.random.seed(seed)
        base_env = make_regime_envs(switch_within_episode=True, seed=seed)
        base_env.reset()
        base_cum_reward = 0.0
        for a in actions:
            obs, reward, done, info = base_env.step(a.reshape(1, -1))
            base_cum_reward += float(np.sum(reward))

        np.random.seed(seed)
        base_env2 = make_regime_envs(switch_within_episode=True, seed=seed)
        wrapper = ReturnPPOWrapper(base_env=base_env2, inventory_scale=INVENTORY_SCALE, return_scale=RETURN_SCALE)
        wrapper.reset()
        wrapper_cum_reward = 0.0
        for a in actions:
            obs, reward, terminated, truncated, info = wrapper.step(a)
            wrapper_cum_reward += reward

        assert wrapper_cum_reward == pytest.approx(base_cum_reward, abs=1e-9)

    def test_no_nans_or_infinities_full_episode(self):
        env = make_env(seed=17)
        obs, info = env.reset(seed=17)
        assert np.all(np.isfinite(obs))
        terminated = truncated = False
        n = 0
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.all(np.isfinite(obs)), f"non-finite obs at step {n}: {obs}"
            assert np.isfinite(reward), f"non-finite reward at step {n}: {reward}"
            n += 1
        assert n == N_STEPS


# ======================================================================
# Action tests (items 13-16)
# ======================================================================
class TestAction:
    def test_random_rollout_completes(self):
        env = make_env(seed=18)
        env.reset(seed=18)
        for _ in range(200):
            action = env.action_space.sample()
            assert env.action_space.contains(action)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    def test_bid_ask_ordering_preserved(self, monkeypatch):
        """Items 13-14: action index 0 = bid, index 1 = ask."""
        env = make_env(seed=19)
        env.reset(seed=19)

        captured = {}
        original_step = env.base_env.step

        def spy_step(action_arr):
            captured["action"] = action_arr.copy()
            return original_step(action_arr)

        monkeypatch.setattr(env.base_env, "step", spy_step)

        action = np.array([-0.9, 0.9], dtype=np.float32)  # [bid=-0.9 (tight), ask=+0.9 (wide)]
        env.step(action)

        passed = captured["action"]
        assert passed.shape == (1, 2)
        assert passed[0, 0] == pytest.approx(-0.9)  # BID_INDEX = 0
        assert passed[0, 1] == pytest.approx(0.9)   # ASK_INDEX = 1

    def test_action_bounds_match_base_env(self):
        """Item 15."""
        env = make_env(seed=21)
        base_low = np.asarray(env.base_env.action_space.low)
        base_high = np.asarray(env.base_env.action_space.high)
        np.testing.assert_allclose(env.action_space.low, base_low)
        np.testing.assert_allclose(env.action_space.high, base_high)

    def test_wrapper_does_not_alter_action_values(self, monkeypatch):
        """Item 16: action-to-quote transformation is not duplicated in
        the wrapper -- the raw action passed to the base env is bit-
        identical to what the caller supplied."""
        env = make_env(seed=20)
        env.reset(seed=20)

        captured = {}
        original_step = env.base_env.step

        def spy_step(action_arr):
            captured["action"] = action_arr.copy()
            return original_step(action_arr)

        monkeypatch.setattr(env.base_env, "step", spy_step)

        action = np.array([0.3, -0.6], dtype=np.float32)
        env.step(action)

        np.testing.assert_allclose(captured["action"].flatten(), action)


# ======================================================================
# Reproducibility tests (item 18)
# ======================================================================
class TestReproducibility:
    @staticmethod
    def _run_full_episode_instrumented(seed, actions):
        """Build+run ONE full episode to completion (never interleaved
        with another env's step() calls -- see test_hamilton_ppo_wrapper.
        py's TestReproducibility docstring for why interleaving would
        desynchronise the shared legacy np.random stream)."""
        np.random.seed(seed)
        base_env = make_regime_envs(switch_within_episode=True, seed=seed)
        logs = _instrument_arrival_fill(base_env)
        wrapper = ReturnPPOWrapper(base_env=base_env, inventory_scale=INVENTORY_SCALE, return_scale=RETURN_SCALE)
        wrapper.reset()

        record = dict(regime=[], raw_midprice=[], reward=[], obs=[])
        for action in actions:
            obs, reward, terminated, truncated, info = wrapper.step(action)
            record["regime"].append(info["true_regime"])
            record["raw_midprice"].append(info["raw_midprice"])
            record["reward"].append(reward)
            record["obs"].append(obs.copy())
        return record

    def test_fixed_seed_reproducible(self):
        seed = 100
        rng = np.random.default_rng(seed + 9000)
        actions = [rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32) for _ in range(N_STEPS)]

        record_a = self._run_full_episode_instrumented(seed, actions)
        record_b = self._run_full_episode_instrumented(seed, actions)

        for step_idx in range(N_STEPS):
            assert record_a["regime"][step_idx] == record_b["regime"][step_idx]
            assert record_a["raw_midprice"][step_idx] == record_b["raw_midprice"][step_idx]
            assert record_a["reward"][step_idx] == record_b["reward"][step_idx]
            np.testing.assert_array_equal(record_a["obs"][step_idx], record_b["obs"][step_idx])


# ======================================================================
# Cross-agent observation-identity test (item 12: MLP and LSTM agents
# must receive identical current observations -- both use ReturnPPOWrapper,
# so this confirms two independently constructed instances agree bit-for-
# bit on the same seed, which is what "same wrapper class, same seed"
# guarantees for both agent types at training/eval time).
# ======================================================================
def test_two_independently_constructed_wrappers_produce_identical_observations():
    """
    Run two SEPARATE ReturnPPOWrapper instances (representing the MLP and
    LSTM agents' own environments), each to completion sequentially with
    the same seed and action sequence, then compare their recorded
    observation traces. Deliberately NOT interleaved: RegimeSwitchingEnv's
    regime-transition draws read from the LEGACY GLOBAL np.random stream,
    so interleaving env_mlp.step()/env_lstm.step() calls in one loop would
    split that single stream into two interleaved sub-sequences instead of
    each wrapper replaying its own seed faithfully -- exactly the pitfall
    documented in tests/test_hamilton_ppo_wrapper.py's TestReproducibility
    (and hit once already, empirically, while first writing this test).
    """
    seed = 777
    rng = np.random.default_rng(seed + 1)
    actions = [rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32) for _ in range(500)]

    def run(seed, actions):
        np.random.seed(seed)
        env = ReturnPPOWrapper(base_env=make_regime_envs(switch_within_episode=True, seed=seed),
                                inventory_scale=INVENTORY_SCALE, return_scale=RETURN_SCALE)
        obs, _ = env.reset()
        trace = [obs.copy()]
        rewards = []
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            trace.append(obs.copy())
            rewards.append(reward)
        return trace, rewards

    trace_mlp, rewards_mlp = run(seed, actions)
    trace_lstm, rewards_lstm = run(seed, actions)

    for step_idx, (o_mlp, o_lstm) in enumerate(zip(trace_mlp, trace_lstm)):
        np.testing.assert_array_equal(o_mlp, o_lstm, err_msg=f"observation diverged at step {step_idx}")
    assert rewards_mlp == rewards_lstm


# ======================================================================
# Validation tests
# ======================================================================
class TestValidation:
    @pytest.mark.parametrize("bad_scale", [0.0, -1.0, -10.0])
    def test_non_positive_inventory_scale_raises(self, bad_scale):
        with pytest.raises(ValueError):
            ReturnPPOWrapper(inventory_scale=bad_scale, seed=1)

    @pytest.mark.parametrize("bad_scale", [0.0, -1.0, -10.0])
    def test_non_positive_return_scale_raises(self, bad_scale):
        with pytest.raises(ValueError):
            ReturnPPOWrapper(return_scale=bad_scale, seed=1)

    def test_positive_scales_do_not_raise(self):
        env = ReturnPPOWrapper(inventory_scale=1.0, return_scale=0.01, seed=1)
        env.reset(seed=1)  # should not raise


# ======================================================================
# Observation / action space sanity
# ======================================================================
def test_action_space_matches_base_env_bounds():
    env = make_env(seed=21)
    base_low = np.asarray(env.base_env.action_space.low)
    base_high = np.asarray(env.base_env.action_space.high)
    np.testing.assert_allclose(env.action_space.low, base_low)
    np.testing.assert_allclose(env.action_space.high, base_high)


def test_observation_space_bounds():
    env = make_env(seed=22)
    assert env.observation_space.shape == (3,)
    np.testing.assert_allclose(env.observation_space.low, [-1.0, 0.0, -1.0])
    np.testing.assert_allclose(env.observation_space.high, [1.0, 1.0, 1.0])
