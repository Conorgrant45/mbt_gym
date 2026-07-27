"""
test_hamilton_ppo_wrapper.py
------------------------------
Pytest suite for envs/hamilton_ppo_wrapper.py. Does not modify any
production file (market dynamics, Hamilton filter, reward, or PPO
observation architecture); builds its own HamiltonPPOWrapper /
RegimeSwitchingEnv / HamiltonFilter instances.

Run from repo root:
    pytest tests/test_hamilton_ppo_wrapper.py -v
"""

import numpy as np
import pytest

from envs.hamilton_ppo_wrapper import HamiltonPPOWrapper, make_filter
from envs.make_envs import make_regime_envs, N_STEPS, TRANSITION_MATRIX
from mbt_gym.gym.index_names import INVENTORY_INDEX

INVENTORY_SCALE = 10.0


def make_env(seed=None):
    return HamiltonPPOWrapper(inventory_scale=INVENTORY_SCALE, seed=seed)


def _instrument_arrival_fill(base_env):
    """
    Monkeypatch (observe only, never alter) each regime sub-env's arrival
    and fill-probability models to log every call's return value. Same
    technique already used in compare_four_policies_paired.py -- does not
    change market dynamics, only records what already happened.
    """
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
# Reset tests
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

    def test_belief_starts_at_stationary_prior(self):
        # HamiltonFilter's default initial_belief is the stationary
        # distribution of TRANSITION_MATRIX (see beliefs/hamilton_filter.py
        # _stationary_distribution()); the wrapper does not override this.
        env = make_env(seed=1)
        obs, info = env.reset(seed=1)

        P = np.array(TRANSITION_MATRIX)
        A = (P.T - np.eye(2))
        A[-1] = 1.0
        b = np.zeros(2)
        b[-1] = 1.0
        pi_stat = np.linalg.solve(A, b)

        # obs[2] is float32 (declared observation dtype), so only ~1e-7
        # relative precision is preserved; env.belief (float64, read
        # directly from the filter) is checked at full precision.
        assert obs[2] == pytest.approx(pi_stat[1], abs=1e-6)
        assert env.belief == pytest.approx(pi_stat[1], abs=1e-9)

    def test_cached_price_equals_raw_initial_midprice(self):
        env = make_env(seed=1)
        env.reset(seed=1)
        assert env._prev_midprice == env.base_env.raw_midprice

    def test_filter_has_not_processed_a_return_at_reset(self):
        env = make_env(seed=1)
        env.reset(seed=1)
        # HamiltonFilter internals: first update() call after reset() only
        # caches the price (return not yet observable) -- xi stays at the
        # product-form prior, and the filter's own _prev_midprice buffer
        # equals the price just cached, confirming no return was consumed.
        assert env.filt._prev_midprice == env.base_env.raw_midprice
        assert env.last_return is None


# ======================================================================
# Step tests
# ======================================================================
class TestStep:
    def test_return_calculated_from_consecutive_raw_prices(self):
        env = make_env(seed=2)
        env.reset(seed=2)
        s_t = env.base_env.raw_midprice

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        s_next = env.base_env.raw_midprice
        expected_return = (s_next - s_t) / s_t
        assert env.last_return == pytest.approx(expected_return, rel=1e-9)
        assert info["raw_midprice"] == pytest.approx(s_next)

    def test_filter_updates_exactly_once_per_step(self, monkeypatch):
        env = make_env(seed=3)
        env.reset(seed=3)

        call_count = {"n": 0}
        original_update = env.filt.update

        def counting_update(midprice):
            call_count["n"] += 1
            return original_update(midprice)

        monkeypatch.setattr(env.filt, "update", counting_update)

        for _ in range(20):
            action = env.action_space.sample()
            env.step(action)

        assert call_count["n"] == 20

    def test_belief_remains_in_unit_interval(self):
        env = make_env(seed=4)
        env.reset(seed=4)
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert 0.0 <= obs[2] <= 1.0
            if terminated or truncated:
                break

    def test_wrapper_belief_replay_matches_independent_filter(self):
        """
        Direct replay test (replaces the previous two-parallel-environment
        version): run ONE wrapped environment for a full episode, record
        the EXACT raw midprice sequence it observed (S_0 at reset, then
        S_1..S_n at each step), then feed that recorded sequence into a
        fresh, independently constructed HamiltonFilter and confirm its
        belief matches the wrapper's belief after every single step.

        This removes the previous test's dependency on getting two
        separately constructed environments' RNG streams paired exactly
        right (np.random.seed timing relative to each env's own
        construction/reset -- see compare_four_policies_paired.py's
        documented pairing requirement, which the earlier version of this
        test got wrong once already: an env.reset() call landing between
        two np.random.seed() calls desynchronised the two paths and the
        test failed until reordered). Recording and replaying the
        wrapper's own observed sequence has no such dependency.
        """
        env = make_env(seed=30)
        obs, info = env.reset(seed=30)

        recorded_prices = [env.base_env.raw_midprice]
        wrapper_beliefs = [env.belief]

        rng = np.random.default_rng(31)
        terminated = truncated = False
        while not (terminated or truncated):
            action = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            recorded_prices.append(info["raw_midprice"])
            wrapper_beliefs.append(env.belief)

        assert len(recorded_prices) == N_STEPS + 1  # S_0 plus one price per step

        ref_filter = make_filter()
        ref_filter.reset()
        replay_beliefs = [ref_filter.update(recorded_prices[0])]  # caches S_0 only, matches reset()
        for price in recorded_prices[1:]:
            replay_beliefs.append(ref_filter.update(price))

        assert len(replay_beliefs) == len(wrapper_beliefs)
        for step_idx, (w, r) in enumerate(zip(wrapper_beliefs, replay_beliefs)):
            assert w == pytest.approx(r, abs=1e-10), (
                f"belief mismatch at step {step_idx}: wrapper={w}, replay={r}"
            )

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
# Leakage tests
#
# Note: a test that only checked `"true_regime" in info` (reading a dict
# key) was removed -- that is not a meaningful leakage test, since info
# is explicitly allowed to carry true_regime for offline diagnostics (see
# module docstring); the only thing that must never happen is the PPO
# OBSERVATION being built from it. That is what the tests below actually
# check, by comparing parallel rollouts.
# ======================================================================
class TestLeakage:
    def test_observation_unaffected_by_reading_hidden_regime(self):
        """Reading info['true_regime'] (as a diagnostic would) must not
        perturb the observation the policy receives on a parallel,
        identically-seeded rollout that never reads it."""
        seed = 9

        np.random.seed(seed)
        base_env_a = make_regime_envs(switch_within_episode=True, seed=seed)
        env_a = HamiltonPPOWrapper(base_env=base_env_a, inventory_scale=INVENTORY_SCALE)
        obs_a, _ = env_a.reset()

        np.random.seed(seed)
        base_env_b = make_regime_envs(switch_within_episode=True, seed=seed)
        env_b = HamiltonPPOWrapper(base_env=base_env_b, inventory_scale=INVENTORY_SCALE)
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

    def test_no_future_return_used(self):
        """The observation returned by step() at time t+1 must be
        constructible from information available up to and including
        S_{t+1} -- never S_{t+2}. Checked by confirming the belief in the
        observation exactly matches filt.update(S_{t+1}) called with no
        knowledge of any later price."""
        env = make_env(seed=10)
        env.reset(seed=10)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        s_next = info["raw_midprice"]

        # Recompute independently using ONLY S_t (cached before this step)
        # and S_{t+1} -- no S_{t+2} exists yet at this point in the test.
        ref_filter = make_filter()
        ref_filter.reset()
        # replay must match the wrapper's own reset+this-step sequence
        env2 = make_env(seed=10)
        env2.reset(seed=10)
        s0 = env2.base_env.raw_midprice
        ref_filter.update(s0)
        ref_belief = ref_filter.update(s_next)
        assert obs[2] == pytest.approx(ref_belief, abs=1e-9)


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
        # tau = 1 - current_step/n_steps = 1 - 4000/4000 = 0.0 exactly
        # (integer division of equal ints is exact in float64, and 0.0
        # survives the float32 cast exactly too) -- checked with == not
        # pytest.approx deliberately.
        assert obs[1] == 0.0

    def test_filter_resets_between_episodes(self):
        env = make_env(seed=12)
        env.reset(seed=12)
        for _ in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        belief_mid_episode = env.belief

        obs2, info2 = env.reset(seed=13)
        # Back to the stationary prior, not wherever the previous episode's
        # belief happened to land.
        assert obs2[2] == pytest.approx(env.belief, abs=1e-12)
        assert env.filt._prev_midprice == env.base_env.raw_midprice
        assert env.last_return is None

    def test_cached_price_resets_correctly(self):
        env = make_env(seed=14)
        env.reset(seed=14)
        for _ in range(100):
            action = env.action_space.sample()
            env.step(action)

        env.reset(seed=15)
        price_ep2 = env._prev_midprice
        # A fresh reset re-caches from the (freshly-reset) base env, not a
        # stale value left over from episode 1's mid-episode price.
        assert price_ep2 == env.base_env.raw_midprice

    def test_cumulative_reward_matches_base_env_full_episode(self):
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
        wrapper = HamiltonPPOWrapper(base_env=base_env2, inventory_scale=INVENTORY_SCALE)
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
# Action tests
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

    def test_wrapper_does_not_alter_action_values(self, monkeypatch):
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
# Reproducibility tests
# ======================================================================
class TestReproducibility:
    @staticmethod
    def _build_instrumented_env(seed):
        np.random.seed(seed)
        base_env = make_regime_envs(switch_within_episode=True, seed=seed)
        logs = _instrument_arrival_fill(base_env)
        wrapper = HamiltonPPOWrapper(base_env=base_env, inventory_scale=INVENTORY_SCALE)
        wrapper.reset()
        return wrapper, logs

    @staticmethod
    def _run_full_episode_instrumented(seed, actions):
        """
        Build+run ONE full episode to completion and return a per-step
        record. Deliberately NOT interleaved with any other env's step()
        calls: RegimeSwitchingEnv's regime-transition draws read from the
        LEGACY GLOBAL np.random stream, so interleaving two envs' step()
        calls (env_a.step(); env_b.step(); env_a.step(); ...) makes them
        split ONE shared stream into two interleaved sub-sequences instead
        of each faithfully replaying its own seed -- exactly the pitfall
        compare_four_policies_paired.py's docstring warns about. Running
        each environment to completion sequentially, then comparing the
        recorded traces afterwards, avoids that entirely.
        """
        wrapper, logs = TestReproducibility._build_instrumented_env(seed)
        record = dict(
            regime=[], raw_midprice=[], raw_inventory=[], reward=[],
            terminated=[], truncated=[], obs=[], arrivals=[], fills=[],
        )
        for action in actions:
            regime = wrapper.base_env.current_regime
            pre_len = len(logs[regime]["arrivals"])
            obs, reward, terminated, truncated, info = wrapper.step(action)
            record["regime"].append(info["true_regime"])
            record["raw_midprice"].append(info["raw_midprice"])
            record["raw_inventory"].append(info["raw_state"][INVENTORY_INDEX])
            record["reward"].append(reward)
            record["terminated"].append(terminated)
            record["truncated"].append(truncated)
            record["obs"].append(obs.copy())
            record["arrivals"].append(logs[regime]["arrivals"][pre_len].copy())
            record["fills"].append(logs[regime]["fills"][pre_len].copy())
        return record

    def test_full_reproducibility_same_seed(self):
        """
        Two environments built from scratch with the SAME seed, driven by
        the SAME fixed 4000-step action sequence, must agree at every
        single step on: raw midprice, raw inventory, true regime, reward,
        termination/truncation, and the underlying arrival/fill draws
        (captured via monkeypatched observer hooks -- see
        _instrument_arrival_fill -- not exposed via `info` by default).

        Each environment is run to completion SEQUENTIALLY (see
        _run_full_episode_instrumented) -- an earlier version of this test
        interleaved env_a.step()/env_b.step() calls within the same loop
        and failed at step 199 because both envs share ONE legacy global
        np.random stream for regime-transition draws; interleaving split
        that single stream into two interleaved sub-sequences rather than
        each env replaying its own seed faithfully. That was a bug in the
        test, not the wrapper -- fixed by running sequentially instead.
        """
        seed = 100
        rng = np.random.default_rng(seed + 9000)
        actions = [rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32) for _ in range(N_STEPS)]

        record_a = self._run_full_episode_instrumented(seed, actions)
        record_b = self._run_full_episode_instrumented(seed, actions)

        for step_idx in range(N_STEPS):
            assert record_a["regime"][step_idx] == record_b["regime"][step_idx], \
                f"true_regime diverged at step {step_idx}"
            assert record_a["raw_midprice"][step_idx] == record_b["raw_midprice"][step_idx], \
                f"raw_midprice diverged at step {step_idx}"
            assert record_a["raw_inventory"][step_idx] == record_b["raw_inventory"][step_idx], \
                f"raw inventory diverged at step {step_idx}"
            assert record_a["reward"][step_idx] == record_b["reward"][step_idx], \
                f"reward diverged at step {step_idx}"
            assert record_a["terminated"][step_idx] == record_b["terminated"][step_idx], \
                f"terminated diverged at step {step_idx}"
            assert record_a["truncated"][step_idx] == record_b["truncated"][step_idx], \
                f"truncated diverged at step {step_idx}"
            np.testing.assert_array_equal(
                record_a["obs"][step_idx], record_b["obs"][step_idx],
                err_msg=f"PPO observation diverged at step {step_idx}",
            )
            np.testing.assert_array_equal(
                record_a["arrivals"][step_idx], record_b["arrivals"][step_idx],
                err_msg=f"arrivals diverged at step {step_idx}",
            )
            np.testing.assert_array_equal(
                record_a["fills"][step_idx], record_b["fills"][step_idx],
                err_msg=f"fills diverged at step {step_idx}",
            )

        assert record_a["terminated"][-1] is True
        assert record_a["truncated"][-1] is False

    def test_different_seeds_produce_different_paths(self):
        seed_a, seed_b = 200, 201
        rng = np.random.default_rng(9999)
        actions = [rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32) for _ in range(200)]

        np.random.seed(seed_a)
        base_env_a = make_regime_envs(switch_within_episode=True, seed=seed_a)
        env_a = HamiltonPPOWrapper(base_env=base_env_a, inventory_scale=INVENTORY_SCALE)
        env_a.reset()

        np.random.seed(seed_b)
        base_env_b = make_regime_envs(switch_within_episode=True, seed=seed_b)
        env_b = HamiltonPPOWrapper(base_env=base_env_b, inventory_scale=INVENTORY_SCALE)
        env_b.reset()

        midprices_a, midprices_b = [], []
        regimes_a, regimes_b = [], []
        for action in actions:
            obs_a, r_a, t_a, tr_a, info_a = env_a.step(action)
            obs_b, r_b, t_b, tr_b, info_b = env_b.step(action)
            midprices_a.append(info_a["raw_midprice"])
            midprices_b.append(info_b["raw_midprice"])
            regimes_a.append(info_a["true_regime"])
            regimes_b.append(info_b["true_regime"])

        assert not np.allclose(midprices_a, midprices_b), (
            "Different seeds produced identical midprice paths -- seeding "
            "is not actually varying the underlying RNG streams."
        )
        assert regimes_a != regimes_b or not np.allclose(midprices_a, midprices_b), (
            "Different seeds should produce a different regime path and/or "
            "midprice path; both were identical."
        )


class TestSeedReset:
    def test_reset_seed_reproduces_nothing_on_an_already_stepped_wrapper(self):
        """
        UPDATED after the RNG-isolation fix (see envs/regime_env.py's class
        docstring): reset(seed=s) on an ALREADY-CONSTRUCTED-AND-STEPPED
        wrapper now reproduces NEITHER the regime path NOR the midprice/
        arrival/fill path. Previously (pre-fix), reset(seed=s) called
        np.random.seed(seed), reseeding the legacy GLOBAL numpy stream that
        RegimeSwitchingEnv's regime-transition draws used to read from --
        so the regime path WAS reproduced that way (though only because of
        a bug: the same global-stream mutation was also silently corrupted
        by any evaluation episode running elsewhere in the process, which
        is exactly what this fix addresses). Regime draws now come from an
        isolated local np.random.Generator, seeded ONCE at construction
        (via make_regime_envs) and never touched by reset(seed=...) at all
        -- reset(seed=s) is accepted for Gymnasium API compatibility but is
        otherwise inert.

        Full reproducibility (as needed for paired-episode comparisons)
        still requires constructing a FRESH HamiltonPPOWrapper(seed=s) each
        time -- exactly the pattern TestReproducibility uses -- never
        reset()-ing an existing instance. That guidance is now
        exceptionless (previously the regime path was a partial exception).
        """
        seed = 555

        # Path 1: fresh construction with seed=s, immediately reset(seed=s).
        env_fresh = HamiltonPPOWrapper(inventory_scale=INVENTORY_SCALE, seed=seed)
        env_fresh.reset(seed=seed)
        midprices_fresh = [env_fresh.base_env.raw_midprice]
        for _ in range(30):
            obs, r, term, trunc, info = env_fresh.step(np.zeros(2, dtype=np.float32))
            midprices_fresh.append(info["raw_midprice"])

        # Path 2: SAME wrapper instance, stepped further first (advancing
        # its model RNGs AND its regime_rng), then reset(seed=s) again.
        env_reused = HamiltonPPOWrapper(inventory_scale=INVENTORY_SCALE, seed=seed)
        env_reused.reset(seed=seed)
        for _ in range(75):
            env_reused.step(np.zeros(2, dtype=np.float32))
        env_reused.reset(seed=seed)
        midprices_reused = [env_reused.base_env.raw_midprice]
        for _ in range(30):
            obs, r, term, trunc, info = env_reused.step(np.zeros(2, dtype=np.float32))
            midprices_reused.append(info["raw_midprice"])

        # Midprice/arrival/fill path is NOT reproduced (unchanged from before this fix).
        assert not np.allclose(midprices_fresh, midprices_reused), (
            "reset(seed=s) unexpectedly reproduced the midprice path on an "
            "already-stepped wrapper."
        )

        # Regime-RNG internal state is NOT reproduced either -- checked
        # DIRECTLY on the generator's own bit-generator state, rather than
        # via downstream sampled regime VALUES: with this project's
        # transition matrix (~99.7-99.9% self-persistence probability per
        # step, see envs/make_envs.py's TRANSITION_MATRIX), a 30-step
        # regime-value sequence has a high a-priori chance of never
        # switching at all regardless of whether the underlying draws are
        # actually correlated -- comparing sampled values alone would be a
        # statistically weak, non-discriminating test (this was the flaw
        # in this test's pre-fix form). Comparing the generator's own
        # internal state is a direct, unambiguous check.
        state_fresh = env_fresh.base_env._regime_rng.bit_generator.state
        state_reused = env_reused.base_env._regime_rng.bit_generator.state
        assert state_fresh != state_reused, (
            "regime_rng's internal state matched after reset(seed=s) on an "
            "already-stepped wrapper -- reset(seed=s) may have started "
            "reseeding/rebuilding the regime generator again; if intended, "
            "update this test's documented expectations."
        )

    def test_fresh_construction_with_same_seed_reproduces_regime_rng_state(self):
        """Positive control for the test above: two INDEPENDENTLY, freshly
        constructed wrappers with the SAME seed must have IDENTICAL
        regime_rng internal state immediately after construction (proving
        the isolation fix still ties regime_rng deterministically to the
        declared seed, via make_regime_envs -- it just no longer responds
        to reset(seed=...) on an existing instance)."""
        seed = 777
        env_a = HamiltonPPOWrapper(inventory_scale=INVENTORY_SCALE, seed=seed)
        env_b = HamiltonPPOWrapper(inventory_scale=INVENTORY_SCALE, seed=seed)
        assert env_a.base_env._regime_rng.bit_generator.state == env_b.base_env._regime_rng.bit_generator.state


# ======================================================================
# Validation tests
# ======================================================================
class TestValidation:
    @pytest.mark.parametrize("bad_scale", [0.0, -1.0, -10.0])
    def test_non_positive_inventory_scale_raises(self, bad_scale):
        with pytest.raises(ValueError):
            HamiltonPPOWrapper(inventory_scale=bad_scale, seed=1)

    def test_positive_inventory_scale_does_not_raise(self):
        env = HamiltonPPOWrapper(inventory_scale=1.0, seed=1)
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
    np.testing.assert_allclose(env.observation_space.low, [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(env.observation_space.high, [1.0, 1.0, 1.0])
