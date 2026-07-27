"""
test_rng_isolation.py
------------------------
Regression coverage for the RNG-isolation audit and fix.

Root cause (fixed): RegimeSwitchingEnv's regime-transition draws
(step()'s and _sample_initial_regime()'s np.random.choice(...)) used the
LEGACY GLOBAL np.random API. HamiltonPPOWrapper/ReturnPPOWrapper's
reset(seed=...) called np.random.seed(seed), mutating that SAME
process-global state. Since PeriodicEvalCallback runs full evaluation
episodes (each constructing a wrapper and calling reset(seed=eval_seed))
interleaved with ongoing training, every periodic evaluation silently
corrupted the training environment's subsequent regime-transition path --
proven empirically (see the audit report) to make two training runs with
different eval_freq values diverge in policy weights despite identical
seeds and hyperparameters, from the very first rollout after an
evaluation fires.

Fix: RegimeSwitchingEnv now owns an explicit, independent
np.random.Generator (self._regime_rng) for regime draws, constructed once
via make_regime_envs()'s existing SeedSequence(seed).spawn() pattern
(extended from 6 to 7 children -- verified not to change the existing 6
midprice/arrival/fill child seeds). reset(seed=...) on the wrappers no
longer touches any global RNG state at all.

Run from repo root:
    pytest tests/test_rng_isolation.py -v
"""

import random

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.make_envs import make_regime_envs
from envs.hamilton_ppo_wrapper import HamiltonPPOWrapper
from envs.return_ppo_wrapper import ReturnPPOWrapper, DEFAULT_RETURN_SCALE
from train_agents import PeriodicEvalCallback, build_train_env, evaluate_policy


# ======================================================================
# SeedSequence.spawn() backward-compatibility (the numeric property the
# fix's child-seed-count change from 6 to 7 relies on)
# ======================================================================
def test_spawn_seven_children_matches_spawn_six_for_first_six():
    """Extending the number of spawned children from 6 to 7 must not
    change the first 6 children's own derived RNG streams -- otherwise
    every existing midprice/arrival/fill realisation for a given seed
    would silently change too."""
    seed = 4242
    six = np.random.SeedSequence(seed).spawn(6)
    seven = np.random.SeedSequence(seed).spawn(7)
    for i in range(6):
        a = np.random.default_rng(six[i]).standard_normal(20)
        b = np.random.default_rng(seven[i]).standard_normal(20)
        np.testing.assert_array_equal(a, b)


# ======================================================================
# regime_rng isolation from the legacy global np.random API
# ======================================================================
class TestRegimeRngIsolation:
    def test_regime_rng_is_a_local_generator(self):
        env = make_regime_envs(switch_within_episode=True, seed=1)
        assert isinstance(env._regime_rng, np.random.Generator)

    def test_external_np_random_seed_does_not_affect_regime_path(self):
        """The core regression: calling np.random.seed(...) from OUTSIDE
        the environment (exactly what evaluation used to do via
        wrapper.reset(seed=...)) must not perturb an already-constructed
        env's subsequent regime draws at all."""
        env_a = make_regime_envs(switch_within_episode=True, seed=99)
        env_a.reset()
        np.random.seed(111)  # simulates an unrelated evaluation episode's reset(seed=...)
        regimes_a = []
        for _ in range(200):
            env_a.step(np.zeros((1, 2), dtype=np.float32))
            regimes_a.append(env_a.regime)

        env_b = make_regime_envs(switch_within_episode=True, seed=99)
        env_b.reset()
        np.random.seed(999)  # a DIFFERENT external reseed, before any stepping
        regimes_b = []
        for _ in range(200):
            env_b.step(np.zeros((1, 2), dtype=np.float32))
            regimes_b.append(env_b.regime)

        assert regimes_a == regimes_b, (
            "Regime path differed depending on an UNRELATED external "
            "np.random.seed() call made between construction and stepping -- "
            "regime_rng is not properly isolated from the global stream."
        )

    def test_same_seed_gives_identical_regime_rng_state_at_construction(self):
        env_a = make_regime_envs(switch_within_episode=True, seed=555)
        env_b = make_regime_envs(switch_within_episode=True, seed=555)
        assert env_a._regime_rng.bit_generator.state == env_b._regime_rng.bit_generator.state

    def test_different_seeds_give_different_regime_rng_state(self):
        env_a = make_regime_envs(switch_within_episode=True, seed=1)
        env_b = make_regime_envs(switch_within_episode=True, seed=2)
        assert env_a._regime_rng.bit_generator.state != env_b._regime_rng.bit_generator.state

    def test_seed_method_reseeds_regime_rng(self):
        """.seed(s) reseeds regime_rng directly as np.random.default_rng(s)
        (same convention as TradingEnvironment.seed()) -- a different,
        simpler derivation than make_regime_envs()'s SeedSequence(seed)
        .spawn() scheme, not expected to match it. What IS expected:
        calling .seed(s) is self-consistent -- the SAME s always produces
        the SAME resulting state, and advances/consumes it during
        stepping like any other reseed."""
        env = make_regime_envs(switch_within_episode=True, seed=1)
        env.step(np.zeros((1, 2), dtype=np.float32))  # advance regime_rng away from its initial state
        env.seed(42)
        state_after_first_reseed = env._regime_rng.bit_generator.state

        env.step(np.zeros((1, 2), dtype=np.float32))  # advance again
        env.seed(42)
        state_after_second_reseed = env._regime_rng.bit_generator.state

        assert state_after_first_reseed == state_after_second_reseed
        assert state_after_first_reseed == np.random.default_rng(42).bit_generator.state


# ======================================================================
# reset(seed=...) is now inert (no longer calls np.random.seed)
# ======================================================================
class TestWrapperResetNoLongerTouchesGlobalRng:
    def test_hamilton_wrapper_reset_does_not_call_np_random_seed(self, monkeypatch):
        calls = []
        original_seed = np.random.seed

        def spy_seed(*a, **kw):
            calls.append((a, kw))
            return original_seed(*a, **kw)

        monkeypatch.setattr(np.random, "seed", spy_seed)
        env = HamiltonPPOWrapper(inventory_scale=10.0, seed=1)
        env.reset(seed=123)
        assert calls == [], f"reset(seed=...) called np.random.seed: {calls}"

    def test_return_wrapper_reset_does_not_call_np_random_seed(self, monkeypatch):
        calls = []
        original_seed = np.random.seed

        def spy_seed(*a, **kw):
            calls.append((a, kw))
            return original_seed(*a, **kw)

        monkeypatch.setattr(np.random, "seed", spy_seed)
        env = ReturnPPOWrapper(inventory_scale=10.0, return_scale=DEFAULT_RETURN_SCALE, seed=1)
        env.reset(seed=123)
        assert calls == [], f"reset(seed=...) called np.random.seed: {calls}"


# ======================================================================
# Evaluation no longer mutates global RNG state (direct, real evaluation)
# ======================================================================
def test_evaluation_episode_does_not_mutate_global_rng_state():
    def _init():
        return Monitor(build_train_env("return_mlp_ppo", 0, 10.0, 0.005))

    vec_env = DummyVecEnv([_init])
    model = PPO("MlpPolicy", vec_env, n_steps=8, batch_size=8,
                policy_kwargs=dict(net_arch=[8, 8]), device="cpu", seed=0, verbose=0)

    np_before = np.random.get_state()
    py_before = random.getstate()
    torch_before = torch.get_rng_state().clone()

    evaluate_policy(model, "return_mlp_ppo", [130_050], 10.0, 0.005, deterministic=True)

    np_after = np.random.get_state()
    py_after = random.getstate()
    torch_after = torch.get_rng_state()

    assert np_before[1].tobytes() == np_after[1].tobytes() and np_before[2:] == np_after[2:], (
        "evaluate_policy mutated the legacy global numpy RNG state."
    )
    assert py_before == py_after, "evaluate_policy mutated python's global random state."
    assert torch.equal(torch_before, torch_after), "evaluate_policy mutated torch's global RNG state."


# ======================================================================
# End-to-end regression: training reproducible regardless of periodic-
# eval interleaving (small-scale, fast version of the audit's A/B/C proof)
# ======================================================================
def _train_small(eval_freq, seed=0):
    def _init():
        return Monitor(build_train_env("return_mlp_ppo", seed, 10.0, 0.005))

    vec_env = DummyVecEnv([_init])
    model = PPO(
        "MlpPolicy", vec_env, n_steps=8, batch_size=8, n_epochs=2,
        policy_kwargs=dict(net_arch=[8, 8]), device="cpu", seed=seed, verbose=0,
    )
    callbacks = []
    if eval_freq is not None:
        callbacks.append(PeriodicEvalCallback(
            agent_type="return_mlp_ppo", eval_seeds=[130_051], inventory_scale=10.0, return_scale=0.005,
            eval_freq=eval_freq, training_curve=[], verbose=0,
        ))
    model.learn(total_timesteps=24, callback=callbacks)
    return next(p for p in model.policy.parameters() if p.dim() >= 2).detach().numpy().copy()


def test_training_bit_identical_regardless_of_eval_freq():
    """The exact regression this whole fix targets, at small scale for
    speed: three 24-env-step trainings (3 rollouts of n_steps=8), seed=0,
    differing only in eval_freq (disabled / 8 / 16) -- final weights must
    be bit-identical."""
    w_disabled = _train_small(eval_freq=None)
    w_freq8 = _train_small(eval_freq=8)
    w_freq16 = _train_small(eval_freq=16)

    np.testing.assert_array_equal(w_disabled, w_freq8)
    np.testing.assert_array_equal(w_disabled, w_freq16)
