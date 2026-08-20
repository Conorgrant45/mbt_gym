"""
test_seed_separation.py
------------------------
Regression coverage for the learner/environment seed separation added to
train_agents.py.

Background: train_agents.py used to conflate two independent sources of
randomness under a single --seed value: PPO/RecurrentPPO's own
construction-time seed (policy/value-network initialisation, action
sampling, minibatch ordering, and the torch/numpy/random global state SB3
seeds internally -- see stable_baselines3.common.utils.set_random_seed)
AND the exogenous training environment's seed (initial regime,
regime-transition draws, Brownian increments, arrivals, jumps, fill draws
-- see envs/make_envs.py's make_regime_envs). Conflating them made it
impossible to hold the environment's realised trajectory fixed while
varying only the learner's initialisation/sampling behaviour (or vice
versa) -- exactly the controlled comparison a seed-robustness study needs.

Fix: --learner-seed and --env-seed (train_agents.py CLI), resolved by the
pure function resolve_seeds(args) (--seed retained for backwards
compatibility, resolved as both learner_seed and env_seed when neither of
the new flags is given). learner_seed is passed to PPO/RecurrentPPO's own
seed=... kwarg; env_seed is passed to build_train_env()/
make_regime_envs(seed=env_seed). The two never interact: env construction
already draws from isolated local np.random.Generator streams (see
tests/test_rng_isolation.py), and reset(seed=...) on either PPO wrapper is
inert (same prior fix) -- so nothing learner_seed touches (directly, or via
PPO's own set_random_seed(), including the VecEnv.seed(learner_seed) call
buried inside it) can reach the environment's RNG, in either direction.

Run from repo root:
    pytest tests/test_seed_separation.py -v
"""

import argparse
import hashlib
import io
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

import shared.train_agents as train_agents
from shared.train_agents import resolve_seeds, build_train_env, PeriodicEvalCallback, evaluate_policy


def _ns(**kw):
    """Minimal argparse.Namespace stand-in for resolve_seeds(), with only
    the three seed-related attributes it reads."""
    base = dict(seed=None, learner_seed=None, env_seed=None)
    base.update(kw)
    return argparse.Namespace(**base)


def hash_state_dict(state_dict) -> str:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]


def _scripted_actions(n, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1.0, 1.0, size=(1, 2)).astype(np.float32) for _ in range(n)]


def _run_scripted(base_env, actions):
    """Step base_env with a FIXED action sequence, recording the purely
    exogenous trace (regime + midprice -- neither depends on the action,
    see envs/regime_env.py's step() and evaluate_agents_common.py's own
    verify_path_pairing()) separately from inventory/reward (which DO
    depend on the action)."""
    base_env.reset()
    regimes, midprices, inventories, rewards = [], [], [], []
    for a in actions:
        obs, reward, done, info = base_env.step(a)
        regimes.append(info["true_regime"])
        midprices.append(info["raw_midprice"])
        inventories.append(float(info["raw_state"][1]))
        rewards.append(float(np.sum(reward)))
    return regimes, midprices, inventories, rewards


# ======================================================================
# Item 8: resolve_seeds() -- pure-function resolution logic
# ======================================================================
class TestResolveSeeds:
    def test_seed_only_resolves_to_both(self):
        """The old command using only --seed remains equivalent to setting
        --learner-seed S --env-seed S."""
        assert resolve_seeds(_ns(seed=7)) == (7, 7)

    def test_learner_and_env_seed_given_directly(self):
        assert resolve_seeds(_ns(learner_seed=3, env_seed=9)) == (3, 9)

    def test_learner_seed_only_defaults_env_seed_to_it(self):
        assert resolve_seeds(_ns(learner_seed=11)) == (11, 11)

    def test_seed_and_learner_seed_agreeing_is_fine(self):
        assert resolve_seeds(_ns(seed=4, learner_seed=4)) == (4, 4)

    def test_seed_and_learner_seed_conflicting_raises(self):
        with pytest.raises(ValueError):
            resolve_seeds(_ns(seed=4, learner_seed=5))

    def test_env_seed_alone_without_learner_seed_or_seed_raises(self):
        with pytest.raises(ValueError):
            resolve_seeds(_ns(env_seed=1))

    def test_nothing_given_at_all_raises(self):
        with pytest.raises(ValueError):
            resolve_seeds(_ns())


# ======================================================================
# Item 1: same env_seed + same scripted actions -> bit-identical exogenous
# trace across two SEPARATELY constructed environments.
# ======================================================================
def test_same_env_seed_same_actions_bit_identical_exogenous_trace():
    actions = _scripted_actions(50)
    env_a = build_train_env("return_mlp_ppo", 42, 10.0, 0.005).base_env
    env_b = build_train_env("return_mlp_ppo", 42, 10.0, 0.005).base_env

    regimes_a, mids_a, _, _ = _run_scripted(env_a, actions)
    regimes_b, mids_b, _, _ = _run_scripted(env_b, actions)

    assert regimes_a == regimes_b
    np.testing.assert_array_equal(mids_a, mids_b)


# ======================================================================
# Item 2: same env_seed, DIFFERENT learner_seed -- exogenous trace unchanged.
# Actually calls SB3's own set_random_seed() (exactly what PPO(seed=...)
# does internally) with two DIFFERENT learner seeds before building each
# env, to prove env construction genuinely does not read any of the
# python/numpy/torch global state that call mutates.
# ======================================================================
def test_same_env_seed_different_learner_seed_leaves_exogenous_trace_unchanged():
    actions = _scripted_actions(50)

    def _build_after_learner_seed(learner_seed, env_seed):
        set_random_seed(learner_seed)
        return build_train_env("return_mlp_ppo", env_seed, 10.0, 0.005).base_env

    env_a = _build_after_learner_seed(learner_seed=100, env_seed=42)
    env_b = _build_after_learner_seed(learner_seed=999, env_seed=42)

    regimes_a, mids_a, _, _ = _run_scripted(env_a, actions)
    regimes_b, mids_b, _, _ = _run_scripted(env_b, actions)

    assert regimes_a == regimes_b
    np.testing.assert_array_equal(mids_a, mids_b)


# ======================================================================
# Item 3: different env_seed -> different exogenous trace.
# ======================================================================
def test_different_env_seed_gives_different_exogenous_trace():
    actions = _scripted_actions(50)
    env_a = build_train_env("return_mlp_ppo", 1, 10.0, 0.005).base_env
    env_b = build_train_env("return_mlp_ppo", 2, 10.0, 0.005).base_env

    regimes_a, mids_a, _, _ = _run_scripted(env_a, actions)
    regimes_b, mids_b, _, _ = _run_scripted(env_b, actions)

    assert regimes_a != regimes_b or not np.array_equal(mids_a, mids_b)


# ======================================================================
# Item 6: different policies (here: different scripted action sequences)
# facing the SAME exogenous stream may have different fills/inventories/
# rewards -- this must not be reported as an exogenous path mismatch.
# ======================================================================
def test_different_actions_same_env_seed_diverge_only_in_action_dependent_quantities():
    actions_a = _scripted_actions(50, seed=0)
    actions_b = _scripted_actions(50, seed=12345)
    assert not all(np.array_equal(a, b) for a, b in zip(actions_a, actions_b)), \
        "test setup bug: scripted action sequences must actually differ"

    env_a = build_train_env("return_mlp_ppo", 7, 10.0, 0.005).base_env
    env_b = build_train_env("return_mlp_ppo", 7, 10.0, 0.005).base_env

    regimes_a, mids_a, inv_a, rew_a = _run_scripted(env_a, actions_a)
    regimes_b, mids_b, inv_b, rew_b = _run_scripted(env_b, actions_b)

    # Exogenous (action-independent) -- must match exactly.
    assert regimes_a == regimes_b
    np.testing.assert_array_equal(mids_a, mids_b)
    # Action-dependent -- expected to differ (not itself a failure, and NOT
    # evidence of an exogenous-path mismatch).
    assert inv_a != inv_b or rew_a != rew_b


# ======================================================================
# Items 4/5: end-to-end training reproducibility given the new seed pair.
# ======================================================================
def _train_tiny(learner_seed, env_seed, eval_freq=None):
    wrapped = build_train_env("return_mlp_ppo", env_seed, 10.0, 0.005)
    vec_env = DummyVecEnv([lambda: Monitor(wrapped)])
    model = PPO(
        "MlpPolicy", vec_env, n_steps=8, batch_size=8, n_epochs=2,
        policy_kwargs=dict(net_arch=[8, 8]), device="cpu", seed=learner_seed, verbose=0,
    )
    initial_hash = hash_state_dict(model.policy.state_dict())

    callbacks = []
    if eval_freq is not None:
        callbacks.append(PeriodicEvalCallback(
            agent_type="return_mlp_ppo", eval_seeds=[130_090], inventory_scale=10.0, return_scale=0.005,
            eval_freq=eval_freq, training_curve=[], verbose=0,
        ))
    model.learn(total_timesteps=24, callback=callbacks)
    final_hash = hash_state_dict(model.policy.state_dict())
    exogenous_final_state = wrapped.base_env._regime_rng.bit_generator.state

    eval_summary, _ = evaluate_policy(model, "return_mlp_ppo", [130_091], 10.0, 0.005, deterministic=True)
    return initial_hash, final_hash, exogenous_final_state, eval_summary["mean_cumulative_reward"]


class TestEndToEndSeedReproducibility:
    def test_same_learner_and_env_seed_bit_identical(self):
        """Item 4: same env_seed AND learner_seed -> identical initial and
        final policy hashes, identical exogenous training trace (the
        training environment's own regime_rng end state) and identical
        evaluation result -- with periodic evaluation interleaved
        (eval_freq=8), the exact scenario the original RNG-isolation bug
        broke."""
        init_a, final_a, exo_a, eval_a = _train_tiny(learner_seed=0, env_seed=42, eval_freq=8)
        init_b, final_b, exo_b, eval_b = _train_tiny(learner_seed=0, env_seed=42, eval_freq=8)
        assert init_a == init_b
        assert final_a == final_b
        assert exo_a == exo_b
        assert eval_a == eval_b

    def test_same_env_seed_different_learner_seed_different_initial_policy_hash(self):
        """Item 5."""
        init_a, _, exo_a, _ = _train_tiny(learner_seed=1, env_seed=42, eval_freq=None)
        init_b, _, exo_b, _ = _train_tiny(learner_seed=2, env_seed=42, eval_freq=None)
        assert init_a != init_b
        # Bonus, restating item 2 in this end-to-end harness: same env_seed
        # -> same exogenous trace regardless of learner_seed.
        assert exo_a == exo_b


# ======================================================================
# Item 7: evaluation callbacks do not mutate the TRAINING environment's own
# RNG state (a direct, environment-object-level check -- complementary to
# test_rng_isolation.py's test_evaluation_episode_does_not_mutate_global_
# rng_state, which only checks *global* python/numpy/torch state).
# ======================================================================
def test_periodic_eval_does_not_mutate_training_env_regime_rng():
    env_seed = 55
    learner_seed = 0

    def _build_and_learn(with_eval: bool):
        wrapped = build_train_env("return_mlp_ppo", env_seed, 10.0, 0.005)
        vec_env = DummyVecEnv([lambda: Monitor(wrapped)])
        model = PPO(
            "MlpPolicy", vec_env, n_steps=8, batch_size=8, n_epochs=2,
            policy_kwargs=dict(net_arch=[8, 8]), device="cpu", seed=learner_seed, verbose=0,
        )
        callbacks = []
        if with_eval:
            callbacks.append(PeriodicEvalCallback(
                agent_type="return_mlp_ppo", eval_seeds=[130_092], inventory_scale=10.0, return_scale=0.005,
                eval_freq=8, training_curve=[], verbose=0,
            ))
        state_before = wrapped.base_env._regime_rng.bit_generator.state
        # 2 rollouts of n_steps=8: rollout 1 trains, rollout 2's
        # on_rollout_start() crosses eval_freq=8 and fires exactly one
        # periodic evaluation (see TestPeriodicEvalTiming in
        # tests/test_train_agents.py for the underlying callback-timing
        # proof this reuses).
        model.learn(total_timesteps=16, callback=callbacks)
        state_after = wrapped.base_env._regime_rng.bit_generator.state
        return state_before, state_after

    _, state_after_with_eval = _build_and_learn(with_eval=True)
    _, state_after_without_eval = _build_and_learn(with_eval=False)

    assert state_after_with_eval == state_after_without_eval, (
        "the training environment's regime_rng ended in a different state depending on whether a "
        "periodic evaluation fired during training -- evaluation is mutating training RNG state."
    )


# ======================================================================
# Item 9: both seeds are correctly recorded in run_config/run_summary JSON
# and the training-curve / per-episode-evaluation CSVs.
# ======================================================================
def test_both_seeds_recorded_in_config_summary_and_csvs(tmp_path, monkeypatch):
    output_dir = tmp_path / "models"
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(train_agents, "REPO_ROOT", tmp_path)

    argv = [
        "train_agents.py",
        "--agent-type", "return_mlp_ppo",
        "--learner-seed", "13",
        "--env-seed", "77",
        "--total-timesteps", "16",
        "--n-steps", "8",
        "--batch-size", "8",
        "--n-epochs", "2",
        "--net-arch", "8", "8",
        "--eval-freq", "1000000",
        "--eval-seeds", "130093",
        "--run-tag", "pytest_seedsep",
        "--output-dir", str(output_dir),
        "--log-dir", str(log_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train_agents.main()

    with open(log_dir / "run_config_pytest_seedsep.json") as f:
        config = json.load(f)
    with open(log_dir / "run_summary_pytest_seedsep.json") as f:
        summary = json.load(f)

    assert config["learner_seed"] == 13
    assert config["training_env_seed"] == 77
    assert summary["learner_seed"] == 13
    assert summary["training_env_seed"] == 77

    curve_df = pd.read_csv(tmp_path / "results" / "return_mlp_ppo_training_curve_pytest_seedsep.csv")
    assert (curve_df["learner_seed"] == 13).all()
    assert (curve_df["training_env_seed"] == 77).all()

    episodes_df = pd.read_csv(tmp_path / "results" / "return_mlp_ppo_eval_episodes_pytest_seedsep.csv")
    assert (episodes_df["learner_seed"] == 13).all()
    assert (episodes_df["training_env_seed"] == 77).all()


# ======================================================================
# evaluate_agents_common.py: training_seed (backwards-compat name) now
# means the learner seed; training_env_seed is a new, separate column.
# ======================================================================
def test_evaluate_agents_common_training_env_seed_column():
    import shared.evaluate_agents_common as EAC

    m = dict(
        evaluation_seed=1, raw_pnl=0.0, full_objective=0.0, spread_revenue=0.0,
        adverse_selection_loss=0.0, running_penalty=0.0, terminal_penalty=0.0,
        mean_abs_inventory=0.0, max_abs_inventory=0.0, mean_signed_inventory=0.0,
        terminal_signed_inventory=0.0, terminal_abs_inventory=0.0, fills=0.0,
        bid_action_mean=0.0, ask_action_mean=0.0, bid_action_std=0.0, ask_action_std=0.0,
        mean_quoted_spread=0.0, episode_length=1, reward_reconciliation_error=0.0,
        recurrent_state_reset_verified=None,
    )
    row = EAC.build_row("hamilton_ppo", 13, None, None, "exogenous_paired_fills_not_claimed_paired", m,
                         training_env_seed=77)
    assert row["training_seed"] == 13
    assert row["training_env_seed"] == 77
    assert set(row.keys()) == set(EAC.SCHEMA_COLUMNS)
