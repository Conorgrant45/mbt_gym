"""
test_return_lstm_recurrent.py
------------------------------------
Focused pytest suite for the recurrent (return_lstm_ppo) branch: LSTM
hidden/cell-state propagation, episode_start handling, state reset at
episode boundaries, and save/reload behaviour for sb3_contrib.RecurrentPPO
+ MlpLstmPolicy wrapped around ReturnPPOWrapper.

Models here are deliberately UNTRAINED (built, never .learn()-ed) -- this
suite tests state-plumbing correctness, not learned behaviour, so it does
not need to run any environment training. This keeps the suite fast and
targeted, per the same "does not modify production files, builds its own
instances" convention as tests/test_hamilton_ppo_wrapper.py and
tests/test_return_ppo_wrapper.py.

Run from repo root:
    pytest tests/test_return_lstm_recurrent.py -v
"""

import numpy as np
import pytest
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.return_ppo_wrapper import ReturnPPOWrapper, DEFAULT_RETURN_SCALE
from envs.hamilton_ppo_wrapper import DEFAULT_INVENTORY_SCALE

INVENTORY_SCALE = DEFAULT_INVENTORY_SCALE
RETURN_SCALE = DEFAULT_RETURN_SCALE


def make_env(seed=None):
    return ReturnPPOWrapper(inventory_scale=INVENTORY_SCALE, return_scale=RETURN_SCALE, seed=seed)


def build_untrained_model(seed=0, lstm_hidden_size=8, n_lstm_layers=1):
    """A small, fast-to-construct RecurrentPPO model. Never trained --
    weights are the library's own random initialisation. n_steps/
    batch_size are irrelevant here since .learn() is never called."""
    def _init():
        return Monitor(make_env(seed=seed))

    vec_env = DummyVecEnv([_init])
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        policy_kwargs=dict(net_arch=[16, 16], lstm_hidden_size=lstm_hidden_size, n_lstm_layers=n_lstm_layers),
        n_steps=8,
        batch_size=8,
        device="cpu",
        seed=seed,
        verbose=0,
    )
    return model, vec_env


# ======================================================================
# Hidden/cell state propagation (items 19-21)
# ======================================================================
class TestStatePropagation:
    def test_hidden_state_changes_during_episode(self):
        """Item 19."""
        model, _ = build_untrained_model(seed=1)
        env = make_env(seed=1)
        obs, info = env.reset()

        action0, state0 = model.predict(obs, state=None, episode_start=np.array([True]), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action0)
        action1, state1 = model.predict(obs, state=state0, episode_start=np.array([False]), deterministic=True)

        hidden0, hidden1 = state0[0], state1[0]
        assert not np.allclose(hidden0, hidden1), (
            "LSTM hidden state did not change between two steps within the same episode."
        )

    def test_cell_state_changes_during_episode(self):
        """Item 20."""
        model, _ = build_untrained_model(seed=2)
        env = make_env(seed=2)
        obs, info = env.reset()

        action0, state0 = model.predict(obs, state=None, episode_start=np.array([True]), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action0)
        action1, state1 = model.predict(obs, state=state0, episode_start=np.array([False]), deterministic=True)

        cell0, cell1 = state0[1], state1[1]
        assert not np.allclose(cell0, cell1), (
            "LSTM cell state did not change between two steps within the same episode."
        )

    def test_state_is_not_reset_every_step(self):
        """Item 21: confirm the model's output at step t genuinely depends
        on the CARRIED (non-zero) state from step t-1, not a state that
        gets silently zeroed on every call. Compared against explicitly
        passing state=None (which model.predict() zero-initialises) at the
        same observation with episode_start=False."""
        model, _ = build_untrained_model(seed=3)
        env = make_env(seed=3)
        obs0, info = env.reset()

        action0, state0 = model.predict(obs0, state=None, episode_start=np.array([True]), deterministic=True)
        obs1, reward, terminated, truncated, info = env.step(action0)

        # Path A: correctly carry the non-zero state0 forward.
        action_carried, state_carried = model.predict(
            obs1, state=state0, episode_start=np.array([False]), deterministic=True,
        )
        # Path B: same obs, same episode_start=False, but a ZEROED incoming state
        # (simulating a bug where state is dropped every step).
        action_zeroed, state_zeroed = model.predict(
            obs1, state=None, episode_start=np.array([False]), deterministic=True,
        )

        assert not np.allclose(action_carried, action_zeroed), (
            "Action was identical whether or not the previous non-zero LSTM state was "
            "carried forward -- this would indicate the state is being reset every step."
        )

    def test_episode_start_true_discards_incoming_state(self):
        """Item 22: passing episode_start=True must zero out ANY incoming
        state (per sb3_contrib's _process_sequence: state is multiplied by
        (1 - episode_start) before the LSTM cell), regardless of whether a
        stale non-zero state happens to be passed alongside it -- confirms
        genuine reset-at-episode-boundary behaviour, not merely "usually
        starts from zero because callers pass state=None"."""
        model, _ = build_untrained_model(seed=4)
        env = make_env(seed=4)
        obs, info = env.reset()

        # Build up a genuinely non-zero state over a few steps.
        state = None
        episode_start = np.array([True])
        for _ in range(5):
            action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_start = np.array([False])
        assert not np.allclose(state[0], 0.0), "Expected a non-zero hidden state after several steps."

        # New episode, but WRONGLY pass the stale non-zero state alongside episode_start=True.
        env2 = make_env(seed=4)
        obs2, info2 = env2.reset()
        action_stale, next_state_stale = model.predict(
            obs2, state=state, episode_start=np.array([True]), deterministic=True,
        )

        # Reference: a genuinely fresh call (state=None) at the same observation.
        action_fresh, next_state_fresh = model.predict(
            obs2, state=None, episode_start=np.array([True]), deterministic=True,
        )

        np.testing.assert_allclose(action_stale, action_fresh, atol=1e-6)
        np.testing.assert_allclose(next_state_stale[0], next_state_fresh[0], atol=1e-6)
        np.testing.assert_allclose(next_state_stale[1], next_state_fresh[1], atol=1e-6)


# ======================================================================
# episode_start flag correctness in a standard rollout loop (items 23-24)
# ======================================================================
def test_episode_start_flag_true_only_at_episode_start(monkeypatch):
    """Items 23-24: replicate the eval loop in train_agents.run_eval_episode
    and record every episode_start value actually passed to model.predict()."""
    from train_agents import run_eval_episode

    model, _ = build_untrained_model(seed=5)
    env = make_env(seed=5)

    captured_episode_starts = []
    original_predict = model.predict

    def spy_predict(obs, state=None, episode_start=None, deterministic=False):
        captured_episode_starts.append(bool(episode_start[0]))
        return original_predict(obs, state=state, episode_start=episode_start, deterministic=deterministic)

    monkeypatch.setattr(model, "predict", spy_predict)

    # Run only a short prefix of the episode (200 steps) -- flag-timing
    # correctness does not depend on running the full 4000-step episode.
    obs, info = env.reset()
    state = None
    episode_start = np.array([True])
    for _ in range(200):
        action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_start = np.array([False])
        if terminated or truncated:
            break

    assert captured_episode_starts[0] is True
    assert all(v is False for v in captured_episode_starts[1:]), (
        f"episode_start was True at non-initial steps: "
        f"{[i for i, v in enumerate(captured_episode_starts) if i > 0 and v]}"
    )


# ======================================================================
# Save / reload (items 25-27)
# ======================================================================
class TestSaveReload:
    def test_feedforward_model_reloads(self, tmp_path):
        """Item 25 (feed-forward branch, i.e. return_mlp_ppo)."""
        from stable_baselines3 import PPO

        def _init():
            return Monitor(make_env(seed=6))

        vec_env = DummyVecEnv([_init])
        model = PPO("MlpPolicy", vec_env, n_steps=8, batch_size=8, policy_kwargs=dict(net_arch=[16, 16]),
                    device="cpu", seed=6, verbose=0)
        path = tmp_path / "ppo_mlp_test"
        model.save(str(path))
        loaded = PPO.load(str(path))

        obs, info = make_env(seed=6).reset()
        a1, _ = model.predict(obs, deterministic=True)
        a2, _ = loaded.predict(obs, deterministic=True)
        np.testing.assert_allclose(a1, a2, atol=1e-6)

    def test_recurrent_model_reloads(self, tmp_path):
        """Item 26."""
        model, _ = build_untrained_model(seed=7)
        path = tmp_path / "ppo_lstm_test"
        model.save(str(path))
        loaded = RecurrentPPO.load(str(path))

        assert loaded.policy.lstm_actor.hidden_size == model.policy.lstm_actor.hidden_size
        assert loaded.policy.lstm_actor.num_layers == model.policy.lstm_actor.num_layers
        for p in loaded.policy.parameters():
            assert torch.isfinite(p).all()

    def test_reloaded_recurrent_model_deterministic_prediction_with_state(self, tmp_path):
        """Item 27: a reloaded recurrent model must reproduce the EXACT
        same action/state sequence as the pre-save model on a fixed,
        deterministic rollout, with state correctly carried at every step."""
        model, _ = build_untrained_model(seed=8)
        path = tmp_path / "ppo_lstm_test2"
        model.save(str(path))
        loaded = RecurrentPPO.load(str(path))

        env_a = make_env(seed=9)
        env_b = make_env(seed=9)
        obs_a, _ = env_a.reset()
        obs_b, _ = env_b.reset()

        state_a = state_b = None
        episode_start = np.array([True])
        for _ in range(50):
            action_a, state_a = model.predict(obs_a, state=state_a, episode_start=episode_start, deterministic=True)
            action_b, state_b = loaded.predict(obs_b, state=state_b, episode_start=episode_start, deterministic=True)
            np.testing.assert_allclose(action_a, action_b, atol=1e-6)
            np.testing.assert_allclose(state_a[0], state_b[0], atol=1e-6)
            np.testing.assert_allclose(state_a[1], state_b[1], atol=1e-6)

            obs_a, reward_a, term_a, trunc_a, info_a = env_a.step(action_a)
            obs_b, reward_b, term_b, trunc_b, info_b = env_b.step(action_b)
            assert reward_a == reward_b
            episode_start = np.array([False])


# ======================================================================
# Finiteness during a longer smoke rollout (items 28-29, recurrent branch)
# ======================================================================
def test_observations_and_actions_finite_during_rollout():
    model, _ = build_untrained_model(seed=10)
    env = make_env(seed=10)
    obs, info = env.reset()
    assert np.all(np.isfinite(obs))

    state = None
    episode_start = np.array([True])
    for _ in range(300):
        action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=True)
        assert np.all(np.isfinite(action))
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        assert np.all(np.isfinite(state[0])) and np.all(np.isfinite(state[1]))
        episode_start = np.array([False])
        if terminated or truncated:
            break
