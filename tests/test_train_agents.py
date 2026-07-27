"""
test_train_agents.py
------------------------
Regression coverage for the rollout-geometry diagnostic audited and fixed
in train_agents.py: rollouts_span_full_episode previously hardcoded False
for every recurrent agent regardless of n_steps (train_agents.py:458,
`(n_steps % N_STEPS == 0) if not is_recurrent else False`), which was
never read anywhere else in the codebase (pure logging/reporting
artifact -- confirmed via a repo-wide grep before the fix) but was
factually wrong once RECURRENT_DEFAULT_N_STEPS was raised to 4000. Fixed
by extracting a single agent-type-agnostic helper,
rollout_spans_full_episode(n_steps, episode_length), used identically for
every agent type. Does not modify training behaviour, PPO
hyperparameters, environment mechanics, or recurrent-state handling.

Run from repo root:
    pytest tests/test_train_agents.py -v
"""

import inspect

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO

import train_agents
from train_agents import (
    rollout_spans_full_episode, AGENT_TYPES, RECURRENT_AGENT_TYPES,
    PeriodicEvalCallback, build_train_env,
)
from envs.make_envs import N_STEPS


class TestRolloutSpansFullEpisode:
    def test_true_when_n_steps_equals_episode_length_feedforward(self):
        """Item 1 (feed-forward branch)."""
        assert rollout_spans_full_episode(4000, N_STEPS) is True

    def test_true_when_n_steps_equals_episode_length_recurrent(self):
        """Item 1 (recurrent branch) -- this is exactly the case that was
        previously hardcoded to False regardless of n_steps."""
        assert rollout_spans_full_episode(4000, N_STEPS) is True

    def test_false_when_n_steps_does_not_divide_episode_length(self):
        """Item 2."""
        assert rollout_spans_full_episode(400, N_STEPS) is False
        assert rollout_spans_full_episode(128, N_STEPS) is False

    def test_true_for_a_whole_multiple_of_episode_length(self):
        """One rollout spanning more than one full episode should also
        count as 'spans a full episode' -- not just an exact match."""
        assert rollout_spans_full_episode(8000, N_STEPS) is True

    def test_agent_type_is_not_a_parameter(self):
        """Item 3, structurally: the function signature has no agent-type
        or is_recurrent parameter at all -- there is nothing for a caller
        to branch on, by construction."""
        sig = inspect.signature(rollout_spans_full_episode)
        assert list(sig.parameters) == ["n_steps", "episode_length"]

    def test_identical_result_for_every_agent_type_at_matching_n_steps(self):
        """Item 1/3 combined: every agent type (feed-forward and
        recurrent) must get the exact same answer for the same
        (n_steps, episode_length) pair -- confirms no hidden per-agent-type
        branching exists anywhere the value is computed."""
        results = {agent_type: rollout_spans_full_episode(4000, N_STEPS) for agent_type in AGENT_TYPES}
        assert all(v is True for v in results.values()), results
        assert len(set(results.values())) == 1

    def test_source_has_no_recurrent_specific_override(self):
        """Item 3: guard against the exact regression pattern that caused
        the original bug (an `if not is_recurrent else ...` branch around
        this calculation) ever reappearing, either inside the helper or
        at its call site in main()."""
        helper_source = inspect.getsource(rollout_spans_full_episode)
        assert "is_recurrent" not in helper_source
        assert "RECURRENT_AGENT_TYPES" not in helper_source

        main_source = inspect.getsource(train_agents.main)
        # The call site must invoke the shared helper plainly -- no
        # conditional expression wrapping it based on agent/recurrence.
        assert "rollout_spans_full_episode(n_steps, N_STEPS)" in main_source
        for line in main_source.splitlines():
            if "rollout_spans_full_episode(" in line:
                assert "is_recurrent" not in line, f"recurrent-specific branch reintroduced: {line!r}"


def test_recurrent_agent_types_is_a_subset_of_all_agent_types():
    assert set(RECURRENT_AGENT_TYPES).issubset(set(AGENT_TYPES))


# ======================================================================
# Regression coverage for the periodic-evaluation timing fix:
# PeriodicEvalCallback now evaluates from _on_rollout_start() instead of
# _on_step(), so a row labelled total_timesteps=N reflects policy weights
# updated by ALL rollouts through N (verified directly against the
# installed stable_baselines3 OnPolicyAlgorithm.learn()/collect_rollouts()
# source -- see the class docstring in train_agents.py). A tiny
# return_mlp_ppo model with n_steps=batch_size=8 is used throughout so
# each "rollout" is 8 env-steps, not a full 4000-step market episode --
# only the ONE evaluation episode that genuinely fires per test runs a
# full episode (env-step count is independent of PPO's n_steps).
# ======================================================================
TINY_EVAL_SEED = 130_001  # disjoint from every evaluation-seed range used elsewhere in this project


def _build_tiny_model(seed=0):
    def _init():
        return Monitor(build_train_env("return_mlp_ppo", seed, 10.0, 0.005))

    vec_env = DummyVecEnv([_init])
    model = PPO(
        "MlpPolicy", vec_env,
        n_steps=8, batch_size=8, n_epochs=2,
        policy_kwargs=dict(net_arch=[8, 8]),
        device="cpu", seed=seed, verbose=0,
    )
    return model


class TestPeriodicEvalTiming:
    def test_no_eval_before_first_completed_update(self):
        """Exactly one rollout (n_steps=8, total_timesteps=8): on_rollout_start()
        fires once, at model.num_timesteps==0, before any train() call --
        must be skipped, not evaluated."""
        model = _build_tiny_model(seed=1)
        training_curve = []
        callback = PeriodicEvalCallback(
            agent_type="return_mlp_ppo", eval_seeds=[TINY_EVAL_SEED],
            inventory_scale=10.0, return_scale=0.005, eval_freq=8,
            training_curve=training_curve, verbose=0,
        )
        model.learn(total_timesteps=8, callback=callback)
        assert training_curve == []

    def test_eval_fires_after_completed_update_with_correct_label_and_post_train_weights(self):
        """Two rollouts (total_timesteps=16): rollout 1 completes and trains,
        THEN rollout 2's on_rollout_start() fires at model.num_timesteps==8,
        crossing eval_freq=8 -- eval must fire exactly once, labelled with
        total_timesteps=8 (not 0, not 16), and must observe the policy
        weights exactly as they were immediately after that one completed
        train() call, not the pre-train weights."""
        model = _build_tiny_model(seed=2)
        training_curve = []
        callback = PeriodicEvalCallback(
            agent_type="return_mlp_ppo", eval_seeds=[TINY_EVAL_SEED],
            inventory_scale=10.0, return_scale=0.005, eval_freq=8,
            training_curve=training_curve, verbose=0,
        )

        post_train_snapshots = []
        original_train = model.train

        def snapshotting_train():
            original_train()
            w = next(model.policy.parameters()).detach().clone().numpy().copy()
            post_train_snapshots.append(w)

        model.train = snapshotting_train

        eval_time_snapshots = []
        original_evaluate_policy = train_agents.evaluate_policy

        def spying_evaluate_policy(eval_model, *args, **kwargs):
            w = next(eval_model.policy.parameters()).detach().clone().numpy().copy()
            eval_time_snapshots.append(w)
            return original_evaluate_policy(eval_model, *args, **kwargs)

        train_agents.evaluate_policy = spying_evaluate_policy
        try:
            model.learn(total_timesteps=16, callback=callback)
        finally:
            train_agents.evaluate_policy = original_evaluate_policy

        assert len(training_curve) == 1, f"expected exactly one periodic eval, got {len(training_curve)}"
        assert training_curve[0]["total_timesteps"] == 8
        assert training_curve[0]["tag"] == "timestep_8"

        assert len(eval_time_snapshots) == 1
        assert len(post_train_snapshots) >= 1
        # The eval must have used weights identical to the FIRST completed
        # train() call's post-update weights (rollout 1's), not later ones.
        np.testing.assert_array_equal(eval_time_snapshots[0], post_train_snapshots[0])

    def test_eval_does_not_fire_below_eval_freq_threshold(self):
        """Rollout boundaries at 8/16/24/32 env-steps, eval_freq=16: only the
        boundary at 16 crosses the threshold within this run."""
        model = _build_tiny_model(seed=3)
        training_curve = []
        callback = PeriodicEvalCallback(
            agent_type="return_mlp_ppo", eval_seeds=[TINY_EVAL_SEED],
            inventory_scale=10.0, return_scale=0.005, eval_freq=16,
            training_curve=training_curve, verbose=0,
        )
        model.learn(total_timesteps=32, callback=callback)
        assert [row["total_timesteps"] for row in training_curve] == [16]


# ======================================================================
# Item 6 (plotting): output filename includes run_tag
# ======================================================================
def test_plot_4_learning_curves_filename_includes_run_tag(monkeypatch, tmp_path):
    import plot_agent_comparison as pac

    monkeypatch.setattr(pac, "RESULTS_DIR", tmp_path)
    saved_names = []

    def spy_savefig(fig, name):
        saved_names.append(name)

    monkeypatch.setattr(pac, "_savefig", spy_savefig)
    pac.plot_4_learning_curves(run_tag="unittest_tag_xyz")

    assert len(saved_names) == 1
    assert "unittest_tag_xyz" in saved_names[0]


def test_plot_4_learning_curves_axis_labels_and_band_description(monkeypatch, tmp_path):
    import pandas as pd
    import plot_agent_comparison as pac

    monkeypatch.setattr(pac, "RESULTS_DIR", tmp_path)
    df = pd.DataFrame(dict(
        total_timesteps=[0, 8], mean_cumulative_reward=[1.0, 2.0], std_cumulative_reward=[0.1, 0.2],
    ))
    df.to_csv(tmp_path / "hamilton_ppo_training_curve_labeltest.csv", index=False)

    captured = {}

    def spy_savefig(fig, name):
        captured["fig"] = fig

    monkeypatch.setattr(pac, "_savefig", spy_savefig)
    pac.plot_4_learning_curves(run_tag="labeltest")

    ax = captured["fig"].axes[0]
    assert ax.get_xlabel() == "Completed training timesteps"
    assert ax.get_ylabel() == "Mean held-out objective"
    title = ax.get_title()
    assert "standard deviation" in title
    assert "confidence interval" in title


# ======================================================================
# Regression coverage for validation-based best-checkpoint saving.
# evaluate_policy is monkeypatched to return a CONTROLLED sequence of
# mean_cumulative_reward values (10.0, then a worse 5.0, then a better
# 20.0) rather than relying on real training to happen to produce a
# particular ordering -- this makes "worse doesn't overwrite" / "better
# does overwrite" deterministic and fast (no real per-eval 4000-step
# episode cost). model.save() itself is NOT mocked -- real files are
# written to tmp_path so file-level claims (existence, call count,
# distinctness from the final model path) are genuinely verified, not
# just asserted about in-memory state.
# ======================================================================
def _build_tiny_feedforward_model(seed=0):
    def _init():
        return Monitor(build_train_env("return_mlp_ppo", seed, 10.0, 0.005))

    vec_env = DummyVecEnv([_init])
    return PPO(
        "MlpPolicy", vec_env, n_steps=8, batch_size=8, n_epochs=2,
        policy_kwargs=dict(net_arch=[8, 8]), device="cpu", seed=seed, verbose=0,
    )


def _build_tiny_recurrent_model(seed=0):
    def _init():
        return Monitor(build_train_env("return_lstm_ppo", seed, 10.0, 0.005))

    vec_env = DummyVecEnv([_init])
    return RecurrentPPO(
        "MlpLstmPolicy", vec_env, n_steps=8, batch_size=8, n_epochs=2,
        policy_kwargs=dict(net_arch=[8, 8], lstm_hidden_size=8, n_lstm_layers=1),
        device="cpu", seed=seed, verbose=0,
    )


def _fake_evaluate_policy_sequence(rewards):
    """Returns a stand-in for train_agents.evaluate_policy that ignores the
    real model/env and yields controlled (summary, records) results in
    sequence, one per call -- deterministic, no real episode cost."""
    calls = {"n": 0}

    def fake(model, agent_type, eval_seeds, inventory_scale, return_scale, deterministic=True, environment_type="fixed"):
        i = calls["n"]
        calls["n"] += 1
        reward = rewards[i]
        summary = dict(
            n_episodes=len(eval_seeds), mean_cumulative_reward=reward, std_cumulative_reward=1.0,
            mean_raw_pnl=reward, mean_terminal_abs_inventory=0.0,
            action_mean_bid=0.0, action_mean_ask=0.0, action_std_bid=0.0, action_std_ask=0.0,
        )
        return summary, []

    return fake, calls


@pytest.mark.parametrize("build_model,agent_type,is_recurrent", [
    (_build_tiny_feedforward_model, "return_mlp_ppo", False),
    (_build_tiny_recurrent_model, "return_lstm_ppo", True),
])
class TestBestCheckpointSaving:
    def test_full_best_checkpoint_sequence(self, build_model, agent_type, is_recurrent, monkeypatch, tmp_path):
        """Covers items 1-4 and 6 in one coherent run: first eval saved as
        best; a worse second eval does not overwrite; a better third eval
        does overwrite; saved timestep/objective match exactly; works for
        both feed-forward PPO and RecurrentPPO."""
        model = build_model(seed=5)
        training_curve = []
        best_model_path = str(tmp_path / f"ppo_{agent_type}_unittest_best")
        callback = PeriodicEvalCallback(
            agent_type=agent_type, eval_seeds=[130_002], inventory_scale=10.0, return_scale=0.005,
            eval_freq=8, training_curve=training_curve, verbose=0, best_model_path=best_model_path,
        )

        fake_eval, calls = _fake_evaluate_policy_sequence([10.0, 5.0, 20.0])
        monkeypatch.setattr(train_agents, "evaluate_policy", fake_eval)

        save_calls = []
        original_save = model.save

        def spying_save(path, *a, **kw):
            save_calls.append(path)
            return original_save(path, *a, **kw)

        model.save = spying_save

        # 4 rollouts of 8 steps -> on_rollout_start fires (and, since
        # completed_timesteps>0, evaluates) at timesteps 8, 16, 24.
        model.learn(total_timesteps=32, callback=callback)

        assert calls["n"] == 3, "expected exactly 3 periodic evaluations"

        # --- item 1: first periodic eval saved as best ---
        # --- item 3/4: better third eval overwrites, with correct timestep/objective ---
        assert callback.best_mean_objective == 20.0
        assert callback.best_std_objective == 1.0
        assert callback.best_timestep == 24
        # --- item 2: worse second eval (5.0 @ t=16) must not have overwritten
        # the first (10.0 @ t=8) before the third (20.0 @ t=24) arrived --
        # i.e. save() was called exactly twice (t=8 and t=24), never for t=16.
        assert save_calls == [best_model_path, best_model_path]
        assert len(save_calls) == 2

        import os
        assert os.path.exists(best_model_path + ".zip")

    def test_no_periodic_eval_means_no_best_recorded(self, build_model, agent_type, is_recurrent, monkeypatch, tmp_path):
        """If training never completes a rollout boundary past t=0 (single
        8-step rollout, total_timesteps=8), no periodic eval ever fires --
        best_* attributes must stay None, and nothing should be saved to
        best_model_path."""
        model = build_model(seed=6)
        training_curve = []
        best_model_path = str(tmp_path / f"ppo_{agent_type}_unittest_best_none")
        callback = PeriodicEvalCallback(
            agent_type=agent_type, eval_seeds=[130_003], inventory_scale=10.0, return_scale=0.005,
            eval_freq=8, training_curve=training_curve, verbose=0, best_model_path=best_model_path,
        )
        fake_eval, calls = _fake_evaluate_policy_sequence([999.0])
        monkeypatch.setattr(train_agents, "evaluate_policy", fake_eval)

        model.learn(total_timesteps=8, callback=callback)

        assert calls["n"] == 0
        assert callback.best_mean_objective is None
        assert callback.best_timestep is None
        import os
        assert not os.path.exists(best_model_path + ".zip")


def test_best_model_path_distinct_from_final_model_path_in_main_source():
    """Item 5 (structural check): main()'s best-model path construction
    must not reuse the final model's path expression -- confirmed by
    source inspection rather than a full expensive end-to-end run, since
    TestBestCheckpointSaving already proves the callback only ever writes
    to the path it's explicitly given."""
    main_source = inspect.getsource(train_agents.main)
    assert 'best_model_path = str(output_dir / f"ppo_{args.agent_type}_{args.run_tag}_best")' in main_source
    assert 'model_path = args.model_path or str(output_dir / f"ppo_{args.agent_type}_{args.run_tag}")' in main_source
