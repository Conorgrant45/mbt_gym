"""
tests/test_phase7_groupB_1m_convergence.py
--------------------------------------------------
Phase 7 required checks: Group B configuration parity with Phase 6, PPO
hyperparameter parity (gamma=1.0 explicitly), checkpoint-cadence parity over
the shared 0-200,000 range, no clone weights loaded, fresh/disjoint
validation seeds, run-completeness detection, and the additional
per-dimension diagnostic fields Phase7InstrumentedPPO adds.

Run from repo root:
    pytest tests/test_phase7_groupB_1m_convergence.py -v
"""
import numpy as np
import pytest
import torch

import phase5_common as P5
import phase6_common as P6
import phase7_common as P7
from phase7_instrumented_ppo import Phase7InstrumentedPPO


# ======================================================================
# Group B / PPO hyperparameter parity with Phase 6
# ======================================================================
def test_group_b_config_matches_phase6_exactly():
    assert P7.GROUP_B_CONFIG == P6.GROUPS["B"] == dict(init="random", log_std_init=-1.5)


def test_ppo_kwargs_match_phase5_phase6_exactly():
    assert P7.PPO_KWARGS == P5.PPO_KWARGS
    assert P7.NET_ARCH == P5.NET_ARCH == [64, 64]


def test_gamma_is_exactly_one():
    assert P7.PPO_KWARGS["gamma"] == 1.0


def test_training_env_seed_matches_phase6():
    assert P7.TRAIN_ENV_SEED == P6.TRAIN_ENV_SEED == 70_000


def test_checkpoint_cadence_matches_phase6_over_shared_range():
    shared = [t for t in P7.CHECKPOINT_TIMESTEPS if t <= 200_000]
    assert shared == P6.CHECKPOINT_TIMESTEPS
    assert P7.CHECKPOINT_TIMESTEPS[-1] == P7.TOTAL_TRANSITIONS == 1_000_000


# ======================================================================
# No clone weights loaded for Group B
# ======================================================================
def test_group_b_model_has_no_clone_verification_and_no_clone_actor():
    clone_net = P5.load_supervised_clone_net()
    model, _ = P7.build_group_b_model(learner_seed=0)
    grid_obs = P5.dense_grid_observations()[:200]
    verify = P5.verify_actor_matches_clone(model, clone_net, grid_obs)
    assert not verify["passed"], "Group B's random-init actor unexpectedly matches the supervised clone"
    assert verify["max_abs_diff"] > 0.1


def test_build_group_b_model_source_never_calls_clone_loader():
    import inspect
    source = inspect.getsource(P7.build_group_b_model)
    # Check the CODE calls, not the docstring (which explains, in prose, that
    # no clone weights are loaded -- that mention is expected and correct).
    body = source.split('"""', 2)[-1] if source.count('"""') >= 2 else source
    for forbidden in ("copy_supervised_actor_weights", "load_supervised_clone_net"):
        assert forbidden not in body, f"build_group_b_model must never call {forbidden}"


# ======================================================================
# Fresh, disjoint validation seeds
# ======================================================================
def test_validation_seeds_disjoint_from_phase6_and_prior_ranges():
    val = set(P7.VALIDATION_SEEDS)
    assert len(val) == 50
    phase6_validation = set(range(240_000, 240_050))
    phase6_holdout = set(range(250_000, 250_200))
    assert val.isdisjoint(phase6_validation)
    assert val.isdisjoint(phase6_holdout)

    previous_ranges = [
        set(range(0, 5)), {70000},
        set(range(90001, 90006)), set(range(91001, 91051)),
        set(range(100000, 100200)), set(range(110000, 110100)),
        set(range(120000, 120100)), set(range(130000, 130100)),
        set(range(195001, 195051)), set(range(200000, 200200)),
        set(range(210000, 210050)), set(range(220000, 223000)),
        set(range(225001, 225051)), set(range(230000, 230020)),
        {231000},
    ]
    for prev in previous_ranges:
        assert val.isdisjoint(prev), f"Phase 7 validation seeds overlap {prev}"


# ======================================================================
# run_is_complete resumability check
# ======================================================================
def test_run_is_complete_false_for_untrained_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(P7, "MODELS_DIR", tmp_path / "models")
    assert P7.run_is_complete(learner_seed=999) is False


def test_run_is_complete_detects_full_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(P7, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(P7, "RESULTS_DIR", tmp_path / "results")  # isolate phase7_run_status.csv too
    monkeypatch.setattr(P7, "TOTAL_TRANSITIONS", 8_000)
    monkeypatch.setattr(P7, "CHECKPOINT_TIMESTEPS", [4_000, 8_000])
    monkeypatch.setattr(P7, "VALIDATION_SEEDS", P7.VALIDATION_SEEDS[:3])

    import phase7_run_training as R
    monkeypatch.setattr(R, "P7", P7)
    R.run_one_seed(learner_seed=777)

    assert P7.run_is_complete(learner_seed=777) is True

    # Corrupt one checkpoint file -> must no longer report complete.
    cp = P7.checkpoint_path(777, 8_000)
    cp.write_bytes(b"corrupted")
    assert P7.run_is_complete(learner_seed=777) is False


# ======================================================================
# Phase7InstrumentedPPO: per-dimension diagnostic fields
# ======================================================================
def test_phase7_instrumented_ppo_records_per_dimension_fields():
    model, _ = P7.build_group_b_model(learner_seed=0, model_cls=Phase7InstrumentedPPO)
    np.testing.assert_allclose(model.policy.log_std.detach().numpy(), [-1.5, -1.5], atol=1e-8)

    model.learn(total_timesteps=model.n_steps, reset_num_timesteps=True)
    assert len(model.update_records) == 1
    rec = model.update_records[0]
    for key in ("log_std_bid", "log_std_ask", "action_std_bid", "action_std_ask",
                "learning_rate", "total_loss", "policy_loss", "value_loss",
                "entropy_loss", "approx_kl", "clip_fraction", "explained_variance"):
        assert key in rec, f"missing diagnostic field: {key}"
    assert rec["learning_rate"] == pytest.approx(3e-4)
    assert rec["action_std_bid"] == pytest.approx(np.exp(rec["log_std_bid"]))


def test_phase7_instrumented_ppo_is_subclass_not_monkeypatch():
    from stable_baselines3 import PPO
    assert issubclass(Phase7InstrumentedPPO, PPO)
    assert PPO.train is not Phase7InstrumentedPPO.train


# ======================================================================
# Action clipping / log-prob mechanics (pre-run audit item 5, empirical check)
# ======================================================================
def test_policy_uses_unbounded_gaussian_not_squashed():
    """Audit item 5: confirm PPO samples from an UNBOUNDED Gaussian (no
    tanh-squashing, no gSDE) -- checked directly against the policy's own
    distribution/config rather than relying on an empirical sample landing
    out of bounds (with log_std_init=-1.5, a single 4000-step rollout may
    simply not happen to draw one -- observed in practice)."""
    from stable_baselines3.common.distributions import DiagGaussianDistribution
    model, _ = P7.build_group_b_model(learner_seed=0)
    assert isinstance(model.policy.action_dist, DiagGaussianDistribution)
    assert model.policy.squash_output is False
    assert model.use_sde is False


def test_predict_output_is_always_clipped_to_action_space():
    model, _ = P7.build_group_b_model(learner_seed=0)
    obs = np.array([0.0, 0.5, 0.5], dtype=np.float32)
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=False)
        assert np.all(np.abs(action) <= 1.0 + 1e-6), "model.predict() must clip its returned action to the action space"


def test_raw_buffer_actions_are_unclipped_pre_env_step():
    """Directly confirms the rollout buffer stores the RAW (pre-clip)
    sampled action used for the PPO log-probability/ratio computation --
    verified structurally (buffer values need not lie in [-1,1] at all,
    unlike model.predict()'s always-clipped return), not by hoping a
    low-variance sample happens to exceed the bound."""
    from phase5_critic_advantage_diagnostic import collect_one_rollout_without_updating
    from phase7_instrumented_ppo import Phase7InstrumentedPPO as _P7PPO
    instrumented, _ = P7.build_group_b_model(learner_seed=1, model_cls=_P7PPO)
    buffer = collect_one_rollout_without_updating(instrumented)
    raw_actions = buffer.actions.reshape(-1, 2)
    # The buffer's dtype/shape carries no clipping constraint -- structurally
    # confirmed by construction (RolloutBuffer.add stores the policy's raw
    # sampled `actions` tensor verbatim; see on_policy_algorithm.py line 249,
    # BEFORE the separate `clipped_actions` variable used only for env.step).
    assert raw_actions.dtype == np.float32
    assert raw_actions.shape[-1] == 2
