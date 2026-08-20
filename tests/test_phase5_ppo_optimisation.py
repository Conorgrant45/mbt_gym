"""
tests/test_phase5_ppo_optimisation.py
--------------------------------------------
Phase 5 required tests (Section 11, items 1-10). Item 11 (all existing
tests remain green) is the full pre-existing suite, run separately, not
duplicated here.

Run from repo root:
    pytest tests/test_phase5_ppo_optimisation.py -v
"""
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

import shared.phase5_common as P5
from shared.phase5_instrumented_ppo import InstrumentedPPO
from shared.phase5_critic_advantage_diagnostic import (
    collect_one_rollout_without_updating, recompute_gae_independently,
)


@pytest.fixture(scope="module")
def clone_net():
    return P5.load_supervised_clone_net()


# ======================================================================
# Item 1: exact supervised-to-PPO actor weight mapping
# ======================================================================
def test_copy_supervised_actor_weights_exact_mapping(clone_net):
    model, _ = P5.build_hamilton_ppo(learner_seed=0)
    report = P5.copy_supervised_actor_weights(model, clone_net)
    assert report["n_layers_copied"] == 3
    supervised_linears = P5.map_supervised_linear_layers(clone_net)
    ppo_linears = P5.map_ppo_actor_linear_layers(model)
    for src, dst in zip(supervised_linears, ppo_linears):
        assert torch.equal(src.weight, dst.weight)
        assert torch.equal(src.bias, dst.bias)


def test_copy_supervised_actor_weights_fails_clearly_on_shape_mismatch():
    model, _ = P5.build_hamilton_ppo(learner_seed=0)
    bad_net = torch.nn.Sequential(torch.nn.Linear(5, 32), torch.nn.Tanh(),
                                   torch.nn.Linear(32, 32), torch.nn.Tanh(),
                                   torch.nn.Linear(32, 2))
    with pytest.raises(AssertionError):
        P5.copy_supervised_actor_weights(model, bad_net)


# ======================================================================
# Item 2: deterministic PPO output equals clone output
# ======================================================================
def test_deterministic_ppo_output_equals_clone_output(clone_net):
    model, _ = P5.build_hamilton_ppo(learner_seed=1)
    P5.copy_supervised_actor_weights(model, clone_net)
    grid_obs = P5.dense_grid_observations()[:500]
    result = P5.verify_actor_matches_clone(model, clone_net, grid_obs)
    assert result["passed"], result
    assert result["max_abs_diff"] < 1e-4


# ======================================================================
# Item 3: actor-only loading leaves critic independently initialised
# ======================================================================
def test_actor_only_transfer_leaves_critic_independent(clone_net):
    model_a, _ = P5.build_hamilton_ppo(learner_seed=0)
    model_b, _ = P5.build_hamilton_ppo(learner_seed=1)
    P5.copy_supervised_actor_weights(model_a, clone_net)
    P5.copy_supervised_actor_weights(model_b, clone_net)

    actor_a = P5.flat_param_vector(P5.get_actor_params(model_a))
    actor_b = P5.flat_param_vector(P5.get_actor_params(model_b))
    np.testing.assert_allclose(actor_a, actor_b, atol=1e-6)

    critic_a = P5.flat_param_vector(P5.get_critic_params(model_a))
    critic_b = P5.flat_param_vector(P5.get_critic_params(model_b))
    assert np.linalg.norm(critic_a - critic_b) > 1e-3, (
        "critics from two different learner seeds are identical -- actor transfer must have "
        "leaked into the critic, or both models share unseeded critic init"
    )


# ======================================================================
# Item 4: t=0 checkpoint is saved correctly
# ======================================================================
def test_t0_checkpoint_saved_and_reloads_to_clone_output(clone_net, tmp_path):
    model, _ = P5.build_hamilton_ppo(learner_seed=0)
    P5.copy_supervised_actor_weights(model, clone_net)
    save_path = tmp_path / "t0_checkpoint"
    model.save(str(save_path))
    assert (tmp_path / "t0_checkpoint.zip").exists()

    reloaded = PPO.load(str(save_path))
    grid_obs = P5.dense_grid_observations()[:200]
    result = P5.verify_actor_matches_clone(reloaded, clone_net, grid_obs)
    assert result["passed"], result


# ======================================================================
# Item 5: frozen control weights do not change
# ======================================================================
def test_frozen_group_c_weights_unchanged_across_evaluations(clone_net):
    model, _ = P5.build_hamilton_ppo(learner_seed=0)
    P5.copy_supervised_actor_weights(model, clone_net)
    actor_before = P5.flat_param_vector(P5.get_actor_params(model))
    critic_before = P5.flat_param_vector(P5.get_critic_params(model))

    # Repeated evaluation (deterministic + stochastic rollouts) must not
    # itself mutate any policy parameter.
    P5.evaluate_checkpoint(model, seeds=P5.DIAGNOSTIC_VALIDATION_SEEDS[:3], torch_seed=123)

    actor_after = P5.flat_param_vector(P5.get_actor_params(model))
    critic_after = P5.flat_param_vector(P5.get_critic_params(model))
    np.testing.assert_array_equal(actor_before, actor_after)
    np.testing.assert_array_equal(critic_before, critic_after)


# ======================================================================
# Item 6: actor and critic parameter-update norms are measured correctly
# ======================================================================
def test_actor_critic_update_norms_measured_correctly(clone_net):
    model, _ = P5.build_hamilton_ppo(learner_seed=0, model_cls=InstrumentedPPO)
    P5.copy_supervised_actor_weights(model, clone_net)

    actor_before = P5.flat_param_vector(P5.get_actor_params(model))
    critic_before = P5.flat_param_vector(P5.get_critic_params(model))

    model.learn(total_timesteps=model.n_steps, reset_num_timesteps=True)

    actor_after = P5.flat_param_vector(P5.get_actor_params(model))
    critic_after = P5.flat_param_vector(P5.get_critic_params(model))
    expected_actor_norm = float(np.linalg.norm(actor_after - actor_before))
    expected_critic_norm = float(np.linalg.norm(critic_after - critic_before))

    assert len(model.update_records) == 1
    rec = model.update_records[0]
    assert rec["actor_param_update_norm"] == pytest.approx(expected_actor_norm, rel=1e-6)
    assert rec["critic_param_update_norm"] == pytest.approx(expected_critic_norm, rel=1e-6)
    assert rec["actor_param_update_norm"] > 0
    assert rec["critic_param_update_norm"] > 0


# ======================================================================
# Item 7: independently recomputed GAE matches the rollout buffer
# ======================================================================
def test_gae_recompute_matches_rollout_buffer(clone_net):
    model, _ = P5.build_hamilton_ppo(learner_seed=0, model_cls=InstrumentedPPO)
    P5.copy_supervised_actor_weights(model, clone_net)
    buffer = collect_one_rollout_without_updating(model)

    from stable_baselines3.common.utils import obs_as_tensor
    with torch.no_grad():
        last_values = model.policy.predict_values(
            obs_as_tensor(model._last_obs, model.device)
        ).cpu().numpy().flatten()

    recomputed = recompute_gae_independently(
        buffer, last_values, model._last_episode_starts, model.gamma, model.gae_lambda,
    ).flatten()
    sb3_advantages = buffer.advantages.flatten()
    max_diff = float(np.max(np.abs(recomputed - sb3_advantages)))
    assert max_diff < 1e-4, f"independent GAE recompute diverges from SB3's own buffer: max diff {max_diff}"


# ======================================================================
# Item 8: action-surface MSE is calculated in the correct physical action
# space (depth units, not raw normalised [-1,1] action units)
# ======================================================================
def test_action_surface_mse_uses_physical_depth_space():
    import shared.phase4_common as P4

    class ConstantModel:
        """A fake model whose deterministic prediction is always the
        normalised action [0.0, 0.0] -- i.e. physical depth MAX_DEPTH/2 on
        both sides."""
        def predict(self, obs, deterministic=True):
            return np.array([0.0, 0.0]), None

    controls = P4.build_analytical_controls()
    q_grid = np.array([0.0])
    tau_grid = np.array([0.5])
    b_grid = np.array([0.5])
    metrics = P5.action_surface_metrics(ConstantModel(), controls, q_grid, tau_grid, b_grid)

    a_bid, a_ask = P4.analytical_belief_weighted_depths(controls, 0.5, 0.0, 0.5)
    expected_bid_depth = (0.0 + 1.0) / 2.0 * P4.MAX_DEPTH
    expected_ask_depth = (0.0 + 1.0) / 2.0 * P4.MAX_DEPTH
    expected_bid_mse = (expected_bid_depth - a_bid) ** 2
    expected_ask_mse = (expected_ask_depth - a_ask) ** 2

    assert metrics["bid_action_depth_mse"] == pytest.approx(expected_bid_mse, rel=1e-6)
    assert metrics["ask_action_depth_mse"] == pytest.approx(expected_ask_mse, rel=1e-6)
    # Sanity: MAX_DEPTH is on the order of a few price units, not O(1) like a
    # raw normalised action -- confirms the comparison happened in depth space.
    assert expected_bid_mse > 1e-3 or expected_ask_mse > 1e-3 or P4.MAX_DEPTH < 0.1


# ======================================================================
# Item 9: model save/reload preserves imported actor weights
# ======================================================================
def test_save_reload_preserves_imported_actor_weights(clone_net, tmp_path):
    model, _ = P5.build_hamilton_ppo(learner_seed=2)
    P5.copy_supervised_actor_weights(model, clone_net)
    path = tmp_path / "clone_transfer_model"
    model.save(str(path))

    reloaded = PPO.load(str(path))
    actor_orig = P5.flat_param_vector(P5.get_actor_params(model))
    actor_reload = P5.flat_param_vector(P5.get_actor_params(reloaded))
    np.testing.assert_allclose(actor_orig, actor_reload, atol=1e-7)


# ======================================================================
# Item 10: fixed-step and existing event-driven behaviour remain unchanged
# ======================================================================
def test_plain_ppo_train_is_not_monkeypatched_by_instrumented_ppo():
    """InstrumentedPPO must be a SUBCLASS, never a monkeypatch of the
    shared PPO.train -- constructing an InstrumentedPPO must not affect a
    plain PPO instance's behaviour (Hamilton/return PPO training elsewhere
    in the project uses plain PPO/RecurrentPPO, untouched by this phase)."""
    import shared.phase4_common as P4
    models = P4.verify_and_load_hamilton_models()
    plain_model = models[("event", 0)]["model"]
    assert type(plain_model).train is PPO.train
    assert PPO.train is not InstrumentedPPO.train


def test_event_driven_wrapper_observation_unchanged():
    """Constructing/using InstrumentedPPO must not alter
    EventDrivenHamiltonPPOWrapper's observation formula (Phase 4's own
    parity test re-run here against a Phase-5-constructed model, to confirm
    this phase introduced no shared-module side effects)."""
    from envs.event_driven_hamilton_ppo_wrapper import EventDrivenHamiltonPPOWrapper
    model, _ = P5.build_hamilton_ppo(learner_seed=0, model_cls=InstrumentedPPO)
    wrapper = EventDrivenHamiltonPPOWrapper(seed=1)
    obs, _ = wrapper.reset()
    expected = np.array([np.tanh(wrapper.base_env.raw_inventory / P5.DEFAULT_INVENTORY_SCALE), 1.0, wrapper.belief],
                         dtype=np.float32)
    np.testing.assert_allclose(obs, expected, atol=1e-6)
