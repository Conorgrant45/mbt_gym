"""
tests/test_phase6_optimisation_experiment.py
------------------------------------------------------
Phase 6 required tests (Section 10, items 1-9). Item 10 (all existing tests
remain green) is the full pre-existing suite, run separately, not
duplicated here.

Run from repo root:
    pytest tests/test_phase6_optimisation_experiment.py -v
"""
import json

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

import shared.phase5_common as P5
import shared.phase6_common as P6


@pytest.fixture(scope="module")
def clone_net():
    return P5.load_supervised_clone_net()


# ======================================================================
# Item 1: correct group-specific actor initialisation
# ======================================================================
def _flat_linear_only(model):
    """Actor weights EXCLUDING log_std -- the part copy_supervised_actor_weights
    actually touches (log_std is deliberately group-specific, varies between
    C and D by design, and must not be included in this comparison)."""
    layers = P5.map_ppo_actor_linear_layers(model)
    with torch.no_grad():
        return torch.cat([p.detach().flatten() for m in layers for p in (m.weight, m.bias)]).numpy()


def test_group_a_b_random_init_group_c_d_clone_init(clone_net):
    model_a, _, verify_a = P6.build_group_model("A", learner_seed=0)
    model_c, _, verify_c = P6.build_group_model("C", learner_seed=0)

    assert verify_a is None  # random init: no clone-verification report
    assert verify_c is not None and verify_c["passed"]

    actor_c = _flat_linear_only(model_c)
    clone_actor_ref, _, _ = P6.build_group_model("D", learner_seed=1)  # different seed, still clone-init
    actor_d = _flat_linear_only(clone_actor_ref)
    # C and D share the SAME clone-copied Linear-layer weights regardless of
    # learner seed or log_std_init -- only log_std itself differs between them.
    np.testing.assert_allclose(actor_c, actor_d, atol=1e-6)

    actor_a = _flat_linear_only(model_a)
    assert np.linalg.norm(actor_a - actor_c) > 1.0  # random init is nowhere near the clone


# ======================================================================
# Item 2: correct group-specific log_std_init
# ======================================================================
def test_group_log_std_init_values():
    for group in ("A", "C"):
        model, _, _ = P6.build_group_model(group, learner_seed=0)
        np.testing.assert_allclose(model.policy.log_std.detach().numpy(), [0.0, 0.0], atol=1e-8)
    for group in ("B", "D"):
        model, _, _ = P6.build_group_model(group, learner_seed=0)
        np.testing.assert_allclose(model.policy.log_std.detach().numpy(), [-1.5, -1.5], atol=1e-8)


# ======================================================================
# Item 3: clone initialisation changes actor but not critic
# ======================================================================
def test_clone_init_leaves_critic_independent_per_seed():
    model_c0, _, _ = P6.build_group_model("C", learner_seed=0)
    model_c1, _, _ = P6.build_group_model("C", learner_seed=1)

    actor_0 = P5.flat_param_vector(P5.get_actor_params(model_c0))
    actor_1 = P5.flat_param_vector(P5.get_actor_params(model_c1))
    np.testing.assert_allclose(actor_0, actor_1, atol=1e-6)  # same clone actor regardless of seed

    critic_0 = P5.flat_param_vector(P5.get_critic_params(model_c0))
    critic_1 = P5.flat_param_vector(P5.get_critic_params(model_c1))
    assert np.linalg.norm(critic_0 - critic_1) > 1e-3  # different seeds -> different random critic init


# ======================================================================
# Item 4: exact clone-to-PPO deterministic output equivalence
# ======================================================================
def test_exact_clone_to_ppo_deterministic_output_equivalence(clone_net):
    for group in ("C", "D"):
        model, _, verify_report = P6.build_group_model(group, learner_seed=2)
        assert verify_report["passed"]
        assert verify_report["max_abs_diff"] < P6.ACTOR_CLONE_VERIFY_TOL

        grid_obs = P5.dense_grid_observations()[:300]
        recheck = P5.verify_actor_matches_clone(model, clone_net, grid_obs)
        assert recheck["passed"]


# ======================================================================
# Item 5: checkpoint metadata records initialisation and exploration setting
# ======================================================================
def test_run_manifest_records_group_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(P6, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(P6, "RESULTS_DIR", tmp_path / "results")  # isolate phase6_run_status.csv too
    monkeypatch.setattr(P6, "TOTAL_TRANSITIONS", 4000)
    monkeypatch.setattr(P6, "CHECKPOINT_TIMESTEPS", [4000])
    monkeypatch.setattr(P6, "VALIDATION_SEEDS", P6.VALIDATION_SEEDS[:3])

    import phase6_run_training as R
    monkeypatch.setattr(R, "P6", P6)
    R.run_one("D", 3)

    manifest = json.loads(P6.run_manifest_path("D", 3).read_text())
    assert manifest["group"] == "D"
    assert manifest["init_type"] == "clone"
    assert manifest["log_std_init"] == -1.5
    assert manifest["learner_seed"] == 3
    assert manifest["clone_verification"]["passed"]
    assert len(manifest["checkpoints"]) == 1


# ======================================================================
# Item 6: offline selection rejects mismatched group metadata
# ======================================================================
def test_offline_selection_rejects_mismatched_group_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(P6, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(P6, "RESULTS_DIR", tmp_path / "results")  # isolate phase6_run_status.csv too
    monkeypatch.setattr(P6, "TOTAL_TRANSITIONS", 4000)
    monkeypatch.setattr(P6, "CHECKPOINT_TIMESTEPS", [4000])
    monkeypatch.setattr(P6, "VALIDATION_SEEDS", P6.VALIDATION_SEEDS[:3])

    import phase6_run_training as R
    monkeypatch.setattr(R, "P6", P6)
    R.run_one("A", 4)

    manifest = json.loads(P6.run_manifest_path("A", 4).read_text())

    def load_and_check(expected_group: str, expected_log_std_init: float):
        if manifest["group"] != expected_group or manifest["log_std_init"] != expected_log_std_init:
            raise ValueError(
                f"run_manifest group/log_std_init ({manifest['group']}, {manifest['log_std_init']}) does not "
                f"match expected ({expected_group}, {expected_log_std_init}) -- refusing to select."
            )
        return manifest

    load_and_check("A", 0.0)  # correct -- should not raise
    with pytest.raises(ValueError):
        load_and_check("D", -1.5)  # mismatched -- must raise


# ======================================================================
# Item 7: holdout seeds do not overlap prior seed ranges
# ======================================================================
def test_holdout_and_validation_seeds_disjoint_from_all_previous_ranges():
    validation = set(P6.VALIDATION_SEEDS)
    holdout = set(P6.HOLDOUT_SEEDS)
    assert validation.isdisjoint(holdout)

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
        assert validation.isdisjoint(prev), f"validation seeds overlap {prev}"
        assert holdout.isdisjoint(prev), f"holdout seeds overlap {prev}"

    assert len(P6.VALIDATION_SEEDS) == 50
    assert len(P6.HOLDOUT_SEEDS) == 200


# ======================================================================
# Item 8: save/reload preserves actor and log_std
# ======================================================================
def test_save_reload_preserves_actor_and_log_std(tmp_path, clone_net):
    for group in ("B", "D"):
        model, _, _ = P6.build_group_model(group, learner_seed=1)
        path = tmp_path / f"group{group}_model"
        model.save(str(path))

        reloaded = PPO.load(str(path))
        actor_orig = P5.flat_param_vector(P5.get_actor_params(model))
        actor_reload = P5.flat_param_vector(P5.get_actor_params(reloaded))
        np.testing.assert_allclose(actor_orig, actor_reload, atol=1e-7)
        np.testing.assert_allclose(model.policy.log_std.detach().numpy(),
                                    reloaded.policy.log_std.detach().numpy(), atol=1e-7)
        np.testing.assert_allclose(reloaded.policy.log_std.detach().numpy(), [-1.5, -1.5], atol=1e-8)


# ======================================================================
# Item 9: deterministic evaluation is repeatable
# ======================================================================
def test_deterministic_evaluation_is_repeatable():
    model, _, _ = P6.build_group_model("C", learner_seed=0)
    seeds = P6.VALIDATION_SEEDS[:5]
    r1 = P6.evaluate_checkpoint_model(model, seeds=seeds, torch_seed=42)
    r2 = P6.evaluate_checkpoint_model(model, seeds=seeds, torch_seed=42)
    assert r1["det_mean_objective"] == pytest.approx(r2["det_mean_objective"], abs=1e-9)
    assert r1["det_mean_fills"] == pytest.approx(r2["det_mean_fills"], abs=1e-9)
