"""
test_evaluate_agents_common.py
------------------------------------
Pytest suite for evaluate_agents_common.py (Phase 3). Does not modify any
production file. Reuses the already-trained Phase 2 smoke models
(models/{agent_type}/ppo_{agent_type}_smoke2.zip) rather than retraining
-- these tests are about evaluation-harness correctness, not learned
policy quality. Kept to a small number of full 4000-step episode runs
(env stepping dominates runtime, not model inference) by reusing results
across assertions where possible.

Run from repo root:
    pytest tests/test_evaluate_agents_common.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

import shared.evaluate_agents_common as EAC
from envs.hamilton_ppo_wrapper import HamiltonPPOWrapper
from envs.return_ppo_wrapper import ReturnPPOWrapper

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

HAMILTON_MODEL = MODELS_DIR / "hamilton_ppo" / "ppo_hamilton_ppo_smoke2.zip"
RETURN_MLP_MODEL = MODELS_DIR / "return_mlp_ppo" / "ppo_return_mlp_ppo_smoke2.zip"
RETURN_LSTM_MODEL = MODELS_DIR / "return_lstm_ppo" / "ppo_return_lstm_ppo_smoke2.zip"

pytestmark = pytest.mark.skipif(
    not (HAMILTON_MODEL.exists() and RETURN_MLP_MODEL.exists() and RETURN_LSTM_MODEL.exists()),
    reason="Phase 2 smoke2 models not found -- run the three train_agents.py smoke commands first.",
)

TEST_SEED = 120500  # disjoint from DEFAULT_HOLDOUT_SEEDS (120000-120099) and every other range used in the project


# ======================================================================
# Item 1: correct wrapper selection per agent type
# ======================================================================
class TestWrapperSelection:
    def test_hamilton_ppo_uses_hamilton_wrapper(self, monkeypatch):
        import shared.hamilton_ppo_eval_lib as HEL
        calls = []
        original_init = HamiltonPPOWrapper.__init__

        def spy_init(self, *a, **kw):
            calls.append(1)
            return original_init(self, *a, **kw)

        monkeypatch.setattr(HamiltonPPOWrapper, "__init__", spy_init)
        model = PPO.load(str(HAMILTON_MODEL))
        EAC.run_hamilton_agent_episode(model, TEST_SEED)
        assert len(calls) == 1

    def test_return_mlp_ppo_uses_return_wrapper_not_hamilton(self, monkeypatch):
        calls_return, calls_hamilton = [], []
        original_return_init = ReturnPPOWrapper.__init__
        original_hamilton_init = HamiltonPPOWrapper.__init__

        def spy_return(self, *a, **kw):
            calls_return.append(1)
            return original_return_init(self, *a, **kw)

        def spy_hamilton(self, *a, **kw):
            calls_hamilton.append(1)
            return original_hamilton_init(self, *a, **kw)

        monkeypatch.setattr(ReturnPPOWrapper, "__init__", spy_return)
        monkeypatch.setattr(HamiltonPPOWrapper, "__init__", spy_hamilton)
        model = PPO.load(str(RETURN_MLP_MODEL))
        EAC.run_return_agent_episode(model, TEST_SEED, is_recurrent=False)
        assert len(calls_return) == 1
        assert len(calls_hamilton) == 0

    def test_return_lstm_ppo_uses_return_wrapper(self, monkeypatch):
        calls = []
        original_init = ReturnPPOWrapper.__init__

        def spy_init(self, *a, **kw):
            calls.append(1)
            return original_init(self, *a, **kw)

        monkeypatch.setattr(ReturnPPOWrapper, "__init__", spy_init)
        model = RecurrentPPO.load(str(RETURN_LSTM_MODEL))
        EAC.run_return_agent_episode(model, TEST_SEED, is_recurrent=True)
        assert len(calls) == 1


# ======================================================================
# Items 2-3: fresh environment construction + deterministic fixed-seed evaluation
# ======================================================================
class TestFreshEnvironmentAndDeterminism:
    def test_return_mlp_same_seed_reproducible(self):
        model = PPO.load(str(RETURN_MLP_MODEL))
        r1 = EAC.run_return_agent_episode(model, TEST_SEED, is_recurrent=False)
        r2 = EAC.run_return_agent_episode(model, TEST_SEED, is_recurrent=False)
        assert r1["raw_pnl"] == pytest.approx(r2["raw_pnl"], abs=1e-9)
        assert r1["full_objective"] == pytest.approx(r2["full_objective"], abs=1e-9)
        assert r1["episode_length"] == r2["episode_length"]
        assert r1["bid_action_mean"] == pytest.approx(r2["bid_action_mean"], abs=1e-9)

    def test_hamilton_ppo_same_seed_reproducible(self):
        model = PPO.load(str(HAMILTON_MODEL))
        r1 = EAC.run_hamilton_agent_episode(model, TEST_SEED)
        r2 = EAC.run_hamilton_agent_episode(model, TEST_SEED)
        assert r1["raw_pnl"] == pytest.approx(r2["raw_pnl"], abs=1e-9)
        assert r1["full_objective"] == pytest.approx(r2["full_objective"], abs=1e-9)


# ======================================================================
# Items 4-6, 16: recurrent state handling within evaluate_agents_common's
# own episode runner (not just sb3-contrib internals, already covered by
# tests/test_return_lstm_recurrent.py)
# ======================================================================
class TestRecurrentStateInEvalHarness:
    def test_recurrent_state_reset_verified_flag_true(self):
        """Item 4/5: run_return_agent_episode's own bookkeeping confirms
        state starts reset (state=None, episode_start=True at t=0) AND
        changes within the episode (carried, not reset every step)."""
        model = RecurrentPPO.load(str(RETURN_LSTM_MODEL))
        r = EAC.run_return_agent_episode(model, TEST_SEED, is_recurrent=True)
        assert r["recurrent_state_reset_verified"] is True

    def test_non_recurrent_agents_report_none_for_recurrent_flag(self):
        """Item 16: non-recurrent agents must not claim any recurrent-state
        property -- the field must be None (not False, not omitted)."""
        model = PPO.load(str(RETURN_MLP_MODEL))
        r = EAC.run_return_agent_episode(model, TEST_SEED, is_recurrent=False)
        assert r["recurrent_state_reset_verified"] is None

        model_h = PPO.load(str(HAMILTON_MODEL))
        r_h = EAC.run_hamilton_agent_episode(model_h, TEST_SEED)
        assert r_h["recurrent_state_reset_verified"] is None

    def test_episode_start_true_only_at_first_call(self, monkeypatch):
        """Item 6."""
        model = RecurrentPPO.load(str(RETURN_LSTM_MODEL))
        captured = []
        original_predict = model.predict

        def spy_predict(obs, state=None, episode_start=None, deterministic=False):
            captured.append(bool(episode_start[0]))
            return original_predict(obs, state=state, episode_start=episode_start, deterministic=deterministic)

        monkeypatch.setattr(model, "predict", spy_predict)
        EAC.run_return_agent_episode(model, TEST_SEED, is_recurrent=True)

        assert captured[0] is True
        assert all(v is False for v in captured[1:])


# ======================================================================
# Item 7: correct model loading
# ======================================================================
class TestModelLoading:
    def test_all_three_model_types_load(self):
        m1 = PPO.load(str(HAMILTON_MODEL))
        m2 = PPO.load(str(RETURN_MLP_MODEL))
        m3 = RecurrentPPO.load(str(RETURN_LSTM_MODEL))
        for m in (m1, m2, m3):
            for p in m.policy.parameters():
                assert np.isfinite(p.detach().numpy()).all()
        assert isinstance(m3.policy.lstm_actor, __import__("torch").nn.LSTM)


# ======================================================================
# Item 8: correct evaluation-seed lists
# ======================================================================
def test_holdout_seeds_disjoint_from_every_previous_range():
    holdout = set(EAC.DEFAULT_HOLDOUT_SEEDS)
    previous_ranges = [
        set(range(0, 5)),                 # training seeds
        set(range(90001, 90006)),         # dev eval seeds
        set(range(100_000, 100_200)),     # single-seed-milestone holdout
        set(range(110_000, 110_100)),     # multiseed holdout
    ]
    for prev in previous_ranges:
        assert holdout.isdisjoint(prev)
    assert len(EAC.DEFAULT_HOLDOUT_SEEDS) == 100
    assert len(holdout) == len(EAC.DEFAULT_HOLDOUT_SEEDS)  # no internal duplicates


# ======================================================================
# Item 9: correct episode lengths + Item 11: finite metrics + Item 12: reconciliation
# ======================================================================
class TestEpisodeLengthFinitenessReconciliation:
    @pytest.mark.parametrize("agent_type,model_path,is_recurrent,runner", [
        ("hamilton_ppo", HAMILTON_MODEL, False, "hamilton"),
        ("return_mlp_ppo", RETURN_MLP_MODEL, False, "return"),
        ("return_lstm_ppo", RETURN_LSTM_MODEL, True, "return"),
    ])
    def test_rl_agent_episode(self, agent_type, model_path, is_recurrent, runner):
        model = (RecurrentPPO if is_recurrent else PPO).load(str(model_path))
        if runner == "hamilton":
            r = EAC.run_hamilton_agent_episode(model, TEST_SEED)
        else:
            r = EAC.run_return_agent_episode(model, TEST_SEED, is_recurrent=is_recurrent)

        assert r["episode_length"] == 4000
        assert r["reward_reconciliation_error"] < EAC.RECONCILIATION_TOL
        numeric_vals = {k: v for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        for k, v in numeric_vals.items():
            assert np.isfinite(v), f"{agent_type}: non-finite value for '{k}': {v}"

    def test_analytic_agent_episode(self):
        controls = {}
        import shared.simulate_belief_weighted as SBW
        for regime, params in SBW.REGIME_PARAMS.items():
            da, db, qag, qbg = SBW.build_optimal_control(**params)
            controls[regime] = {"delta_ask": da, "delta_bid": db, "q_ask": qag, "q_bid": qbg}

        for agent_type in EAC.ANALYTIC_AGENT_TYPES:
            r = EAC.run_analytic_agent_episode(agent_type, controls, TEST_SEED)
            assert r["episode_length"] == 4000
            assert r["reward_reconciliation_error"] < EAC.RECONCILIATION_TOL
            numeric_vals = {k: v for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            for k, v in numeric_vals.items():
                assert np.isfinite(v), f"{agent_type}: non-finite value for '{k}': {v}"


# ======================================================================
# Item 10: correct result columns
# ======================================================================
def test_build_row_matches_schema_columns():
    controls = {}
    import shared.simulate_belief_weighted as SBW
    for regime, params in SBW.REGIME_PARAMS.items():
        da, db, qag, qbg = SBW.build_optimal_control(**params)
        controls[regime] = {"delta_ask": da, "delta_bid": db, "q_ask": qag, "q_bid": qbg}
    m = EAC.run_analytic_agent_episode("naive", controls, TEST_SEED)
    row = EAC.build_row("naive", None, None, None, "exogenous_paired_fills_not_claimed_paired", m)
    assert set(row.keys()) == set(EAC.SCHEMA_COLUMNS)


# ======================================================================
# Item 13: path-pairing claims are verified (not assumed)
# ======================================================================
class TestPathPairingVerification:
    def test_verify_path_pairing_all_exogenous_matched(self):
        result = EAC.verify_path_pairing(seed=888_777, n_steps=100)
        assert result["regime_transition_path_matched"] is True
        assert result["brownian_diffusion_shocks_matched"] is True
        assert result["market_order_arrivals_matched"] is True
        assert result["jump_magnitudes_matched"] is True
        assert result["realised_fills_differ_with_different_actions"] is True

    def test_status_label_never_claims_paired_if_verification_failed(self):
        fake_failed = dict(
            regime_transition_path_matched=True, brownian_diffusion_shocks_matched=False,
            market_order_arrivals_matched=True, jump_magnitudes_matched=True,
        )
        label = EAC.path_pairing_status_label(fake_failed)
        assert "FAILED" in label
        assert label != "exogenous_paired_fills_not_claimed_paired"

    def test_status_label_never_claims_fills_paired(self):
        fake_ok = dict(
            regime_transition_path_matched=True, brownian_diffusion_shocks_matched=True,
            market_order_arrivals_matched=True, jump_magnitudes_matched=True,
        )
        label = EAC.path_pairing_status_label(fake_ok)
        assert "fills_not_claimed_paired" in label


# ======================================================================
# Item 14: no true-regime leakage into the RL policy's observation, as
# exercised by THIS module's own episode runners (not just the wrapper's
# own tests)
# ======================================================================
class TestNoRegimeLeakageInHarness:
    def test_return_mlp_obs_shape_has_no_regime_slot(self, monkeypatch):
        model = PPO.load(str(RETURN_MLP_MODEL))
        captured_shapes = []
        original_predict = model.predict

        def spy_predict(obs, deterministic=False, **kw):
            captured_shapes.append(np.asarray(obs).shape)
            return original_predict(obs, deterministic=deterministic, **kw)

        monkeypatch.setattr(model, "predict", spy_predict)
        EAC.run_return_agent_episode(model, TEST_SEED, is_recurrent=False)
        assert all(s == (3,) for s in captured_shapes)

    def test_hamilton_ppo_obs_shape_has_no_regime_slot(self, monkeypatch):
        import shared.hamilton_ppo_eval_lib as HEL
        model = PPO.load(str(HAMILTON_MODEL))
        captured_shapes = []
        original_predict = model.predict

        def spy_predict(obs, deterministic=False, **kw):
            captured_shapes.append(np.asarray(obs).shape)
            return original_predict(obs, deterministic=deterministic, **kw)

        monkeypatch.setattr(model, "predict", spy_predict)
        EAC.run_hamilton_agent_episode(model, TEST_SEED)
        assert all(s == (3,) for s in captured_shapes)


# ======================================================================
# Item 15: analytic benchmark reproducibility (+ fidelity vs the
# validated original compare_four_policies_paired.run_episode)
# ======================================================================
class TestAnalyticReproducibility:
    def test_same_seed_same_policy_reproducible(self):
        controls = {}
        import shared.simulate_belief_weighted as SBW
        for regime, params in SBW.REGIME_PARAMS.items():
            da, db, qag, qbg = SBW.build_optimal_control(**params)
            controls[regime] = {"delta_ask": da, "delta_bid": db, "q_ask": qag, "q_bid": qbg}
        r1 = EAC.run_analytic_agent_episode("oracle", controls, TEST_SEED)
        r2 = EAC.run_analytic_agent_episode("oracle", controls, TEST_SEED)
        assert r1["raw_pnl"] == r2["raw_pnl"]
        assert r1["full_objective"] == r2["full_objective"]

    def test_fidelity_vs_validated_original(self):
        controls = {}
        import shared.simulate_belief_weighted as SBW
        for regime, params in SBW.REGIME_PARAMS.items():
            da, db, qag, qbg = SBW.build_optimal_control(**params)
            controls[regime] = {"delta_ask": da, "delta_bid": db, "q_ask": qag, "q_bid": qbg}
        diffs = EAC.verify_analytic_reuse_fidelity(controls, [TEST_SEED])
        for agent_type, diff in diffs.items():
            assert diff < EAC.RECONCILIATION_TOL, f"{agent_type}: diverges from validated original by {diff:.3e}"


# ======================================================================
# Item 17: aggregation distinguishes training-seed and evaluation-seed variation
# ======================================================================
def test_hierarchical_summary_distinguishes_within_and_between_seed_variation():
    rng = np.random.default_rng(0)
    rows = []
    for ts in [0, 1, 2]:
        for es in range(5):
            rows.append(dict(
                agent_type="hamilton_ppo", training_seed=ts, evaluation_seed=es,
                raw_pnl=rng.normal() + ts * 10, full_objective=rng.normal() + ts * 10,
                mean_abs_inventory=abs(rng.normal()), terminal_abs_inventory=abs(rng.normal()),
            ))
    df = pd.DataFrame(rows)
    summary = EAC.hierarchical_rl_summary(df, metrics=("raw_pnl",))

    within = summary[summary["level"] == "within_training_seed"]
    between = summary[summary["level"] == "between_training_seeds"]
    assert len(within) == 3  # one row per training seed
    assert len(between) == 1  # one pooled row across the 3 training-seed means
    assert within["n"].tolist() == [5, 5, 5]  # within: n = evaluation seeds per training seed
    assert between["n"].iloc[0] == 3  # between: n = number of training seeds

    # Pooling (aggregate_flat) must NOT reproduce the between-seed n=3 structure --
    # it pools all 15 episodes flat, which is what hierarchical_rl_summary exists to avoid conflating with.
    flat = EAC.aggregate_flat(df, metrics=("raw_pnl",))
    assert flat[flat["agent_type"] == "hamilton_ppo"]["n"].iloc[0] == 15
