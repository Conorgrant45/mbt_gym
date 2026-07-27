"""
test_checkpoint_selection.py
------------------------------------
Focused regression coverage for evaluate_agents_common.py's
--checkpoint-selection {final,best} extension: resolving the correct
model path per selection, failing clearly when a required file is
missing, reading best-validation metadata from run_summary_<run_tag>.json,
and recording checkpoint_selection/loaded_model_path/best_validation_*
correctly on every learned-agent row (never populated for analytic rows).

Does not modify any environment, reward, observation, policy, accounting,
or evaluation-mechanics code -- only the model-path/metadata plumbing
around evaluate_all_agents.

Run from repo root:
    pytest tests/test_checkpoint_selection.py -v
"""

import json
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import evaluate_agents_common as EAC
from envs.return_ppo_wrapper import ReturnPPOWrapper


# ======================================================================
# Pure path-resolution logic (items 3, 4)
# ======================================================================
class TestResolveModelPath:
    def test_final_selection(self):
        p = EAC.resolve_model_path("hamilton_ppo", "seed_0", "final")
        assert p == EAC.MODELS_DIR / "hamilton_ppo" / "ppo_hamilton_ppo_seed_0.zip"

    def test_best_selection(self):
        p = EAC.resolve_model_path("return_lstm_ppo", "seed_0", "best")
        assert p == EAC.MODELS_DIR / "return_lstm_ppo" / "ppo_return_lstm_ppo_seed_0_best.zip"

    def test_final_and_best_paths_are_distinct(self):
        final_p = EAC.resolve_model_path("return_mlp_ppo", "seed_2", "final")
        best_p = EAC.resolve_model_path("return_mlp_ppo", "seed_2", "best")
        assert final_p != best_p
        assert final_p.name == "ppo_return_mlp_ppo_seed_2.zip"
        assert best_p.name == "ppo_return_mlp_ppo_seed_2_best.zip"

    def test_invalid_selection_raises(self):
        with pytest.raises(ValueError):
            EAC.resolve_model_path("hamilton_ppo", "seed_0", "latest")


# ======================================================================
# Item 5: fail clearly before evaluation if a required model file is missing
# ======================================================================
class TestCheckAllModelsExist:
    def test_missing_best_checkpoint_raises_clear_error(self, tmp_path):
        # Only the FINAL file exists -- requesting "best" must fail clearly,
        # not silently fall back or skip.
        agent_dir = tmp_path / "hamilton_ppo"
        agent_dir.mkdir()
        (agent_dir / "ppo_hamilton_ppo_seed_0.zip").touch()

        model_paths = {"hamilton_ppo": agent_dir / "ppo_hamilton_ppo_seed_0_best.zip"}
        with pytest.raises(FileNotFoundError) as exc_info:
            EAC.check_all_models_exist(model_paths, "best", run_tag="seed_0")
        assert "ppo_hamilton_ppo_seed_0_best.zip" in str(exc_info.value)
        assert "best" in str(exc_info.value)

    def test_passes_when_all_present(self, tmp_path):
        agent_dir = tmp_path / "hamilton_ppo"
        agent_dir.mkdir()
        p = agent_dir / "ppo_hamilton_ppo_seed_0.zip"
        p.touch()
        EAC.check_all_models_exist({"hamilton_ppo": p}, "final", run_tag="seed_0")  # must not raise

    def test_lists_every_missing_file_not_just_the_first(self, tmp_path):
        model_paths = {
            "hamilton_ppo": tmp_path / "a" / "missing1.zip",
            "return_mlp_ppo": tmp_path / "b" / "missing2.zip",
        }
        with pytest.raises(FileNotFoundError) as exc_info:
            EAC.check_all_models_exist(model_paths, "best", run_tag="x")
        msg = str(exc_info.value)
        assert "missing1.zip" in msg
        assert "missing2.zip" in msg


# ======================================================================
# Item 6: best-validation metadata read from run_summary_<run_tag>.json
# ======================================================================
class TestLoadBestValidationMetadata:
    def test_missing_run_summary_returns_none_none(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(EAC, "LOGS_DIR", tmp_path)
        result = EAC.load_best_validation_metadata("hamilton_ppo", "nonexistent_tag")
        assert result == (None, None)
        assert "WARNING" in capsys.readouterr().out

    def test_run_summary_without_best_fields_returns_none_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(EAC, "LOGS_DIR", tmp_path)
        log_dir = tmp_path / "hamilton_ppo"
        log_dir.mkdir()
        (log_dir / "run_summary_old_tag.json").write_text(json.dumps({"agent_type": "hamilton_ppo"}))
        assert EAC.load_best_validation_metadata("hamilton_ppo", "old_tag") == (None, None)

    def test_run_summary_with_best_fields_reads_correctly(self, monkeypatch, tmp_path):
        monkeypatch.setattr(EAC, "LOGS_DIR", tmp_path)
        log_dir = tmp_path / "return_lstm_ppo"
        log_dir.mkdir()
        (log_dir / "run_summary_seed_3.json").write_text(json.dumps({
            "best_validation_timestep": 160_000,
            "best_validation_mean_objective": 42.5,
        }))
        timestep, objective = EAC.load_best_validation_metadata("return_lstm_ppo", "seed_3")
        assert timestep == 160_000
        assert objective == 42.5


# ======================================================================
# Item 6/7: row-metadata population via build_row
# ======================================================================
class TestBuildRowCheckpointMetadata:
    FAKE_M = dict(
        evaluation_seed=1, raw_pnl=1.0, full_objective=1.0, spread_revenue=0.0,
        adverse_selection_loss=0.0, running_penalty=0.0, terminal_penalty=0.0,
        mean_abs_inventory=0.0, max_abs_inventory=0.0, mean_signed_inventory=0.0,
        terminal_signed_inventory=0.0, terminal_abs_inventory=0.0, fills=0,
        bid_action_mean=0.0, ask_action_mean=0.0, bid_action_std=0.0, ask_action_std=0.0,
        mean_quoted_spread=0.0, episode_length=4000, reward_reconciliation_error=0.0,
        recurrent_state_reset_verified=None,
    )

    def test_analytic_row_defaults(self):
        """Item 7: analytic rows get checkpoint_selection='analytic' and empty learned-model metadata."""
        row = EAC.build_row("naive", None, None, None, "exogenous_paired_fills_not_claimed_paired", self.FAKE_M)
        assert row["checkpoint_selection"] == "analytic"
        assert row["loaded_model_path"] is None
        assert row["best_validation_timestep"] is None
        assert row["best_validation_mean_objective"] is None

    def test_rl_row_records_full_checkpoint_metadata(self):
        row = EAC.build_row(
            "return_lstm_ppo", 3, "/models/x.zip", 200_000, "exogenous_paired_fills_not_claimed_paired", self.FAKE_M,
            checkpoint_selection="best", loaded_model_path="/models/return_lstm_ppo/ppo_return_lstm_ppo_seed_3_best.zip",
            best_validation_timestep=160_000, best_validation_mean_objective=42.5,
        )
        assert row["checkpoint_selection"] == "best"
        assert row["loaded_model_path"] == "/models/return_lstm_ppo/ppo_return_lstm_ppo_seed_3_best.zip"
        assert row["best_validation_timestep"] == 160_000
        assert row["best_validation_mean_objective"] == 42.5

    def test_final_selection_retains_best_validation_metadata_as_context(self):
        """Item 6, last sentence: a 'final' row still carries the
        best_validation_* contextual fields when available -- it's only
        checkpoint_selection itself that must say 'final', not a blanking
        of the other metadata."""
        row = EAC.build_row(
            "hamilton_ppo", 0, "/models/x.zip", 200_000, "exogenous_paired_fills_not_claimed_paired", self.FAKE_M,
            checkpoint_selection="final", loaded_model_path="/models/hamilton_ppo/ppo_hamilton_ppo_seed_0.zip",
            best_validation_timestep=180_000, best_validation_mean_objective=30.1,
        )
        assert row["checkpoint_selection"] == "final"
        assert row["best_validation_timestep"] == 180_000
        assert row["best_validation_mean_objective"] == 30.1

    def test_row_keys_match_schema_columns(self):
        row = EAC.build_row("naive", None, None, None, "exogenous_paired_fills_not_claimed_paired", self.FAKE_M)
        assert set(row.keys()) == set(EAC.SCHEMA_COLUMNS)


# ======================================================================
# Item 9: real feed-forward and recurrent model loading, final vs best,
# via the exact same resolve_model_path()+model_cls.load() path
# evaluate_all_agents itself uses. Two tiny models with DIFFERENT weights
# are saved under the "final" and "best" naming conventions in a tmp
# models dir; loading via each selection must retrieve the correspondingly
# DIFFERENT weights, proving the right file (not merely a plausible one)
# was loaded for each selection.
# ======================================================================
def _tiny_ppo(seed):
    def _init():
        return Monitor(ReturnPPOWrapper(inventory_scale=10.0, return_scale=0.005, seed=seed))
    vec_env = DummyVecEnv([_init])
    return PPO("MlpPolicy", vec_env, n_steps=8, batch_size=8,
               policy_kwargs=dict(net_arch=[8, 8]), device="cpu", seed=seed, verbose=0)


def _tiny_recurrent_ppo(seed):
    def _init():
        return Monitor(ReturnPPOWrapper(inventory_scale=10.0, return_scale=0.005, seed=seed))
    vec_env = DummyVecEnv([_init])
    return RecurrentPPO("MlpLstmPolicy", vec_env, n_steps=8, batch_size=8,
                         policy_kwargs=dict(net_arch=[8, 8], lstm_hidden_size=8, n_lstm_layers=1),
                         device="cpu", seed=seed, verbose=0)


@pytest.mark.parametrize("agent_type,build_fn,model_cls", [
    ("return_mlp_ppo", _tiny_ppo, PPO),
    ("return_lstm_ppo", _tiny_recurrent_ppo, RecurrentPPO),
])
def test_final_and_best_load_the_correct_distinct_weights(agent_type, build_fn, model_cls, monkeypatch, tmp_path):
    monkeypatch.setattr(EAC, "MODELS_DIR", tmp_path)
    agent_dir = tmp_path / agent_type
    agent_dir.mkdir()

    def first_weight_matrix(model):
        # Skip bias vectors (often zero-initialised regardless of seed) --
        # find the first parameter with a genuinely seed-dependent random
        # initialisation (a 2D+ weight matrix).
        for p in model.policy.parameters():
            if p.dim() >= 2:
                return p.detach().numpy().copy()
        raise AssertionError("no weight-matrix parameter found")

    final_model = build_fn(seed=11)
    best_model = build_fn(seed=22)
    final_path = EAC.resolve_model_path(agent_type, "unittest", "final")
    best_path = EAC.resolve_model_path(agent_type, "unittest", "best")
    final_model.save(str(final_path))
    best_model.save(str(best_path))

    final_w = first_weight_matrix(final_model)
    best_w = first_weight_matrix(best_model)
    assert not np.allclose(final_w, best_w), "test fixture models must have different weights"

    loaded_final = model_cls.load(str(EAC.resolve_model_path(agent_type, "unittest", "final")))
    loaded_best = model_cls.load(str(EAC.resolve_model_path(agent_type, "unittest", "best")))

    np.testing.assert_array_equal(first_weight_matrix(loaded_final), final_w)
    np.testing.assert_array_equal(first_weight_matrix(loaded_best), best_w)
