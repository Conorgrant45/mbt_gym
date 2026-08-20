"""
test_checkpoint_offline_selection.py
--------------------------------------
Regression coverage for save-all-checkpoints (train_agents.py) and offline
checkpoint selection (select_checkpoint_offline.py, plus
evaluate_agents_common.py's new checkpoint_selection="offline_best").

Deliberately small-scale throughout (n_steps=8, total_timesteps=24, 3
rollouts -- matching this project's established tiny-training-test
convention, e.g. tests/test_train_agents.py's TestPeriodicEvalTiming) --
NOT the 200,000-step dissertation experiment, and NOT even the feature
spec's own smoke-diagnostic scale (n_steps=500/total=1500), which is run
separately as a one-off script for the written report rather than as part
of the pytest suite.

Run from repo root:
    pytest tests/test_checkpoint_offline_selection.py -v
"""

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO

import shared.train_agents as train_agents
from shared.train_agents import (
    compute_checkpoint_timesteps, candidate_checkpoint_path, save_candidate_checkpoint,
    hash_policy_state_dict, PeriodicEvalCallback, build_train_env,
)
import shared.select_checkpoint_offline as SCO


# ======================================================================
# compute_checkpoint_timesteps -- pure function, no training
# ======================================================================
class TestComputeCheckpointTimesteps:
    def test_spec_example(self):
        """The exact example from the feature specification."""
        assert compute_checkpoint_timesteps(n_steps=4000, eval_freq=16000, total_timesteps=200000) == (
            [16000, 32000, 48000, 64000, 80000, 96000, 112000, 128000, 144000, 160000, 176000, 192000, 200000]
        )

    def test_smoke_scale_three_candidates(self):
        """Item 1/3 at the smoke-diagnostic scale: n_steps=eval_freq=500,
        total_timesteps=1500 -> exactly the three candidates the smoke
        diagnostic (section 7) expects."""
        assert compute_checkpoint_timesteps(n_steps=500, eval_freq=500, total_timesteps=1500) == [500, 1000, 1500]

    def test_pytest_scale_three_candidates(self):
        assert compute_checkpoint_timesteps(n_steps=8, eval_freq=8, total_timesteps=24) == [8, 16, 24]

    def test_final_timestep_present_exactly_once_when_it_also_crosses_threshold(self):
        """Item 3: total_timesteps must appear exactly once as a candidate,
        even when it ALSO happens to be an exact eval_freq-multiple rollout
        boundary (the case that could, if handled naively, cause the final
        timestep to be double-counted)."""
        candidates = compute_checkpoint_timesteps(n_steps=8, eval_freq=8, total_timesteps=24)
        assert candidates.count(24) == 1

    def test_final_timestep_present_when_not_a_threshold_multiple(self):
        """total_timesteps=200000 is NOT itself a multiple of eval_freq=16000
        (200000/16000 = 12.5) -- it must still appear, exactly once, as the
        last candidate."""
        candidates = compute_checkpoint_timesteps(n_steps=4000, eval_freq=16000, total_timesteps=200000)
        assert candidates.count(200000) == 1
        assert candidates[-1] == 200000

    def test_timestep_zero_never_a_candidate(self):
        """Item 4: the untrained baseline is never eligible."""
        candidates = compute_checkpoint_timesteps(n_steps=8, eval_freq=8, total_timesteps=24)
        assert 0 not in candidates


# ======================================================================
# candidate_checkpoint_path -- naming convention
# ======================================================================
def test_candidate_checkpoint_path_naming_convention(tmp_path):
    p = candidate_checkpoint_path(tmp_path, "hamilton_ppo", "v1", 16000)
    assert p.name == "ppo_hamilton_ppo_v1_t000016000"
    p2 = candidate_checkpoint_path(tmp_path, "hamilton_ppo", "v1", 200000)
    assert p2.name == "ppo_hamilton_ppo_v1_t000200000"


# ======================================================================
# save_candidate_checkpoint -- unit-level overwrite protection + hashing
# ======================================================================
def _tiny_ppo_model(seed=0):
    def _init():
        return Monitor(build_train_env("return_mlp_ppo", seed, 10.0, 0.005))
    vec_env = DummyVecEnv([_init])
    return PPO(
        "MlpPolicy", vec_env, n_steps=8, batch_size=8, n_epochs=2,
        policy_kwargs=dict(net_arch=[8, 8]), device="cpu", seed=seed, verbose=0,
    )


def test_save_candidate_checkpoint_overwrite_protection_and_hash(tmp_path):
    model = _tiny_ppo_model(seed=1)
    checkpoints_dir = tmp_path / "ckpts"

    entry1 = save_candidate_checkpoint(model, checkpoints_dir, "return_mlp_ppo", "ovwtest", 100,
                                        overwrite=False, is_final=False)
    assert Path(entry1["path"]).exists()
    assert entry1["timestep"] == 100
    assert entry1["is_final"] is False
    assert entry1["policy_hash"] == hash_policy_state_dict(model)
    assert len(entry1["policy_hash"]) == 64  # full sha256 hex digest

    # Item 6: refuses to overwrite without the flag.
    with pytest.raises(FileExistsError):
        save_candidate_checkpoint(model, checkpoints_dir, "return_mlp_ppo", "ovwtest", 100,
                                   overwrite=False, is_final=False)

    # ...but succeeds when explicitly allowed.
    entry2 = save_candidate_checkpoint(model, checkpoints_dir, "return_mlp_ppo", "ovwtest", 100,
                                        overwrite=True, is_final=True)
    assert entry2["path"] == entry1["path"]
    assert entry2["is_final"] is True


# ======================================================================
# Item 2: checkpoint saved AFTER its PPO update, never before.
# ======================================================================
def test_checkpoint_saved_matches_post_train_weights_not_pre_train(tmp_path):
    model = _tiny_ppo_model(seed=0)

    post_train_snapshots = []
    original_train = model.train

    def snapshotting_train():
        original_train()
        w = next(model.policy.parameters()).detach().clone().numpy().copy()
        post_train_snapshots.append(w)

    model.train = snapshotting_train

    manifest = []
    callback = PeriodicEvalCallback(
        agent_type="return_mlp_ppo", eval_seeds=[130_201], inventory_scale=10.0, return_scale=0.005,
        eval_freq=8, training_curve=[], verbose=0,
        checkpoints_dir=tmp_path / "ckpts", run_tag="timingtest", checkpoint_manifest=manifest,
    )
    model.learn(total_timesteps=16, callback=callback)  # rollout 1 trains+checkpoints, rollout 2 trains only

    assert len(manifest) == 1
    assert manifest[0]["timestep"] == 8

    reloaded = PPO.load(manifest[0]["path"])
    reloaded_w = next(reloaded.policy.parameters()).detach().numpy()

    np.testing.assert_array_equal(reloaded_w, post_train_snapshots[0])


# ======================================================================
# End-to-end: train_agents.py --save-all-checkpoints, real (tiny) training.
# ======================================================================
def _run_tiny_training(tmp_path, agent_type, run_tag, learner_seed, env_seed,
                        save_all=True, overwrite_checkpoints=False, extra_argv=None):
    """Deliberately does NOT pass --output-dir/--log-dir: train_agents.py's
    own defaults (REPO_ROOT/"models"/agent_type, REPO_ROOT/"logs"/agent_type
    -- the agent_type-nested convention select_checkpoint_offline.py and
    evaluate_agents_common.py both also assume) are exercised via the
    REPO_ROOT patch alone, so paths line up with production usage exactly."""
    argv = [
        "train_agents.py",
        "--agent-type", agent_type,
        "--learner-seed", str(learner_seed),
        "--env-seed", str(env_seed),
        "--total-timesteps", "24",
        "--n-steps", "8",
        "--batch-size", "8",
        "--n-epochs", "2",
        "--net-arch", "8", "8",
        "--eval-freq", "8",
        "--eval-seeds", "130202",
        "--run-tag", run_tag,
    ]
    if agent_type == "return_lstm_ppo":
        argv += ["--lstm-hidden-size", "8", "--n-lstm-layers", "1"]
    if save_all:
        argv.append("--save-all-checkpoints")
    if overwrite_checkpoints:
        argv.append("--overwrite-checkpoints")
    if extra_argv:
        argv += extra_argv

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(train_agents, "REPO_ROOT", tmp_path)
        mp.setattr(sys, "argv", argv)
        train_agents.main()

    return dict(
        tmp_path=tmp_path,
        output_dir=tmp_path / "models" / agent_type,
        log_dir=tmp_path / "logs" / agent_type,
        results_dir=tmp_path / "results",
    )


def test_save_all_checkpoints_end_to_end(tmp_path):
    """Items 1, 3, 4, 5, 7, 18: exact candidate timesteps, final present
    once, timestep-zero excluded, final/best files still produced, manifest
    schema/content correct (including both seed fields)."""
    paths = _run_tiny_training(tmp_path, "return_mlp_ppo", "e2e", learner_seed=7, env_seed=13)

    manifest_path = paths["log_dir"] / "checkpoint_manifest_e2e.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())

    assert manifest["agent_type"] == "return_mlp_ppo"
    assert manifest["run_tag"] == "e2e"
    assert manifest["learner_seed"] == 7          # item 18: learner/env seed separation intact
    assert manifest["training_env_seed"] == 13
    assert manifest["total_timesteps"] == 24
    assert manifest["checkpoint_frequency"] == 8

    timesteps = sorted(c["timestep"] for c in manifest["checkpoints"])
    assert timesteps == [8, 16, 24]                # item 1
    assert timesteps.count(24) == 1                # item 3
    assert 0 not in timesteps                      # item 4

    final_entries = [c for c in manifest["checkpoints"] if c["is_final"]]
    assert [c["timestep"] for c in final_entries] == [24]

    for c in manifest["checkpoints"]:
        assert Path(c["path"]).exists()             # item 7: paths correct
        assert len(c["policy_hash"]) == 64           # item 7: hashes present
        assert "saved_at" in c and c["saved_at"]

    # item 5: final + online-best files still produced, unaffected.
    output_dir = paths["output_dir"]
    assert (output_dir / "ppo_return_mlp_ppo_e2e.zip").exists()
    assert (output_dir / "ppo_return_mlp_ppo_e2e_best.zip").exists()

    # cross-check: the is_final candidate's hash must equal the separately
    # -saved final model's own hash (both represent the identical trained
    # policy, saved with no training in between).
    final_model = PPO.load(str(output_dir / "ppo_return_mlp_ppo_e2e.zip"))
    assert hash_policy_state_dict(final_model) == final_entries[0]["policy_hash"]

    # item 18 (continued): run_config/run_summary still correctly separate
    # learner_seed/training_env_seed (pre-existing feature, unaffected).
    run_summary = json.loads((paths["log_dir"] / "run_summary_e2e.json").read_text())
    assert run_summary["learner_seed"] == 7
    assert run_summary["training_env_seed"] == 13


def test_save_all_checkpoints_overwrite_protection(tmp_path):
    """Item 6, at the main()/CLI level: a second training run with the same
    run-tag and --save-all-checkpoints, without --overwrite-checkpoints,
    must fail before/without silently clobbering the existing candidates."""
    _run_tiny_training(tmp_path, "return_mlp_ppo", "ovw", learner_seed=1, env_seed=1)

    with pytest.raises(FileExistsError):
        _run_tiny_training(tmp_path, "return_mlp_ppo", "ovw", learner_seed=2, env_seed=2)

    # ...but succeeds with --overwrite-checkpoints.
    _run_tiny_training(tmp_path, "return_mlp_ppo", "ovw", learner_seed=2, env_seed=2,
                        overwrite_checkpoints=True)
    manifest = json.loads((tmp_path / "logs" / "return_mlp_ppo" / "checkpoint_manifest_ovw.json").read_text())
    assert manifest["learner_seed"] == 2  # the overwritten run's own metadata, not the first run's


def test_without_save_all_checkpoints_behaves_exactly_as_before(tmp_path):
    """Item 17: omitting --save-all-checkpoints must leave no trace of the
    new feature at all -- no checkpoints/ directory, no manifest -- while
    final/best model files are still produced normally."""
    paths = _run_tiny_training(tmp_path, "return_mlp_ppo", "oldcmd", learner_seed=4, env_seed=9, save_all=False)

    output_dir = paths["output_dir"]
    log_dir = paths["log_dir"]
    assert not (output_dir / "checkpoints").exists()
    assert not (log_dir / "checkpoint_manifest_oldcmd.json").exists()
    assert (output_dir / "ppo_return_mlp_ppo_oldcmd.zip").exists()
    assert (output_dir / "ppo_return_mlp_ppo_oldcmd_best.zip").exists()


def test_save_all_checkpoints_does_not_mutate_training_env_rng(tmp_path):
    """Item 19: with save-all-checkpoints enabled (a NEW code path at the
    same _run_eval() call site as periodic evaluation / online-best
    saving), the training environment's own regime_rng must still end in
    the SAME state as an otherwise-identical run with no periodic
    evaluation/checkpointing at all -- i.e. saving a checkpoint (disk I/O)
    must not perturb any RNG stream any more than evaluation itself does
    (already proven not to, in tests/test_seed_separation.py)."""
    env_seed = 21
    learner_seed = 0

    def _build_and_learn(with_save_all: bool):
        wrapped = build_train_env("return_mlp_ppo", env_seed, 10.0, 0.005)
        vec_env = DummyVecEnv([lambda: Monitor(wrapped)])
        model = PPO(
            "MlpPolicy", vec_env, n_steps=8, batch_size=8, n_epochs=2,
            policy_kwargs=dict(net_arch=[8, 8]), device="cpu", seed=learner_seed, verbose=0,
        )
        callbacks = []
        if with_save_all:
            callbacks.append(PeriodicEvalCallback(
                agent_type="return_mlp_ppo", eval_seeds=[130_203], inventory_scale=10.0, return_scale=0.005,
                eval_freq=8, training_curve=[], verbose=0,
                checkpoints_dir=tmp_path / "rng_ckpts", run_tag="rngtest", checkpoint_manifest=[],
            ))
        model.learn(total_timesteps=16, callback=callbacks)
        return wrapped.base_env._regime_rng.bit_generator.state

    state_with_save_all = _build_and_learn(with_save_all=True)
    state_without = _build_and_learn(with_save_all=False)
    assert state_with_save_all == state_without


# ======================================================================
# select_checkpoint_offline.py -- internals (no full training required)
# ======================================================================
class TestSelectBest:
    def test_highest_mean_selected(self):
        rows = [
            dict(checkpoint_timestep=8, mean_cumulative_reward=1.0),
            dict(checkpoint_timestep=16, mean_cumulative_reward=5.0),
            dict(checkpoint_timestep=24, mean_cumulative_reward=3.0),
        ]
        assert SCO.select_best(rows)["checkpoint_timestep"] == 16

    def test_exact_tie_resolved_to_earlier_timestep(self):
        rows = [
            dict(checkpoint_timestep=24, mean_cumulative_reward=5.0),
            dict(checkpoint_timestep=8, mean_cumulative_reward=5.0),
            dict(checkpoint_timestep=16, mean_cumulative_reward=5.0),
        ]
        assert SCO.select_best(rows)["checkpoint_timestep"] == 8


def test_copy_and_verify_hash_identical_and_overwrite_protection(tmp_path):
    src = tmp_path / "src.zip"
    src.write_bytes(b"fake model bytes for a hashing/copy test")
    dest = tmp_path / "sub" / "dest.zip"

    result = SCO.copy_and_verify(str(src), dest, overwrite=False)
    assert result["source_hash"] == result["copied_hash"]
    assert dest.read_bytes() == src.read_bytes()

    with pytest.raises(FileExistsError):
        SCO.copy_and_verify(str(src), dest, overwrite=False)
    SCO.copy_and_verify(str(src), dest, overwrite=True)  # succeeds


def test_select_checkpoint_offline_never_references_holdout_seeds():
    """Item 14: the selector must not use, inspect, or accept holdout/test
    seeds -- checked functionally (no live dependency on the module that
    owns the holdout-seed range; no CLI argument whose name mentions
    holdout), not by banning the word "holdout" from source text, since
    explaining this constraint in comments/docstrings (as this script's own
    module docstring does) is expected and desirable, not a violation."""
    assert not hasattr(SCO, "DEFAULT_HOLDOUT_SEEDS")
    assert not hasattr(SCO, "evaluate_agents_common")
    assert "evaluate_agents_common" not in sys.modules or not any(
        line.strip().startswith(("import evaluate_agents_common", "from evaluate_agents_common"))
        for line in inspect.getsource(SCO).splitlines()
    )

    parser_source = inspect.getsource(SCO.parse_args)
    assert "--validation-seeds-start" in parser_source
    assert "--validation-seeds-count" in parser_source

    # No CLI flag anywhere in this script's argparse definition mentions "holdout".
    for line in parser_source.splitlines():
        if "add_argument" in line and "--" in line:
            assert "holdout" not in line.lower()


# ======================================================================
# select_checkpoint_offline.py -- end-to-end, on REAL (tiny) checkpoints.
# Module-scoped fixtures: train ONCE, reuse across several assertions.
# ======================================================================
@pytest.fixture(scope="module")
def ff_checkpoints(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("offline_ff")
    paths = _run_tiny_training(tmp_path, "return_mlp_ppo", "offline_ff", learner_seed=3, env_seed=99)
    manifest = json.loads((paths["log_dir"] / "checkpoint_manifest_offline_ff.json").read_text())
    return dict(**paths, manifest=manifest)


@pytest.fixture(scope="module")
def lstm_checkpoints(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("offline_lstm")
    paths = _run_tiny_training(tmp_path, "return_lstm_ppo", "offline_lstm", learner_seed=5, env_seed=101)
    manifest = json.loads((paths["log_dir"] / "checkpoint_manifest_offline_lstm.json").read_text())
    return dict(**paths, manifest=manifest)


def test_offline_evaluate_all_candidates_uses_same_validation_seeds(ff_checkpoints):
    """Item 8."""
    validation_seeds = [140_001, 140_002, 140_003]
    checkpoint_rows, episode_rows = SCO.evaluate_all_candidates(ff_checkpoints["manifest"], validation_seeds)

    assert len(checkpoint_rows) == 3
    for row in checkpoint_rows:
        assert row["n_validation_episodes"] == 3
        assert row["validation_seed_start"] == 140_001
        assert row["validation_seed_count"] == 3

    seeds_per_checkpoint = {}
    for r in episode_rows:
        seeds_per_checkpoint.setdefault(r["checkpoint_timestep"], set()).add(r["evaluation_seed"])
    assert len(seeds_per_checkpoint) == 3
    assert all(s == set(validation_seeds) for s in seeds_per_checkpoint.values())


def test_offline_evaluate_all_candidates_deterministic_rerun(ff_checkpoints):
    """Item 9."""
    validation_seeds = [140_101, 140_102, 140_103]
    rows_a, _ = SCO.evaluate_all_candidates(ff_checkpoints["manifest"], validation_seeds)
    rows_b, _ = SCO.evaluate_all_candidates(ff_checkpoints["manifest"], validation_seeds)

    rows_a = sorted(rows_a, key=lambda r: r["checkpoint_timestep"])
    rows_b = sorted(rows_b, key=lambda r: r["checkpoint_timestep"])
    for a, b in zip(rows_a, rows_b):
        assert a["mean_cumulative_reward"] == b["mean_cumulative_reward"]
        assert a["policy_hash"] == b["policy_hash"]


def _run_offline_selection(tmp_path, agent_type, run_tag, learner_seed, env_seed,
                            validation_start, validation_count, overwrite=False):
    argv = [
        "select_checkpoint_offline.py",
        "--agent-type", agent_type,
        "--run-tag", run_tag,
        "--learner-seed", str(learner_seed),
        "--training-env-seed", str(env_seed),
        "--validation-seeds-start", str(validation_start),
        "--validation-seeds-count", str(validation_count),
    ]
    if overwrite:
        argv.append("--overwrite-selection")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SCO, "RESULTS_DIR", tmp_path / "results")
        mp.setattr(SCO, "MODELS_DIR", tmp_path / "models")
        mp.setattr(SCO, "LOGS_DIR", tmp_path / "logs")
        mp.setattr(sys, "argv", argv)
        SCO.main()


def test_offline_selection_end_to_end_outputs(ff_checkpoints):
    """Items 10, 11 (exercised via select_best, tested directly above; here
    just confirms exactly one row is flagged selected and it is consistent
    with the highest mean), 12."""
    tmp_path = ff_checkpoints["tmp_path"]
    _run_offline_selection(tmp_path, "return_mlp_ppo", "offline_ff", learner_seed=3, env_seed=99,
                            validation_start=140_201, validation_count=3)

    checkpoint_csv = tmp_path / "results" / "return_mlp_ppo_checkpoint_validation_offline_ff.csv"
    episodes_csv = tmp_path / "results" / "return_mlp_ppo_checkpoint_validation_episodes_offline_ff.csv"
    selection_json = tmp_path / "logs" / "return_mlp_ppo" / "offline_selection_offline_ff.json"
    offline_best = tmp_path / "models" / "return_mlp_ppo" / "ppo_return_mlp_ppo_offline_ff_offline_best.zip"

    assert checkpoint_csv.exists() and episodes_csv.exists() and selection_json.exists() and offline_best.exists()

    import pandas as pd
    ckpt_df = pd.read_csv(checkpoint_csv)
    assert len(ckpt_df) == 3
    assert ckpt_df["selected"].sum() == 1
    best_row = ckpt_df.loc[ckpt_df["selected"]].iloc[0]
    assert best_row["mean_cumulative_reward"] == ckpt_df["mean_cumulative_reward"].max()

    episodes_df = pd.read_csv(episodes_csv)
    assert len(episodes_df) == 3 * 3  # 3 checkpoints x 3 validation episodes

    selection = json.loads(selection_json.read_text())
    assert selection["selected_timestep"] == int(best_row["checkpoint_timestep"])
    assert selection["validation_seeds"] == [140_201, 140_202, 140_203]
    assert selection["candidate_timesteps"] == [8, 16, 24]
    assert selection["selection_metric"] == "mean_objective"
    assert selection["tie_break_rule"] == "earlier_timestep"

    # item 12: copied model hash-identical to its selected source.
    assert selection["hash_verified"] is True
    assert selection["source_policy_hash"] == selection["copied_policy_hash"]
    assert offline_best.read_bytes() == Path(best_row["checkpoint_path"]).read_bytes()

    # overwrite protection on re-run.
    with pytest.raises(FileExistsError):
        _run_offline_selection(tmp_path, "return_mlp_ppo", "offline_ff", learner_seed=3, env_seed=99,
                                validation_start=140_201, validation_count=3)
    _run_offline_selection(tmp_path, "return_mlp_ppo", "offline_ff", learner_seed=3, env_seed=99,
                            validation_start=140_201, validation_count=3, overwrite=True)


def test_offline_selection_learner_env_seed_mismatch_raises(ff_checkpoints):
    with pytest.raises(ValueError):
        _run_offline_selection(ff_checkpoints["tmp_path"], "return_mlp_ppo", "offline_ff",
                                learner_seed=999, env_seed=99, validation_start=140_301, validation_count=2)


# ======================================================================
# Item 13: recurrent evaluation resets LSTM state correctly at episode
# boundaries, through the offline-selection code path.
# ======================================================================
def test_offline_selection_recurrent_uses_recurrent_ppo(lstm_checkpoints, monkeypatch):
    load_calls = []
    original_load = RecurrentPPO.load

    def spy_load(*a, **kw):
        load_calls.append(a)
        return original_load(*a, **kw)

    monkeypatch.setattr(RecurrentPPO, "load", spy_load)

    validation_seeds = [140_401, 140_402]
    checkpoint_rows, episode_rows = SCO.evaluate_all_candidates(lstm_checkpoints["manifest"], validation_seeds)

    assert len(load_calls) == len(lstm_checkpoints["manifest"]["checkpoints"])
    assert len(checkpoint_rows) == len(lstm_checkpoints["manifest"]["checkpoints"])
    for r in episode_rows:
        assert r["steps"] == train_agents.N_STEPS  # completed a full episode without crashing


def test_recurrent_episode_results_independent_of_evaluation_order(lstm_checkpoints):
    """LSTM state must be reset (never carried over) at every episode
    boundary -- proven directly: a given validation seed's result must be
    IDENTICAL regardless of what seed was evaluated immediately before it
    within the same evaluate_policy() call."""
    checkpoint_path = sorted(lstm_checkpoints["manifest"]["checkpoints"], key=lambda c: c["timestep"])[0]["path"]
    model = RecurrentPPO.load(checkpoint_path)

    _, records_ab = train_agents.evaluate_policy(model, "return_lstm_ppo", [140_501, 140_502], 10.0, 0.005,
                                                  deterministic=True)
    _, records_ba = train_agents.evaluate_policy(model, "return_lstm_ppo", [140_502, 140_501], 10.0, 0.005,
                                                  deterministic=True)

    reward_ab_502 = next(r["cumulative_reward"] for r in records_ab if r["seed"] == 140_502)
    reward_ba_502 = next(r["cumulative_reward"] for r in records_ba if r["seed"] == 140_502)
    assert reward_ab_502 == reward_ba_502


# ======================================================================
# evaluate_agents_common.py: offline_best loading (items 15, 16).
# ======================================================================
def test_evaluate_agents_common_resolves_and_loads_offline_best(ff_checkpoints, monkeypatch):
    import shared.evaluate_agents_common as EAC

    tmp_path = ff_checkpoints["tmp_path"]
    _run_offline_selection(tmp_path, "return_mlp_ppo", "offline_ff", learner_seed=3, env_seed=99,
                            validation_start=140_601, validation_count=2, overwrite=True)

    monkeypatch.setattr(EAC, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(EAC, "LOGS_DIR", tmp_path / "logs")

    path = EAC.resolve_model_path("return_mlp_ppo", "offline_ff", "offline_best")
    assert path == tmp_path / "models" / "return_mlp_ppo" / "ppo_return_mlp_ppo_offline_ff_offline_best.zip"
    assert path.exists()

    meta = EAC.load_offline_selection_metadata("return_mlp_ppo", "offline_ff")
    assert meta["selected_timestep"] in (8, 16, 24)
    assert meta["validation_seed_count"] == 2

    m = dict(
        evaluation_seed=1, raw_pnl=0.0, full_objective=0.0, spread_revenue=0.0,
        adverse_selection_loss=0.0, running_penalty=0.0, terminal_penalty=0.0,
        mean_abs_inventory=0.0, max_abs_inventory=0.0, mean_signed_inventory=0.0,
        terminal_signed_inventory=0.0, terminal_abs_inventory=0.0, fills=0.0,
        bid_action_mean=0.0, ask_action_mean=0.0, bid_action_std=0.0, ask_action_std=0.0,
        mean_quoted_spread=0.0, episode_length=1, reward_reconciliation_error=0.0,
        recurrent_state_reset_verified=None,
    )
    row = EAC.build_row(
        "return_mlp_ppo", 3, path, 24, "exogenous_paired_fills_not_claimed_paired", m,
        checkpoint_selection="offline_best", loaded_model_path=path, training_env_seed=99,
        selected_checkpoint_timestep=meta["selected_timestep"],
        selected_validation_mean=meta["selected_mean_cumulative_reward"],
        selected_validation_std=meta["selected_std_cumulative_reward"],
        validation_seed_count=meta["validation_seed_count"],
    )
    assert row["checkpoint_selection"] == "offline_best"
    assert row["selected_checkpoint_timestep"] == meta["selected_timestep"]
    assert row["validation_seed_count"] == 2
    assert set(row.keys()) == set(EAC.SCHEMA_COLUMNS)


def test_load_offline_selection_metadata_missing_raises_not_silent_fallback(tmp_path, monkeypatch):
    """Item 16."""
    import shared.evaluate_agents_common as EAC
    monkeypatch.setattr(EAC, "LOGS_DIR", tmp_path / "logs")
    with pytest.raises(FileNotFoundError):
        EAC.load_offline_selection_metadata("return_mlp_ppo", "nonexistent_run_tag")


def test_check_all_models_exist_raises_for_missing_offline_best_file(tmp_path, monkeypatch):
    """Item 16, the other failure point: a resolved offline_best path that
    does not exist on disk must also raise (via the existing
    check_all_models_exist gate), not silently fall back."""
    import shared.evaluate_agents_common as EAC
    monkeypatch.setattr(EAC, "MODELS_DIR", tmp_path / "models")
    model_paths = {"return_mlp_ppo": EAC.resolve_model_path("return_mlp_ppo", "nope", "offline_best")}
    with pytest.raises(FileNotFoundError):
        EAC.check_all_models_exist(model_paths, "offline_best", "nope")
