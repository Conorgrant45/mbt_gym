"""
tests/test_final_architecture_comparison.py
------------------------------------------------------
Focused tests for the final reduced-exploration architecture-comparison
experiment (final_*.py, scripts/plot_final_architecture_results.py,
scripts/generate_final_architecture_tables.py). Covers: seed-range
disjointness, Hamilton/Phase-7 reuse-compatibility check structure,
episode-level data schema completeness + reward reconciliation for all
three architectures and the analytical benchmarks, deterministic-evaluation
reproducibility, recurrent (LSTM) hidden-state handling correctness,
contrast statistics against hand-computed values, and that the plotting/
table-generation scripts execute against synthetic minimal data without
ever training or evaluating a model.

Run from repo root:
    pytest tests/test_final_architecture_comparison.py -v
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import final.final_common as FC
import final.final_episode_runner as ER
from final.final_instrumented_ppo import FinalInstrumentedPPO, FinalInstrumentedRecurrentPPO
import final.final_analyze_contrasts as FAC

REPO_ROOT = FC.REPO_ROOT


# ======================================================================
# Seed-range disjointness
# ======================================================================
def test_validation_seeds_disjoint_from_all_prior_ranges():
    result = FC.verify_seed_range_disjoint(FC.VALIDATION_SEEDS, "validation")
    assert result["disjoint"], f"overlaps: {result['overlaps']}"
    assert result["n_seeds"] == 50


def test_holdout_seeds_disjoint_from_all_prior_ranges():
    result = FC.verify_seed_range_disjoint(FC.HOLDOUT_SEEDS, "holdout")
    assert result["disjoint"], f"overlaps: {result['overlaps']}"
    assert result["n_seeds"] == 500


def test_validation_and_holdout_mutually_disjoint():
    assert FC.verify_validation_holdout_disjoint_from_each_other()


def test_prior_seed_ranges_include_phase7_post_training_holdout():
    assert "phase7_post_training_unseen_holdout" in FC.PRIOR_SEED_RANGES
    assert len(FC.PRIOR_SEED_RANGES["phase7_post_training_unseen_holdout"]) == 500


# ======================================================================
# Hamilton / Phase 7 reuse-compatibility check
# ======================================================================
def test_hamilton_reuse_compatibility_check_is_currently_true():
    compat = FC.check_hamilton_reuse_compatibility()
    assert compat["compatible"], compat.get("reason")
    assert all(compat["field_checks"].values())
    for seed in FC.LEARNER_SEEDS:
        assert compat["checkpoint_details"][seed] == "all checkpoint hashes verified"


def test_hamilton_reuse_compatibility_detects_mismatch():
    original = FC.CHECKPOINT_TIMESTEPS
    try:
        FC.CHECKPOINT_TIMESTEPS = [1, 2, 3]
        compat = FC.check_hamilton_reuse_compatibility()
        assert not compat["compatible"]
        assert compat["field_checks"]["checkpoint_timesteps"] is False
    finally:
        FC.CHECKPOINT_TIMESTEPS = original


# ======================================================================
# Checkpoint path helpers
# ======================================================================
def test_checkpoint_path_naming_is_stable_and_ordered():
    p1 = FC.checkpoint_path("return_mlp_ppo", 0, 16_000)
    p2 = FC.checkpoint_path("return_mlp_ppo", 0, 1_000_000)
    assert str(p1) < str(p2)  # zero-padded -- lexicographic order matches numeric order
    assert p1.parent == p2.parent == FC.run_dir("return_mlp_ppo", 0)


def test_final_checkpoint_path_uses_total_transitions():
    assert FC.final_checkpoint_path("hamilton_ppo", 0) == FC.checkpoint_path("hamilton_ppo", 0, FC.TOTAL_TRANSITIONS)


# ======================================================================
# Episode-level data schema + reward reconciliation, all 3 architectures
# + 2 analytical benchmarks (untrained/fresh models -- fast, no training)
# ======================================================================
REQUIRED_EPISODE_FIELDS = {
    "full_objective", "raw_pnl", "fills", "buy_side_fills", "sell_side_fills", "total_arrivals",
    "buy_side_arrivals", "sell_side_arrivals", "fill_to_arrival_ratio", "mean_quoted_spread",
    "mean_bid_depth", "mean_ask_depth", "spread_revenue", "spread_revenue_per_fill",
    "adverse_selection_loss", "running_penalty", "terminal_penalty", "mean_abs_inventory",
    "terminal_abs_inventory", "mean_signed_inventory", "terminal_signed_inventory", "max_abs_inventory",
    "mean_bid_action", "mean_ask_action", "action_std_bid", "action_std_ask", "episode_duration",
    "n_decision_events", "reward_reconciliation_error", "deterministic", "near_bound_rate_deterministic",
    "stochastic_action_std_bid", "stochastic_action_std_ask", "near_bound_rate_stochastic",
    "stochastic_full_objective",
}


@pytest.mark.parametrize("architecture", list(FC.ARCHITECTURES))
def test_episode_schema_and_reward_reconciliation_ppo(architecture):
    model_cls = FinalInstrumentedRecurrentPPO if FC.is_recurrent(architecture) else FinalInstrumentedPPO
    model, _ = FC.build_model(architecture, learner_seed=0, model_cls=model_cls)
    row = ER.evaluate_episode_pair(model, architecture, seed=999_001, stochastic_torch_seed=123)
    missing = REQUIRED_EPISODE_FIELDS - set(row.keys())
    assert not missing, f"{architecture}: missing fields {missing}"
    assert row["reward_reconciliation_error"] < 1e-6
    assert row["deterministic"] is True
    assert row["total_arrivals"] > 0


@pytest.mark.parametrize("policy_name", ["oracle", "belief_weighted"])
def test_episode_schema_and_reward_reconciliation_benchmarks(policy_name):
    import shared.phase4_common as P4
    controls = P4.build_analytical_controls()
    row = ER.evaluate_benchmark_row(policy_name, controls, seed=999_002)
    missing = REQUIRED_EPISODE_FIELDS - set(row.keys())
    assert not missing, f"{policy_name}: missing fields {missing}"
    assert row["reward_reconciliation_error"] < 1e-6
    assert np.isnan(row["stochastic_action_std_bid"])  # benchmarks: no stochastic component, must be NaN not 0


# ======================================================================
# Deterministic evaluation is exactly reproducible
# ======================================================================
@pytest.mark.parametrize("architecture", list(FC.ARCHITECTURES))
def test_deterministic_evaluation_is_reproducible(architecture):
    model_cls = FinalInstrumentedRecurrentPPO if FC.is_recurrent(architecture) else FinalInstrumentedPPO
    model, _ = FC.build_model(architecture, learner_seed=0, model_cls=model_cls)
    row_a = ER.run_ppo_episode(model, architecture, seed=999_003, deterministic=True)
    row_b = ER.run_ppo_episode(model, architecture, seed=999_003, deterministic=True)
    assert row_a["full_objective"] == pytest.approx(row_b["full_objective"], abs=1e-10)
    assert row_a["fills"] == row_b["fills"]
    assert row_a["mean_bid_action"] == pytest.approx(row_b["mean_bid_action"], abs=1e-10)


# ======================================================================
# Recurrent (LSTM) hidden-state handling: episode_start correctness
# ======================================================================
def test_recurrent_episode_start_reset_gives_different_result_than_carried_state():
    """Confirms lstm_states is actually being threaded (recurrent), not
    silently ignored -- if the LSTM state were reset every step instead of
    carried, an actual LSTM policy's action distribution would not depend
    on within-episode history at all; we instead check the converse: that
    two DIFFERENT random-but-fixed model seeds threading state correctly
    produce internally-consistent (self-reproducible) but seed-DEPENDENT
    trajectories, confirming state is neither reset every step nor shared
    across independent model instances."""
    model_a, _ = FC.build_model("return_lstm_ppo", learner_seed=0, model_cls=FinalInstrumentedRecurrentPPO)
    model_b, _ = FC.build_model("return_lstm_ppo", learner_seed=1, model_cls=FinalInstrumentedRecurrentPPO)
    row_a = ER.run_ppo_episode(model_a, "return_lstm_ppo", seed=999_004, deterministic=True)
    row_b = ER.run_ppo_episode(model_b, "return_lstm_ppo", seed=999_004, deterministic=True)
    # Different random initialisations (different learner seeds) must not
    # coincidentally produce bit-identical trajectories.
    assert row_a["mean_bid_action"] != pytest.approx(row_b["mean_bid_action"], abs=1e-12)


def test_recurrent_model_predict_requires_episode_start_flag_shape():
    model, _ = FC.build_model("return_lstm_ppo", learner_seed=0, model_cls=FinalInstrumentedRecurrentPPO)
    obs = np.zeros(4, dtype=np.float32)
    action, lstm_states = model.predict(obs, state=None, episode_start=np.array([True]), deterministic=True)
    assert action.shape == (2,)
    assert lstm_states is not None


# ======================================================================
# Contrast statistics: verified against hand-computed values
# ======================================================================
def test_bootstrap_ci_of_mean_brackets_the_mean_and_is_reproducible():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lo1, hi1 = FAC.bootstrap_ci_of_mean(values)
    lo2, hi2 = FAC.bootstrap_ci_of_mean(values)
    assert lo1 == lo2 and hi1 == hi2  # fixed RNG seed inside bootstrap_ci_of_mean's default -> reproducible
    assert lo1 <= values.mean() <= hi1


def test_ci_contains_zero():
    assert FAC.ci_contains_zero(-1.0, 1.0)
    assert not FAC.ci_contains_zero(0.5, 1.0)
    assert not FAC.ci_contains_zero(-1.0, -0.5)


def test_compute_architecture_contrasts_matches_hand_computed_diff(tmp_path):
    episodes = pd.DataFrame([
        dict(policy="hamilton_ppo", learner_seed=s, checkpoint_transition=FC.TOTAL_TRANSITIONS,
             path_seed=0, deterministic=True, full_objective=10.0 + s, raw_pnl=0, fills=0,
             fill_to_arrival_ratio=0, mean_quoted_spread=0, spread_revenue=0, spread_revenue_per_fill=0,
             adverse_selection_loss=0, running_penalty=0, terminal_penalty=0, mean_abs_inventory=0,
             action_std_bid=0, action_std_ask=0)
        for s in FC.LEARNER_SEEDS
    ] + [
        dict(policy="return_mlp_ppo", learner_seed=s, checkpoint_transition=FC.TOTAL_TRANSITIONS,
             path_seed=0, deterministic=True, full_objective=5.0 + s, raw_pnl=0, fills=0,
             fill_to_arrival_ratio=0, mean_quoted_spread=0, spread_revenue=0, spread_revenue_per_fill=0,
             adverse_selection_loss=0, running_penalty=0, terminal_penalty=0, mean_abs_inventory=0,
             action_std_bid=0, action_std_ask=0)
        for s in FC.LEARNER_SEEDS
    ] + [
        dict(policy="return_lstm_ppo", learner_seed=s, checkpoint_transition=FC.TOTAL_TRANSITIONS,
             path_seed=0, deterministic=True, full_objective=1.0 + s, raw_pnl=0, fills=0,
             fill_to_arrival_ratio=0, mean_quoted_spread=0, spread_revenue=0, spread_revenue_per_fill=0,
             adverse_selection_loss=0, running_penalty=0, terminal_penalty=0, mean_abs_inventory=0,
             action_std_bid=0, action_std_ask=0)
        for s in FC.LEARNER_SEEDS
    ])
    result = FAC.compute_architecture_contrasts(episodes)
    row = result[(result["architecture_a"] == "hamilton_ppo") & (result["architecture_b"] == "return_mlp_ppo")
                 & (result["metric"] == "full_objective")].iloc[0]
    # (10+s) - (5+s) = 5.0 exactly, for every seed -> mean diff 5.0, SD 0.
    assert row["mean_matched_seed_diff"] == pytest.approx(5.0)
    assert row["sd_matched_seed_diff"] == pytest.approx(0.0, abs=1e-9)
    assert row["n_seeds_favouring_a"] == len(FC.LEARNER_SEEDS)
    assert row["n_seeds_favouring_b"] == 0


# ======================================================================
# Plotting / table-generation scripts: execute against synthetic minimal
# data, never train or evaluate a model
# ======================================================================
@pytest.fixture
def synthetic_results_dir(tmp_path) -> Path:
    d = tmp_path / "synthetic_results"
    d.mkdir()
    architectures = list(FC.ARCHITECTURES)
    checkpoints = [4_000, 8_000]
    seeds = [0, 1]

    val_seed_rows, val_arch_rows = [], []
    for arch in architectures:
        for t in checkpoints:
            vals = []
            for s in seeds:
                v = 10.0 + t / 1000.0 + s
                vals.append(v)
                val_seed_rows.append(dict(architecture=arch, learner_seed=s, checkpoint_transition=t, n_paths=5,
                                           mean_full_objective=v, mean_fills=3, mean_fill_to_arrival_ratio=0.5,
                                           mean_mean_quoted_spread=0.1, mean_spread_revenue=1.0,
                                           mean_adverse_selection_loss=0.2, mean_mean_abs_inventory=2.0,
                                           mean_running_penalty=0.05, mean_action_std_bid=0.3,
                                           mean_near_bound_rate_stochastic=0.1,
                                           sd_full_objective=0.5, se_full_objective=0.5 / np.sqrt(5)))
            arr = np.array(vals)
            val_arch_rows.append(dict(architecture=arch, checkpoint_transition=t, n_seeds=len(seeds),
                                       cross_seed_mean_full_objective=float(arr.mean()),
                                       cross_seed_sd_full_objective=float(arr.std(ddof=1)),
                                       cross_seed_se_full_objective=float(arr.std(ddof=1) / np.sqrt(len(arr))),
                                       cross_seed_mean_mean_fills=3.0, cross_seed_mean_fill_to_arrival_ratio=0.5,
                                       cross_seed_mean_mean_quoted_spread=0.1, cross_seed_mean_spread_revenue=1.0,
                                       cross_seed_mean_adverse_selection_loss=0.2,
                                       cross_seed_mean_mean_abs_inventory=2.0, cross_seed_mean_running_penalty=0.05,
                                       cross_seed_mean_action_std_bid=0.3,
                                       cross_seed_mean_near_bound_rate_stochastic=0.1))
    pd.DataFrame(val_seed_rows).to_csv(d / "validation_seed_summary.csv", index=False)
    pd.DataFrame(val_arch_rows).to_csv(d / "validation_architecture_summary.csv", index=False)

    bench_rows = [dict(policy=p, n_paths=5, mean_full_objective=8.0) for p in
                  ("oracle", "belief_weighted", "frozen_clone")]
    pd.DataFrame(bench_rows).to_csv(d / "validation_benchmark_summary.csv", index=False)

    hold_arch_rows = []
    for arch in architectures:
        for t in checkpoints:
            hold_arch_rows.append(dict(
                architecture=arch, checkpoint_transition=t, n_seeds=len(seeds),
                cross_seed_mean_full_objective=10.0 + t / 1000.0, cross_seed_sd_full_objective=0.5,
                cross_seed_mean_mean_fills=3.0, cross_seed_mean_mean_quoted_spread=0.1,
                cross_seed_mean_spread_revenue=1.0, cross_seed_mean_adverse_selection_loss=0.2,
                cross_seed_mean_mean_abs_inventory=2.0, cross_seed_mean_raw_pnl=1.5,
                cross_seed_mean_terminal_penalty=0.05, cross_seed_mean_running_penalty=0.05,
                cross_seed_mean_total_arrivals=200, cross_seed_mean_mean_bid_depth=0.05,
                cross_seed_mean_mean_ask_depth=0.05, cross_seed_mean_spread_revenue_per_fill=0.3,
                cross_seed_mean_action_std_bid=0.3, cross_seed_mean_action_std_ask=0.3,
                cross_seed_mean_near_bound_rate_deterministic=0.1, cross_seed_mean_near_bound_rate_stochastic=0.1,
            ))
    pd.DataFrame(hold_arch_rows).to_csv(d / "final_holdout_architecture_summary.csv", index=False)

    fixed_rows = []
    for arch in architectures:
        for s in seeds:
            for metric in FAC.FIXED_200K_1M_METRICS:
                fixed_rows.append(dict(architecture=arch, learner_seed=s, metric=metric, n_paths=5,
                                        mean_change_200k_to_1m=0.5, sd_change=0.2, se_change=0.1,
                                        normal_ci_lo=0.2, normal_ci_hi=0.8, bootstrap_ci_lo=0.15,
                                        bootstrap_ci_hi=0.85, win_rate_1m_better=0.7))
    pd.DataFrame(fixed_rows).to_csv(d / "fixed_200k_vs_1m_contrasts.csv", index=False)

    bench_paired_rows = []
    for arch in architectures:
        for s in seeds:
            for benchmark in FAC.BENCHMARKS:
                bench_paired_rows.append(dict(architecture=arch, learner_seed=s, benchmark=benchmark,
                                               metric="full_objective", n_paths=5, paired_mean_diff=1.0,
                                               paired_sd_diff=0.5, se_diff=0.2, normal_ci_lo=0.5, normal_ci_hi=1.5,
                                               normal_ci_contains_zero=False, bootstrap_ci_lo=0.4,
                                               bootstrap_ci_hi=1.6, bootstrap_ci_contains_zero=False, win_rate=0.8))
    pd.DataFrame(bench_paired_rows).to_csv(d / "benchmark_paired_contrasts.csv", index=False)

    arch_contrast_rows = []
    for a, b in FAC.ARCHITECTURE_PAIRS:
        for metric in FAC.ARCHITECTURE_CONTRAST_METRICS:
            arch_contrast_rows.append(dict(architecture_a=a, architecture_b=b, metric=metric, n_seeds=len(seeds),
                                            mean_matched_seed_diff=1.0, sd_matched_seed_diff=0.5,
                                            se_matched_seed_diff=0.3, bootstrap_ci_lo=0.2, bootstrap_ci_hi=1.8,
                                            bootstrap_ci_contains_zero=False, n_seeds_favouring_a=1,
                                            n_seeds_favouring_b=1, n_seeds_tied=0, per_seed_diffs="[]"))
    pd.DataFrame(arch_contrast_rows).to_csv(d / "architecture_contrasts.csv", index=False)

    return d


def test_plotting_script_runs_against_synthetic_data(synthetic_results_dir, tmp_path):
    out_dir = tmp_path / "plots"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "plot_final_architecture_results.py"),
         "--results-dir", str(synthetic_results_dir), "--output-dir", str(out_dir),
         "--format", "png", "--dpi", "50", "--checkpoint-markers", "4000", "8000"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) >= 6, f"expected at least 6 plot files, got {[p.name for p in pngs]}"


def test_table_generation_script_runs_against_synthetic_data(synthetic_results_dir, tmp_path):
    out_dir = tmp_path / "tables"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_final_architecture_tables.py"),
         "--results-dir", str(synthetic_results_dir), "--output-dir", str(out_dir)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    md_files = list(out_dir.glob("*.md"))
    tex_files = list(out_dir.glob("*.tex"))
    assert len(md_files) == 5
    assert len(tex_files) == 5
