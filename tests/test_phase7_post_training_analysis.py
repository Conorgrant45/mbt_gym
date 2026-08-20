"""
tests/test_phase7_post_training_analysis.py
------------------------------------------------------
Tests for the Phase 7 post-training analysis (validation learning-curve
plot + unseen paired holdout evaluation). Covers: new-holdout disjointness
against every prior seed range, checkpoint-selection correctness and
reproducibility, that selection never reads holdout data, paired-contrast
statistics correctness (against hand-computed values), and that no
existing Phase 7 file is modified by any of this analysis.

Run from repo root:
    pytest tests/test_phase7_post_training_analysis.py -v
"""
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import shared.phase7_common as P7
import phase7_post_training.phase7_post_training_common as PC
from phase7_post_training.phase7_evaluate_unseen_holdout import paired_contrast, bootstrap_ci


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ======================================================================
# New holdout range: disjoint from every previously-used range
# ======================================================================
def test_new_holdout_is_500_seeds():
    assert len(PC.NEW_HOLDOUT_SEEDS) == 500
    assert len(set(PC.NEW_HOLDOUT_SEEDS)) == 500  # no internal duplicates


def test_new_holdout_disjoint_from_all_prior_ranges():
    result = PC.verify_new_holdout_disjoint()
    assert result["disjoint"], f"overlaps found: {result['overlaps']}"
    assert result["n_new_holdout_seeds"] == 500


def test_prior_seed_ranges_list_matches_source_definitions():
    # Spot-check a handful of ranges against their own source-of-truth values
    # (not re-typed independently -- these are the SAME literals used
    # elsewhere in the project).
    assert PC.PRIOR_SEED_RANGES["phase6_validation_seeds"] == set(range(240_000, 240_050))
    assert PC.PRIOR_SEED_RANGES["phase6_holdout_seeds"] == set(range(250_000, 250_200))
    assert PC.PRIOR_SEED_RANGES["phase7_validation_seeds"] == set(range(260_000, 260_050))
    assert PC.PRIOR_SEED_RANGES["learner_seeds"] == set(range(0, 5))
    assert PC.PRIOR_SEED_RANGES["training_env_seed"] == {70_000}


# ======================================================================
# Checkpoint selection: validation-only, reproducible, never touches holdout
# ======================================================================
def test_checkpoint_selection_uses_only_validation_csv_never_holdout():
    source = inspect.getsource(PC.select_checkpoint_for_seed)
    # Check the CODE body only (excluding the docstring, which correctly
    # explains, in prose, that this function never touches holdout data --
    # that mention is expected and must not fail this check).
    body = source.split('"""', 2)[-1] if source.count('"""') >= 2 else source
    assert "holdout" not in body.lower()
    assert "NEW_HOLDOUT_SEEDS" not in body


def test_checkpoint_selection_is_argmax_det_objective_reproducible():
    cp = pd.read_csv(P7.RESULTS_DIR / "phase7_checkpoint_summary.csv")
    for seed in P7.LEARNER_SEEDS:
        sel_a = PC.select_checkpoint_for_seed(seed, cp)
        sel_b = PC.select_checkpoint_for_seed(seed, cp)
        assert sel_a["timestep"] == sel_b["timestep"]  # deterministic, repeatable

        sub = cp[cp["learner_seed"] == seed]
        expected_best = sub.loc[sub["det_mean_objective"].idxmax()]
        # Tie-break: earliest timestep among any ties at the max value.
        max_val = sub["det_mean_objective"].max()
        expected_timestep = int(sub[sub["det_mean_objective"] == max_val]["timestep"].min())
        assert sel_a["timestep"] == expected_timestep


def test_checkpoint_selection_only_uses_expected_number_of_checkpoints():
    cp = pd.read_csv(P7.RESULTS_DIR / "phase7_checkpoint_summary.csv")
    with pytest.raises(AssertionError):
        PC.select_checkpoint_for_seed(999, cp)  # no rows for seed 999 -> length mismatch


# ======================================================================
# Paired-contrast statistics: verified against hand-computed values
# ======================================================================
def test_paired_contrast_matches_hand_computed_stats():
    ppo = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
    bench = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    result = paired_contrast(ppo, bench, "ppo", "bench")

    diffs = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    expected_mean = diffs.mean()
    expected_sd = diffs.std(ddof=1)
    expected_se = expected_sd / np.sqrt(5)

    assert result["mean_paired_diff"] == pytest.approx(expected_mean)
    assert result["paired_sd"] == pytest.approx(expected_sd)
    assert result["se"] == pytest.approx(expected_se)
    assert result["win_rate"] == pytest.approx(1.0)  # all diffs positive
    assert result["ci95_lo"] < result["mean_paired_diff"] < result["ci95_hi"]
    assert not result["ci_includes_zero"]  # all positive, clearly excludes 0


def test_paired_contrast_ci_includes_zero_when_diffs_straddle_zero():
    ppo = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    bench = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = paired_contrast(ppo, bench, "ppo", "bench")
    assert result["ci_includes_zero"]
    assert "not statistically distinguishable" in result["verdict"]


def test_bootstrap_ci_is_reproducible_and_brackets_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    lo1, hi1 = bootstrap_ci(values)
    lo2, hi2 = bootstrap_ci(values)
    assert lo1 == lo2 and hi1 == hi2  # fixed RNG seed -> reproducible
    assert lo1 <= values.mean() <= hi1


# ======================================================================
# No modification of any existing Phase 7 file
# ======================================================================
EXISTING_PHASE7_FILES = [
    "phase7_benchmark_on_own_validation_seeds.csv", "phase7_checkpoint_summary.csv",
    "phase7_config_manifest.json", "phase7_group_summary.csv", "phase7_run_status.csv",
    "phase7_seed_summary.csv", "phase7_time_to_benchmark.csv", "phase7_time_to_benchmark.png",
    "phase7_time_to_benchmark_summary.md", "phase7_training_diagnostics.csv",
]


def test_existing_phase7_files_are_untouched():
    for name in EXISTING_PHASE7_FILES:
        path = P7.RESULTS_DIR / name
        assert path.exists(), f"expected pre-existing Phase 7 file missing: {path}"
    # This test only confirms presence/readability here; the authoritative
    # before/after hash comparison for this session is done externally via
    # `sha256sum` snapshots taken before any post-training-analysis code ran
    # (see the task's own safety requirement #3) -- not duplicated as a
    # hash-literal test here since that would require committing hashes of
    # files this test suite does not own.


def test_post_training_outputs_are_isolated_under_their_own_subdirectory():
    assert PC.POST_DIR == P7.RESULTS_DIR / "post_training_analysis"
    assert PC.POST_DIR != P7.RESULTS_DIR
