"""
test_stats_helpers.py
-----------------------
Toy-data unit tests for stats_helpers.py. Plain assert-based (no pytest
dependency) -- run directly:

    python test_stats_helpers.py

Every figure script in this set imports stats_helpers; this file must pass
before any of those results are trusted.
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import stats_helpers as SH


def test_paired_diff_by_path_correctness():
    df = pd.DataFrame({
        "policy": ["a", "a", "a", "b", "b", "b"],
        "path": [10, 20, 30, 30, 10, 20],   # deliberately out of order for b
        "value": [1.0, 2.0, 3.0, 30.0, 10.0, 20.0],
    })
    out = SH.paired_diff_by_path(df, "policy", "value", "path", "a", "b")
    out = out.sort_values("path").reset_index(drop=True)
    # path=10: a=1, b=10 -> diff=-9 ; path=20: a=2,b=20 -> diff=-18 ; path=30: a=3,b=30 -> diff=-27
    assert list(out["path"]) == [10, 20, 30]
    assert np.allclose(out["diff"].to_numpy(), [-9.0, -18.0, -27.0])
    print("test_paired_diff_by_path_correctness: PASS")


def test_paired_diff_by_path_raises_on_mismatched_paths():
    df = pd.DataFrame({
        "policy": ["a", "a", "b", "b"],
        "path": [1, 2, 1, 3],   # path=2 missing for b, path=3 missing for a
        "value": [1.0, 2.0, 10.0, 30.0],
    })
    try:
        SH.paired_diff_by_path(df, "policy", "value", "path", "a", "b")
        raised = False
    except ValueError:
        raised = True
    assert raised, "expected ValueError on mismatched path sets"
    print("test_paired_diff_by_path_raises_on_mismatched_paths: PASS")


def test_paired_diff_by_path_raises_on_duplicates():
    df = pd.DataFrame({
        "policy": ["a", "a", "a", "b", "b"],
        "path": [1, 1, 2, 1, 2],   # path=1 duplicated for a
        "value": [1.0, 1.5, 2.0, 10.0, 20.0],
    })
    try:
        SH.paired_diff_by_path(df, "policy", "value", "path", "a", "b")
        raised = False
    except ValueError:
        raised = True
    assert raised, "expected ValueError on duplicate path within one policy"
    print("test_paired_diff_by_path_raises_on_duplicates: PASS")


def test_cvar_known_value():
    x = np.arange(1, 11)  # 1..10
    # alpha=0.10 -> k=ceil(0.1*10)=1 -> mean of the single lowest value = 1
    assert SH.cvar(x, alpha=0.10) == 1.0
    # alpha=0.30 -> k=3 -> mean of [1,2,3] = 2.0
    assert SH.cvar(x, alpha=0.30) == 2.0
    print("test_cvar_known_value: PASS")


def test_paired_path_bootstrap_mean_recovers_true_mean():
    # A percentile bootstrap CI is constructed around the OBSERVED sample
    # statistic, not the population parameter -- testing against the true
    # generating mean is flaky (~5% chance of legitimate exclusion at n=2000
    # by sampling variability alone). Test against the sample mean instead,
    # which the bootstrap distribution is centred on by construction.
    rng = np.random.default_rng(42)
    d = rng.normal(loc=5.0, scale=2.0, size=2000)
    sample_mean = d.mean()
    lo, hi = SH.paired_path_bootstrap(d, B=4000, seed=1)
    assert lo < sample_mean < hi, f"95% CI [{lo},{hi}] should bracket the sample mean {sample_mean}"
    # sanity: CI width should be small relative to the mean at n=2000
    assert (hi - lo) < 1.0
    # and the true generating mean (5.0) should be close to the CI (loose sanity check,
    # not a strict bracket -- see comment above)
    assert lo - 0.3 < 5.0 < hi + 0.3
    print(f"test_paired_path_bootstrap_mean_recovers_true_mean: PASS (CI=[{lo:.3f},{hi:.3f}], "
          f"sample_mean={sample_mean:.3f})")


def test_paired_path_bootstrap_matches_normal_theory_se():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, size=5000)
    lo, hi = SH.paired_path_bootstrap(x, B=5000, seed=2)
    se_theory = x.std(ddof=1) / np.sqrt(len(x))
    width_theory = 2 * 1.96 * se_theory
    width_boot = hi - lo
    # bootstrap CI width should roughly match the normal-theory width (loose tolerance)
    assert abs(width_boot - width_theory) / width_theory < 0.15, (width_boot, width_theory)
    print(f"test_paired_path_bootstrap_matches_normal_theory_se: PASS (boot width={width_boot:.4f}, "
          f"theory width={width_theory:.4f})")


def test_paired_path_bootstrap_two_sided_cvar_contrast():
    # Two IDENTICAL distributions -> CI of CVaR(x)-CVaR(y) should bracket 0.
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, size=1000)
    y = rng.normal(0, 1, size=1000)
    lo, hi = SH.paired_path_bootstrap(x, y, stat_fn=lambda a: SH.cvar(a, alpha=0.10), B=3000, seed=4)
    assert lo < 0 < hi, f"CI [{lo},{hi}] should bracket 0 for two draws from the same distribution"
    print(f"test_paired_path_bootstrap_two_sided_cvar_contrast: PASS (CI=[{lo:.3f},{hi:.3f}])")


def test_hierarchical_bootstrap_point_matches_direct_computation():
    # Toy world: 2 "paths" values per architecture-seed, contrast = mean(arch A) - mean(arch B)
    rng = np.random.default_rng(0)
    n_paths = 50
    data = {
        "A": {s: rng.normal(10, 1, size=n_paths) for s in range(3)},
        "B": {s: rng.normal(7, 1, size=n_paths) for s in range(3)},
    }
    arch_seeds = {"A": [0, 1, 2], "B": [0, 1, 2]}

    def compute_stat(path_idx, seed_choice):
        a_vals = np.concatenate([data["A"][s][path_idx] for s in seed_choice["A"]])
        b_vals = np.concatenate([data["B"][s][path_idx] for s in seed_choice["B"]])
        return a_vals.mean() - b_vals.mean()

    result = SH.hierarchical_bootstrap(compute_stat, n_paths, arch_seeds, B=500, seed=5)
    # point estimate (no resampling) should equal the direct computation
    direct = np.concatenate([data["A"][s] for s in [0, 1, 2]]).mean() - \
             np.concatenate([data["B"][s] for s in [0, 1, 2]]).mean()
    assert abs(result["point"] - direct) < 1e-9
    assert result["ci_lo"] < result["point"] < result["ci_hi"]
    assert len(result["replicates"]) == 500
    print(f"test_hierarchical_bootstrap_point_matches_direct_computation: PASS "
          f"(point={result['point']:.3f}, CI=[{result['ci_lo']:.3f},{result['ci_hi']:.3f}])")


def test_hierarchical_bootstrap_shared_path_resample_preserves_pairing():
    # If A and B are PERFECTLY correlated per-path (B = A - 3 exactly, same
    # path indices), the contrast mean(A)-mean(B) should be EXACTLY 3 in
    # every single replicate (zero variance), because the same path_idx is
    # applied to both -- this is the direct test that pairing is preserved
    # at the resample level, not just on average.
    rng = np.random.default_rng(11)
    n_paths = 40
    base = rng.normal(0, 5, size=n_paths)
    data = {"A": {0: base.copy()}, "B": {0: base - 3.0}}
    arch_seeds = {"A": [0], "B": [0]}

    def compute_stat(path_idx, seed_choice):
        return data["A"][0][path_idx].mean() - data["B"][0][path_idx].mean()

    result = SH.hierarchical_bootstrap(compute_stat, n_paths, arch_seeds, B=200, seed=6)
    assert np.allclose(result["replicates"], 3.0, atol=1e-9), \
        "paired resample should give EXACTLY 3.0 every replicate when B=A-3 pathwise"
    print("test_hierarchical_bootstrap_shared_path_resample_preserves_pairing: PASS")


def test_per_seed_paired_ttest_matches_scipy():
    rng = np.random.default_rng(9)
    d = rng.normal(2.0, 3.0, size=500)
    mine = SH.per_seed_paired_ttest(d)
    ref = scipy_stats.ttest_1samp(d, popmean=0.0)
    assert abs(mine["t"] - ref.statistic) < 1e-9
    assert abs(mine["p"] - ref.pvalue) < 1e-9
    assert mine["df"] == 499
    print(f"test_per_seed_paired_ttest_matches_scipy: PASS (t={mine['t']:.4f}, p={mine['p']:.4g})")


def test_cross_seed_onesample_ttest_matches_scipy_and_df4():
    seed_means = np.array([1.2, 0.8, 1.5, -0.3, 2.0])  # 5 seeds -> df=4
    mine = SH.cross_seed_onesample_ttest(seed_means)
    ref = scipy_stats.ttest_1samp(seed_means, popmean=0.0)
    assert mine["df"] == 4
    assert abs(mine["t"] - ref.statistic) < 1e-9
    assert abs(mine["p"] - ref.pvalue) < 1e-9
    assert mine["ci_lo"] < mine["mean"] < mine["ci_hi"]
    print(f"test_cross_seed_onesample_ttest_matches_scipy_and_df4: PASS (t={mine['t']:.4f}, "
          f"p={mine['p']:.4g}, CI=[{mine['ci_lo']:.3f},{mine['ci_hi']:.3f}])")


def main():
    tests = [
        test_paired_diff_by_path_correctness,
        test_paired_diff_by_path_raises_on_mismatched_paths,
        test_paired_diff_by_path_raises_on_duplicates,
        test_cvar_known_value,
        test_paired_path_bootstrap_mean_recovers_true_mean,
        test_paired_path_bootstrap_matches_normal_theory_se,
        test_paired_path_bootstrap_two_sided_cvar_contrast,
        test_hierarchical_bootstrap_point_matches_direct_computation,
        test_hierarchical_bootstrap_shared_path_resample_preserves_pairing,
        test_per_seed_paired_ttest_matches_scipy,
        test_cross_seed_onesample_ttest_matches_scipy_and_df4,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
