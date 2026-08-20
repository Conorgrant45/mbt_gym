"""
stats_helpers.py
------------------
Shared statistical helpers for the fig01-fig08 / figA1-figA3 thesis figure
set. Unit-tested on toy data in test_stats_helpers.py (run that file
directly to verify before trusting any figure built on top of this module).

Design principles enforced throughout this module (per the implementation
brief):
  - All differencing is PAIRED (same path_seed / same learner_seed on both
    sides of a difference), via an explicit merge with a 1:1 validation
    assert -- never positional alignment, never an unpaired two-sample SE.
  - Nonlinear statistics (e.g. CVaR) are NEVER differenced by independently
    bootstrapping each side and subtracting bootstrap distributions -- the
    same resampled path indices are applied to both sides inside one
    resampling loop, preserving the pairing at the resample level too.
  - The hierarchical bootstrap resamples path indices ONCE per replicate
    and applies that SAME resample to every policy (preserving pairing),
    then independently resamples learner seeds per architecture.
    Benchmarks (no seed dimension) are affected only by the path resample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _stats


# ======================================================================
# Paired merge with explicit validation (never positional alignment)
# ======================================================================
def paired_diff_by_path(df: pd.DataFrame, policy_col: str, value_col: str, path_col: str,
                         a: str, b: str, extra_eq_filters: dict | None = None) -> pd.DataFrame:
    """Returns a DataFrame with columns [path_col, 'value_a', 'value_b', 'diff']
    for policy `a` minus policy `b`, matched 1:1 on `path_col`. `extra_eq_filters`
    is an optional dict of {column: value} applied to BOTH sides before merging
    (e.g. {'checkpoint_transition': 1_000_000}). Raises if the merge is not
    exactly 1:1 (i.e. if either side has a duplicate or missing path)."""
    sub = df
    if extra_eq_filters:
        for col, val in extra_eq_filters.items():
            if pd.isna(val):
                sub = sub[sub[col].isna()]
            else:
                sub = sub[sub[col] == val]
    da = sub[sub[policy_col] == a][[path_col, value_col]].rename(columns={value_col: "value_a"})
    db = sub[sub[policy_col] == b][[path_col, value_col]].rename(columns={value_col: "value_b"})
    if da[path_col].duplicated().any() or db[path_col].duplicated().any():
        raise ValueError(
            f"paired_diff_by_path: duplicate {path_col!r} values found for policy={a!r} or {b!r} "
            f"after filtering {extra_eq_filters} -- cannot form a 1:1 pairing (see the "
            f"validation_path_level.csv duplicate-row finding in FIGURES_REPORT.md if this file "
            f"is the source; dedupe with .drop_duplicates() before calling this)."
        )
    merged = da.merge(db, on=path_col, how="inner", validate="1:1")
    if len(merged) != len(da) or len(merged) != len(db):
        raise ValueError(
            f"paired_diff_by_path: {len(da)} paths for {a!r}, {len(db)} for {b!r}, "
            f"{len(merged)} after inner-join on {path_col!r} -- path sets are not identical."
        )
    merged["diff"] = merged["value_a"] - merged["value_b"]
    return merged


# ======================================================================
# Paired path-level bootstrap (single learner seed, ~500 paths)
# ======================================================================
def paired_path_bootstrap(x: np.ndarray, y: np.ndarray | None = None, stat_fn=np.mean,
                           B: int = 10_000, seed: int = 0, confidence: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI, resampling PATH INDICES with replacement.

    If y is None: x is already the array of per-path paired differences;
    returns the CI of stat_fn(x) (default: the mean).

    If y is provided: x and y are the two policies' per-path values, ALREADY
    aligned so x[i] and y[i] are the same path (e.g. via paired_diff_by_path's
    'value_a'/'value_b' columns). The SAME resampled index set is applied to
    both x and y inside each replicate before computing stat_fn(x_i) -
    stat_fn(y_i) -- required for nonlinear stat_fn (e.g. CVaR), where
    bootstrapping each side independently and subtracting would give the
    wrong variance.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    if y is not None:
        y = np.asarray(y, dtype=float)
        if len(y) != n:
            raise ValueError(f"x and y must be the same length (paired), got {n} vs {len(y)}")

    boot_vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        if y is None:
            boot_vals[b] = stat_fn(x[idx])
        else:
            boot_vals[b] = stat_fn(x[idx]) - stat_fn(y[idx])

    alpha = 1.0 - confidence
    lo, hi = np.percentile(boot_vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ======================================================================
# CVaR (lower-tail expected shortfall)
# ======================================================================
def cvar(x: np.ndarray, alpha: float = 0.10) -> float:
    """Mean of the lowest alpha-fraction of values in x (e.g. alpha=0.10 ->
    mean of the worst 10% of per-path objectives within one (policy, seed))."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("cvar: empty input")
    k = max(1, int(np.ceil(alpha * n)))
    return float(np.sort(x)[:k].mean())


# ======================================================================
# Hierarchical bootstrap: resample paths ONCE per replicate (shared across
# every policy, preserving pairing), independently resample learner seeds
# per architecture, recompute an arbitrary contrast statistic.
# ======================================================================
def hierarchical_bootstrap(compute_stat, n_paths: int, arch_seeds: dict[str, list[int]],
                            B: int = 10_000, seed: int = 0, confidence: float = 0.95) -> dict:
    """
    compute_stat(path_idx: np.ndarray[int], seed_choice: dict[str, np.ndarray[int]]) -> float
        Callable the caller provides; must use path_idx (integer positions
        into whatever fixed-order per-path arrays the caller closed over)
        and seed_choice[architecture] (resampled seed IDs, with replacement,
        same length as arch_seeds[architecture]) to recompute the statistic
        for one replicate. Benchmarks have no seed dimension, so a caller
        whose compute_stat only touches benchmark data will naturally ignore
        seed_choice and be affected only by the shared path resample.
    n_paths: total number of paths (e.g. 500).
    arch_seeds: {architecture: [seed ids]} -- e.g. {"hamilton_ppo": [0,1,2,3,4], ...}.

    Returns dict(point=..., ci_lo=..., ci_hi=..., replicates=np.ndarray) where
    `point` is compute_stat evaluated on the UNRESAMPLED data (path_idx =
    arange(n_paths), seed_choice = the original seed lists), and ci_lo/hi is
    the percentile CI over B replicates.
    """
    rng = np.random.default_rng(seed)
    replicates = np.empty(B)
    for b in range(B):
        path_idx = rng.integers(0, n_paths, size=n_paths)
        seed_choice = {arch: rng.choice(seeds, size=len(seeds), replace=True) for arch, seeds in arch_seeds.items()}
        replicates[b] = compute_stat(path_idx, seed_choice)

    point_seed_choice = {arch: np.asarray(seeds) for arch, seeds in arch_seeds.items()}
    point = compute_stat(np.arange(n_paths), point_seed_choice)

    alpha = 1.0 - confidence
    lo, hi = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(point=float(point), ci_lo=float(lo), ci_hi=float(hi), replicates=replicates)


# ======================================================================
# t-tests
# ======================================================================
def per_seed_paired_ttest(d: np.ndarray) -> dict:
    """d: per-path paired differences for ONE learner seed (~500 values).
    Two-sided one-sample t-test of mean(d) against 0, df = len(d) - 1."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    df = n - 1
    t = mean / se if se > 0 else float("inf") if mean != 0 else 0.0
    p = float(2 * _stats.t.sf(abs(t), df)) if np.isfinite(t) else 0.0
    return dict(n=n, mean=mean, se=se, t=float(t), df=df, p=p)


def cross_seed_onesample_ttest(seed_means: np.ndarray, confidence: float = 0.95) -> dict:
    """seed_means: the per-seed paired means (typically 5 values, one per
    learner seed). One-sample t-test of their mean against 0, df = n - 1
    (df=4 for 5 seeds)."""
    seed_means = np.asarray(seed_means, dtype=float)
    n = len(seed_means)
    mean = float(seed_means.mean())
    se = float(seed_means.std(ddof=1) / np.sqrt(n))
    df = n - 1
    t = mean / se if se > 0 else float("inf") if mean != 0 else 0.0
    p = float(2 * _stats.t.sf(abs(t), df)) if np.isfinite(t) else 0.0
    tcrit = float(_stats.t.ppf(1 - (1 - confidence) / 2, df))
    return dict(n=n, mean=mean, se=se, t=float(t), df=df, p=p,
                ci_lo=mean - tcrit * se, ci_hi=mean + tcrit * se)
