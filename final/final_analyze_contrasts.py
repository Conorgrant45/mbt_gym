"""
final/final_analyze_contrasts.py
--------------------------------
Reads final_holdout_episode_level.csv (written by final_evaluate_holdout.py
-- one row per (policy, learner_seed, checkpoint_transition, path_seed),
same 500 holdout paths for every policy) and produces the three contrast
deliverables, keeping the two uncertainty sources explicitly separate:

  architecture_contrasts.csv     -- matched-LEARNER-SEED contrasts between
                                     architectures at the fixed 1,000,000
                                     checkpoint (Hamilton-MLP, Hamilton-LSTM,
                                     LSTM-MLP). Uncertainty source: the 5
                                     learner seeds (bootstrap over seeds).
  benchmark_paired_contrasts.csv -- PATH-LEVEL paired contrasts of each
                                     architecture/seed at the fixed
                                     1,000,000 checkpoint against each of
                                     oracle / belief_weighted / frozen_clone,
                                     paired by path_seed. Uncertainty
                                     source: the 500 holdout paths (both
                                     normal-theory and paired bootstrap CIs).
  fixed_200k_vs_1m_contrasts.csv -- PATH-LEVEL paired contrasts of each
                                     architecture/seed's 200,000- vs
                                     1,000,000-transition checkpoint on the
                                     SAME 500 holdout paths. Uncertainty
                                     source: the 500 holdout paths.

Never evaluates a model or reads a checkpoint file -- purely a read-only
aggregation/statistics pass over the already-produced episode-level CSV,
reproducible at any time.

Run from repo root:
    python final/final_analyze_contrasts.py
"""
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

import numpy as np
import pandas as pd

import final.final_common as FC

N_BOOT = 10_000
BOOTSTRAP_SEED = 20260101
Z_95 = 1.959963984540054

ARCHITECTURE_CONTRAST_METRICS = [
    "full_objective", "raw_pnl", "fills", "fill_to_arrival_ratio",
    "mean_quoted_spread", "spread_revenue", "spread_revenue_per_fill",
    "adverse_selection_loss", "running_penalty", "terminal_penalty",
    "mean_abs_inventory", "action_std_bid", "action_std_ask",
]
ARCHITECTURE_PAIRS = [
    ("hamilton_ppo", "return_mlp_ppo"),
    ("hamilton_ppo", "return_lstm_ppo"),
    ("return_lstm_ppo", "return_mlp_ppo"),
]

FIXED_200K_1M_METRICS = [
    "full_objective", "fills", "mean_quoted_spread", "spread_revenue",
    "adverse_selection_loss", "mean_abs_inventory", "running_penalty", "terminal_penalty",
]

BENCHMARKS = ("oracle", "belief_weighted", "frozen_clone")


def bootstrap_ci_of_mean(values: np.ndarray, n_boot: int = N_BOOT, rng: np.random.Generator = None) -> tuple:
    if rng is None:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def ci_contains_zero(lo: float, hi: float) -> bool:
    return lo <= 0.0 <= hi


# ======================================================================
# 1. Matched-learner-seed architecture contrasts (uncertainty: 5 seeds)
# ======================================================================
def compute_architecture_contrasts(episodes: pd.DataFrame) -> pd.DataFrame:
    fixed_checkpoint = FC.FIXED_200K_1M_TIMESTEPS[-1]
    fixed = episodes[(episodes["checkpoint_transition"] == fixed_checkpoint)
                      & episodes["policy"].isin(FC.ARCHITECTURES) & episodes["deterministic"]]

    seed_means = (fixed.groupby(["policy", "learner_seed"])[ARCHITECTURE_CONTRAST_METRICS]
                  .mean().reset_index())

    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for arch_a, arch_b in ARCHITECTURE_PAIRS:
        sa = seed_means[seed_means["policy"] == arch_a].set_index("learner_seed").sort_index()
        sb = seed_means[seed_means["policy"] == arch_b].set_index("learner_seed").sort_index()
        assert list(sa.index) == list(FC.LEARNER_SEEDS) == list(sb.index), (
            f"{arch_a} vs {arch_b}: learner-seed index mismatch -- both architectures must have "
            f"all {len(FC.LEARNER_SEEDS)} seeds' holdout evaluations at the fixed checkpoint"
        )
        for metric in ARCHITECTURE_CONTRAST_METRICS:
            diffs = (sa[metric] - sb[metric]).to_numpy()
            lo, hi = bootstrap_ci_of_mean(diffs, rng=rng)
            rows.append(dict(
                architecture_a=arch_a, architecture_b=arch_b, metric=metric,
                n_seeds=len(diffs),
                mean_matched_seed_diff=float(diffs.mean()),
                sd_matched_seed_diff=float(diffs.std(ddof=1)),
                se_matched_seed_diff=float(diffs.std(ddof=1) / np.sqrt(len(diffs))),
                bootstrap_ci_lo=lo, bootstrap_ci_hi=hi,
                bootstrap_ci_contains_zero=ci_contains_zero(lo, hi),
                n_seeds_favouring_a=int((diffs > 0).sum()),
                n_seeds_favouring_b=int((diffs < 0).sum()),
                n_seeds_tied=int((diffs == 0).sum()),
                per_seed_diffs=list(np.round(diffs, 6)),
            ))
    return pd.DataFrame(rows)


# ======================================================================
# 2. Paired path-level contrasts vs each benchmark (uncertainty: 500 paths)
# ======================================================================
def compute_benchmark_paired_contrasts(episodes: pd.DataFrame) -> pd.DataFrame:
    fixed_checkpoint = FC.FIXED_200K_1M_TIMESTEPS[-1]
    fixed = episodes[(episodes["checkpoint_transition"] == fixed_checkpoint)
                      & episodes["policy"].isin(FC.ARCHITECTURES) & episodes["deterministic"]]
    bench = episodes[episodes["policy"].isin(BENCHMARKS) & episodes["deterministic"]]

    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for architecture in FC.ARCHITECTURES:
        for seed in FC.LEARNER_SEEDS:
            ppo_df = fixed[(fixed["policy"] == architecture) & (fixed["learner_seed"] == seed)]
            ppo_df = ppo_df.set_index("path_seed").sort_index()
            assert list(ppo_df.index) == sorted(FC.HOLDOUT_SEEDS), (
                f"{architecture} seed {seed}: holdout path_seed index incomplete/misaligned"
            )
            for benchmark in BENCHMARKS:
                bench_df = bench[bench["policy"] == benchmark].set_index("path_seed").sort_index()
                assert list(bench_df.index) == sorted(FC.HOLDOUT_SEEDS), (
                    f"{benchmark}: holdout path_seed index incomplete/misaligned"
                )
                diffs = (ppo_df["full_objective"] - bench_df["full_objective"]).to_numpy()
                sd = float(diffs.std(ddof=1))
                se = sd / np.sqrt(len(diffs))
                mean_diff = float(diffs.mean())
                normal_lo, normal_hi = mean_diff - Z_95 * se, mean_diff + Z_95 * se
                boot_lo, boot_hi = bootstrap_ci_of_mean(diffs, rng=rng)
                rows.append(dict(
                    architecture=architecture, learner_seed=seed, benchmark=benchmark,
                    metric="full_objective", n_paths=len(diffs),
                    paired_mean_diff=mean_diff, paired_sd_diff=sd, se_diff=se,
                    normal_ci_lo=normal_lo, normal_ci_hi=normal_hi,
                    normal_ci_contains_zero=ci_contains_zero(normal_lo, normal_hi),
                    bootstrap_ci_lo=boot_lo, bootstrap_ci_hi=boot_hi,
                    bootstrap_ci_contains_zero=ci_contains_zero(boot_lo, boot_hi),
                    win_rate=float((diffs > 0).mean()),
                ))
    return pd.DataFrame(rows)


# ======================================================================
# 3. Fixed 200k-vs-1m paired contrasts (uncertainty: 500 paths, SAME paths
#    for both checkpoints)
# ======================================================================
def compute_fixed_200k_vs_1m_contrasts(episodes: pd.DataFrame) -> pd.DataFrame:
    ppo = episodes[episodes["policy"].isin(FC.ARCHITECTURES) & episodes["deterministic"]]
    checkpoint_200k, checkpoint_1m = FC.FIXED_200K_1M_TIMESTEPS[0], FC.FIXED_200K_1M_TIMESTEPS[-1]

    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for architecture in FC.ARCHITECTURES:
        for seed in FC.LEARNER_SEEDS:
            df_200k = ppo[(ppo["policy"] == architecture) & (ppo["learner_seed"] == seed)
                          & (ppo["checkpoint_transition"] == checkpoint_200k)].set_index("path_seed").sort_index()
            df_1m = ppo[(ppo["policy"] == architecture) & (ppo["learner_seed"] == seed)
                        & (ppo["checkpoint_transition"] == checkpoint_1m)].set_index("path_seed").sort_index()
            assert list(df_200k.index) == list(df_1m.index) == sorted(FC.HOLDOUT_SEEDS), (
                f"{architecture} seed {seed}: 200k/1m holdout path_seed indices must match and be complete"
            )
            for metric in FIXED_200K_1M_METRICS:
                diffs = (df_1m[metric] - df_200k[metric]).to_numpy()
                sd = float(diffs.std(ddof=1))
                se = sd / np.sqrt(len(diffs))
                mean_diff = float(diffs.mean())
                normal_lo, normal_hi = mean_diff - Z_95 * se, mean_diff + Z_95 * se
                boot_lo, boot_hi = bootstrap_ci_of_mean(diffs, rng=rng)
                rows.append(dict(
                    architecture=architecture, learner_seed=seed, metric=metric, n_paths=len(diffs),
                    mean_change_200k_to_1m=mean_diff, sd_change=sd, se_change=se,
                    normal_ci_lo=normal_lo, normal_ci_hi=normal_hi,
                    normal_ci_contains_zero=ci_contains_zero(normal_lo, normal_hi),
                    bootstrap_ci_lo=boot_lo, bootstrap_ci_hi=boot_hi,
                    bootstrap_ci_contains_zero=ci_contains_zero(boot_lo, boot_hi),
                    win_rate_1m_better=float((diffs > 0).mean()),
                ))
    return pd.DataFrame(rows)


def main():
    path = FC.RESULTS_DIR / "final_holdout_episode_level.csv"
    assert path.exists(), f"final_holdout_episode_level.csv not found at {path} -- run final_evaluate_holdout.py first"
    episodes = pd.read_csv(path)

    arch_df = compute_architecture_contrasts(episodes)
    arch_path = FC.RESULTS_DIR / "architecture_contrasts.csv"
    arch_df.to_csv(arch_path, index=False)
    print(f"Saved {arch_path} ({len(arch_df)} rows)")

    bench_df = compute_benchmark_paired_contrasts(episodes)
    bench_path = FC.RESULTS_DIR / "benchmark_paired_contrasts.csv"
    bench_df.to_csv(bench_path, index=False)
    print(f"Saved {bench_path} ({len(bench_df)} rows)")

    fixed_df = compute_fixed_200k_vs_1m_contrasts(episodes)
    fixed_path = FC.RESULTS_DIR / "fixed_200k_vs_1m_contrasts.csv"
    fixed_df.to_csv(fixed_path, index=False)
    print(f"Saved {fixed_path} ({len(fixed_df)} rows)")


if __name__ == "__main__":
    main()
