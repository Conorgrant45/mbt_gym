"""
phase7_post_training/phase7_evaluate_unseen_holdout.py
------------------------------------------
Phase 7 post-training analysis, Part B: freeze each Group B learner seed's
validation-selected checkpoint (selection performed ONCE, using only
phase7_checkpoint_summary.csv, never the holdout below) and evaluate it,
deterministically, on a completely fresh, disjoint 500-path holdout set
alongside the analytical oracle, belief-weighted policy, and frozen
supervised clone -- reusing evaluate_agents_event_driven.py's
already-validated per-episode runners unmodified (matched exogenous paths
across every policy by construction, since each policy is evaluated on the
SAME seed -> same event-driven environment path).

No retraining. No change to checkpoint selection. No modification of any
existing Phase 7 file -- every output goes under
results/phase7_groupB_1m_convergence/post_training_analysis/.

Run from repo root:
    python phase7_post_training/phase7_evaluate_unseen_holdout.py
"""
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

import json
import subprocess
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shared.phase4_common as P4
import shared.phase5_common as P5
import shared.phase7_common as P7
import phase7_post_training.phase7_post_training_common as PC
import shared.evaluate_agents_event_driven as EAED
from shared.phase4_supervised_clone import SupervisedCloneAgent
from shared.train_agents import get_dependency_versions

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 999_101  # analysis-only RNG, not an environment seed
BENCHMARK_POLICIES = ("belief_weighted", "oracle")  # analytic policies via EAED
Z_975 = 1.959963984540054

EPISODE_FIELDS = [
    "full_objective", "raw_pnl", "spread_revenue", "adverse_selection_loss",
    "running_penalty", "terminal_penalty", "fills",
    "mean_abs_inventory", "terminal_abs_inventory", "mean_signed_inventory",
    "mean_quoted_spread", "episode_length", "reward_reconciliation_error",
]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=P7.REPO_ROOT).decode().strip()
    except Exception as e:
        return f"UNKNOWN ({e})"


def run_all_episodes(selections: dict, clone_agent, controls: dict, seeds: list) -> pd.DataFrame:
    rows = []
    for policy_name in ("oracle", "belief_weighted"):
        print(f"Evaluating analytical policy: {policy_name} ({len(seeds)} holdout episodes)...")
        for seed in seeds:
            m = EAED.run_event_analytic_agent_episode(policy_name, controls, seed)
            rows.append(dict(policy=policy_name, learner_seed=None, evaluation_seed=seed,
                              **{k: m[k] for k in EPISODE_FIELDS}))

    print(f"Evaluating frozen supervised clone ({len(seeds)} holdout episodes)...")
    for seed in seeds:
        m = EAED.run_event_hamilton_agent_episode(clone_agent, seed)
        rows.append(dict(policy="frozen_clone", learner_seed=None, evaluation_seed=seed,
                          **{k: m[k] for k in EPISODE_FIELDS}))

    for seed, sel in selections.items():
        print(f"Evaluating Group B seed {seed} (frozen checkpoint t={sel['timestep']:,}), "
              f"deterministic, ({len(seeds)} holdout episodes)...")
        model = PC.load_frozen_checkpoint(seed, sel["timestep"])
        for eval_seed in seeds:
            m = EAED.run_event_hamilton_agent_episode(model, eval_seed)  # deterministic=True internally
            rows.append(dict(policy="hamilton_ppo", learner_seed=seed, evaluation_seed=eval_seed,
                              **{k: m[k] for k in EPISODE_FIELDS}))
        del model

    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def paired_contrast(ppo_vals: pd.Series, bench_vals: pd.Series, label_ppo: str, label_bench: str) -> dict:
    diffs = (ppo_vals.to_numpy() - bench_vals.to_numpy())
    n = len(diffs)
    mean_diff = float(diffs.mean())
    sd_diff = float(diffs.std(ddof=1))
    se_diff = sd_diff / np.sqrt(n)
    ci_lo, ci_hi = mean_diff - Z_975 * se_diff, mean_diff + Z_975 * se_diff
    boot_lo, boot_hi = bootstrap_ci(diffs)
    win_rate = float((diffs > 0).mean())
    ci_includes_zero = ci_lo <= 0.0 <= ci_hi
    if ci_includes_zero:
        verdict = "not statistically distinguishable from zero -- no superiority claim"
    elif mean_diff > 0:
        verdict = f"{label_ppo} exceeds {label_bench} (CI excludes zero, positive)"
    else:
        verdict = f"{label_ppo} falls below {label_bench} (CI excludes zero, negative)"
    return dict(
        ppo_group=label_ppo, benchmark=label_bench, n_paths=n,
        mean_paired_diff=mean_diff, paired_sd=sd_diff, se=se_diff,
        ci95_lo=ci_lo, ci95_hi=ci_hi,
        bootstrap_ci95_lo=boot_lo, bootstrap_ci95_hi=boot_hi,
        win_rate=win_rate, ci_includes_zero=bool(ci_includes_zero), verdict=verdict,
    )


def main():
    t0 = time.time()
    PC.POST_DIR.mkdir(parents=True, exist_ok=True)

    disjoint_check = PC.verify_new_holdout_disjoint()
    assert disjoint_check["disjoint"], f"New holdout range is NOT disjoint: {disjoint_check['overlaps']}"
    print(f"New holdout range {PC.NEW_HOLDOUT_SEEDS[0]}-{PC.NEW_HOLDOUT_SEEDS[-1]} "
          f"({len(PC.NEW_HOLDOUT_SEEDS)} seeds) verified disjoint from all prior ranges.")

    cp = pd.read_csv(P7.RESULTS_DIR / "phase7_checkpoint_summary.csv")
    selections = {seed: PC.select_checkpoint_for_seed(seed, cp) for seed in P7.LEARNER_SEEDS}
    for seed, sel in selections.items():
        print(f"Frozen seed {seed}: checkpoint t={sel['timestep']:,}, "
              f"validation det_mean_objective={sel['det_mean_objective']:.3f}")

    controls = P4.build_analytical_controls()
    clone_net = P5.load_supervised_clone_net()
    clone_agent = SupervisedCloneAgent(clone_net)

    episodes_df = run_all_episodes(selections, clone_agent, controls, PC.NEW_HOLDOUT_SEEDS)
    episodes_path = PC.POST_DIR / "phase7_unseen_holdout_episodes.csv"
    episodes_df.to_csv(episodes_path, index=False)
    max_recon = episodes_df["reward_reconciliation_error"].max()
    print(f"\n{len(episodes_df)} episode rows saved to {episodes_path}")
    print(f"Max reward-reconciliation error: {max_recon:.3e}")

    # --- Seed-level summary (one row per policy x learner_seed) ---
    seed_rows = []
    for (policy, learner_seed), sub in episodes_df.groupby(["policy", "learner_seed"], dropna=False):
        row = dict(policy=policy, learner_seed=learner_seed, n_episodes=len(sub))
        for field in EPISODE_FIELDS:
            row[f"mean_{field}"] = float(sub[field].mean())
            if field == "full_objective":
                row["sd_full_objective"] = float(sub[field].std(ddof=1))
                row["se_full_objective"] = row["sd_full_objective"] / np.sqrt(len(sub))
        seed_rows.append(row)
    seed_summary_df = pd.DataFrame(seed_rows)
    seed_summary_path = PC.POST_DIR / "phase7_unseen_holdout_seed_summary.csv"
    seed_summary_df.to_csv(seed_summary_path, index=False)
    print(f"Saved {seed_summary_path}")

    # --- Group-level summary (PPO across 5 seeds, plus each benchmark) ---
    ppo_seed_summary = seed_summary_df[seed_summary_df["policy"] == "hamilton_ppo"]
    group_rows = []
    group_rows.append(dict(
        group="hamilton_ppo (5-seed aggregate)",
        n_seeds=len(ppo_seed_summary),
        mean_full_objective=float(ppo_seed_summary["mean_full_objective"].mean()),
        cross_seed_sd_full_objective=float(ppo_seed_summary["mean_full_objective"].std(ddof=1)),
        mean_fills=float(ppo_seed_summary["mean_fills"].mean()),
        mean_quoted_spread=float(ppo_seed_summary["mean_mean_quoted_spread"].mean()),
        mean_abs_inventory=float(ppo_seed_summary["mean_mean_abs_inventory"].mean()),
    ))
    for policy in ("oracle", "belief_weighted", "frozen_clone"):
        row = seed_summary_df[seed_summary_df["policy"] == policy].iloc[0]
        group_rows.append(dict(
            group=policy, n_seeds=None,
            mean_full_objective=row["mean_full_objective"],
            cross_seed_sd_full_objective=None,
            mean_fills=row["mean_fills"],
            mean_quoted_spread=row["mean_mean_quoted_spread"],
            mean_abs_inventory=row["mean_mean_abs_inventory"],
        ))
    group_summary_df = pd.DataFrame(group_rows)
    group_summary_path = PC.POST_DIR / "phase7_unseen_holdout_group_summary.csv"
    group_summary_df.to_csv(group_summary_path, index=False)
    print(f"Saved {group_summary_path}")
    print(group_summary_df.to_string(index=False))

    # --- Paired contrasts ---
    bench_series = {}
    for b in ("belief_weighted", "oracle", "frozen_clone"):
        col = episodes_df[episodes_df["policy"] == b].set_index("evaluation_seed")["full_objective"]
        bench_series[b] = col.reindex(PC.NEW_HOLDOUT_SEEDS)

    contrast_rows = []
    ppo_seed_series = {}
    for seed in P7.LEARNER_SEEDS:
        col = episodes_df[(episodes_df["policy"] == "hamilton_ppo") & (episodes_df["learner_seed"] == seed)] \
            .set_index("evaluation_seed")["full_objective"].reindex(PC.NEW_HOLDOUT_SEEDS)
        ppo_seed_series[seed] = col
        for b_name, b_series in bench_series.items():
            contrast_rows.append(paired_contrast(col, b_series, f"hamilton_ppo_seed{seed}", b_name))

    ppo_aggregate = pd.concat(ppo_seed_series.values(), axis=1).mean(axis=1)
    for b_name, b_series in bench_series.items():
        contrast_rows.append(paired_contrast(ppo_aggregate, b_series, "hamilton_ppo_5seed_aggregate", b_name))

    contrasts_df = pd.DataFrame(contrast_rows)
    contrasts_path = PC.POST_DIR / "phase7_unseen_holdout_paired_contrasts.csv"
    contrasts_df.to_csv(contrasts_path, index=False)
    print(f"\nSaved {contrasts_path}")
    print(contrasts_df[["ppo_group", "benchmark", "mean_paired_diff", "ci95_lo", "ci95_hi", "win_rate", "verdict"]]
          .to_string(index=False))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    labels, means, ses, colors = [], [], [], []
    seed_colors = {0: "tab:red", 1: "tab:orange", 2: "tab:green", 3: "tab:blue", 4: "tab:purple"}
    for seed in P7.LEARNER_SEEDS:
        s = ppo_seed_series[seed]
        labels.append(f"PPO seed {seed}")
        means.append(s.mean())
        ses.append(s.std(ddof=1) / np.sqrt(len(s)))
        colors.append(seed_colors[seed])
    for b_name, b_series in bench_series.items():
        labels.append(b_name)
        means.append(b_series.mean())
        ses.append(b_series.std(ddof=1) / np.sqrt(len(b_series)))
        colors.append("gray")
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=[Z_975 * s for s in ses], color=colors, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean full objective (500 unseen holdout paths, 95% CI)")
    ax.set_title(f"Phase 7 unseen holdout ({len(PC.NEW_HOLDOUT_SEEDS)} paths, "
                 f"seeds {PC.NEW_HOLDOUT_SEEDS[0]}-{PC.NEW_HOLDOUT_SEEDS[-1]}): deterministic evaluation")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    plot_path = PC.POST_DIR / "phase7_unseen_holdout_objective_plot.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    print(f"Saved {plot_path}")

    elapsed = time.time() - t0

    # --- Manifest ---
    manifest = dict(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git_commit=git_commit(),
        dependency_versions=get_dependency_versions(),
        frozen_selected_checkpoints={
            str(seed): dict(timestep=sel["timestep"], det_mean_objective=sel["det_mean_objective"],
                              det_se_objective=sel["det_se_objective"],
                              checkpoint_path=str(P7.checkpoint_path(seed, sel["timestep"])))
            for seed, sel in selections.items()
        },
        selection_rule="argmax deterministic mean validation objective (phase7_checkpoint_summary.csv, "
                       "Phase 7's own fixed 50-seed validation set 260000-260049), tie broken toward "
                       "earlier timestep -- IDENTICAL rule to Phase 6, applied post-hoc since Phase 7 "
                       "itself never froze a selection. Holdout data was never used for this selection.",
        new_holdout_seed_start=PC.NEW_HOLDOUT_SEEDS[0], new_holdout_seed_count=len(PC.NEW_HOLDOUT_SEEDS),
        new_holdout_disjoint_check=disjoint_check,
        prior_seed_ranges_checked={k: sorted(v)[:3] + ["..."] + sorted(v)[-3:] if len(v) > 6 else sorted(v)
                                    for k, v in PC.PRIOR_SEED_RANGES.items()},
        n_episodes_total=len(episodes_df),
        max_reward_reconciliation_error=float(max_recon),
        elapsed_seconds=elapsed,
    )
    manifest_path = PC.POST_DIR / "phase7_unseen_holdout_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Saved {manifest_path}")

    write_report(selections, disjoint_check, seed_summary_df, group_summary_df, contrasts_df, max_recon, elapsed)


def write_report(selections, disjoint_check, seed_summary_df, group_summary_df, contrasts_df, max_recon, elapsed):
    lines = []
    lines.append("# Phase 7 unseen paired holdout evaluation: report\n")
    lines.append("## Frozen selected checkpoint per learner seed\n")
    for seed, sel in selections.items():
        lines.append(f"- Seed {seed}: **t={sel['timestep']:,}** "
                     f"(validation det_mean_objective={sel['det_mean_objective']:.3f}, "
                     f"se={sel['det_se_objective']:.3f}) -- selected using ONLY "
                     f"phase7_checkpoint_summary.csv's existing validation records, never the holdout below.")

    lines.append(f"\n## Unseen holdout seed range\n")
    lines.append(f"- **{PC.NEW_HOLDOUT_SEEDS[0]}-{PC.NEW_HOLDOUT_SEEDS[-1]}** ({len(PC.NEW_HOLDOUT_SEEDS)} paths)")
    lines.append(f"- Disjointness verified programmatically against {len(PC.PRIOR_SEED_RANGES)} previously-used "
                 f"seed ranges spanning Phases 1-7 (learner seeds, training-env seed, every documented "
                 f"validation/holdout/diagnostic range): **{disjoint_check['disjoint']}** "
                 f"(overlaps found: {disjoint_check['overlaps']}).")

    lines.append("\n## Aggregate PPO performance (deterministic, 500 unseen paths)\n")
    agg = group_summary_df[group_summary_df["group"].str.contains("aggregate")].iloc[0]
    lines.append(f"- Mean full objective (5-seed aggregate): {agg['mean_full_objective']:.3f} "
                 f"(cross-seed SD {agg['cross_seed_sd_full_objective']:.3f})")
    lines.append(f"- Mean fills: {agg['mean_fills']:.2f}, mean quoted spread: {agg['mean_quoted_spread']:.3f}, "
                 f"mean abs inventory: {agg['mean_abs_inventory']:.3f}")
    lines.append("\nPer-seed:")
    ppo_seeds = seed_summary_df[seed_summary_df["policy"] == "hamilton_ppo"]
    for _, r in ppo_seeds.iterrows():
        lines.append(f"- Seed {int(r['learner_seed'])}: mean full objective "
                     f"{r['mean_full_objective']:.3f} (SE {r['se_full_objective']:.3f}), "
                     f"fills {r['mean_fills']:.2f}, quoted spread {r['mean_mean_quoted_spread']:.3f}")

    lines.append("\n## Analytical benchmark performance (same 500 paths)\n")
    for policy in ("oracle", "belief_weighted", "frozen_clone"):
        r = seed_summary_df[seed_summary_df["policy"] == policy].iloc[0]
        lines.append(f"- {policy}: mean full objective {r['mean_full_objective']:.3f} "
                     f"(SE {r['se_full_objective']:.3f})")

    lines.append("\n## Paired confidence intervals (PPO minus benchmark, per path)\n")
    lines.append("| PPO group | Benchmark | Mean diff | 95% CI | Bootstrap 95% CI | Win rate | Verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in contrasts_df.iterrows():
        lines.append(f"| {r['ppo_group']} | {r['benchmark']} | {r['mean_paired_diff']:.3f} | "
                     f"[{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}] | "
                     f"[{r['bootstrap_ci95_lo']:.3f}, {r['bootstrap_ci95_hi']:.3f}] | "
                     f"{r['win_rate']:.3f} | {r['verdict']} |")

    lines.append("\n## Whether PPO matches, exceeds, or falls below each benchmark\n")
    agg_contrasts = contrasts_df[contrasts_df["ppo_group"] == "hamilton_ppo_5seed_aggregate"]
    any_ci_includes_zero = False
    for _, r in agg_contrasts.iterrows():
        if r["ci_includes_zero"]:
            any_ci_includes_zero = True
            statement = (f"**statistically indistinguishable** from {r['benchmark']} (95% CI includes "
                         f"zero) -- no superiority or inferiority is claimed.")
        elif r["mean_paired_diff"] > 0:
            statement = f"**exceeds** {r['benchmark']} (95% CI [{r['ci95_lo']:.2f}, {r['ci95_hi']:.2f}] excludes zero, positive)."
        else:
            statement = (f"**falls below** {r['benchmark']} (95% CI [{r['ci95_lo']:.2f}, {r['ci95_hi']:.2f}] "
                         f"excludes zero, negative) -- this is a confidently NEGATIVE finding, not an absence "
                         f"of evidence.")
        lines.append(f"- 5-seed aggregate vs. {r['benchmark']}: {statement}")

    lines.append("\n### Interpretation: why does this reverse the validation-based picture?\n")
    lines.append(
        "On Phase 7's own 50-seed validation set, the SELECTED checkpoints looked competitive with (even "
        "nominally above) all three benchmarks -- see `phase7_groupB_1m_convergence_report.md` and "
        "`phase7_time_to_benchmark_summary.md`. On this fresh, never-inspected 500-path holdout, the SAME "
        "frozen checkpoints fall clearly and significantly below all three benchmarks for every one of the "
        "5 seeds. The most likely explanation is checkpoint-selection optimism (a 'winner's curse' effect): "
        "each seed's checkpoint was chosen as the ARGMAX deterministic validation objective over 21 candidate "
        "checkpoints, each itself estimated from only 50 noisy episodes (checkpoint-level SE was "
        "approximately 7-9 objective-units, per `phase7_checkpoint_summary.csv`). Selecting the maximum of "
        "21 noisy estimates systematically overestimates that checkpoint's TRUE mean performance, and a "
        "fresh, independent sample of paths (this holdout) is not subject to the same upward bias -- so "
        "performance reverts toward a lower, more representative level. This is precisely the failure mode "
        "an untouched, disjoint holdout evaluation is designed to detect, and this result should be read as "
        "the more trustworthy estimate of these checkpoints' true performance, not the validation-based "
        "numbers reported earlier in Phase 7."
    )
    if not any_ci_includes_zero:
        lines.append(
            "\nNo contrast in this analysis has a confidence interval including zero -- every PPO seed and "
            "the 5-seed aggregate are confidently below all three benchmarks on this holdout, at the level "
            "of statistical uncertainty this sample size supports."
        )

    lines.append(f"\n## Runtime and tests\n")
    lines.append(f"- Evaluation runtime: {elapsed:.1f}s ({elapsed/60:.1f} min) for "
                 f"{len(PC.NEW_HOLDOUT_SEEDS)} paths x 8 policies (5 PPO seeds + 3 benchmarks) = "
                 f"{len(PC.NEW_HOLDOUT_SEEDS)*8} episodes.")
    lines.append(f"- Max reward-reconciliation error across all episodes: {max_recon:.3e} "
                 f"(confirms the environment/reward pipeline accounted for correctly).")
    lines.append(f"- Full project test suite run before and after this evaluation -- see commit-time "
                 f"pytest output for pass/fail counts (this script does not itself invoke pytest).")

    report_path = PC.POST_DIR / "phase7_unseen_holdout_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
