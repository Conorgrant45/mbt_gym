"""
phase7_analyze_time_to_benchmark.py
------------------------------------------
Pure analysis of ALREADY-COMPLETED Phase 7 results -- no retraining, no new
holdout evaluation, no modification of any existing Phase 7 CSV/report.
Reads phase7_checkpoint_summary.csv, phase7_benchmark_on_own_validation_seeds.csv,
and phase7_run_status.csv (all produced by the prior phase7_run_training.py /
phase7_evaluate_holdout-style runs) and estimates, per learner seed, how many
transitions and how much PURE TRAINING wall-clock time were needed to reach
belief-weighted-benchmark-level deterministic validation performance.

Pure-training vs evaluation time split: the original run only recorded ONE
cumulative wall-clock timestamp per checkpoint (`wall_clock_elapsed_seconds`,
covering training AND that checkpoint's own evaluation together -- see
phase7_run_training.run_one_seed). To separate them without retraining, this
script measures the evaluation-only cost EMPIRICALLY by loading three
ALREADY-SAVED checkpoints (no training step) and timing
phase7_common.evaluate_checkpoint_model on each -- this loads a frozen model
and runs the SAME 50-seed deterministic+stochastic validation episodes the
original run already used, so it is read-only with respect to existing
results (measuring wall-clock cost only, not producing new performance
numbers). The three measurements agree closely (32-34s), consistent with
evaluation cost being policy-independent in this event-driven environment
(fixed validation seeds -> fixed event/arrival sequences; only tiny MLP
forward passes vary). Their mean is used as a constant per-checkpoint
evaluation-time estimate, subtracted (once per checkpoint reached) from the
recorded cumulative wall-clock time to estimate pure training time.

Run from repo root:
    python phase7_analyze_time_to_benchmark.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import phase7_common as P7

# Empirically measured (see module docstring): evaluate_checkpoint_model on
# three already-saved checkpoints (seed0/t=200000, seed2/t=500000,
# seed4/t=1000000) took 34.10s / 33.94s / 32.26s respectively. No training
# occurred in this measurement -- these are freshly-loaded frozen models.
EVAL_SECONDS_PER_CHECKPOINT_MEASUREMENTS = [34.10, 33.94, 32.26]
EVAL_SECONDS_PER_CHECKPOINT = float(np.mean(EVAL_SECONDS_PER_CHECKPOINT_MEASUREMENTS))

Z_975 = 1.959963984540054  # 1.96, full precision


def combined_threshold(benchmark_mean: float, benchmark_se: float, ppo_se: float) -> float:
    return benchmark_mean - Z_975 * np.sqrt(ppo_se ** 2 + benchmark_se ** 2)


def find_first_qualifying_checkpoint(cp_seed: pd.DataFrame, benchmark_mean: float, benchmark_se: float) -> dict:
    """cp_seed: this seed's checkpoints, sorted by timestep ascending, with a
    boolean 'qualifies' column already attached. Returns the earliest
    checkpoint (by position) that qualifies AND for which at least 2 of the
    next 3 AVAILABLE checkpoints also qualify -- or a dict with found=False
    if no checkpoint satisfies this."""
    n = len(cp_seed)
    qualifies = cp_seed["qualifies"].to_numpy()
    for i in range(n):
        if not qualifies[i]:
            continue
        window = qualifies[i + 1:i + 4]  # next up to 3 available checkpoints
        n_available = len(window)
        n_pass = int(window.sum())
        required = min(2, n_available)  # if fewer than 2 remain, require all of what remains
        if n_available == 0 or n_pass >= required:
            return dict(found=True, index=i, n_confirm_available=n_available, n_confirm_pass=n_pass)
    return dict(found=False, index=None, n_confirm_available=None, n_confirm_pass=None)


def main():
    cp = pd.read_csv(P7.RESULTS_DIR / "phase7_checkpoint_summary.csv")
    bench = pd.read_csv(P7.RESULTS_DIR / "phase7_benchmark_on_own_validation_seeds.csv")
    run_status = pd.read_csv(P7.RESULTS_DIR / "phase7_run_status.csv")

    bw = bench[bench["policy"] == "belief_weighted"]["full_objective"]
    benchmark_mean = float(bw.mean())
    benchmark_se = float(bw.std(ddof=1) / np.sqrt(len(bw)))
    print(f"Belief-weighted benchmark (n={len(bw)}, phase7 validation seeds "
          f"{P7.VALIDATION_SEEDS[0]}-{P7.VALIDATION_SEEDS[-1]}): "
          f"mean={benchmark_mean:.4f}  se={benchmark_se:.4f}")
    print(f"Eval-time-per-checkpoint estimate (mean of 3 measurements "
          f"{EVAL_SECONDS_PER_CHECKPOINT_MEASUREMENTS}): {EVAL_SECONDS_PER_CHECKPOINT:.2f}s")

    rows = []
    for seed in P7.LEARNER_SEEDS:
        cp_seed = cp[cp["learner_seed"] == seed].sort_values("timestep").reset_index(drop=True)
        cp_seed["checkpoint_index"] = np.arange(1, len(cp_seed) + 1)  # 1-based count of checkpoints reached
        cp_seed["est_eval_seconds_cumulative"] = cp_seed["checkpoint_index"] * EVAL_SECONDS_PER_CHECKPOINT
        cp_seed["est_pure_training_seconds"] = (
            cp_seed["wall_clock_elapsed_seconds"] - cp_seed["est_eval_seconds_cumulative"]
        )
        cp_seed["threshold"] = combined_threshold(benchmark_mean, benchmark_se, cp_seed["det_se_objective"])
        cp_seed["qualifies"] = cp_seed["det_mean_objective"] >= cp_seed["threshold"]

        result = find_first_qualifying_checkpoint(cp_seed, benchmark_mean, benchmark_se)
        final_row = cp_seed.iloc[-1]

        if result["found"]:
            q_row = cp_seed.iloc[result["index"]]
            later = cp_seed.iloc[result["index"] + 1:]
            later_competitive_frac = float(later["qualifies"].mean()) if len(later) else float("nan")
            rows.append(dict(
                learner_seed=seed,
                first_qualifying_timestep=int(q_row["timestep"]),
                est_pure_training_minutes_at_qualifying=float(q_row["est_pure_training_seconds"] / 60.0),
                wall_clock_minutes_at_qualifying=float(q_row["wall_clock_elapsed_seconds"] / 60.0),
                det_objective_at_qualifying=float(q_row["det_mean_objective"]),
                det_se_at_qualifying=float(q_row["det_se_objective"]),
                benchmark_mean=benchmark_mean, benchmark_se=benchmark_se,
                n_confirm_checkpoints_available=result["n_confirm_available"],
                n_confirm_checkpoints_passed=result["n_confirm_pass"],
                frac_later_checkpoints_competitive=later_competitive_frac,
                final_1m_det_objective=float(final_row["det_mean_objective"]),
                final_1m_det_se=float(final_row["det_se_objective"]),
                final_1m_qualifies=bool(final_row["qualifies"]),
                qualified=True,
            ))
        else:
            rows.append(dict(
                learner_seed=seed,
                first_qualifying_timestep=None,
                est_pure_training_minutes_at_qualifying=None,
                wall_clock_minutes_at_qualifying=None,
                det_objective_at_qualifying=None,
                det_se_at_qualifying=None,
                benchmark_mean=benchmark_mean, benchmark_se=benchmark_se,
                n_confirm_checkpoints_available=None,
                n_confirm_checkpoints_passed=None,
                frac_later_checkpoints_competitive=None,
                final_1m_det_objective=float(final_row["det_mean_objective"]),
                final_1m_det_se=float(final_row["det_se_objective"]),
                final_1m_qualifies=bool(final_row["qualifies"]),
                qualified=False,
            ))

    out_df = pd.DataFrame(rows)
    out_path = P7.RESULTS_DIR / "phase7_time_to_benchmark.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(out_df.to_string(index=False))

    qualified = out_df[out_df["qualified"]]
    n_qualified = len(qualified)
    mean_transitions = float(qualified["first_qualifying_timestep"].mean()) if n_qualified else float("nan")
    median_transitions = float(qualified["first_qualifying_timestep"].median()) if n_qualified else float("nan")
    mean_minutes = float(qualified["est_pure_training_minutes_at_qualifying"].mean()) if n_qualified else float("nan")
    median_minutes = float(qualified["est_pure_training_minutes_at_qualifying"].median()) if n_qualified else float("nan")

    # --- Time-budget breakdown (item 5): training vs evaluation vs tests ---
    total_run_wallclock = float(run_status["duration_seconds"].sum())
    n_checkpoints_total = len(cp)
    total_eval_estimate = n_checkpoints_total * EVAL_SECONDS_PER_CHECKPOINT
    total_training_estimate = total_run_wallclock - total_eval_estimate

    summary_lines = []
    summary_lines.append("# Phase 7: time-to-benchmark analysis\n")
    summary_lines.append(
        f"Belief-weighted benchmark on Phase 7's own validation seeds "
        f"({P7.VALIDATION_SEEDS[0]}-{P7.VALIDATION_SEEDS[-1]}, n={len(bw)}): "
        f"mean={benchmark_mean:.4f}, SE={benchmark_se:.4f}.\n"
    )
    summary_lines.append(
        f"Criterion: earliest checkpoint with det_mean_objective >= benchmark_mean - "
        f"1.96*sqrt(PPO_SE^2 + benchmark_SE^2), confirmed by >=2 of the next 3 available "
        f"checkpoints also qualifying.\n"
    )
    summary_lines.append(
        f"**Caveat, stated plainly**: the belief-weighted benchmark's own per-episode standard "
        f"deviation on this validation set is large (SD~44.8 over n=50 episodes, SE={benchmark_se:.2f}) "
        f"-- an inherent feature of this environment's per-episode P&L variance, not a flaw in the "
        f"benchmark estimate. Combined with each checkpoint's own SE (~4.4-9.9), the resulting "
        f"statistical-indistinguishability threshold is wide (roughly benchmark_mean-13 to -15 in "
        f"absolute terms), so it is satisfied as soon as PPO's performance is broadly in the same "
        f"range as the benchmark -- which happens at the FIRST evaluated checkpoint (16,000 "
        f"transitions) for every seed. This is the mechanically correct answer to the criterion as "
        f"specified, but it should NOT be read as 'the policy matches the benchmark's performance "
        f"level after 16,000 transitions' -- the raw point estimate keeps rising substantially for "
        f"hundreds of thousands of further transitions (see phase7_groupB_1m_convergence_report.md "
        f"and phase7_checkpoint_summary.csv), and only becomes statistically DISTINGUISHABLE from "
        f"pure noise around the benchmark much later, if at all, given how wide the benchmark's own "
        f"uncertainty is. This criterion measures 'not implausible given the noise,' not 'has "
        f"converged to the benchmark's point estimate.'\n"
    )
    summary_lines.append("## Per-seed results\n")
    for _, r in out_df.iterrows():
        if r["qualified"]:
            summary_lines.append(
                f"- **Seed {int(r['learner_seed'])}**: first qualifies at "
                f"**{int(r['first_qualifying_timestep']):,} transitions** "
                f"(~{r['est_pure_training_minutes_at_qualifying']:.1f} min pure training, "
                f"~{r['wall_clock_minutes_at_qualifying']:.1f} min wall-clock incl. evaluation). "
                f"Objective {r['det_objective_at_qualifying']:.2f} (SE {r['det_se_at_qualifying']:.2f}) vs. "
                f"benchmark {benchmark_mean:.2f} (SE {benchmark_se:.2f}). "
                f"Confirmation: {int(r['n_confirm_checkpoints_passed'])}/{int(r['n_confirm_checkpoints_available'])} "
                f"of the next checkpoints also qualified; "
                f"{r['frac_later_checkpoints_competitive']*100:.0f}% of ALL later checkpoints remained competitive. "
                f"Final (1,000,000 transitions): {r['final_1m_det_objective']:.2f} "
                f"(SE {r['final_1m_det_se']:.2f}), qualifies={r['final_1m_qualifies']}."
            )
        else:
            summary_lines.append(
                f"- **Seed {int(r['learner_seed'])}**: never reached a confirmed qualifying checkpoint. "
                f"Final (1,000,000 transitions): {r['final_1m_det_objective']:.2f} "
                f"(SE {r['final_1m_det_se']:.2f}), qualifies={r['final_1m_qualifies']}."
            )
    summary_lines.append("\n## Aggregate (seeds that reached benchmark-level performance)\n")
    summary_lines.append(f"- Seeds qualifying: {n_qualified}/{len(out_df)}")
    summary_lines.append(f"- Mean transitions to benchmark: {mean_transitions:,.0f}")
    summary_lines.append(f"- Median transitions to benchmark: {median_transitions:,.0f}")
    summary_lines.append(f"- Mean pure-training minutes to benchmark: {mean_minutes:.1f}")
    summary_lines.append(f"- Median pure-training minutes to benchmark: {median_minutes:.1f}")
    summary_lines.append("\n## Time-budget breakdown (Phase 7, all 5 seeds combined)\n")
    summary_lines.append(
        f"- Total recorded run wall-clock (training + checkpoint evaluation, from "
        f"phase7_run_status.csv): {total_run_wallclock:.1f}s ({total_run_wallclock/60:.1f} min)."
    )
    summary_lines.append(
        f"- Estimated evaluation-only time ({n_checkpoints_total} checkpoints x "
        f"{EVAL_SECONDS_PER_CHECKPOINT:.2f}s/checkpoint, measured empirically on 3 already-saved "
        f"checkpoints without retraining): {total_eval_estimate:.1f}s ({total_eval_estimate/60:.1f} min)."
    )
    summary_lines.append(
        f"- Estimated pure-training-only time (remainder): {total_training_estimate:.1f}s "
        f"({total_training_estimate/60:.1f} min) -- "
        f"{100*total_training_estimate/total_run_wallclock:.1f}% of total run wall-clock."
    )
    summary_lines.append(
        "- Full project test-suite runtime (333 tests, includes far more than Phase 7's own 15): "
        "~674.5s pre-run + ~554.2s post-run (from this session's own pytest invocations; not a "
        "Phase-7-only figure, reported for completeness)."
    )
    summary_lines.append(
        "- Plotting (phase7_generate_plots.py) and report-generation time: not separately "
        "instrumented in the original run; both are negligible (seconds, not minutes) relative to "
        "training and are not re-run here per this task's instruction not to modify existing outputs."
    )

    summary_path = P7.RESULTS_DIR / "phase7_time_to_benchmark_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"\nSaved {summary_path}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    seed_colors = {0: "tab:red", 1: "tab:orange", 2: "tab:green", 3: "tab:blue", 4: "tab:purple"}
    ax = axes[0]
    for seed in P7.LEARNER_SEEDS:
        sub = cp[cp["learner_seed"] == seed].sort_values("timestep")
        ax.plot(sub["timestep"], sub["det_mean_objective"], marker="o", markersize=3,
                color=seed_colors[seed], label=f"seed {seed}")
    ax.axhline(benchmark_mean, color="black", lw=1.5, label="belief-weighted benchmark")
    ax.axhspan(benchmark_mean - Z_975 * benchmark_se, benchmark_mean + Z_975 * benchmark_se,
               color="black", alpha=0.08)
    for _, r in qualified.iterrows():
        ax.axvline(r["first_qualifying_timestep"], color=seed_colors[int(r["learner_seed"])],
                    ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("PPO transitions")
    ax.set_ylabel("Deterministic validation objective")
    ax.set_title("Time-to-benchmark: dashed lines mark each seed's first qualifying checkpoint")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    bars_x = [f"seed {int(s)}" for s in qualified["learner_seed"]]
    bars_y = qualified["first_qualifying_timestep"].to_numpy()
    colors = [seed_colors[int(s)] for s in qualified["learner_seed"]]
    ax2.bar(bars_x, bars_y, color=colors)
    if n_qualified:
        ax2.axhline(mean_transitions, color="black", ls="--", lw=1.2, label=f"mean={mean_transitions:,.0f}")
        ax2.axhline(median_transitions, color="gray", ls=":", lw=1.2, label=f"median={median_transitions:,.0f}")
    ax2.set_ylabel("Transitions to first qualifying checkpoint")
    ax2.set_title("Transitions required to reach benchmark-level performance")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    plot_path = P7.RESULTS_DIR / "phase7_time_to_benchmark.png"
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
