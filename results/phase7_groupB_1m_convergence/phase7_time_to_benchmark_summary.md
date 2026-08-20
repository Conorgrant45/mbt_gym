# Phase 7: time-to-benchmark analysis

Belief-weighted benchmark on Phase 7's own validation seeds (260000-260049, n=50): mean=45.6013, SE=6.3329.

Criterion: earliest checkpoint with det_mean_objective >= benchmark_mean - 1.96*sqrt(PPO_SE^2 + benchmark_SE^2), confirmed by >=2 of the next 3 available checkpoints also qualifying.

**Caveat, stated plainly**: the belief-weighted benchmark's own per-episode standard deviation on this validation set is large (SD~44.8 over n=50 episodes, SE=6.33) -- an inherent feature of this environment's per-episode P&L variance, not a flaw in the benchmark estimate. Combined with each checkpoint's own SE (~4.4-9.9), the resulting statistical-indistinguishability threshold is wide (roughly benchmark_mean-13 to -15 in absolute terms), so it is satisfied as soon as PPO's performance is broadly in the same range as the benchmark -- which happens at the FIRST evaluated checkpoint (16,000 transitions) for every seed. This is the mechanically correct answer to the criterion as specified, but it should NOT be read as 'the policy matches the benchmark's performance level after 16,000 transitions' -- the raw point estimate keeps rising substantially for hundreds of thousands of further transitions (see phase7_groupB_1m_convergence_report.md and phase7_checkpoint_summary.csv), and only becomes statistically DISTINGUISHABLE from pure noise around the benchmark much later, if at all, given how wide the benchmark's own uncertainty is. This criterion measures 'not implausible given the noise,' not 'has converged to the benchmark's point estimate.'

## Per-seed results

- **Seed 0**: first qualifies at **16,000 transitions** (~0.4 min pure training, ~0.9 min wall-clock incl. evaluation). Objective 41.94 (SE 6.35) vs. benchmark 45.60 (SE 6.33). Confirmation: 3/3 of the next checkpoints also qualified; 100% of ALL later checkpoints remained competitive. Final (1,000,000 transitions): 47.61 (SE 7.65), qualifies=True.
- **Seed 1**: first qualifies at **16,000 transitions** (~0.4 min pure training, ~0.9 min wall-clock incl. evaluation). Objective 41.75 (SE 5.83) vs. benchmark 45.60 (SE 6.33). Confirmation: 3/3 of the next checkpoints also qualified; 100% of ALL later checkpoints remained competitive. Final (1,000,000 transitions): 48.79 (SE 9.88), qualifies=True.
- **Seed 2**: first qualifies at **16,000 transitions** (~0.2 min pure training, ~0.8 min wall-clock incl. evaluation). Objective 40.43 (SE 4.40) vs. benchmark 45.60 (SE 6.33). Confirmation: 3/3 of the next checkpoints also qualified; 100% of ALL later checkpoints remained competitive. Final (1,000,000 transitions): 43.02 (SE 9.91), qualifies=True.
- **Seed 3**: first qualifies at **16,000 transitions** (~0.5 min pure training, ~1.0 min wall-clock incl. evaluation). Objective 40.78 (SE 4.91) vs. benchmark 45.60 (SE 6.33). Confirmation: 3/3 of the next checkpoints also qualified; 100% of ALL later checkpoints remained competitive. Final (1,000,000 transitions): 46.00 (SE 8.75), qualifies=True.
- **Seed 4**: first qualifies at **16,000 transitions** (~0.2 min pure training, ~0.8 min wall-clock incl. evaluation). Objective 41.51 (SE 5.48) vs. benchmark 45.60 (SE 6.33). Confirmation: 3/3 of the next checkpoints also qualified; 100% of ALL later checkpoints remained competitive. Final (1,000,000 transitions): 46.46 (SE 9.99), qualifies=True.

## Aggregate (seeds that reached benchmark-level performance)

- Seeds qualifying: 5/5
- Mean transitions to benchmark: 16,000
- Median transitions to benchmark: 16,000
- Mean pure-training minutes to benchmark: 0.3
- Median pure-training minutes to benchmark: 0.4

## Time-budget breakdown (Phase 7, all 5 seeds combined)

- Total recorded run wall-clock (training + checkpoint evaluation, from phase7_run_status.csv): 9466.1s (157.8 min).
- Estimated evaluation-only time (105 checkpoints x 33.43s/checkpoint, measured empirically on 3 already-saved checkpoints without retraining): 3510.5s (58.5 min).
- Estimated pure-training-only time (remainder): 5955.6s (99.3 min) -- 62.9% of total run wall-clock.
- Full project test-suite runtime (333 tests, includes far more than Phase 7's own 15): ~674.5s pre-run + ~554.2s post-run (from this session's own pytest invocations; not a Phase-7-only figure, reported for completeness).
- Plotting (phase7_generate_plots.py) and report-generation time: not separately instrumented in the original run; both are negligible (seconds, not minutes) relative to training and are not re-run here per this task's instruction not to modify existing outputs.
