# Phase 7 validation learning curve: summary

Source data: `phase7_checkpoint_summary.csv` (21 checkpoints x 5 seeds, deterministic validation objective on Phase 7's own fixed 50-seed validation set, seeds 260000-260049) and `phase7_benchmark_on_own_validation_seeds.csv` (oracle/belief-weighted/frozen-clone evaluated on the SAME 50 validation seeds). No retraining or new evaluation was performed to produce this plot/summary.

Benchmark means on these validation seeds: oracle=45.774, belief_weighted=45.601, frozen_clone=45.627.

## Selected checkpoint per seed (validation-only selection, Phase 6's rule reused)

- Seed 0: t=1,000,000, det_mean_objective=47.611, det_se=7.648
- Seed 1: t=800,000, det_mean_objective=49.646, det_se=8.541
- Seed 2: t=400,000, det_mean_objective=46.996, det_se=7.932
- Seed 3: t=900,000, det_mean_objective=48.686, det_se=7.654
- Seed 4: t=400,000, det_mean_objective=47.520, det_se=7.822

## Descriptive evidence: still improving beyond 200,000 transitions?

**Descriptive observation**: the cross-seed mean deterministic validation objective at 200,000 transitions was 43.729; it reaches a maximum of 47.037 at t=400,000 transitions, and its final (1,000,000-transition) value is 46.376 -- both clearly above the 200,000-transition value. This is descriptive evidence that mean validation performance continued to rise materially beyond 200,000 transitions in this run. It is NOT a claim that the policy has converged or that this trend would continue indefinitely -- no asymptotic/convergence test is performed here (see phase7_groupB_1m_convergence_report.md for the slope analysis over the final 200,000 transitions, which found the rate of improvement had become close to flat by the end of the run).

## Earliest benchmark-level checkpoint per seed

Using the belief-weighted benchmark and the same statistical-indistinguishability criterion as `phase7_time_to_benchmark.csv` (PPO mean >= benchmark mean - 1.96*sqrt(PPO_SE^2 + benchmark_SE^2), confirmed by >=2 of the next 3 checkpoints): every one of the 5 seeds first qualifies at the earliest evaluated checkpoint, 16,000 transitions (see that file for the full per-seed detail and the important caveat that this reflects the belief-weighted benchmark's own wide per-episode uncertainty, not convergence to its point estimate).

**Distinguishing descriptive evidence from convergence claims**: the statement 'mean validation objective is higher at 1,000,000 transitions than at 200,000 transitions' is descriptive and directly measured. The statement 'this policy has converged' is an inferential claim this analysis does NOT make -- the near-flat slope in the final 200,000 transitions (established in the main Phase 7 report) is consistent with either an approaching plateau or a temporary slow patch, and this plot alone cannot distinguish those without further training.
