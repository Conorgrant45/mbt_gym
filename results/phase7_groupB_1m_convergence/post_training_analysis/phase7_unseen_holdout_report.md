# Phase 7 unseen paired holdout evaluation: report

## Frozen selected checkpoint per learner seed

- Seed 0: **t=1,000,000** (validation det_mean_objective=47.611, se=7.648) -- selected using ONLY phase7_checkpoint_summary.csv's existing validation records, never the holdout below.
- Seed 1: **t=800,000** (validation det_mean_objective=49.646, se=8.541) -- selected using ONLY phase7_checkpoint_summary.csv's existing validation records, never the holdout below.
- Seed 2: **t=400,000** (validation det_mean_objective=46.996, se=7.932) -- selected using ONLY phase7_checkpoint_summary.csv's existing validation records, never the holdout below.
- Seed 3: **t=900,000** (validation det_mean_objective=48.686, se=7.654) -- selected using ONLY phase7_checkpoint_summary.csv's existing validation records, never the holdout below.
- Seed 4: **t=400,000** (validation det_mean_objective=47.520, se=7.822) -- selected using ONLY phase7_checkpoint_summary.csv's existing validation records, never the holdout below.

## Unseen holdout seed range

- **270000-270499** (500 paths)
- Disjointness verified programmatically against 18 previously-used seed ranges spanning Phases 1-7 (learner seeds, training-env seed, every documented validation/holdout/diagnostic range): **True** (overlaps found: {}).

## Aggregate PPO performance (deterministic, 500 unseen paths)

- Mean full objective (5-seed aggregate): 35.354 (cross-seed SD 1.311)
- Mean fills: 37.19, mean quoted spread: 2.699, mean abs inventory: 4.071

Per-seed:
- Seed 0: mean full objective 37.471 (SE 2.652), fills 41.14, quoted spread 2.560
- Seed 1: mean full objective 35.690 (SE 2.554), fills 35.89, quoted spread 2.740
- Seed 2: mean full objective 34.211 (SE 2.381), fills 38.12, quoted spread 2.681
- Seed 3: mean full objective 34.459 (SE 2.398), fills 34.28, quoted spread 2.799
- Seed 4: mean full objective 34.940 (SE 2.416), fills 36.49, quoted spread 2.715

## Analytical benchmark performance (same 500 paths)

- oracle: mean full objective 43.729 (SE 1.930)
- belief_weighted: mean full objective 43.666 (SE 1.938)
- frozen_clone: mean full objective 43.697 (SE 1.930)

## Paired confidence intervals (PPO minus benchmark, per path)

| PPO group | Benchmark | Mean diff | 95% CI | Bootstrap 95% CI | Win rate | Verdict |
|---|---|---|---|---|---|---|
| hamilton_ppo_seed0 | belief_weighted | -6.195 | [-10.055, -2.335] | [-10.039, -2.381] | 0.398 | hamilton_ppo_seed0 falls below belief_weighted (CI excludes zero, negative) |
| hamilton_ppo_seed0 | oracle | -6.258 | [-10.100, -2.416] | [-10.085, -2.449] | 0.400 | hamilton_ppo_seed0 falls below oracle (CI excludes zero, negative) |
| hamilton_ppo_seed0 | frozen_clone | -6.227 | [-10.087, -2.367] | [-10.070, -2.416] | 0.402 | hamilton_ppo_seed0 falls below frozen_clone (CI excludes zero, negative) |
| hamilton_ppo_seed1 | belief_weighted | -7.976 | [-11.800, -4.151] | [-11.742, -4.257] | 0.378 | hamilton_ppo_seed1 falls below belief_weighted (CI excludes zero, negative) |
| hamilton_ppo_seed1 | oracle | -8.039 | [-11.851, -4.227] | [-11.794, -4.347] | 0.384 | hamilton_ppo_seed1 falls below oracle (CI excludes zero, negative) |
| hamilton_ppo_seed1 | frozen_clone | -8.007 | [-11.828, -4.187] | [-11.780, -4.297] | 0.380 | hamilton_ppo_seed1 falls below frozen_clone (CI excludes zero, negative) |
| hamilton_ppo_seed2 | belief_weighted | -9.454 | [-12.894, -6.014] | [-12.856, -6.102] | 0.338 | hamilton_ppo_seed2 falls below belief_weighted (CI excludes zero, negative) |
| hamilton_ppo_seed2 | oracle | -9.517 | [-12.959, -6.076] | [-12.913, -6.135] | 0.334 | hamilton_ppo_seed2 falls below oracle (CI excludes zero, negative) |
| hamilton_ppo_seed2 | frozen_clone | -9.486 | [-12.907, -6.065] | [-12.872, -6.130] | 0.336 | hamilton_ppo_seed2 falls below frozen_clone (CI excludes zero, negative) |
| hamilton_ppo_seed3 | belief_weighted | -9.207 | [-12.801, -5.613] | [-12.758, -5.708] | 0.366 | hamilton_ppo_seed3 falls below belief_weighted (CI excludes zero, negative) |
| hamilton_ppo_seed3 | oracle | -9.270 | [-12.863, -5.677] | [-12.837, -5.767] | 0.362 | hamilton_ppo_seed3 falls below oracle (CI excludes zero, negative) |
| hamilton_ppo_seed3 | frozen_clone | -9.239 | [-12.823, -5.654] | [-12.776, -5.739] | 0.366 | hamilton_ppo_seed3 falls below frozen_clone (CI excludes zero, negative) |
| hamilton_ppo_seed4 | belief_weighted | -8.726 | [-12.236, -5.216] | [-12.192, -5.251] | 0.356 | hamilton_ppo_seed4 falls below belief_weighted (CI excludes zero, negative) |
| hamilton_ppo_seed4 | oracle | -8.789 | [-12.295, -5.283] | [-12.264, -5.362] | 0.362 | hamilton_ppo_seed4 falls below oracle (CI excludes zero, negative) |
| hamilton_ppo_seed4 | frozen_clone | -8.758 | [-12.260, -5.255] | [-12.194, -5.308] | 0.360 | hamilton_ppo_seed4 falls below frozen_clone (CI excludes zero, negative) |
| hamilton_ppo_5seed_aggregate | belief_weighted | -8.311 | [-11.546, -5.077] | [-11.529, -5.142] | 0.352 | hamilton_ppo_5seed_aggregate falls below belief_weighted (CI excludes zero, negative) |
| hamilton_ppo_5seed_aggregate | oracle | -8.375 | [-11.601, -5.149] | [-11.579, -5.238] | 0.360 | hamilton_ppo_5seed_aggregate falls below oracle (CI excludes zero, negative) |
| hamilton_ppo_5seed_aggregate | frozen_clone | -8.343 | [-11.569, -5.118] | [-11.534, -5.179] | 0.352 | hamilton_ppo_5seed_aggregate falls below frozen_clone (CI excludes zero, negative) |

## Whether PPO matches, exceeds, or falls below each benchmark

- 5-seed aggregate vs. belief_weighted: **falls below** belief_weighted (95% CI [-11.55, -5.08] excludes zero, negative) -- this is a confidently NEGATIVE finding, not an absence of evidence.
- 5-seed aggregate vs. oracle: **falls below** oracle (95% CI [-11.60, -5.15] excludes zero, negative) -- this is a confidently NEGATIVE finding, not an absence of evidence.
- 5-seed aggregate vs. frozen_clone: **falls below** frozen_clone (95% CI [-11.57, -5.12] excludes zero, negative) -- this is a confidently NEGATIVE finding, not an absence of evidence.

### Interpretation: why does this reverse the validation-based picture?

On Phase 7's own 50-seed validation set, the SELECTED checkpoints looked competitive with (even nominally above) all three benchmarks -- see `phase7_groupB_1m_convergence_report.md` and `phase7_time_to_benchmark_summary.md`. On this fresh, never-inspected 500-path holdout, the SAME frozen checkpoints fall clearly and significantly below all three benchmarks for every one of the 5 seeds. The most likely explanation is checkpoint-selection optimism (a 'winner's curse' effect): each seed's checkpoint was chosen as the ARGMAX deterministic validation objective over 21 candidate checkpoints, each itself estimated from only 50 noisy episodes (checkpoint-level SE was approximately 7-9 objective-units, per `phase7_checkpoint_summary.csv`). Selecting the maximum of 21 noisy estimates systematically overestimates that checkpoint's TRUE mean performance, and a fresh, independent sample of paths (this holdout) is not subject to the same upward bias -- so performance reverts toward a lower, more representative level. This is precisely the failure mode an untouched, disjoint holdout evaluation is designed to detect, and this result should be read as the more trustworthy estimate of these checkpoints' true performance, not the validation-based numbers reported earlier in Phase 7.

No contrast in this analysis has a confidence interval including zero -- every PPO seed and the 5-seed aggregate are confidently below all three benchmarks on this holdout, at the level of statistical uncertainty this sample size supports.

## Runtime and tests

- Evaluation runtime: 1249.6s (20.8 min) for 500 paths x 8 policies (5 PPO seeds + 3 benchmarks) = 4000 episodes.
- Max reward-reconciliation error across all episodes: 2.558e-13 (confirms the environment/reward pipeline accounted for correctly).
- Full project test suite run before and after this evaluation -- see commit-time pytest output for pass/fail counts (this script does not itself invoke pytest).
