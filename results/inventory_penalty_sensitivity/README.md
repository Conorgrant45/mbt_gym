# Inventory-Penalty Sensitivity Experiment

Supervisor-requested controlled experiment testing whether increasing the
inventory penalties changes the **inventory behaviour** learned by the three
PPO architectures (Hamilton belief-state MLP, raw-return MLP, raw-return
LSTM), relative to the original calibration used throughout this project.

## Penalty calibrations

| Calibration | phi (running) | alpha (terminal) |
|---|---|---|
| `original` | 0.01 | 0.001 |
| `high_penalty` | 0.10 | 0.010 |

Both penalties are multiplied by exactly 10, preserving their relative
weighting. Every other environment parameter, PPO hyperparameter, network
architecture, training budget (1,000,000 observable-event transitions),
learner-seed count (5), training-environment seed (70,000), and the 500-path
holdout evaluation set are held **identical** to the original experiment
(`results/final_reduced_exploration_architecture_comparison/`) -- this
experiment changes phi/alpha and nothing else.

The original experiment's own checkpoints, logs, tables, and results are
**never modified** by this experiment -- everything here is new, additive
output under a separate directory tree
(`results/inventory_penalty_sensitivity/`, `models/inventory_penalty_sensitivity/`,
`logs/inventory_penalty_sensitivity/`).

## Reproduction commands

Run from the repository root (PowerShell or bash).

```bash
# 1. Validation tests (fast, no training)
python -m pytest tests/test_inventory_penalty_sensitivity.py -v

# 2. Train the high-penalty frozen supervised clone (~2 minutes)
python ips_train_clone.py

# 3. Train all 3 PPO architectures x 5 learner seeds x 1,000,000 transitions
#    under the high-penalty reward (long-running -- see "Compute budget"
#    below; resumable at (architecture, learner_seed) granularity, safe to
#    re-run/interrupt/resume)
python ips_run_training.py

# 4. Evaluate both calibrations on the same 500 holdout paths (re-evaluates
#    the ORIGINAL calibration's EXISTING checkpoints read-only, plus the
#    freshly-trained high-penalty checkpoints; resumable per (calibration,
#    policy))
python ips_evaluate_holdout.py --calibrations original
python ips_evaluate_holdout.py --calibrations high_penalty   # after step 3 finishes

# 5. Aggregate summary statistics and the paired original-vs-high-penalty
#    comparison table
python ips_aggregate_summary.py

# 6. Consolidated experiment manifest (config, seeds, checkpoint paths+hashes,
#    code version)
python ips_write_manifest.py

# 7. Paste-ready LaTeX (numerical tables only, no interpretation)
python ips_generate_latex.py
```

### Monitoring a running training batch

```powershell
Get-Content results/inventory_penalty_sensitivity/run_status.csv -Tail 10 -Wait
```
```bash
tail -f results/inventory_penalty_sensitivity/run_status.csv
```

## Compute budget

Training is the dominant cost. Per-seed wall-clock times observed in this
project (event-driven environment, 1,000,000 transitions, same machine):

| Architecture | Approx. time / seed | 5 seeds |
|---|---|---|
| `hamilton_ppo` | ~30-48 min | ~2.5-4 h |
| `return_mlp_ppo` | ~13-18 min | ~1.1-1.5 h |
| `return_lstm_ppo` | ~164-177 min | ~13.7-14.75 h |

Total: **roughly 17-20 hours**, dominated by `return_lstm_ppo`. This cannot
complete within a single interactive session; `ips_run_training.py` is
resumable at (architecture, learner_seed) granularity (matching the
established convention from Phases 6/7/the original experiment) -- if
interrupted, re-running the same command skips every run already verified
complete (final-checkpoint hash-matched against its manifest entry) and
resumes with the next incomplete one.

## Directory contents

- `training_config_manifest.json` -- high-penalty training configuration
  (architectures, seeds, PPO/RecurrentPPO hyperparameters, checkpoint
  semantics).
- `experiment_manifest.json` -- consolidated manifest (both calibrations):
  configuration, seeds, checkpoint paths + SHA-256 hashes, clone paths, code
  version, dependency versions, progress status. Written by
  `ips_write_manifest.py`, safe to re-run at any point.
- `run_status.csv` -- one row per completed (architecture, learner_seed)
  high-penalty training run: status, timestamps, duration, final transition
  count.
- `training_log_long.csv` -- per-PPO-update diagnostics (policy/value loss,
  approx KL, clip fraction, explained variance, action std, gradient norms,
  ...) for every high-penalty training run, identical schema to the original
  experiment's own `training_log_long.csv`.
- `clone_high_penalty_supervised_training.csv` /
  `clone_high_penalty_supervised_test_metrics.csv` /
  `clone_high_penalty_supervised_test_region_breakdown.csv` -- high-penalty
  frozen-clone supervised training curves, held-out test-set fit quality, and
  a breakdown by inventory/tau/belief region.
- `event_level/<calibration>/<policy_tag>.parquet` -- event-level rows (one
  per observable market-order arrival or terminal event), one file per
  (calibration, policy[, learner_seed]). Sufficient to reconstruct the
  complete piecewise-constant inventory path for every evaluated episode.
  Schema: `policy, architecture, penalty_calibration, learner_seed,
  holdout_episode_seed, event_index, event_time, elapsed_inter_event_time,
  inventory_before, inventory_after, bid_depth, ask_depth, fill_bid,
  fill_ask, latent_regime, filtered_regime_belief, wealth_change,
  running_penalty_contribution, terminal_penalty_contribution,
  realised_stage_reward`.
- `event_level/<calibration>/<policy_tag>_episode_summary.parquet` --
  per-holdout-path episode-level summary for that one policy (used to
  rebuild `episode_level.csv` without re-simulating).
- `episode_level.csv` -- combined episode-level dataset, one row per
  (calibration, policy, learner_seed, holdout_episode_seed): time-averaged
  signed/absolute inventory, integrated squared inventory, RMS inventory,
  max |inventory|, terminal inventory (signed/abs/squared), inventory-
  boundary-contact count, bid/ask/total fill counts, pre-penalty
  mark-to-market wealth, running/terminal penalty contributions, final
  realised objective, PLUS the original experiment's own (event-weighted)
  mean-inventory definitions retained under an explicit `*_event_weighted`
  suffix, PLUS every other field from the original experiment's episode
  schema (spread revenue, adverse-selection loss, fill-to-arrival ratio,
  action statistics, reward-reconciliation error, ...).
- `seed_summary.csv` -- per (calibration, policy, learner_seed) means over
  the 500 holdout paths.
- `policy_summary.csv` -- per (calibration, policy): cross-seed mean/SD/SE
  over the 5 learner seeds for the 3 PPO architectures (same aggregation
  convention as the original experiment's own
  `final_holdout_architecture_summary.csv` -- never a flat pool of
  2,500 path-level observations); single-evaluation summary for the 3
  reference policies (oracle, belief_weighted, frozen_clone).
- `comparison_original_vs_high_penalty.csv` -- PATH-LEVEL PAIRED (same 500
  holdout paths under both calibrations) high-penalty-minus-original
  differences for every inventory-behaviour / monetary-penalty /
  pre-penalty-wealth / penalised-objective metric, with normal-theory and
  bootstrap 95% CIs (reusing the original experiment's own paired-contrast
  procedure, `final_analyze_contrasts.bootstrap_ci_of_mean`, verbatim).
  `metric_group` distinguishes behavioural, monetary, wealth, and objective
  metrics explicitly -- a behavioural change is never inferred solely from a
  monetary-penalty change.
- `holdout_evaluation_manifest.json` -- provenance for the holdout
  evaluation pass (seed range, calibrations evaluated, row counts, max
  reward-reconciliation error).

## Model / checkpoint locations

- `models/inventory_penalty_sensitivity/<architecture>_seed<N>/` -- the
  high-penalty calibration's fresh checkpoints: `..._t000000000_initial.zip`
  (pre-training) and `..._t001000000.zip` (the ONLY checkpoint ever
  evaluated -- no periodic validation / offline checkpoint selection is
  performed for this experiment, per the brief).
- Original-calibration checkpoints are **reused read-only** from
  `models/final_reduced_exploration_architecture_comparison/` (`return_mlp_ppo`,
  `return_lstm_ppo`) and `models/phase7_groupB_1m_convergence/` (`hamilton_ppo`)
  -- never copied, moved, or retrained.
- `logs/inventory_penalty_sensitivity/supervised_clone_high_penalty_best.pt`
  -- the frozen high-penalty clone's weights (same `[64,64]` Tanh MLP
  architecture as the original clone at
  `logs/phase4_policy_diagnostic/supervised_clone_best.pt`, reused read-only
  for the original calibration).

## LaTeX output

`inventory_penalty_sensitivity.tex` (repository root) -- standalone, paste-
ready `\subsection{Inventory-Penalty Sensitivity}`. Numerical tables only
(no interpretation), generated purely from `policy_summary.csv` and
`comparison_original_vs_high_penalty.csv` by `ips_generate_latex.py`.
