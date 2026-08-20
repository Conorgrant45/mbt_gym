# Data inventory: `final_reduced_exploration_architecture_comparison`

What this experiment is: three PPO architectures (`hamilton_ppo` belief-state
MLP, `return_mlp_ppo` raw-return MLP, `return_lstm_ppo` raw-return LSTM),
5 learner seeds each (0-4), log_std_init=-1.5, 1,000,000 transitions,
event-driven environment only. Hamilton's official results reuse Phase 7's
already-trained checkpoints read-only (re-evaluated on this experiment's own
validation/holdout seeds); `return_mlp_ppo`/`return_lstm_ppo` were trained
fresh here. Checkpoint cadence: 21 timesteps — every 16,000 up to 200,000,
then every 100,000 up to 1,000,000. Full config: `experiment_manifest.md`/
`.json`.

Two seed ranges, disjoint from every prior phase and from each other:
**validation** = 280000-280049 (50 seeds, used at every one of the 21
checkpoints throughout training/selection) and **holdout** = 290000-290499
(500 seeds, touched only at the two pre-registered fixed checkpoints
200,000 and 1,000,000 — never used for checkpoint selection).

Everything below is under
`results/final_reduced_exploration_architecture_comparison/` unless stated
otherwise.

---

## 1. Model artifacts (`models/final_reduced_exploration_architecture_comparison/`)

- `{architecture}_seed{0-4}/` — one dir per (architecture, seed). Contains
  the initial untrained checkpoint (`..._t000000000_initial.zip`) plus all
  21 trained checkpoints (`..._t{timestep:09d}.zip`) and `run_manifest.json`.
  **`hamilton_ppo_seed{N}/` dirs are empty** — Hamilton's checkpoints are
  reused read-only from Phase 7 (`results/phase7_groupB_1m_convergence/`),
  never copied here; `checkpoint_manifest.csv`'s `path`/`source` columns
  point at the originals.
- `hamilton_diagnostics_only/` — **NOT part of the official experiment.**
  **STATUS: COMPLETE** (v2, all 5 seeds, `final_train_hamilton_diagnostics.py`
  using the same interleaved checkpoint+eval loop `final_run_training.py`
  uses officially for MLP/LSTM). Contains all 21 checkpoints per seed
  (`hamilton_diag_seed{N}_t{timestep:09d}.zip`) plus `seed{N}_done.json`
  markers. Its final validation objective is ~14-25 points below Phase 7's
  original Hamilton run for every seed — LARGER than v1's gap, not smaller.
  Root-caused precisely (not just empirically-observed): Phase 7's original
  eval routine (`phase6_common.evaluate_checkpoint_model`) calls
  `torch.manual_seed()` **once per checkpoint** (21 times/run); this
  experiment's own official eval routine
  (`final_run_training.evaluate_checkpoint_and_log`, what v2 correctly
  replicates to match MLP/LSTM's protocol) calls it **once per validation
  path per checkpoint** (21×50=1,050 times/run). Since each call resets
  PyTorch's global RNG (which PPO's rollout collection also draws from),
  v2's far higher perturbation count makes it diverge from Phase 7's
  original MORE than v1's zero-perturbation run did — this is expected, not
  a bug. **v2 is protocol-consistent with how MLP/LSTM's diagnostics were
  collected within this experiment** (identical eval seeds, identical
  interleaving cadence) and is the right choice for the training-dynamics
  comparison (figA2) on that basis — it was never going to reproduce Phase
  7's specific numbers, because Phase 7 used a different, lighter-touch eval
  protocol that no longer exists anywhere else in this codebase.
- `hamilton_diagnostics_only_bak_non_interleaved_protocol/` — the **v1**
  diagnostics-only run (single uninterrupted `model.learn()` call per seed,
  no interleaved eval at all, 5 final checkpoints only), preserved for
  reference/comparison only. **Superseded by v2 above; do not use
  this v1 run for anything** — kept only as a record of the finding.

## 2. Configuration / manifests

| File | What it is |
|---|---|
| `experiment_manifest.json` / `.md` | Full config: architectures, seeds, PPO/RecurrentPPO hyperparameters, net arch, checkpoint cadence, seed ranges + disjointness proof, Hamilton reuse-compatibility check. Human-readable `.md` version. |
| `final_holdout_evaluation_manifest.json` | Provenance for the holdout evaluation pass (which checkpoints/seeds were evaluated, when). |
| `run_status.csv` | 15 rows (3 architectures × 5 seeds). Per-run status, start/finish time, wall-clock duration, final transitions reached. Source for the wall-clock cost figures (e.g. LSTM ~10,000s/seed vs MLP ~800s/seed vs Hamilton reuse+re-eval ~700-900s/seed). |
| `checkpoint_manifest.csv` | 315 rows (3 arch × 5 seeds × 21 checkpoints). Path, SHA-256 hash, and provenance (`trained_this_experiment` vs `phase7_reused`) for every checkpoint file. |

## 3. Per-update training diagnostics (fine-grained, ~every 4,000 transitions)

| File | Rows | Covers |
|---|---|---|
| `training_log_long.csv` | 2,500 (2 arch × 5 seeds × 250 updates) | `return_mlp_ppo`, `return_lstm_ppo` **only** — official runs. |
| `hamilton_training_log_long.csv` | 1,250 (5 seeds × 250 updates) | `hamilton_ppo`, v2 (interleaved protocol) — **COMPLETE.** Same column schema/meaning as `training_log_long.csv` and directly comparable to it (unlike v1) — used by `results/thesis_final_plots/figA2_ppo_diagnostics.py`, which auto-detects completeness and includes it with a footnote. |
| `hamilton_training_log_long.csv.bak_non_interleaved_protocol` | 1,250 (5 seeds × 250 updates) | `hamilton_ppo` v1 — **do not use, see §1.** Kept only for reference. |
| `hamilton_diagnostics_only_validation_path_level.csv` | 5,250 (5 seeds × 21 checkpoints × 50 paths) | `hamilton_ppo` v2 only. Per-(seed, checkpoint, validation path) episode rows from the interleaved eval loop — same 39-col schema and same 50-seed validation set as the official `validation_path_level.csv`, but for this diagnostics-only run. Not present for v1. |
| `hamilton_training_diagnostics_sanity_check.csv` | 5 (one per seed) | v2's check of this run's final deterministic validation mean vs. Phase 7's original, per seed — gap is -14 to -25 points, LARGER than v1's -11 to -15 (see §1: this is expected, not a regression — v2 undergoes far more RNG-perturbing eval calls than Phase 7's original protocol ever did). |
| `hamilton_training_diagnostics_sanity_check.csv.bak_non_interleaved_protocol` | 5 | v1's version of the same check. |

Both training logs have the same 29 columns per PPO update: `n_updates`,
`num_timesteps`, `policy_loss`, `value_loss`, `entropy_loss`, `approx_kl`,
`clip_fraction`, `explained_variance`, `log_std_mean` (= mean **action std**,
i.e. `exp(log_std)`, despite the name — see `final_instrumented_ppo.py`),
`advantage_mean/std/min/max`, `return_mean/std` (rollout-buffer GAE-bootstrapped
return, **not** per-episode reward), `actor_grad_norm_mean`,
`critic_grad_norm_mean`, `actor_param_update_norm`, `critic_param_update_norm`,
`frac_sampled_action_clipped`, `frac_sampled_action_near_bound`,
`log_std_bid/ask`, `action_std_bid/ask`, `learning_rate`, `total_loss`.

**Known-fixed data-integrity bug**: `training_log_long.csv`'s `return_lstm_ppo`
rows originally had columns 10-29 silently mislabeled (a column-order mismatch
when appending to the CSV — see `scripts/fix_training_log_lstm_column_bug.py`
for the full root-cause and the exact repair mapping). This has been repaired
in place; the pre-fix original is preserved at
`training_log_long.csv.bak_before_lstm_column_fix`. `final_run_training.py`
was also patched to reindex-by-column-name before any future append, so this
can't recur.

Plotting scripts (read-only, in `scripts/`): `plot_final_training_return_curve.py`
(return_mean vs. transitions, MLP+LSTM only, separate y-axis per architecture),
`plot_final_training_diagnostics.py` (4-panel optimisation diagnostics —
explained variance, log_std_mean, approx_kl, actor grad norm — MLP+LSTM only).

## 4. Validation-set data (fixed 50-seed set, all 21 checkpoints, every architecture)

Used for checkpoint monitoring throughout training — **never** the basis for
the final reported numbers (those come from the holdout set, §5).

| File | Grain |
|---|---|
| `validation_path_level.csv` (16,341 rows) | One row per (architecture, seed, checkpoint, validation path). Full per-episode metric set (39 cols: objective decomposition, fills, spread/inventory/action stats, `deterministic` flag, stochastic companion-pass fields) + `wall_clock_elapsed_seconds`. |
| `validation_seed_summary.csv` (315 rows) | One row per (architecture, seed, checkpoint) — means/SD/SE over the 50 validation paths. |
| `validation_architecture_summary.csv` (63 rows) | One row per (architecture, checkpoint) — cross-seed means over the 5 seeds. **Source for the combined learning-curve plots** (`plots/final_learning_curve_*.png`). |
| `validation_benchmark_episode_level.csv` / `validation_benchmark_summary.csv` | Same, for the 3 analytical benchmarks (oracle, belief_weighted, frozen_clone) — no seed/checkpoint dimension, evaluated once each on the same 50 validation paths. |

## 5. Final holdout data (fixed 500-seed set, ONLY the two pre-registered checkpoints 200k/1M)

This is the basis for the actual reported results.

| File | Grain |
|---|---|
| `final_holdout_episode_level.csv` (16,500 rows) | One row per (policy, seed, checkpoint ∈ {200k,1M}, holdout path). Same 39-col schema as the validation episode-level file. Includes the 3 analytical benchmarks (`checkpoint_transition` = NaN for those). **This is the file the ECDF, scatter, and outlier-path analyses in this conversation were built from.** |
| `final_holdout_seed_summary.csv` (33 rows) | Per (policy, seed, checkpoint) means over 500 paths. Source for `01_training_budget_slope.py` and `06_behavioral_metrics_budget_slope.py`. |
| `final_holdout_architecture_summary.csv` (6 rows) | Per (architecture, checkpoint) cross-seed means. |

## 6. Pre-computed statistical contrasts (all read-only, paired-by-path or paired-by-seed — never naive pooling across non-independent observations)

| File | Rows | Comparison |
|---|---|---|
| `fixed_200k_vs_1m_contrasts.csv` | 120 (3 arch × 5 seeds × 8 metrics) | Paired (same 500 holdout paths) 200k→1M change per metric, with normal + bootstrap CIs and win-rate. **The source for the "does the training budget beyond 200k help?" discussion** — full_objective: CI excludes zero for 0-20% of seeds; every behavioral metric: 100%. |
| `benchmark_paired_contrasts.csv` | 45 (3 arch × 5 seeds × 3 benchmarks) | Paired RL-minus-benchmark objective difference at 1M, same holdout paths. Source for `02_paired_benchmark_forest.py`. |
| `architecture_contrasts.csv` | 39 (architecture-pair × 13 metrics) | Matched-seed architecture-vs-architecture differences at 1M, with per-seed diff list retained. |

## 7. Behavioral / action diagnostics

| File | Grain |
|---|---|
| `action_diagnostics.csv` (348 rows) | Per (policy, seed, checkpoint): mean bid/ask action, action std, near-bound rates (deterministic + stochastic), stochastic full_objective. All 21 checkpoints, all policies including benchmarks. |
| `trajectory_subset.csv` (5,648 rows) | **Single-path deep-dive only**: seed 0, checkpoint 1,000,000, one holdout path, step-by-step event log (time, event type, arrival side, hidden regime, Hamilton belief, inventory, quoted depths, fills, cash, mark-to-market wealth, reward decomposition, recurrent hidden-state norm for LSTM). For 4 policies (hamilton_ppo, return_lstm_ppo, return_mlp_ppo, belief_weighted) — not oracle/frozen_clone. Used for `05_inventory_time_profile.py`. |

## 8. Generated figures

- `plots/` (in this results dir) — ad-hoc PNG+PDF pairs from `scripts/plot_final_architecture_results.py`, `scripts/plot_final_training_return_curve.py`, `scripts/plot_final_training_diagnostics.py`: architecture comparison, behavioural decomposition, benchmark paired differences, fixed 200k-vs-1M comparison, per-architecture + combined learning curves, training-return curve, training diagnostics.
- `results/thesis_final_plots/` — the numbered, thesis-ready figure series (`01_training_budget_slope.py` … `06_behavioral_metrics_budget_slope.py`), each self-contained with its own `CONFIGURATION` block; see that directory's own `README.md` for a script-by-script description. Outputs in `results/thesis_final_plots/figures/`.

## 9. Scripts that produced this data (all at repo root unless noted)

`final_common.py` (shared config/paths/model-building), `final_run_training.py`
(orchestrator: trains MLP/LSTM, reuses+re-evaluates Hamilton), `final_instrumented_ppo.py`
(diagnostic PPO/RecurrentPPO subclasses), `final_episode_runner.py` (episode
simulation + metric decomposition, shared by validation/holdout/benchmark
evaluation), `final_evaluate_holdout.py`, `final_evaluate_validation_benchmarks.py`,
`final_aggregate_validation.py`, `final_aggregate_holdout.py`,
`final_build_action_diagnostics.py`, `final_analyze_contrasts.py`
(all the contrast CSVs in §6), `scripts/generate_final_architecture_tables.py`,
`final_train_hamilton_diagnostics.py` (the diagnostics-only Hamilton side run,
§1/§3 caveat).
