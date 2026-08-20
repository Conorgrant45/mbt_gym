# Phase 4: Hamilton PPO Policy Diagnostic

Branch `event-driven-environment`, base commit `e37271deb3f2fa6f88fd6fccfabc305e60420049`
(clean at the start of this phase). This phase changed **no** neural architecture,
PPO hyperparameter, reward, action space, economic model, or Hamilton filter, and
launched **no** new PPO training run. All findings below come from (a) a static
diagnostic grid evaluated against already-trained, already-verified checkpoints,
(b) fresh rollouts of those same frozen checkpoints, and (c) one new supervised
(non-RL) regression model trained purely to test architecture capacity.

Full test suite: **297/297 passing** (283 pre-existing + 14 new Phase 4 tests in
`tests/test_phase4_policy_diagnostic.py`).

## 0. Scope and inputs

Read before starting: `rl_architecture_audit_report.md`,
`event_driven_agent_integration_phase2_report.md`, `event_driven_phase3_results.md`.
Phase 3 concluded RESULTS INCONCLUSIVE on why Hamilton PPO remains below the
analytical belief-weighted benchmark; this phase investigates why.

## 1. Model provenance

All 10 Hamilton PPO checkpoints (`offlinecv_env70000_learner_{0..4}` fixed-step,
`phase3_event_env70000_learner_{0..4}` event-driven) were re-verified directly from
their own `run_config`/`offline_selection`/`checkpoint_manifest` JSON, not assumed:
environment_type, learner_seed, 200,000 training transitions, `[64,64]` Tanh MLP
architecture, observation shape `(3,)`, action shape `(2,)`, and a genuine
(non-zero, manifest-listed) selected checkpoint timestep — all confirmed
(`phase4_common.verify_and_load_hamilton_models`). Every model loaded is an
`_offline_best.zip`, never a final-timestep checkpoint.

## 2-3. Action-surface diagnostic (9,471-state grid; Section 3 metrics)

Grid: q ∈ {-40,...,40} step 2 (41), τ ∈ {0,...,1.0} step 0.1 (11), belief ∈
{0,...,1.0} step 0.05 (21) — extended slightly beyond the task's `{-30,...,30}`
example after checking Phase 3 holdout inventory reached ±46 (fixed) / ±26 (event).
Exact production observation `[tanh(q/10), τ, belief]` and action transform
`(a+1)/2·MAX_DEPTH`, applied exactly once (verified by test item 2).

**Every one of the 10 checkpoints quotes wider than analytical at every grid
state** (`mean_quoted_spread_error_signed` is positive for all 10, range
1.06–1.61). Several show a **negative correlation** with the analytical ask-depth
surface (fixed_seed3: both bid and ask negative; event_seed0: both negative
≈ -0.51) — this is a structural surface mismatch, not noisy imitation of the right
shape. `frac_wrong_inventory_skew` ranges from 1.7% (event_seed1, good) to 99.1%
(event_seed0 — nearly always skews the wrong way). No checkpoint saturates the
action bounds on this grid (`frac_near_action_bound = 0.0` throughout). See
`results/phase4_policy_diagnostic/phase4_action_surface_metrics.csv` and
`plots/skew_heatmap_tau{0.1,0.5}.png`, `plots/depth_vs_{inventory,tau,belief}_*.png`.

## 4. Fill-selectivity diagnostic

Converting the SAME grid's quote depths to expected fill probabilities
(`p = exp(-κ·depth)`) isolates "wider quotes" from "different state visitation":
at identical states, learned policies achieve only **28–58%** of analytical's
expected fill probability, directly attributable to spreads **1.57–1.88×** wider
than analytical (not to visiting different, less fill-friendly states — those
states are identical here by construction). See `phase4_fill_selectivity.csv`.

## 5. State-visitation diagnostic (fresh seeds 210000–210049)

Per-step instrumented rollouts of all frozen checkpoints (all 3 architectures) on
seeds never used for training, validation, or the Phase 3 holdout, using
incremental (O(1)-memory) statistics only. Realised fill rates for RL agents
(≈0.005–0.027/step fixed, ≈0.06–0.12/step event) are consistently below
analytical (0.0155 fixed, 0.236 event) — confirming Section 4's grid-level finding
holds at actually-visited states too. This diagnostic independently reproduced,
via a completely different methodology, the two catastrophic-collapse seeds
already known from Phase 3 (`return_lstm_ppo` fixed seed0: mean inventory +40.4,
std 23.6; `return_mlp_ppo` fixed seed0: mean inventory +26.5, std 18.5) — strong
cross-validation. One bug was found and fixed during this diagnostic: the
fixed-step analytical fill-detection initially compared inventory reconstructed
from a normalised observation against raw inventory, producing spurious
near-100%-of-steps "fills" from floating-point round-trip noise; fixed to compare
two raw readings directly (see `phase4_state_visitation_diagnostic.py`). See
`phase4_state_visitation.csv`.

## 6. Supervised representability test

The **exact** production architecture ([64,64] Tanh MLP, 2D output) was trained by
plain supervised regression to imitate the analytical belief-weighted policy's
own action surface (input `[tanh(q/10), τ, belief]`, target = analytical actions in
the normalised `[-1,1]` action space), using data collected from the event-driven
environment (100/30/40 train/val/test episodes, seeds 220000/221000/222000+,
3 independent init seeds). Best model selected by **validation MSE only**:

| Metric | Value |
|---|---|
| Test MSE | 2.80×10⁻⁶ |
| Bid / ask test MSE | 2.70×10⁻⁶ / 2.90×10⁻⁶ |
| Max abs. test error | 0.0142 |
| Fraction of test outputs near ±1 bound | 0.0 |
| Save/reload prediction diff | 0.0 (exact) |

This is an extremely strong fit with no saturation — the architecture has ample
capacity to represent the analytical policy. See `phase4_supervised_test_metrics.csv`,
`phase4_supervised_training.csv`, `phase4_supervised_test_region_breakdown.csv`.

## 7. Evaluating the frozen clone as a policy (fresh seeds 225001–225050)

This is the decisive check: low imitation error does not by itself imply good
control performance (a pipeline/timing mismatch could still exist). The frozen
clone — never re-tuned after Section 6 — was evaluated through the identical
Hamilton observation/belief-filter/action-transformation pipeline as real
Hamilton PPO (duck-typed `.predict()` shim, so `evaluate_agents_common`/
`evaluate_agents_event_driven`'s already-validated runners are reused unmodified),
on 50 fresh seeds in both environments:

| Policy | Env | Mean J (se) | Fills | Spread | Mean \|Q\| |
|---|---|---|---|---|---|
| Analytical belief-weighted | fixed | 48.99 (4.75) | 60.8 | 2.06 | 2.91 |
| **Supervised clone** | fixed | **48.62 (4.76)** | **60.8** | 3.06 | 2.91 |
| Hamilton PPO (5 seeds) | fixed | 33.4–43.9 | 18.2–35.4 | 2.86–4.59 | 2.4–16.0 |
| Analytical belief-weighted | event | 35.09 (6.41) | 64.1 | 2.05 | 3.31 |
| **Supervised clone** | event | **35.51 (6.20)** | **64.1** | 2.05 | 3.28 |
| Hamilton PPO (5 seeds) | event | 28.9–33.5 | 23.1–30.2 | 3.00–3.45 | 2.1–3.8 |

The clone matches analytical's full objective to within 1 standard error in
**both** environments and reproduces its fill count essentially exactly, while
every real (PPO-trained) checkpoint underperforms both by a wide margin. Maximum
reward-reconciliation error across all 700 evaluation episodes: 1.1×10⁻¹¹
(confirms the shared pipeline is computing consistently). See
`phase4_policy_evaluation.csv` / `phase4_policy_evaluation_summary.csv`.

## 9. Architecture-independent diagnostics, all 3 RL agents

The analytical-imitation test (Sections 6–7) is specific to Hamilton PPO's
belief-state input and was **not** applied to `return_mlp_ppo`/`return_lstm_ppo`.
Instead, spread/fill/inventory/saturation diagnostics were computed for all three
architectures from already-existing data (Phase 3's 1000-episode-per-cell holdout
plus this phase's fresh-seed state-visitation instrumentation):

| Agent | Env | Mean J | Fills | Spread | Mean Q (signed) |
|---|---|---|---|---|---|
| Hamilton PPO | fixed | 31.25 | 29.7 | 3.45 | +4.50 |
| Return MLP PPO | fixed | 29.44 | 35.6 | 3.42 | +6.33 |
| Return LSTM PPO | fixed | 24.75 | 48.6 | 3.52 | +11.10 |
| Hamilton PPO | event | 34.25 | 26.6 | 3.22 | -0.51 |
| Return MLP PPO | event | 28.92 | 23.1 | 3.48 | +1.01 |
| Return LSTM PPO | event | 31.27 | 23.6 | 3.37 | -0.11 |
| Analytical | fixed | 41.98 | 63.8 | 2.06 | -0.99 |
| Analytical | event | 45.04 | 65.2 | 2.03 | +0.05 |

All three architectures share the same qualitative failure: spreads ~1.6–1.7×
analytical, far fewer fills, and a positive/net-long inventory bias that is worst
in the fixed-step environment (all 3 architectures) — this is not unique to the
belief-state input, arguing against a Hamilton-specific representation problem.
Episode-mean action saturation is 0.0 in every cell. See
`phase4_section9_holdout_behavioral_summary.csv`,
`phase4_section9_visitation_behavioral_summary.csv`.

## 10. Required tests

14 new tests in `tests/test_phase4_policy_diagnostic.py` cover all 12 required
items (grid/production observation parity, single action transform, bid/ask
ordering, physical-depth fill probabilities, normalised supervised targets,
save/reload equality, belief-timing-preserving reconciliation for the clone,
pure/matched grid functions, disjoint fresh-seed ranges, evaluation-order
invariance, O(1) incremental-stats memory). All 14 pass; full suite is 297/297.

## 11. Outputs

All required files present under `results/phase4_policy_diagnostic/`:
`phase4_action_surface_metrics.csv`, `phase4_action_surface_grid.csv`,
`phase4_state_visitation.csv`, `phase4_fill_selectivity.csv`,
`phase4_supervised_training.csv`, `phase4_supervised_test_metrics.csv`,
`phase4_policy_evaluation.csv`, `phase4_diagnostic_tables.tex`, this report, plus
plots under `results/phase4_policy_diagnostic/plots/` and two supplementary
Section 9 summary CSVs.

## 12. Interpretation

- **Do the learned Hamilton policies match the analytical action surface?** No —
  wider quotes everywhere, several with negative surface correlation, and
  inventory-skew errors up to 99% of grid states on the worst checkpoint.
- **Can the MLP represent the analytical policy?** Yes, essentially exactly
  (test MSE 2.8×10⁻⁶, max error 0.014, no saturation).
- **Does a clone with that fit approach analytical control performance?** Yes —
  within 1 SE of analytical full objective and fill count in both environments.
- **Why do learned policies generate fewer fills?** Wider quotes at matched
  states (spread ratio 1.57–1.88×), not different, less-fillable state
  visitation — confirmed independently at the grid level (Section 4) and at
  actually-visited states (Section 5).
- **Quote levels vs. state visitation?** Quote levels are the primary driver;
  state visitation differences (net-long drift) are a downstream *consequence*
  of the wider/skewed quotes, not an independent cause.
- **Do fixed-step and event-driven Hamilton PPO fail the same way?** Yes —
  both show universally wider spreads, majority-negative-or-mixed surface
  correlation on the worst seeds, and reduced fill rates; neither environment
  type shows a qualitatively different failure mode.
- **Do all 3 RL agents share the same behavioral failure?** Yes (Section 9):
  wider spreads, fewer fills, net-long inventory bias worst in fixed-step, no
  saturation — common to Hamilton PPO, return_mlp_ppo, and return_lstm_ppo alike.
- **Next justified step:** PPO optimisation changes (e.g., entropy/learning-rate
  schedule, longer training, exploration tuning) or training-procedure fixes —
  not an architecture change, not a reward/action-space change, and not a
  Hamilton-filter change. The representation is sufficient; the gap is in how
  PPO is finding (or failing to find) that representation's optimum.

**REPRESENTATION SUFFICIENT — PPO OPTIMISATION IS THE PRIMARY BOTTLENECK**
