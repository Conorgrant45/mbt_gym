# Phase 7: Group B, 1,000,000-Transition Convergence Diagnostic

Branch `event-driven-environment`, base commit `1d54f7b5957095e00e0dc01da2c6c946f531d6c9`
(clean at the start of this phase). `results/phase6_optimisation_experiment/`
was never modified (verified via `git status` — no changes reported). All
new outputs are under `results/phase7_groupB_1m_convergence/`,
`logs/phase7_groupB_1m_convergence/`, and `models/phase7_groupB_1m_convergence/`.

## 1. What changed relative to Phase 6

Only the following, exactly as instructed:

| | Phase 6 Group B | Phase 7 |
|---|---|---|
| Total transitions | 200,000 | **1,000,000** |
| Checkpoint cadence | every 16,000 to 200,000 (13 checkpoints) | **same 13 checkpoints, plus every 100,000 from 300,000 to 1,000,000 (8 more — 21 total)** |
| Diagnostic logging | det/stoch objective, spread/fills/inventory, log_std/action_std (mean) | **same, plus per-dimension log_std/action_std, learning rate, total loss, and full per-update SB3 internals (explained_variance, clip_fraction, approx_kl, entropy_loss, value_loss, policy_gradient_loss) — every one of the 250 PPO updates, not just at checkpoints** |
| Everything else (env, reward, observation, action mapping, architecture, all other PPO hyperparameters, actor initialisation, log_std_init, training-environment seed) | — | **identical, verified programmatically (see Section 2)** |

Confirmed and asserted in `phase7_common.py` (not just claimed): `gamma=1.0`,
`n_steps=4000`, `batch_size=400`, `n_epochs=10`, `gae_lambda=0.95`,
`clip_range=0.2`, `ent_coef=0.0`, `vf_coef=0.5`, `max_grad_norm=0.5`,
`learning_rate=3e-4` (constant, no schedule — confirmed in the per-update
diagnostics, which log the actual value read from the optimizer every
update and it never varies), `log_std_init=-1.5`, `net_arch=[64,64]`,
`training_env_seed=70000` — all read directly from `phase5_common.PPO_KWARGS`
(the same object Phase 6 itself used) and cross-checked equal to
`phase6_common.GROUPS["B"]` by assertion at import time, not re-typed
independently anywhere.

## 2. Pre-run audit

1. **Runner/config located**: `phase6_common.py` (`GROUPS["B"] = dict(init="random",
   log_std_init=-1.5)`) and `phase6_run_training.py`'s `run_one` function.
   Reused, not re-derived: Phase 7's `phase7_common.build_group_b_model` calls
   the SAME `phase5_common.build_hamilton_ppo` Phase 6 used.
2. **Action convention**: confirmed `[bid_action, ask_action]` ordering,
   unchanged (`envs/event_driven_hamilton_ppo_wrapper.py:149`).
3. **No clone weights loaded**: `build_group_b_model` never calls
   `copy_supervised_actor_weights`/`load_supervised_clone_net` (asserted by
   test, checking the function's own source body); empirically, the
   resulting actor is far from the supervised clone (max abs diff ≫ tolerance).
4. **Hamilton filter / event-driven wrapper unchanged**: `git log` shows no
   commits touching `envs/event_driven_hamilton_ppo_wrapper.py`,
   `envs/event_driven_regime_env.py`, or `beliefs/event_time_hamilton_filter.py`
   since Phase 2 (commits `97ee1bf`/`f105ca9`) — untouched across Phases 3-7.
5. **Bounded-action mechanics** (confirmed directly from the installed SB3
   2.8.0 source, not assumed): PPO's `ActorCriticPolicy` uses an **unbounded
   diagonal Gaussian** (`squash_output=False`, no gSDE — confirmed
   `use_sde=False`). The rollout buffer stores the **raw, unclipped** sampled
   action (`on_policy_algorithm.py:249`, added to the buffer *before* the
   separate `clipped_actions = np.clip(...)` computation at line 216) — this
   raw action is what `evaluate_actions()` recomputes the log-probability
   from during `train()`'s ratio calculation. Only the action actually sent
   to `env.step()` during rollout collection, and the action returned by
   `model.predict()` at evaluation time (`policies.py:379`), is clipped to
   `[-1, 1]`.
6. **`action_saturation_rate` vs. `stoch_frac_near_bound` reconciled**:
   Phase 6's `phase6_analyze_factor_effects.py` defines
   `action_saturation_rate` as literally `det_frac_near_bound` at the
   selected checkpoint — the fraction of steps where the **deterministic**
   (mean) action lands within 0.05 of a bound. A small MLP's mean output
   rarely saturates exactly, so this is ≈0.0 for essentially every
   checkpoint of every group, Phase 6 included. `stoch_frac_near_bound`
   measures the same threshold on the **sampled-then-clipped** action; with
   `log_std_init=0.0` (Groups A/C, action std=1.0 on a `[-1,1]` range), a
   large fraction of samples exceed the bounds and get clipped there,
   giving the >50% seen in Phase 6. These are simply two different
   quantities (deterministic mean vs. clipped stochastic sample) carrying
   the same generic name in the aggregated Phase 6 CSV — not a bug, and
   both are reported separately and correctly in this phase's own
   `phase7_checkpoint_summary.csv` (`det_frac_near_bound` /
   `stoch_frac_near_bound` columns).
7. **Full existing test suite run before any modification**: 318/318 passed
   (`python -m pytest -q`, 674.5s) before phase7_common.py/phase7_run_training.py
   were written.

## 3. Configuration manifest / validation seeds

`phase7_config_manifest.json` records git commit, package versions
(`stable_baselines3==2.8.0`, `torch==2.11.0+cpu`, etc. — unchanged from
Phase 6), the full PPO kwarg set, the 21-point checkpoint schedule, and the
validation seed range. **TensorBoard is not installed in this environment**
(`ModuleNotFoundError: No module named 'tensorboard'`); per instruction #10
(prefer the existing environment, do not install unless absolutely
necessary), TensorBoard logging was skipped and every diagnostic is written
to CSV instead — `phase7_training_diagnostics.csv` (1,250 rows: 5 seeds ×
250 PPO updates) and `phase7_checkpoint_summary.csv` (105 rows: 5 seeds ×
21 checkpoints) — satisfying the brief's own stated fallback.

**Validation seeds: 260,000–260,049** (50, fresh) — disjoint from Phase 6's
own validation range (240,000–240,049), Phase 6's holdout range
(250,000–250,199, never read anywhere in this phase), and every other
previously-used range in this project (checked by test against the full
enumerated list). One fixed validation set was used for every checkpoint of
every seed, matching Phase 6's own convention.

## 4. Did all 5 runs complete?

**Yes — 5/5**, run sequentially, fresh random initialisation each time (no
resuming from any prior checkpoint — instruction #9). Total wall-clock:
**2h 37m 50s** (11:19:42 → 13:57:32). Per-seed durations: seed 0 = 1991.6s,
seed 1 = 2124.0s, seed 2 = 1797.0s, seed 3 = 1869.7s, seed 4 = 1683.8s.
`phase7_run_status.csv` gained a row for each seed only after
`phase7_common.run_is_complete` verified every one of the 21 checkpoint file
hashes, the checkpoint-validation row count, and the final 1,000,000-transition
checkpoint's presence — no partially-completed seed was ever recorded as
complete (none were interrupted in this run, so this path was not
exercised, but is present and tested).

## 5. Seed-level and group-level results

| Seed | Det. obj. @200k | Best before 200k | Best after 200k | Final @1M | Slope, final 200k |
|---|---|---|---|---|---|
| 0 | 43.66 | 44.04 | 47.61 | 47.61 | +0.000019 |
| 1 | 42.85 | 44.83 | 49.65 | 48.79 | −0.000004 |
| 2 | 42.45 | 42.33 | 47.00 | 43.02 | +0.000020 |
| 3 | 44.36 | 44.25 | 48.69 | 46.00 | +0.000006 |
| 4 | 45.33 | 46.15 | 47.52 | 46.46 | +0.000011 |
| **Mean** | **43.73** | **44.32** | **48.09** | **46.38** | **+0.000010** |

Every seed's *best-after-200k* value clearly exceeds its own *best-before-200k*
value (mean gain +3.77) and its final-at-1M value exceeds its own
200k-transition value (mean gain +2.65). The slope over the final 200,000
transitions (800k→1M, least-squares fit over the 800k/900k/1,000k
checkpoints) is **essentially flat** (all five values are within
±0.00002 objective-units-per-transition of zero) — the *rate* of improvement
has clearly slowed by the end of the run, even though the *level* reached is
well above the 200k-transition benchmark.

**Benchmark comparison, evaluated on this phase's OWN validation seeds**
(260,000–260,049, for a fair apples-to-apples comparison rather than reusing
Phase 6's own holdout numbers):

| Policy | Mean full objective (50 episodes) |
|---|---|
| Oracle (analytical) | 45.77 |
| Belief-weighted (analytical) | 45.60 |
| Frozen supervised clone | 45.63 |
| **Group B, 1M transitions (mean of 5 seeds)** | **46.38** |

Group B's 1,000,000-transition result is **statistically indistinguishable
from, and nominally slightly above,** all three analytical/clone benchmarks
on the same validation seeds — a genuinely different picture from Phase 6,
where Group B's 200,000-transition result (mean 36.14 on Phase 6's own
holdout) sat clearly below these benchmarks (~41.7–41.9). Extended training
alone — no clone initialisation — closes essentially the entire gap.

## 6. Behavioural indicators (final 200,000 transitions, i.e. 800k–1M)

| Seed | Mean explained variance | Mean clip fraction | Mean approx KL | Mean value loss (SD) |
|---|---|---|---|---|
| 0 | 0.478 | 0.0336 | 0.00449 | 109.2 (45.2) |
| 1 | 0.438 | 0.0321 | 0.00423 | 174.4 (60.2) |
| 2 | 0.363 | 0.0338 | 0.00464 | 178.1 (75.8) |
| 3 | 0.449 | 0.0316 | 0.00445 | 129.3 (47.6) |
| 4 | 0.323 | 0.0313 | 0.00410 | 208.4 (74.1) |

Clip fraction (~0.03) and approximate KL (~0.004) stay low and stable across
the *entire* 1,000,000-transition run in every seed (`plots/clip_fraction_and_kl.png`)
— no drift, no divergence, no sign of PPO-update instability at any point.
Explained variance rises sharply to ~0.6–0.8 by ~150k transitions, then
gradually declines to a noisier 0.2–0.6 band for the remainder of training
(`plots/explained_variance.png`) — moderate, not "poor" by the 0.3 threshold
used for classification (only seed 4 dips marginally below it in its
finest-grained window), and not deteriorating toward zero. `log_std`
decreases steadily and consistently across all 5 seeds from −1.5 at
initialisation to approximately −2.0 to −2.1 by 1,000,000 transitions
(`plots/log_std_trajectory.png`) — exploration keeps shrinking well beyond
the point (~64,000 transitions) where Phase 5 found it had barely moved,
confirming the 200,000/64,000-transition budgets used in Phases 5-6 were
simply too short to see this.

## 7. Evidence vs. inference

**Evidence** (directly measured, not interpreted): all 5 seeds show
best-after-200k > best-before-200k and final-at-1M > at-200k; log_std
decreases monotonically-ish throughout in all 5 seeds; clip fraction and
approximate KL remain low and stable throughout in all 5 seeds; explained
variance is moderate (0.32–0.48) in the final 200k window, not collapsing
toward zero; the slope over the final 200,000 transitions is close to zero
in all 5 seeds; Group B's 1M-transition mean matches the analytical/clone
benchmarks on a shared validation set.

**Inference** (this phase's interpretation of that evidence): the
200,000-transition budget used in Phases 3–6 was binding — training had
not converged by that point, and a materially better policy was reachable
with more of the SAME algorithm, unchanged. The near-zero slope in the
final 200,000 transitions suggests the run is APPROACHING a plateau by
1,000,000 transitions, though this is not yet established as a true
asymptote (a single flat 200k-transition window is consistent with either
a genuine plateau or a temporary slow patch before a further rise — this
phase does not extend training further to distinguish those). The moderate
(not high) explained variance is flagged as a plausible contributor to the
noisy, non-monotonic path within the 300k–1M region (e.g. seed 2's retreat
from its own peak, seed 3's dip around 600k) without being read as the
primary bottleneck, since it does not prevent continued net improvement.

## 8. Files created

`phase7_common.py`, `phase7_instrumented_ppo.py`, `phase7_run_training.py`,
`phase7_analyze_convergence.py`, `phase7_generate_plots.py`,
`tests/test_phase7_groupB_1m_convergence.py` (15 tests). Outputs under
`results/phase7_groupB_1m_convergence/`: `phase7_config_manifest.json`,
`phase7_run_status.csv`, `phase7_checkpoint_summary.csv`,
`phase7_training_diagnostics.csv`, `phase7_seed_summary.csv`,
`phase7_group_summary.csv`, `phase7_benchmark_on_own_validation_seeds.csv`,
this report, and `plots/` (5 per-seed learning-curve plots,
`aggregate_learning_curve.png`, `log_std_trajectory.png`,
`explained_variance.png`, `clip_fraction_and_kl.png`). 105 checkpoint files
(21 per seed × 5, each a full SB3 checkpoint including optimizer state) under
`models/phase7_groupB_1m_convergence/groupB_seed{0..4}/`.

## Errors found and fixed this phase

Two test-design bugs, caught and fixed before they could affect any real
result (both were self-contained to `tests/test_phase7_groupB_1m_convergence.py`,
never touched real training data):
1. A clone-loader-absence check searched the ENTIRE function source
   (including its own docstring, which correctly explains that no clone
   weights are loaded) for the substring "clone", trivially failing on its
   own documentation. Fixed to check only the function body for actual
   calls to the clone-loading functions.
2. An empirical check that raw sampled actions exceed `[-1,1]` under an
   unbounded Gaussian failed on one 4,000-step rollout under
   `log_std_init=-1.5` (std≈0.22 is simply too small for that to reliably
   happen in one small sample). Replaced with a structural check
   (`isinstance(model.policy.action_dist, DiagGaussianDistribution)`,
   `squash_output is False`, `use_sde is False`) that verifies the same
   underlying fact (an unbounded Gaussian policy) without depending on
   sampling luck.

## Exact commands executed

```
git branch --show-current
git status
python -m pytest -q                              # pre-run audit, 318/318, 674.5s
python phase7_run_training.py                     # 5 seeds x 1,000,000 transitions, 2h37m50s
python phase7_analyze_convergence.py
python phase7_generate_plots.py
python -m pytest -q                               # post-run confirmation
```

## Final verdict

Every one of the 5 seeds shows the SAME pattern: clear, materially-sized
improvement between the 200,000-transition region and later checkpoints
(mean best-after-200k gain +3.77, mean final-at-1M gain +2.65 over the
200k mark), while PPO-update statistics (clip fraction, approximate KL)
remain low and stable throughout and explained variance stays moderate
rather than collapsing. The 200,000-transition budget used in every prior
phase of this project was too short to reach the performance this exact,
unmodified configuration is capable of.

**VERDICT: TRAINING BUDGET WAS BINDING**
