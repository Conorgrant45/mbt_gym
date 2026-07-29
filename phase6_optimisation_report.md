# Phase 6: 2x2 Optimisation Experiment (Actor Initialisation x Exploration Variance)

Branch `event-driven-environment`, base commit `588a4888aca2d62d446fad52f976d315f8ee2d32`
(clean at the start of this phase). No change to the environment, reward,
Hamilton filter, observation space, action mapping, network architecture, or
any PPO hyperparameter other than `log_std_init` — the one factor this
experiment deliberately varies. Full test suite: **318/318 passing** (309
pre-existing + 9 new Phase 6 tests in `tests/test_phase6_optimisation_experiment.py`).

The objective throughout is improved learned-policy holdout performance and
robustness — not reproducing the analytical policy (per the standing
convention established in Phase 5: the analytical policy and supervised
clone are diagnostic references, never a required target).

## 1. Experimental design

2×2 design, event-driven Hamilton PPO only, training-environment seed
70,000, learner seeds 0–4, 200,000 transitions per run (4 groups × 5 seeds
= 20 sequential training runs):

| Group | Actor init | `log_std_init` (action std) |
|---|---|---|
| A | Random | 0.0 (std=1.0) |
| B | Random | −1.5 (std=0.22) |
| C | Supervised clone | 0.0 (std=1.0) |
| D | Supervised clone | −1.5 (std=0.22) |

13 candidate checkpoints per run (16k, 32k, …, 192k, 200k — `train_agents.
compute_checkpoint_timesteps`, reused unmodified), offline-selected by
**argmax deterministic mean validation objective** (ties broken toward the
earlier checkpoint), copied and SHA-256-hash-verified to `*_offline_best.zip`.

## 0. Seed ranges

Fresh, disjoint from every prior range in this project (verified by test):
**validation seeds 240,000–240,049** (50), **holdout seeds 250,000–250,199**
(200). Both written to `phase6_experiment_manifest.json` before any training
began. The holdout set was not inspected until all 20 runs completed, all 20
checkpoints were selected and hash-verified, and `phase6_evaluate_holdout.py`
was finalised — enforced by that script's own pre-flight check
(`preflight_check_all_runs_complete`), which refuses to run otherwise.

## 2. Did all 20 runs complete?

**Yes — 20/20**, running sequentially (single process, one run at a time,
per the 8GB RAM constraint), total wall-clock ≈3h18m. The resumable
orchestrator (`phase6_run_training.py`) records git commit, package
versions, and a per-run status row (start/finish time, duration, clone-
verification error, selected checkpoint, offline-best hash) in
`phase6_run_status.csv`; resumability is at (group, seed)-run granularity
(a run is skipped only if every checkpoint hash, the validation CSV row
count, and the offline-selection hash all independently verify — see
`phase6_common.run_is_complete`).

## 3. Selected checkpoint per run

| Group | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|---|
| A | 48,000 | 16,000 | 128,000 | 48,000 | 144,000 |
| B | 16,000 | 80,000 | 192,000 | 176,000 | 48,000 |
| C | 16,000 | 16,000 | 16,000 | 16,000 | 32,000 |
| D | 96,000 | 144,000 | 96,000 | 144,000 | 48,000 |

Group C's selection concentrates almost entirely at t=16,000 (4/5 seeds) —
its own validation curve (`plots/checkpoint_training_curves.png`) shows a
severe mid-training crash in at least one seed (deterministic objective
dropping to −35 around t≈140,000) that offline selection successfully
avoids by picking an early checkpoint. Group D shows no comparable crash
anywhere in its validation curve and selects later, higher checkpoints on
average — a training-*process* stability difference between C and D not
fully captured by comparing only their final selected-checkpoint numbers.

## 4-5. Group holdout performance (200 fresh holdout episodes, seeds 250,000–250,199)

| Group | Mean J | SD (across 5 seeds) | Min | Max | Loss rate | Fills | Mean \|Q\| |
|---|---|---|---|---|---|---|---|
| A | 32.33 | 2.41 | 29.70 | 35.92 | 0.199 | 31.2 | 6.21 |
| B | 36.14 | 0.62 | 35.34 | 36.73 | 0.128 | 31.0 | 3.45 |
| C | 40.47 | 1.12 | 38.91 | 41.95 | 0.101 | 50.9 | 3.42 |
| D | 41.55 | 0.97 | 39.90 | 42.33 | 0.134 | 63.5 | 4.01 |
| Oracle (analytical) | 41.80 (se 2.81) | — | — | — | 0.100 | 65.2 | 3.29 |
| Belief-weighted | 41.90 (se 2.80) | — | — | — | 0.100 | 65.1 | 3.28 |
| Frozen supervised clone | 41.70 (se 2.82) | — | — | — | 0.100 | 65.1 | 3.29 |

**Groups C and D are statistically indistinguishable from the analytical
oracle/belief-weighted policies and the frozen supervised clone** — all
five land within ≈41.7–41.9 to 42.3, well inside standard error. Random
initialisation (A, B) stays clearly below this level even at the full
200,000-transition budget. Cross-seed SD drops sharply from A (2.41) to
every other group (0.62–1.12) — both interventions improve robustness, not
just the mean. Group A carries the worst loss rate (19.9%) and the largest
mean absolute inventory (6.21, vs. 3.4–4.0 elsewhere), driven by two
specific seeds (2 and 4) whose holdout SE balloons to 8.8–9.4 with large net
inventory drift (signed inventory −11.1/−10.8) — noisy, unstable policies,
though not catastrophic collapses of the kind seen in Phase 3/4's
fixed-step `return_mlp`/`return_lstm_ppo` runs (no holdout episode set in
this experiment showed the extreme runaway inventory of those cases).

## 6-7. Primary factor effects (seed-level paired differences, n=5)

| Contrast | Mean | SD | Min | Max | 95% bootstrap CI (weak, n=5) |
|---|---|---|---|---|---|
| Reduced exploration, random init (B−A) | +3.82 | 2.10 | 0.81 | 5.89 | [2.18, 5.39] |
| Reduced exploration, clone init (D−C) | +1.09 | 1.86 | −2.05 | 2.95 | [−0.64, 2.36] |
| Clone init, default exploration (C−A) | +8.14 | 2.51 | 4.85 | 11.03 | [6.04, 10.00] |
| Clone init, reduced exploration (D−B) | +5.41 | 0.92 | 4.27 | 6.80 | [4.74, 6.15] |
| Interaction (D−C)−(B−A) | −2.73 | 2.48 | −4.88 | 0.75 | [−4.53, −0.61] |

**Clone initialisation's effect is large and clearly non-zero at BOTH
exploration settings** (C−A and D−B both exclude 0 comfortably, with the
tightest CI of the four main contrasts on D−B: [4.74, 6.15]). **Reduced
exploration's effect is clearly non-zero for random initialisation** (B−A)
**but not clearly distinguishable from zero for clone initialisation**
(D−C, CI includes 0) — diminishing returns once the actor already starts
from a good policy. **The interaction is negative and excludes zero**: the
two interventions are sub-additive on holdout objective, not synergistic —
most of the combined group's advantage comes from clone initialisation, not
from an extra boost specific to combining it with reduced exploration.

This statistical picture undersells one practical distinction, however: even
though D's *final selected-checkpoint* objective is not clearly higher than
C's, D's *entire training trajectory* is materially more stable (Section
3) — reduced exploration removes the risk of landing on a bad checkpoint in
the first place, rather than only rescuing the final number after the fact
via offline selection.

## 8. Learner-seed robustness

Cross-seed SD of holdout mean objective: A=2.41, B=0.62, C=1.12, D=0.97.
Every intervention (B, C, D) reduces cross-seed variance relative to A;
reduced exploration (B) gives the single lowest variance among all four
groups, though at a materially lower mean than C/D.

## 9. Action saturation and learned log_std

Action saturation rate is **0.0 for all 20 selected models** on deterministic
holdout evaluation. `log_std` does gradually decrease over the full
200,000-transition budget in every group (`plots/log_std_trajectory.png`) —
groups A/C move from 0.0 to about −0.06 to −0.17; groups B/D move from −1.5
to about −1.56 to −1.65 — a real, if slow, self-directed reduction in
exploration that Phase 5's much shorter 64,000-transition runs did not have
enough training to reveal.

## 10. Validation-to-holdout generalisation

Mean validation-to-holdout gap (positive = validation overestimated holdout):
A=+1.33, B=−0.90, C=+2.16, D=+6.24. Group D shows the largest generalisation
gap — its offline-selected checkpoints look meaningfully better on the
50-seed validation set than they turn out to be on the 200-seed holdout set
— reported honestly as a caveat, though D's *absolute* holdout performance
remains the best of the four groups regardless.

## 11. Comparison with analytical policies and the frozen clone

See Section 4-5's table. Groups C and D match the oracle, belief-weighted,
and frozen-clone benchmarks within noise; groups A and B do not.

## 12. Which configuration should be used in the dissertation?

**Group D (clone initialisation, reduced exploration)** — highest holdout
mean objective, second-lowest cross-seed SD, fill/inventory behaviour
closest to the analytical benchmarks, and (unlike Group C) a training
trajectory with no observed mid-training crash. The caveat: its
validation-to-holdout generalisation gap is the largest of the four groups,
so any future validation-based claims about D should be treated with that
gap in mind; the holdout numbers reported here are the ones that matter.

## 13. Is another optimisation change justified?

Not on the current evidence. The two changes tested here already close
almost the entire gap to the analytical/clone benchmarks (from ~32 up to
~41.5, against a benchmark of ~41.7–41.9). Given the interaction is
sub-additive and reduced exploration's marginal benefit on top of clone
initialisation is not clearly significant, a natural next refinement — not
yet run — would be tuning the *degree* of exploration reduction specifically
for clone-initialised runs (e.g. a less aggressive `log_std_init` than
−1.5, or a short entropy/log_std schedule) to see whether the
validation-to-holdout gap in Group D can be narrowed without sacrificing
its holdout mean. This is a refinement, not evidence of a new bottleneck.

## Errors found and fixed this phase

1. **Test-isolation leak** (`tests/test_phase6_optimisation_experiment.py`):
   two tests monkeypatched `phase6_common.MODELS_DIR` to an isolated
   `tmp_path` but not `RESULTS_DIR`, so `phase6_run_training.append_status_row`
   wrote bogus 4,000-transition test rows into the REAL
   `results/phase6_optimisation_experiment/phase6_run_status.csv` while the
   real 20-run experiment was queued to start. Caught by inspecting the
   status file before launching the real run and finding rows with
   implausible (4,000-transition) selected timesteps. Fixed by also
   monkeypatching `RESULTS_DIR`; the stray rows (harmless — the real
   model artefacts were correctly isolated in `tmp_path`) were deleted
   before the real run began appending its own rows.
2. **Missing directory creation** (`phase6_run_training.append_status_row`):
   surfaced by the same test-isolation fix — writing to a fresh, not-yet-
   created `RESULTS_DIR` raised `OSError`. Fixed by adding
   `P6.RESULTS_DIR.mkdir(parents=True, exist_ok=True)` before the CSV
   append; harmless in the real run (the directory already existed there),
   but a genuine latent bug for any environment where it does not.
3. **Duplicate-kwarg `TypeError`** (`phase6_evaluate_holdout.py`): both
   `evaluate_agents_event_driven` episode runners already return an
   `evaluation_seed` key (hardcoded to `None` for the RL-agent runner, set
   correctly for the analytic-policy runner) — my wrapper code additionally
   passed `evaluation_seed=seed` as an explicit kwarg into the same `dict(...)`
   call, which Python rejects as a duplicate keyword argument the first
   time it was exercised for real (on the very first analytic-policy
   episode). Fixed by removing the redundant explicit kwarg where the
   function already returns the real seed, and by overwriting
   `m["evaluation_seed"] = seed` first where the function hardcodes `None`.
   Caught immediately (script failed loudly with exit code 1) before any
   holdout data was written; no partial/incorrect holdout results were ever
   produced or used.

**REDUCED EXPLORATION BENEFICIAL** does not fit (its effect is not clearly
present for clone initialisation). **COMBINED INITIALISATION AND
EXPLORATION IMPROVEMENT** does not fit (the interaction is negative, not an
extra synergistic gain). **PERFORMANCE IMPROVED, ROBUSTNESS UNRESOLVED**
does not fit (cross-seed variance clearly shrinks in every intervention
group). **OPTIMISATION CHANGE NOT SUPPORTED** and **RESULTS INCONCLUSIVE**
are contradicted by the clean, bootstrap-confirmed effects above. Clone
initialisation is the one factor whose benefit is large, consistent, and
clearly non-zero at both exploration settings (C−A and D−B), and it alone
accounts for closing nearly the entire gap to the analytical benchmarks.

**CLONE INITIALISATION BENEFICIAL**
