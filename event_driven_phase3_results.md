# Phase 3 — Event-Driven vs. Fixed-Step PPO: Controlled Comparison

Branch: `event-driven-environment`, confirmed active with a clean working
tree before any changes were made. Full test suite (281/281) confirmed
passing before training was launched, and (283/283, +2 regression tests)
after the bug fix described in Section 3 below.

## 1. Exact seed ranges

| Purpose | Range | Notes |
|---|---|---|
| Training-environment seed (fixed **and** event) | 70000 | Same value in both, per spec |
| Learner seeds | 0, 1, 2, 3, 4 | Same 5 values in both environments |
| Fixed-step offline-best validation (pre-existing) | 91001–91050 | Unchanged, reused as-is |
| Event-driven offline-best validation (new) | 195001–195050 (50 seeds) | Chosen fresh after inspecting every prior range in the project (training 0–4/70000; dev-monitor 90001–90005; fixed validation 91001–91050; `hamilton_ppo_eval_lib` 100000–100199/100000–100049/110000–110099; `evaluate_agents_common` 120000–120099; `evaluate_agents_event_driven` default 130000–130099, unconsumed; Phase 1 diagnostic 900000–900499/950000–950499; Phase 2 smoke 91401–91408/160000s/170000–170029) |
| **Holdout (frozen before any evaluation)** | **200000–200199 (200 seeds)** | Never used for training, monitoring, validation or checkpoint selection anywhere in this project; frozen in `phase3_experiment_manifest.json` before `phase3_holdout_evaluation.py` was ever run |

## 2. Git commit and package versions

Git commit at launch: `97ee1bf8f63ee1a41d9295d42241dd7b279266c2`. torch
2.11.0+cpu, stable_baselines3 2.8.0, sb3_contrib 2.8.0, gymnasium 1.2.3, gym
0.26.2, numpy 2.2.4, pandas 3.0.0, scipy 1.17.0, Python 3.13.14. Full record
in `results/phase3_event_vs_fixed/phase3_experiment_manifest.json`.

## 3. Existing fixed-step models: reused, not retrained

All 15 fixed-step `_offline_best.zip` models (`offlinecv_env70000_learner_{0..4}`
× 3 architectures) were located and verified via their own `run_config`/
`offline_selection` JSON metadata: `environment_type` absent (⇒ pre-Phase-2
⇒ fixed, correct), `training_env_seed=70000`, `learner_seed` matching,
`total_timesteps=200000`, `checkpoint_selection`/path ending in
`_offline_best.zip`. **All 15 passed; none were retrained.**

**A genuine implementation bug was found and fixed during this phase**, per
the explicit instruction to stop rather than silently patch and continue.
While reviewing the holdout results (before drawing any conclusion from
them), the event-driven analytical benchmarks showed naive (44.46)
outperforming oracle/belief-weighted/randomised (~32.1–32.2) — the reverse
of both CJ theory and the fixed-step pattern, and consistent with the same
(then-unnoticed) anomaly already present in Phase 2's own 200-episode
diagnostic. Root cause: `evaluate_agents_event_driven.py`'s
`run_event_analytic_agent_episode` divided the event-driven environment's
already-raw inventory by `SBW.INV_UNIT` (1/10000) a second time.
`INV_UNIT` exists solely to convert the **fixed-step** environment's
*normalised* observation component back to raw share units
(`inv_sc = obs_flat[1] / INV_UNIT`, used identically in
`evaluate_agents_common.py`, `compare_four_policies_paired.py` and
`simulate_belief_weighted.py`); the event-driven environment's
`raw_inventory` was never normalised in the first place. This inflated
`inv_sc` by 10,000× (e.g. 0.5 shares → `inv_sc=5000`), saturating
`SBW.get_control`'s inventory-grid clipping (range ≈[-50, 50]) to the
boundary for almost any nonzero inventory — confirmed numerically:
raw inventories of 0.5, 1.0 and 2.0 shares all produced the *identical*
saturated control (ask=0.069, bid=1.264) under the bug, versus a smooth,
distinct, correctly-skewed response after the fix. This is isolated to that
one analytic-policy evaluation function — it does **not** touch the
environment, the RL observation wrappers, RL training, or RL holdout
evaluation, all of which use entirely different code paths and were
unaffected. Fixed (one line: `inv_sc = q` instead of `inv_sc = q /
SBW.INV_UNIT`), two regression tests added
(`test_event_analytic_policy_inv_sc_matches_raw_inventory_not_inflated`,
`test_event_analytic_policy_control_responds_smoothly_to_small_inventory`),
full suite re-confirmed green (283/283), and **only** the affected data
(the 4×200 event-driven analytic-benchmark holdout episodes) was
regenerated — the 15+15 RL holdout evaluations and all 15 event-driven
training runs were computed by an unaffected code path and were **not**
mixed with pre-fix numbers. Every number in this report reflects the
post-fix code. Phase 2's report cited the same pre-fix analytic numbers;
an erratum has been appended there pointing to this corrected version.

## 4. Event-driven runs completed

**15/15** (3 architectures × 5 learner seeds), all trained, checkpointed
(13 candidates each, every 16,000 transitions plus final), offline-selected
against 50 validation episodes, and hash-verified. Verified independently
against the actual artefacts on disk (not only the orchestrator's own
self-report) via `phase3_orchestrate_event_driven_batch.run_is_complete`
for all 15 (agent_type, seed) pairs: all `True`.

## 5. Selected checkpoint for every run

| Architecture | Seed | Fixed timestep | Event timestep |
|---|---|---|---|
| Hamilton PPO | 0 | 16000 | 128000 |
| Hamilton PPO | 1 | 144000 | 32000 |
| Hamilton PPO | 2 | 176000 | 16000 |
| Hamilton PPO | 3 | 176000 | 16000 |
| Hamilton PPO | 4 | 16000 | 80000 |
| Return MLP PPO | 0 | 160000 | 16000 |
| Return MLP PPO | 1 | 96000 | 96000 |
| Return MLP PPO | 2 | 128000 | 96000 |
| Return MLP PPO | 3 | 32000 | 16000 |
| Return MLP PPO | 4 | 64000 | 80000 |
| Return LSTM PPO | 0 | 160000 | 32000 |
| Return LSTM PPO | 1 | 48000 | 200000 |
| Return LSTM PPO | 2 | 16000 | 112000 |
| Return LSTM PPO | 3 | 80000 | 200000 |
| Return LSTM PPO | 4 | 112000 | 32000 |

Selected timesteps are scattered across the full 16,000–200,000 range in
**both** environments — checkpoint sensitivity (Section 10) is not visibly
reduced by event-driven training; see Section 10 below.

## 6. Fixed-step vs. event-driven performance by architecture

Across-seed mean holdout `full_objective` (± across-seed std, n=5 learner
seeds; full per-seed table in `phase3_results_tables.tex`):

| Architecture | Fixed | Event | Diff (event−fixed) | Bootstrap 95% CI (n=5, weak) |
|---|---|---|---|---|
| Hamilton PPO | 31.25 (4.33) | 34.25 (**1.30**) | +3.00 | [-0.01, 6.68] |
| Return MLP PPO | 29.44 (5.91) | 28.92 (5.60) | -0.52 | [-7.20, 7.83] |
| Return LSTM PPO | 24.75 (**20.46**) | 31.27 (**5.95**) | +6.52 | [-6.31, 24.95] |

None of the three bootstrap CIs excludes zero comfortably (Hamilton's
barely does, at the boundary) — **with n=5 learner seeds, none of these
differences should be read as a statistically definitive win**, exactly as
instructed. What *is* striking and highly consistent, independent of any
significance test: **the across-seed standard deviation collapses for
Hamilton PPO (4.33→1.30) and for Return LSTM PPO (20.46→5.95)** in the
event-driven environment. This is driven by a concrete, visible mechanism,
not a statistical artefact: **two of the five fixed-step seeds are
catastrophically collapsed policies** (`return_lstm_ppo` seed 0: mean
objective **-11.65**, std **483**, loss rate 50.5%; `return_mlp_ppo` seed
0: mean **19.33**, std **328**, loss rate 48.0%) — the exact seed-fragility
pathology flagged in the prior RL architecture audit. **No event-driven
seed, in any of the 15 runs, shows this pattern** (worst event-driven seed
across all three architectures: `return_mlp_ppo` seed 2 at mean 22.01, std
40.7, loss rate 22.0% — far from catastrophic).

## 7. Gap to the analytical belief-weighted policy

| Architecture | Fixed gap | Event gap | Change |
|---|---|---|---|
| Hamilton PPO | 10.73 | 10.79 | +0.06 (unchanged) |
| Return MLP PPO | 12.53 | 16.11 | +3.58 (worse) |
| Return LSTM PPO | 17.23 | 13.77 | -3.46 (better) |

The analytical benchmarks themselves are **not** identical across
environments (belief-weighted: 41.98 fixed vs. 45.04 event — the two
environments are statistically equivalent, not pathwise identical, per
Phase 1; this is exactly why Section 9 mandated using the benchmark-relative
gap, not the raw objective, as the primary cross-environment comparison).
For the **primary comparison (Hamilton PPO)**, the gap to the belief-weighted
policy is essentially **unchanged** (10.73 → 10.79) — Hamilton PPO's own
objective improved by roughly the same amount the benchmark's ceiling moved
up by. This does **not** support the strong form of David's hypothesis
(that removing uninformative fixed-grid transitions would let Hamilton PPO
approach the analytical policy). Return LSTM PPO's gap narrows moderately;
Return MLP PPO's gap widens.

Gap to the oracle (regime-conditioned benchmark, not claimed globally
optimal — see `phase3_benchmark_comparison.csv` for the full oracle-relative
table) shows the same qualitative pattern, since oracle and belief-weighted
are themselves only ~0.2–0.7 points apart in both environments (tiny
information gap, consistent with the prior RL architecture audit).

## 8. Spread, fill and inventory differences

Across-seed mean, fixed → event (all three architectures):

| Metric | Hamilton | Return MLP | Return LSTM |
|---|---|---|---|
| Quoted spread | 3.454→3.220 (narrower) | 3.424→3.484 (wider) | 3.518→3.374 (narrower) |
| Fills | 29.7→26.6 (**fewer**) | 35.6→23.1 (**fewer**) | 48.6→23.6 (**fewer**) |
| Mean signed inventory | +4.500→**-0.513** | +6.326→**+1.012** | +11.103→**-0.107** |
| Terminal \|inventory\| | 9.80→4.50 | 13.84→4.57 | 21.80→5.85 |
| Running penalty | 0.866→0.174 | 2.240→0.134 | 4.768→0.203 |
| Adverse-selection loss | 10.13→8.04 | 12.49→7.87 | 17.09→7.86 |
| Loss rate | 26.9%→14.8% | 21.3%→16.9% | 28.5%→17.6% |

Two of Section 1's five named failure modes are **not** resolved by
event-driven training — spreads only marginally narrow (and *widen* for
Return MLP), and **fills decrease for all three architectures**, the
opposite of what "more informative transitions → better fill decisions"
would naively predict. But **inventory bias is dramatically reduced or
eliminated for all three architectures** (the persistent net-long bias
identified in the RL architecture audit essentially disappears for
Hamilton and Return LSTM, and shrinks by ~84% for Return MLP), terminal
inventory risk falls 46–73%, running penalty falls 80–96%, adverse-selection
loss falls 21–54%, and loss rate improves for all three. Per Section 12's
explicit interpretation rule ("if spreads narrow and fills increase but the
objective does not improve, investigate adverse-selection exposure and
inventory penalties rather than declaring success") — the actual pattern
observed is almost the inverse: fills *decrease* while several risk/bias
metrics improve substantially and the objective improves for two of three
architectures. The most defensible reading is that event-driven agents are
being **more selective** about which quotes lead to fills (fewer, better-
timed fills, less inventory accumulation, less adverse-selection exposure
per unit of inventory carried) rather than simply learning a uniformly
"better" or "worse" policy.

## 9. Learner-seed sensitivity

Per-seed range within an architecture remains wide in both environments
(e.g. fixed Hamilton PPO: 24.1–34.6; event Return LSTM PPO: 24.1–40.5) —
learner-seed sensitivity is **not** eliminated by event-driven training.
What changes is the *shape* of that sensitivity: fixed-step's worst seeds
are catastrophic outliers (Section 6); event-driven's worst seeds are
merely below-average, not collapsed.

## 10. Checkpoint sensitivity

Selected checkpoint timesteps range from 16,000 to 200,000 in **both**
environments (Section 5's table) — training curves remain non-monotonic and
checkpoint-selection-sensitive under event-driven training too. This
specific pathology (flagged in the prior RL architecture audit as a
plausible driver of the fixed-step gap) is **not** visibly improved by
switching to event-time transitions.

## 11. Training and evaluation wall-clock cost

Per-run training wall-clock (200,000 transitions): Hamilton PPO ≈600–730s
fixed vs. ≈240–255s event (≈2.5–2.9× faster); Return MLP PPO ≈380–540s fixed
vs. ≈120–135s event (≈3–4× faster); Return LSTM PPO ≈1090–1130s fixed vs.
≈2100–2720s event (≈2× **slower**, not faster — recurrent per-step forward-
pass cost dominates over the cheaper event-driven environment step. See
`phase3_checkpoint_summary.csv`'s `total_elapsed_seconds` column for exact
per-run figures). Holdout evaluation (200 episodes/model): fixed-step
≈340–930s per RL model (4000 steps/episode) vs. event-driven ≈14–47s per RL
model (~280 steps/episode in expectation) — 15–65× faster, the direct
computational benefit of decision-only-at-arrivals. Total holdout
evaluation wall-clock: ≈11,400s (≈3.2h), dominated by the 15 fixed-step RL
models and the 4 fixed-step analytic benchmarks (≈10,300s combined).

## 12. Is David's event-driven hypothesis supported?

Evaluated failure-mode by failure-mode (Section 1):

| Failure mode | Resolved? |
|---|---|
| Excessively wide quoted spreads | **No** — marginal narrowing (2/3), one widens |
| Too few fills | **No** — fills *decrease* for all three |
| High objective variance | **Partially** — dramatic reduction for Hamilton and LSTM; unchanged for MLP |
| Persistent net-long inventory | **Yes** — dramatic reduction/near-elimination, all three |
| Large gap to belief-weighted | **No** — unchanged (Hamilton, the primary comparison), worse (MLP), better (LSTM) |

Per Section 12's interpretation rules: Hamilton PPO does **not** measurably
approach the analytical belief-weighted policy (gap unchanged) — the
specific, strong form of the hypothesis this experiment was designed to
test is **not supported**. The improvement is **not uniform** across the
three architectures (MLP does not improve; Hamilton and LSTM improve
unevenly) — this weighs against a single common "more informative
transitions help credit assignment uniformly" explanation, though the
inventory-bias and catastrophic-collapse-avoidance findings (Sections 6, 8)
are uniform across all three and are too large and mechanistically clear
(concrete outlier seeds, concrete inventory-bias numbers) to dismiss as
noise. This is a genuinely mixed result: a real, substantial, reproducible
secondary benefit (variance/tail-risk/inventory-bias reduction) alongside a
clear failure of the primary, headline hypothesis (closing the gap to the
analytical policy) and an unresolved or worsened outcome on two of the five
named failure modes (spreads, fills).

## 13. Remaining limitations

1. **n=5 learner seeds is weak evidence** for any of the event-vs-fixed
   differences — explicitly not claimed as statistically definitive.
2. **The continuous-time analytic-policy control lookup remains
   nearest-grid-point** (Phase 2's documented ≤1.25e-4 timing-error
   approximation), now additionally confirmed correct on the inventory axis
   too (Section 3's bug fix) but still not a genuinely re-solved
   continuous-time HJB.
3. **Checkpoint sensitivity is not resolved** by event-driven training
   (Section 10) — the same non-monotonic training-curve behaviour persists.
4. **Fills decreasing is unexplained** beyond the "more selective" reading
   offered in Section 8 — this would need a dedicated action-surface/fill-
   timing diagnostic (Section 10 of the audit brief) to characterise
   precisely, not attempted in this phase.
5. **The pre-existing fixed-step deterministic-evaluation anomaly** (flagged
   unresolved in both the RL architecture audit and the Phase 2 report)
   remains unresolved — out of scope here too.
6. **This experiment's own bug (Section 3) was caught only by inspecting
   results that looked implausible**, not by a test written in advance —
   the two new regression tests close this specific gap for the future, but
   this is a reminder that analytic-policy integration code for a new
   environment needs the same "does this number make economic sense"
   scrutiny as the environment/filter code itself.

## 14. Is a sigma=0 ablation justified as the next experiment?

**Not yet.** The sigma=0 (jump-only) ablation was explicitly out of scope
for this phase and remains a reasonable *future* experiment, but it should
not be the immediate next step: the current results raise a more pressing,
cheaper question first — why do event-driven agents fill *less* often
despite (for 2/3 architectures) narrower spreads, and why does checkpoint
selection remain just as unstable under event-driven training. Both are
answerable with the existing sigma=0.01 setup and the artefacts already on
disk (action-surface diagnostics per Section 10 of the audit brief), and
would sharpen what a sigma=0 ablation is actually testing for. Recommended
order: (a) action-surface/fill-selectivity diagnostic on the existing
event-driven checkpoints; (b) a checkpoint-sensitivity investigation
(Section 10, both environments); (c) only then, sigma=0.

---

## RESULTS INCONCLUSIVE
