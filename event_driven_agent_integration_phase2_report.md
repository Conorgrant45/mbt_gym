# Event-Driven Agent Integration — Phase 2 Report

Branch: `event-driven-environment`, confirmed active and with a clean working
tree before any changes were made. Phase 1 baseline: commit `f105ca9`. No
full dissertation training run was launched; all training in this phase was
short smoke training (16,000 event transitions per agent) explicitly for
plumbing verification, not a learning result.

## 1. Files inspected

`train_agents.py` (996→1200 lines after edits), `evaluate_agents_common.py`
(935 lines, read in full), `select_checkpoint_offline.py` (271 lines, read
in full), `envs/event_driven_regime_env.py`, `envs/regime_env.py`,
`envs/make_envs.py`, `beliefs/hamilton_filter.py`,
`envs/hamilton_ppo_wrapper.py`, `envs/return_ppo_wrapper.py`,
`simulate_belief_weighted.py` (full read: `build_optimal_control`,
`get_control`, `REGIME_PARAMS`, `make_filter`, `NAIVE_DEPTH`,
`normalise_depth`), `mbt_gym/mbt_gym/gym/TradingEnvironment.py`,
`ModelDynamics.py`, `RewardFunctions.py`, `index_names.py` (all re-verified
unchanged since Phase 1 via `git log`), the full existing test suite, and
`event_driven_environment_phase1_report.md`.

## 2. Files created or changed

**Created:**
- `beliefs/event_time_hamilton_filter.py` — `EventTimeHamiltonFilter`.
- `envs/event_driven_hamilton_ppo_wrapper.py` — `EventDrivenHamiltonPPOWrapper`.
- `envs/event_driven_return_ppo_wrapper.py` — `EventDrivenReturnPPOWrapper`.
- `evaluate_agents_event_driven.py` — event-driven evaluator (analytic
  policies, RL-agent runners, exogenous path hash, incremental stats).
- `event_driven_statistical_diagnostic_phase2.py` — Section 11 diagnostics.
- `tests/test_event_driven_agent_integration.py` — 46 tests (Section 10
  items 1-34 plus supporting checks).
- This report.

**Changed (additive/parametrised, default behaviour preserved exactly):**
- `train_agents.py`: added `--environment-type {fixed,event}` (default
  `fixed`); `build_train_env`/`build_eval_env`/`make_train_env_fn`/
  `run_eval_episode`/`evaluate_policy`/`PeriodicEvalCallback` all gained an
  `environment_type` parameter defaulting to `"fixed"`; added `RunningStats`
  (incremental mean/std accumulator); `main()`'s `output_dir`/`log_dir`
  resolve to `models/<agent_type>_event/`/`logs/<agent_type>_event/` **only**
  when `--environment-type event` is passed, otherwise resolving to the
  exact pre-existing paths; `run_config`/`checkpoint_manifest`/`run_summary`
  now record `environment_type`; the event path's rollout-geometry block
  explicitly documents that one rollout buffer spans several event-driven
  episodes, not one (see Section 7).
- `select_checkpoint_offline.py`: added `--environment-type`; `load_manifest`
  now resolves the environment-type-specific directory and raises
  `ValueError` if the manifest's own recorded `environment_type` disagrees
  with the CLI argument; `checkpoint_rows`/`selection_record` now carry
  `environment_type`.
- `envs/event_driven_regime_env.py` (Phase 1 file): **one bug fix**. The
  fill-probability `rng.uniform()` draw was previously short-circuited by
  Python's `and` operator whenever inventory was already at the
  `max_inventory`/`-max_inventory` bound, meaning the draw's *occurrence*
  (not just its outcome) depended on the action-dependent inventory
  path — a real, if practically dormant (bound is 10,000, never reached at
  these economic scales), violation of exogenous-path independence. Fixed
  by drawing unconditionally, then gating only the *outcome*. Covered by
  `tests/test_event_driven_agent_integration.py::test_max_inventory_fill_suppression_does_not_desync_rng`
  and the full existing `tests/test_event_driven_regime_env.py` suite (still
  26/26 green).
- `tests/test_train_agents.py`: one line. `_fake_evaluate_policy_sequence`'s
  mock `fake()` function needed to accept the new `environment_type` keyword
  (added by `PeriodicEvalCallback._run_eval`'s call site) — a test-fixture
  update, not a behavioural change; the mock still returns the exact same
  controlled sequence it always did.

**Not touched:** `envs/regime_env.py`, `envs/hamilton_ppo_wrapper.py`,
`envs/return_ppo_wrapper.py`, `beliefs/hamilton_filter.py`,
`simulate_belief_weighted.py`, `evaluate_agents_common.py`, any `mbt_gym/`
file, the fixed-step environment's economic model or event ordering.

## 3. Exact event-time information set

At each observable event $n$ (arrival or terminal), the agent's decision may
use only: current inventory $Q_n$ (`info['inventory_after']`), elapsed time
$\tau_n$ = `terminal_time - info['raw_state'][TIME_INDEX]` expressed as a
remaining-fraction, the belief $b_n$ (Hamilton agent) or the raw return
$r_n$ and $\Delta\tau_n$ (return agents) computed from
`info['price_after']`/`info['price_before']`/`info['elapsed_time']`. It may
**never** use `info['true_regime']`, `info['regime_at_event']`,
`info['number_internal_regime_switches']`, or `info['integrated_variance']`
— all four are privileged/diagnostic-only fields, verified absent from every
observation vector by construction and by
`tests/test_event_driven_agent_integration.py`'s leakage tests (items 19-20).
Chronology (traced against source, matching Phase 1's fixed-step timing
exactly): `action chosen from obs_n` → `env.step()` → `info` (return +
event-time) → `filter/return-feature update` → `obs_{n+1}` → next action.
No step ever uses information from beyond its own `info` dict.

## 4. Event-time Hamilton-filter derivation

**Is event timing itself regime-informative?** No, for the production
calibration: `envs/make_envs.py` constructs `PoissonArrivalModel(intensity=
[[LAMBDA, LAMBDA]])` **identically** for both regime sub-environments — total
arrival intensity (280) does not depend on the hidden regime at all. This was
verified, not assumed:
`EventTimeHamiltonFilter.__init__` raises `NotImplementedError` if ever
constructed with regime-dependent intensities (tested directly), and the
production constructor call passes the same `LAMBDA` to both. Consequently
the CTMC prediction is the *exact* (not approximate) closed form
$b^-_n = b_{n-1}\exp(Q\Delta\tau_n)$, with no additional waiting-time
likelihood term. Prediction is exact even across multiple hidden switches
within one interval, since the matrix exponential integrates over every
possible switch path by construction.

**Emission**, at an observable arrival (guaranteed, unlike the fixed-step
filter's per-tick "maybe no arrival" branch):
$f(r_n\mid Z=0)=N(0,\sigma_0^2\Delta\tau_n)$;
$f(r_n\mid Z=1)=p_{\text{buy}}\,\mathrm{EMG}(r_n;\sigma_1\sqrt{\Delta\tau_n},\epsilon)+p_{\text{sell}}\,\mathrm{EMG}(-r_n;\ldots)$,
with $p_{\text{buy}}=p_{\text{sell}}=0.5$ in production. At the terminal
event, both hypotheses collapse to the pure Gaussian (no jump possible
regardless of regime, matching the environment's own terminal handling).
Computed in the log domain via `scipy.special.logsumexp` (item: "perform
likelihood calculations in the log domain where necessary").

**Return-only information set preserved**: `update()`'s signature is
`(price, delta_tau, is_arrival)` — it has **no** `arrival_side` parameter at
all (verified by a test inspecting the signature directly), so the mixture
above marginalises over which side arrived using the a-priori split, exactly
mirroring how the existing fixed-step filter marginalises over
arrival/no-arrival rather than conditioning on the environment's own draws.
This was a deliberate design decision, not an oversight: the environment
*does* reveal `info['arrival_side']`, and it would have been easy to
silently condition on it — doing so would add information the thesis's
existing filter never had.

**Equal-sigma guard**: production has `R0_VOLATILITY == R1_VOLATILITY ==
0.01`, verified directly by the constructor (raises `NotImplementedError` if
unequal) — exact filtering for unequal regime-dependent diffusion spanning
hidden switches within one interval is a harder continuous-time filtering
problem, explicitly not implemented, not silently approximated.

## 5. Whether event timing itself is regime-informative

No — see Section 4. This was checked against the actual constructed
intensities, not assumed from the model description.

## 6. Observation definitions for all three agents

| Agent | Shape | Components | dtype |
|---|---|---|---|
| Hamilton PPO | (3,) | `[tanh(Q/10), 1-t/T, belief]` | float32 |
| Return MLP/LSTM | (4,) | `[tanh(Q/10), 1-t/T, tanh(r/EPSILON_PCT), tanh(Δτ/(1/280))]` | float32 |

`tests/test_event_driven_agent_integration.py` items 15-20 verify ordering,
bounds ($[-1,1]\times[0,1]\times[0,1]$ / $[-1,1]\times[0,1]\times[-1,1]\times[-1,1]$),
dtype, reset values (`belief`=stationary prior; `r`=`Δτ`=0.0 exactly), and
that changing `Δτ` alone changes only the 4th return-agent component (proving
the two pieces of information are not conflated, per the brief's own warning
about a raw return without `Δτ` confounding a large move over a long wait
with one over a short wait).

## 7. PPO rollout behaviour under variable episode lengths

Confirmed via a short `model.learn(total_timesteps=2000)` call
(`test_variable_episode_lengths_recorded_by_monitor`): `Monitor`'s
`ep_info_buffer` records **different** episode lengths across the episodes
completed in one short rollout (never 4000), proving SB3's rollout
collection does not assume a fixed episode length — episode boundaries are
determined entirely by the wrapper's own `terminated`/`truncated` flags, and
GAE/`episode_starts` bookkeeping (SB3-internal, not reimplemented) resets
correctly at each one regardless of length. `gamma=1.0` is passed through
unchanged (never modified for the event-driven path).

**Rollout geometry is honestly reported as changed, not glossed over**: with
`n_steps=4000` (unchanged default) and a mean of `2*LAMBDA=280`
transitions/episode, one rollout buffer now spans **~14.3 event-driven
episodes** (confirmed in the Hamilton smoke run's printed rollout geometry:
`mean_transitions_per_episode=280`, `approx_episodes_per_rollout=14.29`,
`rollouts_span_full_episode=False`, `rollouts_span_multiple_episodes=True`),
versus exactly 1 fixed-step episode per rollout previously. `n_steps`/
`batch_size`/`n_epochs` were **not** tuned for this — they are the
unmodified Phase-2 feed-forward/recurrent defaults, carried over
unquestioned, per the brief's explicit instruction not to tune PPO in this
phase.

LSTM-specific: state persists across observable events within an episode and
resets only at `episode_start=True` (items 23-24, both tested directly with
an untrained tiny model — `test_lstm_state_persists_within_episode_event_driven`,
`test_lstm_state_resets_when_episode_start_true`); internal hidden regime
switches never reach the wrapper at all, so they cannot possibly reset
recurrent state (this follows structurally from Phase 1's environment design
— hidden switches produce no `step()` call).

## 8. Analytical-policy integration

The oracle/belief-weighted/randomised/naive policies were adapted via
`continuous_time_control()`: the *same, unmodified*
`simulate_belief_weighted.build_optimal_control`/`get_control` solver and
table (no re-derivation) are looked up at the nearest fixed-step grid row to
the continuous decision time (`t_idx = round(elapsed_time/STEP_SIZE)`,
clipped), a timing error of at most `STEP_SIZE/2 = 1.25e-4`, negligible
against the ~0.0036 mean inter-arrival gap. The belief-weighted policy uses
`EventTimeHamiltonFilter.belief` — the **same** posterior class Hamilton PPO
uses (not a re-derived one). The oracle ("regime-conditioned benchmark" —
**never** labelled globally optimal, consistent with the existing
dissertation convention already established for the fixed-step oracle, see
[[project_regime_switching_mm]]) reads `env.current_regime` at decision time
only, with no lookahead. All four analytic policies and all three RL agents
share: the same event times (driven by the one shared
`EventDrivenRegimeSwitchingEnv` instance per episode), the same
`inventory_after`, the same `terminal_time`, the same `[bid, ask]` action
ordering and `denormalise_depth`/`normalise_depth` convention (cross-checked
byte-for-byte against `simulate_belief_weighted.MAX_DEPTH`/`normalise_depth`),
the same reward, and the same terminal treatment (exactly one terminal
penalty, confirmed by the Phase 1 environment tests, unchanged).

200-episode diagnostic (analytic policies, event-driven evaluator):

| agent | full_objective (mean±se) | raw_pnl | fills | terminal|inv| |
|---|---|---|---|---|
| oracle | 30.45 ± 0.81 | 30.46 | 76.0 | 1.19 |
| belief_weighted | 30.41 ± 0.81 | 30.42 | 75.9 | 1.20 |
| randomised | 30.31 ± 0.81 | 30.32 | 76.0 | 1.18 |
| naive | 34.39 ± 4.61 | 34.82 | 70.3 | 6.86 |

Compared against Phase 1's own equivalence-diagnostic naive-policy numbers
(different seed range, same policy): full_objective z=-1.31, raw_pnl
z=-1.31, fills z=-0.71 — no flag (threshold `|z|>3`), i.e. the new Phase 2
evaluator layer introduces no detectable drift versus Phase 1's raw-environment
diagnostic.

> **Erratum (added during Phase 3, see `event_driven_phase3_results.md`
> Section 3):** the `oracle`/`belief_weighted`/`randomised` rows in the
> table above are **wrong**. `run_event_analytic_agent_episode` divided the
> event-driven environment's already-raw inventory by `SBW.INV_UNIT` a
> second time (that constant exists only to un-normalise the *fixed-step*
> environment's observation), inflating it 10,000× and saturating
> `SBW.get_control`'s inventory-grid clipping to the boundary for almost any
> nonzero inventory. This was not caught at the time because the numbers
> were merely compared against Phase 1's *naive*-policy diagnostic (naive
> does not depend on `inv_sc` at all, so it was correctly unaffected and
> matched) — the comparison above never checked oracle/belief-weighted/
> randomised against each other or against theory, which would have shown
> naive implausibly beating them. Fixed in Phase 3; two regression tests
> added (`tests/test_event_driven_agent_integration.py`). The `naive` row
> above is unaffected and correct. Corrected oracle/belief_weighted/
> randomised numbers (200 holdout episodes, seeds 200000-200199): oracle
> 45.21±2.99, belief_weighted 45.04±2.97, randomised 45.14±2.97 — all
> properly above naive (44.46), matching CJ theory and the fixed-step
> ordering, as they should.

## 9. Exogenous path-pairing proof

`compute_exogenous_path_hash(seed, action_fn)` hashes only
`event_type`/`arrival_side`/`regime_at_event`/`elapsed_time`/
`brownian_increment`/`jump_increment` at every step (never fills, cash or
inventory, which legitimately depend on actions).
`verify_exogenous_path_pairing()` confirms this hash is **identical** across
four structurally different action sequences (all-zero, all-max, naive
constant, uniform-random) for the same seed — `all_matched=True`, verified
both in the diagnostic script and as a standalone test
(`test_exogenous_path_hash_matches_across_policies`), and a companion test
confirms the hash **differs** across different seeds
(`test_exogenous_path_hash_differs_across_seeds`), ruling out a
degenerate always-equal implementation. This is also what led to finding and
fixing the max-inventory RNG short-circuit bug in Section 2.

## 10. Checkpoint compatibility

Verified end-to-end with the three smoke-trained models (`--save-all-checkpoints`,
Section 12): periodic checkpoints saved post-update via
`PeriodicEvalCallback._on_rollout_start` (unchanged timing logic, only
`environment_type` threaded through); final checkpoint saved explicitly by
`main()` after `.learn()` returns (unchanged); manifests contain
`environment_type: "event"` for all three; `select_checkpoint_offline.py`
evaluated each manifest's candidates using the event-driven environment
(verified by `evaluate_all_candidates` reading `environment_type` from the
manifest itself, not from a CLI-only assumption) and wrote
`ppo_<agent>_<tag>_offline_best.zip`, hash-verified identical to its source
checkpoint, for all three agents. The environment-type mismatch guard was
exercised twice: (a) live, by pointing `select_checkpoint_offline.py
--environment-type fixed` at an event-driven run-tag — rejected via
`FileNotFoundError` (directory separation alone already prevents it); (b) a
targeted unit test that copies an event-driven manifest into the `fixed`
directory to force the **explicit** `environment_type` field check (not just
the path-based guard) to fire — raises `ValueError` as designed.

## 11. Targeted test commands and pass counts

```
python -m pytest tests/test_event_driven_agent_integration.py -v
# 46 passed in 32.3s
```

Covers Section 10 items 1-34 (item 35 is the full-suite run below). One test
(`test_terminal_event_uses_no_jump_emission`) initially failed due to an
unrealistic return magnitude in the *test's own* independent reference
calculation (causing underflow to `0/0` in the test, not the filter — the
filter's own `eps`-guarded fallback handles this correctly); fixed by scaling
the test's return to the diffusion scale actually implied by its
`delta_tau`, no production code changed for this fix.

## 12. Full-suite command and pass count

```
python -m pytest tests/ test_adverse_selection_identity.py test_hamilton_filter.py -q
# 281 passed, 3 warnings in 394.70s (0:06:34)
#   (235 pre-existing/Phase-1 + 46 new Phase 2; 0 failures)
```

One regression was found and fixed during this run: an existing mock
(`tests/test_train_agents.py::_fake_evaluate_policy_sequence`) did not
accept the new `environment_type` keyword now passed by
`PeriodicEvalCallback._run_eval`. Fixed by widening the mock's signature
(one line); re-ran the full suite afterward to confirm 281/281 green — not
inferred from the partial fix alone.

## 13. Smoke-training commands and outcomes

```
python train_agents.py --agent-type hamilton_ppo    --environment-type event --seed 0 --env-seed 70600 --learner-seed 0 --total-timesteps 16000 --eval-freq 8000 --save-all-checkpoints --run-tag phase2_smoke_hamilton
python train_agents.py --agent-type return_mlp_ppo   --environment-type event --seed 0 --env-seed 70600 --learner-seed 0 --total-timesteps 16000 --eval-freq 8000 --save-all-checkpoints --run-tag phase2_smoke_mlp
python train_agents.py --agent-type return_lstm_ppo  --environment-type event --seed 0 --env-seed 70600 --learner-seed 0 --total-timesteps 16000 --eval-freq 8000 --save-all-checkpoints --run-tag phase2_smoke_lstm
```

All three: "Run complete: PASSED", finite losses, no NaN/inf parameters,
save/reload equivalence exact (`max|action diff|=0.0e+00`,
`max|reward diff|=0.0e+00`). `select_checkpoint_offline.py` run against all
three manifests: all "Offline checkpoint selection complete: PASSED",
hash-verified. A subsequent `evaluate_agents_event_driven.py` run against the
three `_offline_best` models (30 fresh holdout seeds) confirmed finite,
reconciled (`reward_reconciliation_error` ~1e-14) output for every agent —
plumbing works end to end. **These smoke numbers are not, and must not be
read as, a dissertation result**: 16,000 timesteps is far below any
meaningful training budget (the existing fixed-step dissertation runs use
200,000), no multi-seed replication was done, and one smoke run
(`hamilton_ppo`) happened to show a higher mean than the analytic policies
purely as an artifact of 16k-step noise and a 30-episode holdout — this is
not evidence of anything and is called out explicitly here so it is not
later mistaken for one.

## 14. Deterministic-evaluation discrepancy (previously reported)

The RL architecture audit (see [[project_rl_architecture_audit_findings]])
flagged an unresolved anomaly: the periodic `timestep_200000` evaluation and
the post-`.learn()` `final_post_training` evaluation disagreed for every one
of 15 fixed-step training runs, despite both claiming to deterministically
evaluate the same weights on the same seeds. This phase did **not**
specifically re-investigate that anomaly (it is a fixed-step,
200,000-timestep-run phenomenon; reproducing it would require a fixed-step
run at that scale, out of scope for "no full dissertation training run").
It was, however, indirectly probed on the event-driven path: all four
smoke/probe training runs in this phase showed **exact** (`0.0e+00`)
agreement between save/reload evaluations using the identical eval-seed
mechanism — this does not resolve the original anomaly (different code path
and scale) but does not surface a *new* instance of it either, at the scale
tested. The original anomaly remains open and unresolved; it is not ignored,
but genuinely out of this phase's scope to fix.

## 15. Remaining assumptions or limitations

1. **Continuous-time control lookup is nearest-grid-point, not a genuinely
   continuous-time re-solve.** Documented error bound (≤1.25e-4) is small
   relative to the mean inter-arrival gap, but this is an approximation, not
   an exact continuous-time optimal control.
2. **Equal-sigma and regime-independent-intensity guards are the ONLY
   supported cases** — by design, not oversight, but any future recalibration
   that breaks either assumption will raise `NotImplementedError`, not
   silently produce a wrong filter.
3. **The 2-state-chain shortcut carried over from Phase 1** (regime switch
   destination hardcoded as `1-z`) is unchanged and still only valid for
   exactly two regimes.
4. **Fixed-step `run_eval_episode`'s save/reload check still stores full
   per-episode actions** (unchanged, small-scale, ≤5 eval seeds) —
   deliberately NOT migrated to `RunningStats`, since Section 8's "no full
   trajectory storage" requirement was scoped to the large-scale evaluators
   (`evaluate_agents_event_driven.py`, `select_checkpoint_offline.py`'s
   multi-candidate loop) where the actual historical MemoryError risk lives;
   this scoping decision is stated here explicitly rather than left
   implicit.
5. **Smoke-training budget (16,000 transitions) is far too small to say
   anything about learned performance** — reiterated from Section 13,
   because the numbers exist on disk and could otherwise be
   misread.
6. **The pre-existing fixed-step deterministic-evaluation anomaly (Section
   14) remains unresolved** — flagged, not fixed, in this phase.

## 16. Is Phase 2 safe for a controlled learning experiment?

All required correctness invariants pass (46/46 new tests, 281/281 full
suite, exogenous-path-hash proof, exact reward reconciliation through every
wrapper, save/reload exactness, checkpoint-selection integrity, filter
prediction/emission verified against independent reference calculations, no
information leakage). The event-time Hamilton filter's calibration
(200-episode privileged diagnostic) is strong: accuracy 99.82%, Brier score
0.00145, log loss 0.00523, mean belief 0.005/0.998 in true regime 0/1
respectively, mean switch-detection lag 0.0035s (~1 mean inter-arrival gap),
and calibration bins track the diagonal closely wherever sample size is
non-trivial. All three agent types (Hamilton PPO, return MLP PPO, return
LSTM PPO) construct, train, checkpoint, reload and evaluate correctly
end-to-end under the event-driven environment.

**SAFE TO PROCEED TO PHASE 3**

with the six limitations in Section 15 carried forward explicitly, and with
the understanding that Phase 3's controlled learning experiment is the FIRST
phase where actual learned performance should be assessed — nothing in
Phase 1 or Phase 2 has done so, by design.

## 17. Recommended Phase 3 experiment design

1. A genuine (not smoke) training budget per agent — the brief's own
   instruction not to launch 200,000-transition runs in *this* phase implies
   Phase 3 is where that gate lifts; recommend starting with an intermediate
   budget (e.g. 50,000-100,000 transitions, ~180-360 episodes) before
   committing to the full 200,000, given the event-driven episode is ~14x
   cheaper per rollout-buffer than the fixed-step one.
2. Multi-seed replication (learner-seed × training-env-seed, matching the
   existing fixed-step 5-seed convention) before any performance claim.
3. Reuse `select_checkpoint_offline.py`'s offline-validated-best checkpoint
   for every reported number — do **not** repeat the fixed-step project's
   own finding (see [[project_rl_architecture_audit_findings]]) of reporting
   raw-final-checkpoint numbers where a validated-best was available but
   unused.
4. A true event-driven vs. fixed-step head-to-head on the SAME agent
   architectures and a comparable training budget, to test whether
   event-time decisions (higher information density per decision, ~14x
   fewer decisions per unit wall-clock/compute) close any of the
   representation/optimisation gap identified in the RL architecture audit.
5. Only after (1)-(4): revisit the sigma=0 (jump-only) filter ablation,
   explicitly deferred out of both Phase 1 and Phase 2.
