# RL Architecture Audit — Hamilton PPO vs. Analytical Benchmarks

Audit date: 2026-07-27. Scope: full RL pipeline (Hamilton filter → observation
wrapper → PPO → reward → evaluation → checkpoint selection), per the 17-section
diagnostic brief. This report separates **information gap**,
**representation gap**, and **optimisation gap**, and does not conclude "Hamilton
PPO fails to match the oracle" without that separation.

No economic-model changes were made. No full training run was launched. All
numbers below come from either (a) code inspection, (b) the existing test
suite (executed, not re-derived), or (c) the already-produced 5-seed final
comparison dataset (`results/agent_comparison_final_all_seeds_episodes.csv`,
1900 episodes, pure re-aggregation, no new rollouts).

---

## 1. Files inspected

Core pipeline (read in full): `beliefs/hamilton_filter.py`,
`envs/hamilton_ppo_wrapper.py`, `envs/return_ppo_wrapper.py`,
`envs/regime_env.py`, `envs/make_envs.py`, `envs/arrival_jump_midprice.py`,
`train_hamilton_ppo.py`, `train_agents.py` (996 lines, full),
`select_checkpoint_offline.py`, `hamilton_ppo_eval_lib.py`,
`aggregate_final_results.py`, `evaluate_agents_common.py` (schema/Part-2/3
sections). Test suite (11 files, ~209 tests) read for coverage mapping;
`tests/test_rl_architecture_audit.py` and `tests/test_hamilton_filter_audit.py`
read in full. No files were changed.

## 2. Runtime data-flow (as built, not as assumed)

```
RegimeSwitchingEnv (global clock, two synced sub-TradingEnvironments)
  -> raw_state (cash, inventory, time, midprice)  [info['raw_state'], never the normalised obs]
  -> HamiltonPPOWrapper._build_obs():
       q_scaled = tanh(raw_inventory / 10.0)            float32, [-1,1]
       tau      = 1 - current_step / 4000               float32, [0,1]
       b_t      = HamiltonFilter.update(raw_midprice)    float32, [0,1]
     obs = [q_scaled, tau, b_t]                          Box(3,), float32
  -> PPO MlpPolicy (net_arch=[64,64], Tanh) -> Gaussian mean/std
  -> sampled (train) / mean (deterministic eval) action, shape (2,), [-1,1]
  -> passed straight through to base_env (TradingEnvironment denormalises
     internally: action=[bid_action, ask_action] -> depth=(a+1)/2*MAX_DEPTH)
  -> arrivals (Poisson, action-independent) x fills (Exponential(kappa*depth))
  -> RunningInventoryPenalty: reward_t = pnl_t - PHI*dt*q_t^2 - ALPHA*q_T^2*1[terminal]
  -> Monitor -> DummyVecEnv -> PPO RolloutBuffer (n_steps=4000 = exactly one
     episode) -> GAE(gamma=1, lambda=0.95) -> 10 epochs x 10 minibatches (400)
     -> clipped surrogate + value loss -> Adam update
```

`ReturnPPOWrapper` is structurally identical except the third observation
component is `tanh(raw_return/EPSILON_PCT)` instead of the filter belief, and
no `HamiltonFilter` is instantiated at all.

Belief timing (traced against the code, cross-checked against
`tests/test_hamilton_filter_audit.py`): `reset()` calls `filt.update(S_0)`,
which short-circuits on `_prev_midprice is None` and returns the **unchanged
prior** — it does not consume $S_0$ as a return. Each `step(a_t)` computes
$b_{t+1}$ from $S_{t+1}$ *after* `a_t` has already been applied to the
environment, so the action never uses information from its own step's return.
Exactly one Bayes update per env step (confirmed by
`test_exactly_one_bayes_update_per_update_call`).

## 3. Confirmed invariants (empirically, not just by reading)

- **Reward-decomposition identity**: `cumulative_reward == raw_pnl -
  running_inventory_penalty - terminal_inventory_penalty` holds to
  `max|diff| = 2.3e-11` across all 1900 episodes in the final comparison
  dataset (not a synthetic check — the actual evaluation data).
- **Episode length**: exactly 4000 steps on every one of the 1900 episodes
  (`episode_length` column, single unique value).
- **Path pairing**: `path_pairing_status` is
  `exogenous_paired_fills_not_claimed_paired` on every row — regime path,
  diffusion, arrivals and jumps are confirmed action-independent and
  identical across all seven agent types on a shared seed; fills are
  correctly *not* claimed paired.
- **Filter/policy same posterior**: `simulate_belief_weighted.make_filter()`
  and `hamilton_ppo_wrapper.make_filter()` build `HamiltonFilter` from
  identical `envs.make_envs` constants (same transition matrix, sigma, jump
  params, r=0) — the belief-weighted analytical policy is a genuine
  same-information comparison for Hamilton PPO, not a different posterior.
- **Benchmark parameter parity**: `tests/test_rl_architecture_audit.py`
  confirms the analytical policies' phi/alpha/MAX_DEPTH are literal aliases
  of the environment's own reward/fill constants (not independently
  redefined values that could silently drift).
- **RNG isolation**: `RegimeSwitchingEnv`'s regime draws, and every
  midprice/arrival/fill model, use independent `SeedSequence`-spawned
  generators (`tests/test_rng_isolation.py`, `test_seed_separation.py`,
  both green) — evaluation callbacks cannot perturb the training stream.
- **Full existing test suite**: **209 passed, 0 failed** (9m16s), covering
  filter hand-calculations, no-lookahead timing, observation/action ordering
  and bounds, save/reload bit-equivalence, recurrent state handling, and
  checkpoint-selection hashing.

## 4. Information gap vs. representation/optimisation gap (the core question)

Using the already-produced 5-training-seed × 100-holdout-seed comparison
(`results/agent_comparison_final_all_seeds_episodes.csv`, seeds 120000–120099,
disjoint from every training/dev range):

| agent | mean `full_objective` | std | mean quoted spread | fills/episode | mean signed inventory |
|---|---|---|---|---|---|
| oracle | 44.74 | 36.8 | 2.05 | 64.0 | 0.25 |
| belief_weighted | 44.05 | 37.3 | 2.05 | 62.7 | 0.00 |
| randomised | 43.67 | 38.3 | 2.05 | 63.6 | 0.07 |
| naive | 43.00 | 58.9 | 1.83 | 70.2 | 0.01 |
| **hamilton_ppo** | **25.13** | 107.2 | 3.36 | 25.6 | 4.34 |
| **return_lstm_ppo** | **28.18** | 130.6 | 3.36 | 33.5 | 5.98 |
| **return_mlp_ppo** | **17.20** | 158.7 | 3.20 | 36.9 | 3.45 |

- **Information gap** = oracle − belief_weighted = **0.69** (≈1.5% of the
  objective). Partial observability, given a well-calibrated Hamilton filter,
  costs almost nothing here. (Filter accuracy was previously verified at
  0.90 per [[project_regime_switching_mm]]; this result is consistent with that.)
- **Representation + optimisation gap** = belief_weighted − hamilton_ppo =
  **18.92** (≈43% of the analytical policy's objective). This is the entire
  size of the reported problem, and it is **not** an information problem.
- **All three RL architectures fail in the same qualitative way**, regardless
  of what information they receive: they quote far wider than optimal
  (spread 3.2–3.4 vs. 1.8–2.05), fill 2–2.7× less often, capture roughly half
  the spread revenue (33–41 vs. 59–64), run 3–4× the objective variance of
  every analytical policy, and carry a **persistent net-long inventory bias**
  (+3.4 to +6.0 shares mean, vs. ≈0 for every analytical policy) that is
  absent from the reward's own incentives (`PER_STEP_INVENTORY_AVERSION` is
  symmetric). This directional bias is visible directly in the action means:
  `hamilton_ppo` quotes bid=+0.124, ask=+0.312 (both wider than the
  belief-weighted policy's ≈−0.33/−0.33, and asymmetrically wider on the ask
  side), consistent with a policy that undersells and thus drifts long.
- `return_lstm_ppo` — which receives **no belief and no regime information at
  all**, only a raw one-step return via LSTM memory — performs
  statistically indistinguishably from (and numerically slightly above)
  `hamilton_ppo` (paired diff −3.05, CI [−8.9, 2.8], not significant). An
  agent with strictly less structured information does no worse than the
  belief-fed agent. This is strong evidence **against** an information-gap
  explanation for the shortfall and **for** a shared PPO
  training/optimisation limitation across architectures.

## 5. Root-cause findings, with severity

**Finding 1 (high severity, confirmed): training-curve non-monotonicity /
checkpoint instability.** For every one of the 15 final training runs
inspected (3 agent types × 5 seeds), the training curve is **not**
monotonically improving: e.g. `hamilton_ppo` seed 0's periodic evaluation
mean_reward goes baseline 43.4 → 31.2 (40k) → 31.2 (80k) → 31.9 (120k) → 48.6
(160k) → 61.1 (200k, periodic) → 70.6 (final). The policy gets *worse* than
an untrained baseline for the first 120k of 200k timesteps on this seed. This
pattern recurs across seeds/agents. Independently, the separately-run
5-seed offline checkpoint-selection experiment (`offlinecv_env70000_learner_*`,
a proper validation-set sweep over 13 candidate checkpoints per seed) selects
the **earliest** candidate (16,000 of 200,000 timesteps — 8% of training) as
best for 2 of 5 seeds, meaning performance actively **degrades** with
continued training on those seeds. This is consistent with PPO optimisation
instability (possibility #5), not an information or representation problem,
and confirms that checkpoint-selection noise (possibility #6) is real and
consequential — which checkpoint is used materially changes the reported
number.

**Finding 2 (high severity, needs follow-up): the actual final 5-seed
dissertation comparison used the raw final-timestep model, not a validated
best/offline-selected checkpoint.** `agent_comparison_final_all_seeds_episodes.csv`'s
`model_path` for every hamilton_ppo/return_mlp_ppo/return_lstm_ppo row points
to `ppo_<agent>_final_seed_<n>.zip` — the plain post-`.learn()` save, not a
`_best` or `_offline_best` file (which don't exist for the `final_seed_*` run
tags; no `--save-all-checkpoints` manifest was produced for these specific
runs). Given Finding 1's non-monotonicity, this means the reported RL numbers
are plausibly *not* each seed's best achievable performance. This is a
methodological gap in the comparison procedure (possibility #7), separate
from the RL algorithm itself, and the project already has working,
209/209-tested infrastructure (`select_checkpoint_offline.py`) that was
simply never pointed at these specific runs.

**Finding 3 (medium severity, confirmed anomaly, root cause not yet
isolated): periodic vs. final evaluation disagree at the same claimed
timestep.** For all 15 training runs, the `timestep_200000` row (evaluated
from `PeriodicEvalCallback._on_rollout_start`) and the `final_post_training`
row (evaluated in `main()` after `model.learn()` returns) disagree — by as
much as 62.7 (return_mlp_ppo seed 0) or 70.1 (return_lstm_ppo seed 4) reward
units — despite both being documented as deterministic evaluations of "the"
model at `total_timesteps=200000` on the identical 5 fixed eval seeds. Two
deterministic evaluations of literally the same weights on the same seeds
cannot differ; this proves the two calls are evaluating **different**
weights. Per the module's own rollout-boundary arithmetic
(`n_steps=4000`, `total_timesteps=200000` ⇒ exactly 50 rollouts, so
`on_rollout_start` should only ever fire at `num_timesteps ∈
{0,4000,...,196000}`), a periodic evaluation firing at exactly `200000`
should not be possible under the mechanism described in the code's own
docstring — meaning either that reasoning about the installed SB3 version's
rollout-boundary semantics is incomplete, or there is a real timing bug.
Separately, `best_validation_mean_objective` is `None` in every one of the 5
`run_summary_final_seed_*.json` files despite periodic evaluations
demonstrably having fired (the `timestep_40000` etc. rows exist with real
values) — meaning the online best-checkpoint tracker is not populating
`run_summary` correctly for these runs, even though the equivalent unit tests
(`tests/test_checkpoint_selection.py`, `tests/test_train_agents.py`, 209/209
green) pass. This needs a dedicated, isolated repro (a short synthetic run
with instrumented callback firing) before being labelled a bug or an
already-covered/inert code path — flagged, not fixed, per instructions not to
silently patch during the audit.

**Finding 4 (low severity, reproduced live, not yet regression-tested):
MemoryError from full-trajectory action storage.** A historical crash exists
on disk (`logs/offline_selection_batch/seed3_holdout_error.txt`):
`evaluate_agents_common.run_return_agent_episode` raised `MemoryError` at
`actions.append(action.copy())` mid-batch. This session independently
reproduced a `MemoryError` on an unrelated tiny (3.1 MiB) numpy allocation
in `simulate_belief_weighted.build_optimal_control` while running under this
machine's actual operating conditions (8 GB total RAM, ~470–620 MB free
during this session). This is consistent with genuine resource fragility
under low headroom, not a logic bug, but Section 13's mandated regression
test for the original MemoryError does not appear in the test suite (grepped
across `tests/`, not found) — this is an open item, not fixed.

**Finding 5 (not a bug, a scope gap): no privileged/true-regime PPO
diagnostic exists.** Grepped the full repo for `true_regime`,
`TrueRegime`, `oracle_ppo`, `privileged` — `true_regime` appears only as
critic-adjacent/logging metadata (`info['true_regime']`, offline belief-
accuracy scoring), never as an actual policy observation. Section 9's
diagnostic (an observation of `[q_scaled, tau, true_regime]` trained with
identical PPO hyperparameters/budget) has not been built. Given Finding 1
already shows the *raw-return* agent (strictly less informative than the
belief agent) performs on par with the belief agent, the true-regime
diagnostic is unlikely to change the qualitative conclusion (optimisation,
not information, dominates) — but it has not been run, so this remains an
inference from a proxy result, not a direct confirmation.

**Finding 6 (inconclusive, not a finding either way): supervised-imitation
representability diagnostic did not complete.** `diagnostic_imitation_hamilton_belief.py`
(Section 8's exact required diagnostic — already written, not authored this
session) crashed twice with `MemoryError` when run in this session, on the
same 8 GB/low-free-RAM machine as Finding 4. It has **not** produced a
train/test action-MSE result. Do not treat the representation question as
resolved in either direction until this is re-run successfully (e.g. after
closing other processes, or with smaller `N_TRAIN_EPISODES`).

## 6. What this does and does not establish

- **Established**: the reported "Hamilton PPO hasn't approached the oracle"
  symptom is overwhelmingly a **representation-and/or-optimisation** problem,
  not an information problem — the belief-weighted policy loses almost
  nothing (0.69 of 44.74) to partial observability, while Hamilton PPO loses
  18.92. The failure mode (too-wide quotes, low fill rate, persistent long
  bias, high tail variance, non-monotonic training curves, checkpoint
  instability) is common to all three independently-trained architectures,
  which is the strongest single piece of evidence that this is a shared PPO
  training/optimisation limitation (possibly compounded by the small
  effective training budget — 200k timesteps = 50 independent 4000-step
  episodes per seed — and by checkpoint-selection practice) rather than
  something specific to the Hamilton filter, the belief observation, or an
  implementation bug in the reward/action/observation pipeline, all of which
  passed every invariant check run in this audit.
- **Not established**: whether the gap is *representation* (architecture
  cannot express the analytical policy) or *pure optimisation* (architecture
  can express it, PPO hasn't found it) — Finding 6 is unresolved. Given
  Finding 1's non-monotonic/checkpoint-sensitive training curves, pure
  optimisation instability is the leaner explanation, but this is not yet
  directly confirmed.
- **Not established**: the exact mechanism behind Finding 3's eval-timing
  disagreement.
- **No implementation bug was found** in the Hamilton filter, the
  observation/action wrappers, the reward decomposition, or the
  environment/analytical-policy parameter parity — all of these passed
  either an existing targeted test, a hand-recomputation against the actual
  1900-episode dataset, or both.

## 7. Whether existing dissertation results remain valid

The 5-seed final comparison's **qualitative** conclusion (all three RL
agents underperform every analytical policy, including naive) is well
supported and internally consistent (reward identity holds to 1e-11, path
pairing verified, parameter parity verified). The **magnitude** of the RL
agents' shortfall is likely pessimistic given Findings 1–3: a validated
best/offline-selected checkpoint, and a resolved eval-timing anomaly, could
each independently change the reported numbers. These results should be
labelled as using **final-timestep, not best-validated, checkpoints** if
reported as-is.

## 8. Recommended next actions, in order

1. Re-run `diagnostic_imitation_hamilton_belief.py` to completion (Finding 6)
   — cheapest, most decisive remaining diagnostic; needs more free RAM than
   was available this session.
2. Isolate Finding 3 with a short (e.g. 8000-timestep, 2-rollout) synthetic
   PPO run with print-instrumented callback firing, to determine whether
   `timestep_200000` genuinely evaluates stale weights or whether
   `best_validation_mean_objective`'s `None` result is a separate reporting
   bug.
3. Re-run `select_checkpoint_offline.py`-style validation (or re-train with
   `--save-all-checkpoints`) for the `final_seed_*` run tags specifically, to
   get a best-validated (not raw-final) number for the actual dissertation
   comparison — this is training-run-adjacent work and should be scoped and
   approved separately, not launched as part of this audit.
4. Only after (1)–(3): build the Section 9 true-regime PPO diagnostic, if
   the representation/optimisation split from (1) still leaves it
   informative.

## 9. Test commands and results

```
python -m pytest tests/ test_adverse_selection_identity.py test_hamilton_filter.py -q
# 209 passed, 3 warnings in 556.12s (0:09:16)

python aggregate_final_results.py
# pure re-aggregation of results/agent_comparison_final_all_seeds_episodes.csv, no new rollouts
```

No hyperparameters, code, or economic-model parameters were changed during
this audit.
