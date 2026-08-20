# Event-Driven Environment — Phase 1 Report

Branch: `event-driven-environment` (confirmed active before any changes were made).
No PPO training was launched. The fixed-step environment was not modified or
removed.

## 1. Files inspected

`envs/regime_env.py` (`RegimeSwitchingEnv`), `envs/make_envs.py`,
`envs/arrival_jump_midprice.py`, `mbt_gym/mbt_gym/gym/TradingEnvironment.py`,
`mbt_gym/mbt_gym/gym/ModelDynamics.py` (`LimitOrderModelDynamics`),
`mbt_gym/mbt_gym/stochastic_processes/arrival_models.py`
(`PoissonArrivalModel`), `mbt_gym/mbt_gym/stochastic_processes/
fill_probability_models.py` (`ExponentialFillFunction`),
`mbt_gym/mbt_gym/stochastic_processes/midprice_models.py`
(`BrownianMotionMidpriceModel`), `mbt_gym/mbt_gym/rewards/RewardFunctions.py`
(`PnL`, `RunningInventoryPenalty`), `mbt_gym/mbt_gym/gym/index_names.py`,
`simulate_belief_weighted.py` (`MAX_DEPTH`, `normalise_depth`, `NAIVE_DEPTH`),
`beliefs/hamilton_filter.py`, `compare_four_policies_paired.py`
(`instrument_env`, reused read-only), the existing test suite (`tests/` +
`test_adverse_selection_identity.py` + `test_hamilton_filter.py`).

## 2. Files created or changed

**Created:**
- `envs/event_driven_regime_env.py` — `EventDrivenRegimeSwitchingEnv`.
- `tests/test_event_driven_regime_env.py` — 26 tests (Section 5 items 1-20
  plus 6 supporting parity/constant checks).
- `equivalence_diagnostic_event_driven_vs_fixed_step.py` — Monte Carlo
  equivalence diagnostic.
- `results/equivalence_event_driven_vs_fixed_step.csv` — diagnostic output.
- This report.

**Changed (additive only, one line + one comment block):**
- `envs/make_envs.py` — added `TRANSITION_GENERATOR = _Q.copy()` immediately
  after the existing continuous-time generator `_Q` is finalised (before it
  is re-discretised into `TRANSITION_MATRIX`). No existing line, value, or
  behaviour was altered; this only exposes an already-computed local variable
  under a public name so the new environment does not re-derive it via a
  second `logm()` call. Verified: `expm(TRANSITION_GENERATOR * STEP_SIZE)`
  (renormalised) reproduces `TRANSITION_MATRIX` exactly.

**Not touched:** `envs/regime_env.py`, `envs/hamilton_ppo_wrapper.py`,
`envs/return_ppo_wrapper.py`, `beliefs/hamilton_filter.py`, any `mbt_gym/`
file, any training script.

## 3. Exact event ordering (confirmed from source, not inferred)

Read directly from `TradingEnvironment._update_state` →
`LimitOrderModelDynamics.get_arrivals_and_fills`/`update_state` →
stochastic-process `.update()` → `RunningInventoryPenalty.calculate`:

1. Arrivals and fills are drawn from the state as it stands **before** this
   step's market update (fills are independent Bernoulli(exp(-kappa·depth))
   draws using the chosen depths, independent of arrivals).
2. **Agent state (cash, inventory) is updated first**, using the
   **current, pre-diffusion, pre-jump** midprice: a bid fill executes at
   `midprice − bid_depth`; an ask fill executes at `midprice + ask_depth`.
3. **Market state (midprice) is updated second**: Brownian diffusion, then
   (regime 1 only) the arrival-triggered permanent jump. The fill price
   therefore never includes that same step's own jump.
4. Reward = `(cash_next + inv_next·price_next) − (cash_cur + inv_cur·price_cur)
   − dt·phi·inv_next² − alpha·is_terminal·inv_next²` — the running penalty
   uses the fill-updated *next* inventory scaled by that step's own `dt`, the
   terminal penalty (last step only) likewise, applied exactly once.

`EventDrivenRegimeSwitchingEnv` reproduces this ordering exactly, generalised
from a fixed `dt` to a variable elapsed interval Δτ: fills are computed and
applied at the pre-jump price, the jump is applied strictly after, and the
running penalty for an interval of constant inventory `Q` is exactly
`phi·Q²·Δτ` — exact, not approximate, because inventory is provably constant
between arrival events (only an arrival's own fill can change it, and the
interval's penalty is charged against the old, correct, constant inventory
before that fill is processed).

Also confirmed and preserved: `BID_INDEX=0`/`ASK_INDEX=1`; arrivals[BID] =
sell-MO arrival (can fill the agent's bid), arrivals[ASK] = buy-MO arrival
(can fill the agent's ask); jump sign = `+Exponential(epsilon)` on a buy
arrival, `−Exponential(epsilon)` on a sell arrival, **regime 0 has no jump
term at all**; jump magnitude is independent-exponential (not fixed) and
drawn from arrivals regardless of fill; `T=1`, `lambda_bid=lambda_ask=140`,
`kappa=1.5`, `EPSILON=0.5`, `PER_STEP_INVENTORY_AVERSION=0.01`,
`TERMINAL_INVENTORY_AVERSION=0.001` — all imported directly from
`envs/make_envs.py`, none redefined.

## 4. Competing-event simulation

From hidden regime `z`, four competing events race every iteration: buy MO
arrival (`Exponential(lambda_ask)`), sell MO arrival
(`Exponential(lambda_bid)`), regime switch (`Exponential(−Q[z,z])`; for this
2-state chain the only destination is `1−z`), and the deterministic horizon
(`terminal_time − current_time`). The earliest wins. A regime switch loops
internally (no RL step, same outstanding quotes); an arrival or the horizon
returns one transition to the agent.

Because Brownian increments over disjoint intervals sum to a Gaussian with
summed variances, the environment accumulates
`integrated_variance = Σ_j sigma_{z_j}² · Δτ_j` across every internal
sub-interval and draws **one** `Z ~ N(0,1)` per RL step
(`brownian_increment = sqrt(integrated_variance) · Z`) rather than one
Gaussian per sub-interval — statistically identical, cheaper. A single
`np.random.Generator` per environment instance drives every draw, in a fixed,
documented per-iteration order (see the module docstring), which is what
makes `test_identical_seeds_reproduce_identical_event_paths` meaningful.

## 5. Test commands and pass counts

```
python -m pytest tests/test_event_driven_regime_env.py -v
# 26 passed in 9.52s

python -m pytest tests/ test_adverse_selection_identity.py test_hamilton_filter.py -q
# 235 passed, 3 warnings in 481.22s (0:08:01)
#   (209 pre-existing + 26 new; 0 failures — item 20 satisfied)
```

All 20 required test categories from Section 5 are covered, plus: MAX_DEPTH
parity with `simulate_belief_weighted.MAX_DEPTH`, `denormalise_depth`
inverting `normalise_depth` exactly, the continuous generator's stationary
distribution matching the discrete chain's, and generator-row-sums-to-zero.

## 6. Monte Carlo equivalence diagnostic

`equivalence_diagnostic_event_driven_vs_fixed_step.py`, 500 independent
episodes per environment, shared naive constant policy (`SBW.NAIVE_DEPTH` on
both sides, same physical depth in both environments). Fixed-step episodes
took 364.7s (4000 steps × 500), event-driven episodes took 5.2s (~280 steps ×
500) — reflecting the intended efficiency gain of decision-only-at-arrivals.

| metric | fixed-step (mean ± se) | event-driven (mean ± se) | z |
|---|---|---|---|
| n_buy_arrivals | 140.586 ± 0.508 | 140.402 ± 0.514 | −0.25 |
| n_sell_arrivals | 140.330 ± 0.515 | 140.892 ± 0.527 | +0.76 |
| n_fills | 71.122 ± 0.364 | 70.754 ± 0.379 | −0.70 |
| time_in_regime_1 | 0.7186 ± 0.0068 | 0.7150 ± 0.0074 | −0.36 |
| terminal_price | 99.181 ± 0.442 | 99.652 ± 0.455 | +0.74 |
| running_penalty | 0.3600 ± 0.0182 | 0.3590 ± 0.0178 | −0.04 |
| terminal_inventory | −0.110 ± 0.384 | 0.342 ± 0.379 | +0.84 |
| raw_pnl | 43.520 ± 2.711 | 41.761 ± 2.612 | −0.47 |
| full_objective | 43.086 ± 2.714 | 41.331 ± 2.619 | −0.47 |

No metric exceeded the pre-registered `|z| > 3` flag threshold — every
comparison is within `|z| < 1`. Theoretical stationary `P(regime=1) =
0.7143`; both environments' empirical time-in-regime-1 (0.7186 fixed-step,
0.7150 event-driven) land within a few tenths of a percentage point of that
value, on either side, with no systematic bias in either direction. As a
cross-check, both `full_objective` figures (43.09, 41.33) land close to the
naive-policy figure independently established in the prior RL architecture
audit (mean 42.99 over a different, larger holdout set) — consistent, not
just internally.

## 7. Discrepancies from the fixed-step model

None material. All nine compared statistics are statistically
indistinguishable at conventional confidence levels. No sign errors, no
systematic drift, no discretisation bias large enough to register against
Monte Carlo noise at n=500 episodes per side.

## 8. Assumptions requiring confirmation

1. **Fill-suppression at `max_inventory`.** The fixed-step environment
   suppresses a fill that would breach `max_inventory`/`-max_inventory`
   (`TradingEnvironment._remove_max_inventory_fills`). This environment
   replicates that exact guard (same default bound, 10,000), but at the
   realised inventory scales in this project (single digits to low tens) it
   never binds in either environment — not exercised by the equivalence
   diagnostic. Flagged, not independently stress-tested.
2. **RNG architecture is intentionally different, not equivalent.** The
   fixed-step environment uses several independently-seeded sub-process
   RNGs (midprice/arrival/fill/regime, each `SeedSequence`-spawned); this new
   environment uses one `Generator` per instance. This is a deliberate
   design choice for a genuinely new environment (documented in the module
   docstring) rather than an attempt to replicate the old architecture, and
   does not affect the ordering/economic-model guarantees above — but it
   means "same seed" does **not** mean "same episode" across the two
   environments, only within each one. Confirm this is the intended scope
   for Phase 1 (task did not ask for pathwise-identical seeding across the
   two environments, only distributional agreement, which is what Section 6
   verifies).
3. **Regime-occupation test (item 7) uses a documented small-bias proxy.**
   Because arrivals (rate 280) are far more frequent than regime switches
   (rate ≈10.4 / ≈4.1) at the real calibration, attributing each RL step's
   `elapsed_time` to `regime_at_event` (rather than splitting it across
   whichever regimes were actually active during that step's internal
   sub-intervals) is a very good but not exact proxy for true occupation
   time. `tests/test_event_driven_regime_env.py::test_regime_occupation_matches_ctmc_stationary_distribution`
   uses a 0.05 absolute tolerance to accommodate this; the equivalence
   diagnostic's `time_in_regime_1` metric (which uses the fixed-step
   environment's own exact per-tick regime label) does not have this bias
   and independently confirms occupation matches theory.
4. **Two-state-chain shortcut.** The regime-switch destination is hardcoded
   as `1 - z` (embedded-chain sampling is trivial for exactly two states).
   If a future phase generalises to more than two regimes, this line (not
   the waiting-time logic) needs to become a proper categorical draw over
   off-diagonal rates.

## 9. Is Phase 1 safe to proceed to agent integration?

**Yes**, with the above assumptions logged for Phase 2 to pick up. All
invariants required by the audit brief pass: exact termination at `T`, no
agent-visible steps from hidden switches, quotes held fixed through internal
switches, correct exponential timing and buy/sell proportions, correct mean
arrival count (~280), correct regime occupation, correct Brownian variance
scaling, independent arrival/fill draws, jump tied to arrival not fill with
correct mean and sign, exact running/terminal penalty accounting, exact
reward-decomposition identity, correct bid/ask cash/inventory signs, seed
reproducibility (both identical- and different-seed behaviour), no regime
leakage into the observation, and full backward compatibility (235/235 tests
green, including all 209 pre-existing ones). The Monte Carlo equivalence
diagnostic found no discrepancy between the two environments large enough to
warrant investigation before Phase 2.

Recommended next step: build the Phase 2 wrappers (event-time Hamilton
filter/likelihood, belief and raw-return observations for the event-driven
environment) only after this report is reviewed — per the task brief, no
training, hyperparameter change, or filter change was made in this phase.
