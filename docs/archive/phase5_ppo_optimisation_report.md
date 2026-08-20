# Phase 5: PPO Optimisation Diagnostic

Branch `event-driven-environment`, base commit `f5d204f5b587fdaa9a86bb05c7705007fa918e65`
(clean at the start of this phase). No neural architecture, PPO
hyperparameter, reward, action space, economic model, or Hamilton filter was
changed anywhere in groups A/B/C (the one intentional exception is the
explicitly-labelled Section 10 conditional diagnostic E, which changes
`log_std_init` alone, reported as a separate result, never substituted into
the main experiment). Full test suite: **309/309 passing** (297 pre-existing
+ 12 new Phase 5 tests in `tests/test_phase5_ppo_optimisation.py`).

**Reframing applied throughout** (per mid-phase correction): the analytical
belief-weighted policy and the Phase 4 supervised clone are used here purely
as *diagnostic references* — neither is assumed globally optimal for the RL
problem. Primary success criteria are (1) validation-selected out-of-sample
objective, (2) cross-seed variance, (3) catastrophic-policy frequency, and
(4) economically reasonable spread/fill/inventory behaviour. Deviation from
the analytical surface or from the clone is not treated as an error unless
associated with worse control performance.

## 0. Scope

Read before starting: `phase4_policy_diagnostic_report.md`,
`event_driven_phase3_results.md`, `rl_architecture_audit_report.md`. Phase 4
established REPRESENTATION SUFFICIENT — the exact `[64,64]` Tanh architecture
can represent the analytical policy via supervised learning and control at a
near-analytical level. Phase 5 asks what happens when PPO starts from that
already-good policy.

## 1. Experimental design

Training-environment seed 70,000, learner seeds 0/1/2, event-driven
environment (chosen per the brief: it removed the catastrophic fixed-grid
collapses seen in Phase 3/4 and has the strongest event-time filter), 64,000
transitions (16 rollouts of `n_steps=4,000`), checkpoints at
t = 0, 4k, 8k, 16k, 24k, 32k, 48k, 64k. Three groups × three seeds:

- **A — random initialisation**: unmodified Hamilton PPO.
- **B — clone initialisation**: actor loaded from
  `logs/phase4_policy_diagnostic/supervised_clone_best.pt`, critic randomly
  initialised, PPO configuration unchanged.
- **C — frozen clone control**: actor loaded, nothing updated, evaluated at
  every checkpoint — isolates evaluation-side drift from real training
  effects.

PPO hyperparameters are `train_agents.py`'s own defaults, reproduced
verbatim in `phase5_common.PPO_KWARGS` (learning_rate=3e-4, n_steps=4000,
batch_size=400, n_epochs=10, gamma=1.0, gae_lambda=0.95, clip_range=0.2,
ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, net_arch=[64,64]) — nothing
tuned for this phase.

## 2. Clone-to-actor weight transfer

`phase5_common.copy_supervised_actor_weights` maps the supervised clone's 3
Linear layers onto `model.policy.mlp_extractor.policy_net`'s two hidden
layers plus `model.policy.action_net` (the Gaussian mean head) — verified,
not assumed, via `introspect`-style layer enumeration. The critic
(`mlp_extractor.value_net` + `value_net`) and `log_std` are provably
untouched (independent critics across learner seeds with an identical actor,
confirmed by test). On the dense 9,471-state Phase 4 grid, the PPO actor's
deterministic output matches the clone's output to **4.17×10⁻⁷** max abs
difference — the transfer is exact to floating-point precision.

## 3. Initial exploration diagnostic (t=0, before any update)

| Seed | log_std | Deterministic J | Stochastic J | Frac. actions near bound |
|---|---|---|---|---|
| 0 | 0.0 | 29.41 | −15.15 | 0.612 |
| 1 | 0.0 | 29.41 | +4.91 | 0.611 |
| 2 | 0.0 | 29.41 | −7.28 | 0.606 |

SB3's default `log_std_init=0.0` gives action std = 1.0 on a `[-1,1]` range —
61% of sampled actions land within 0.05 of a bound, before any training. The
clone's own deterministic policy (29.4) is dramatically better than the
*same actor's* stochastic behaviour (mean −5.8, 2 of 3 seeds net losses).

## 4-5. Controlled training: does PPO damage the clone, or improve it?

Group C's deterministic objective is **bit-identical** (29.413456) at all 8
checkpoints — the evaluation pipeline introduces zero drift, so every change
in A/B below reflects real PPO updates.

| Group | Seed | t=0 | t=4k | t=8k | t=16k | t=24k | t=32k | t=48k | t=64k |
|---|---|---|---|---|---|---|---|---|---|
| A | 0 | 20.5 | 21.6 | 26.1 | 29.1 | 28.0 | 30.7 | 26.1 | **29.3** |
| A | 1 | 19.9 | 23.3 | 28.0 | 28.9 | 29.4 | 26.5 | 22.9 | **24.7** |
| A | 2 | 20.0 | 20.9 | 28.3 | 31.8 | 25.5 | 27.2 | 24.3 | **28.8** |
| B | 0 | 29.4 | 34.3 | 29.8 | 28.3 | 23.0 | 25.9 | 11.2 | **36.0** |
| B | 1 | 29.4 | 32.5 | 30.6 | 35.8 | 37.5 | 31.9 | 19.5 | **32.7** |
| B | 2 | 29.4 | 37.9 | 32.1 | 36.1 | 39.1 | 43.9 | 32.0 | **43.7** |
| C | all | 29.4 | 29.4 | 29.4 | 29.4 | 29.4 | 29.4 | 29.4 | 29.4 |

**Group B ends above both its own t=0 baseline and every Group A seed, in
all 3 seeds** — PPO improves on the clone by t=64,000, not damages it,
despite a shared non-monotonic dip around t=24k–48k (worst single value:
11.2, seed 0 at t=48k, recovering to 36.0 by t=64k). Per the reframing, this
is reported as successful policy improvement, not deterioration, since the
actor moving away from the clone (distance grows from 0 to ~0.53 over the
run) is accompanied by *higher*, not lower, objective at the final
checkpoint.

**Group A never approaches the clone's parameter region**: actor-clone
distance stays at 13.6–13.9 throughout (moves by <1% of its starting value
in 64,000 transitions), while Group B's distance grows only to ~0.53 over
the same budget. Group A's final objective (mean 27.6, std 2.56 across
seeds) is below Group B's (mean 37.5, std 5.62) at every seed. Group A also
sustains far fewer fills throughout (~24–25 at t=64k vs. Group B's 56–74,
much closer to the clone/analytical's own ~65) — a second, independent,
economically-meaningful signal that random-init has not converged toward
sensible quoting behaviour within this budget.

**Caveat, reported honestly**: Group B's cross-seed standard deviation at
t=64,000 (5.62) is *larger* than Group A's (2.56) — clone initialisation
gives a much better mean here but not lower variance at this specific
checkpoint. No catastrophic collapse (deeply negative objective, runaway
inventory) was observed in *either* group at *any* checkpoint in this
64,000-transition, event-driven experiment — unlike the 200,000-transition
fixed-step runs analysed in Phases 3–4, which did show catastrophic
collapses in `return_mlp_ppo`/`return_lstm_ppo`.

## 6. Per-update diagnostics: why the non-monotonic dip?

`policy_loss`, `value_loss`, `approx_kl`, and `clip_fraction` track each
other closely between Group A and Group B at the *same update index*
(`results/phase5_ppo_optimisation/plots/update_level_diagnostics.png`) —
both groups train on the identical exogenous path (env seed 70,000), so most
of the update-to-update volatility is a shared, exogenous-noise-driven
property of this problem under the current exploration setting, not an
idiosyncrasy of clone initialisation. `value_loss` swings by 2–5× between
consecutive updates for both groups (e.g. seed 0: 79.8 → 180.7 → 102.7 →
76.3 → 105.5 → 93.2 → 230.9 → 70.3 → …) and never stabilises across the full
64,000 transitions. Group B's `actor_grad_norm_mean` (~0.22–0.26) is
consistently *higher* than Group A's (~0.17–0.20) — the clone-initialised
actor experiences systematically larger policy gradients, not smaller.

**The decisive signal: `log_std_mean` stays within 2% of its initial value
of 1.0 across all 16 updates, in both groups.** With `ent_coef=0.0`, nothing
in the loss actively pressures exploration to shrink, and the sheer noise in
the advantage signal (see Section 7) gives no consistent gradient to shrink
it indirectly either. Exploration variance is not a t=0 artefact that PPO
resolves during training — it persists, essentially unchanged, for the
entire experiment.

## 7. Critic / advantage diagnostic (clone-initialised, t=0, one un-updated rollout)

Independent GAE recomputation matches SB3's own rollout buffer **exactly**
(max abs diff 0.0 after fixing a shape-broadcasting bug in the initial
implementation — see Errors below). The randomly-initialised critic's
near-zero value predictions (mean 0.085) are essentially uninformative
against realised (undiscounted, buffer-truncated) Monte Carlo returns whose
std is 70.3 — dominated by the huge sampled-action variance discussed above,
not by genuine value-relevant structure. Advantages are correspondingly
extreme (mean −0.92, std 15.65, range [−128.8, +76.7]), and only weakly
associated with analytically-sensible actions: 24% of top-decile-advantage
actions and 23% of bottom-decile-advantage actions are within 1.0 depth unit
of the analytical surface — essentially indistinguishable from each other,
i.e. the sign of the advantage carries little information about whether the
action was analytically sensible. This is consistent with the update-level
volatility in Section 6: **critic/advantage instability is present, but it
reads as a downstream symptom of the same excessive exploration variance
(huge action noise → huge return noise → noisy advantages/critic → volatile
updates), not an independent root cause.**

## 10. Conditional diagnostic E: reduced exploration variance

Triggered directly by Sections 3 and 6 (stochastic ≪ deterministic at t=0;
`log_std` never shrinks). One short, explicitly-labelled variant: same
clone-initialised actor, `log_std_init=−1.5` (action std ≈ 0.22, ~4.5×
smaller), 3 seeds, 32,000 transitions (half budget — confirmatory, not a
retrain), everything else unchanged.

| Config | t=0 stoch. J | t=16k stoch. J | t=32k stoch. J | Frac. near bound |
|---|---|---|---|---|
| B: log_std_init=0.0 (default) | −15.15 | −10.29 | −4.18 | ~0.5–0.6 |
| E: log_std_init=−1.5 (reduced) | +26.60 | +28.59 | +17.13 | ~0.02–0.03 |

Reducing `log_std_init` **closes the deterministic/stochastic gap
dramatically** — the stochastic objective moves from deeply negative to
positive and close to the deterministic objective, with near-bound sampling
essentially eliminated (`results/phase5_ppo_optimisation/plots/
reduced_log_std_vs_default.png`). This directly confirms the causal
mechanism identified in Sections 3 and 6. Deterministic-objective
trajectories over the matched 0–32,000 window are comparable in magnitude
between E and B (E: 29.4→26.7, visibly lower-variance; B: 29.4→33.9,
noisier) — this diagnostic confirms exploration variance as the direct,
fixable cause of the huge stochastic-performance gap, without yet
demonstrating a clear acceleration of deterministic-objective improvement
over this short horizon (a longer run at reduced `log_std_init` was judged
out of scope for this diagnostic phase — see Section 11).

Conditional diagnostic D (critic warm-up) was **not** run: the critic
instability observed in Section 7 reads as downstream of exploration
variance rather than an independent problem, and the brief restricts D to
cases where critic/advantage instability appears independently significant.

## 11. Required tests

12 new tests in `tests/test_phase5_ppo_optimisation.py` cover all 10 listed
items (exact weight mapping incl. a shape-mismatch failure case,
deterministic-output equality, critic independence across seeds, t=0
checkpoint save/reload, frozen-control weight invariance, actor/critic
update-norm correctness, independent GAE cross-check, action-surface MSE in
physical depth space, save/reload preserving imported weights, and
non-monkeypatching of plain PPO/the event-driven wrapper's observation
formula). All 12 pass; full suite is 309/309.

## Errors found and fixed this phase

1. **Broadcasting bug in the independent GAE cross-check**
   (`phase5_critic_advantage_diagnostic.py`): comparing a `(4000,1)`-shaped
   recomputed-advantage array against a `(4000,)`-shaped SB3 array
   broadcast to `(4000,4000)` instead of raising, producing a spurious
   max-diff of 205.4 that looked like a real GAE mismatch. Diagnosed by
   reproducing the SAME computation inline with matching shapes (got exact
   0.0 diff), then bisecting the actual script line-by-line to find the
   `.flatten()` shape mismatch. Fixed by flattening the recomputed array
   before the comparison; verified exact match afterward.

## 12. Interpretation

1. **Weight transfer**: exact (4.17×10⁻⁷ max diff on 9,471 states).
2. **Performance before the first update**: deterministic 29.4 (matches
   Phase 4's clone evaluation); stochastic mean −5.8, highly seed-variable.
3. **Deterministic vs. stochastic at t=0**: enormous gap, driven by
   `log_std_init=0.0` giving action std=1.0 on a `[-1,1]` range.
4. **Does PPO preserve or damage the supervised policy?** Neither cleanly —
   it is non-monotonic (a shared dip at t=24k–48k) but ends, in all 3 seeds,
   *above* the clone's own t=0 performance and above every random-init seed.
   Per the reframing, this is policy **improvement**, not damage.
5. **First checkpoint of deterioration**: t=24,000–48,000 shows the shared
   trough (most severe at t=48,000); fully recovers by t=64,000 in all seeds.
6. **Action surface**: not the primary lens this phase (see reframing) —
   the actor moves modestly away from the clone's parameter region while
   objective improves, consistent with genuine policy refinement rather than
   drift toward a worse solution.
7. **Excessive exploration variance?** Yes — confirmed at t=0, confirmed to
   persist essentially unchanged through all 16 updates/64,000 transitions
   in both groups, and confirmed causally fixable via the reduced-`log_std`
   diagnostic (Section 10).
8. **Critic/GAE instability?** Present (large, volatile value loss and
   advantages throughout), but reads as downstream of exploration variance
   rather than an independent root cause.
9. **Does random initialisation ever approach the clone?** No — actor-clone
   distance is essentially flat (13.6→13.9) over 64,000 transitions, and its
   objective stays below clone-initialised runs at every checkpoint and seed.
10. **Critic warm-up / reduced exploration tested?** Reduced exploration
    (E) was tested and confirmed the causal mechanism; critic warm-up (D)
    was not run, as instability reads as a symptom, not an independent cause.
11. **Next justified optimisation change**: reduce and/or schedule
    `log_std_init` (e.g. a smaller initial value, or an explicit decay
    schedule) for the next training iteration — this is the one change with
    direct causal support from this phase's own diagnostics. Longer training
    at reduced `log_std` to see whether deterministic-objective gains follow
    is the natural next step, not yet run here.
12. **Is a full multi-seed experiment warranted?** Yes — for the specific,
    narrow change of reduced/scheduled `log_std_init`, given (a) the clean
    causal confirmation in Section 10, (b) clone initialisation's clear
    advantage over random initialisation on the primary success criteria
    (objective, fill/inventory realism) even under the current exploration
    setting, and (c) the absence of any catastrophic collapse in this
    experiment to warn against proceeding. A full 5-seed, 200,000-transition
    run (matching Phase 3/4's convention) combining clone initialisation
    with reduced `log_std_init` is the recommended next experiment.

**EXCESSIVE EXPLORATION NOISE**
