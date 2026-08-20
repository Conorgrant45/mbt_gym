# Figures report: fig01-fig08 / figA1-figA3

Companion report to the implementation brief. Summarises the §1 schema
reconnaissance, reconciliation discrepancies found, which optional compute
jobs were run (none, pending go-ahead), and headline numbers per figure.
All figures live in `figures/` as `<name>.pdf` + `<name>.png` (300dpi) +
`<name>_stats.json` (every number a caption might need). Read
`DATA_INVENTORY.md` first for the full data catalogue this all sits on top of.

## §1 schema reconnaissance — findings

- **`final_holdout_episode_level.csv`**: 16,500 rows = 3×5×2×500 (RL) +
  3×500 (benchmarks), exactly as expected. Path-identity confirmed: all 6
  policies × their seed/checkpoint groups share the identical 500
  `path_seed` values. `deterministic` is `True` for every row (this file
  only stores the deterministic pass; the stochastic companion pass is
  folded into `stochastic_full_objective`/`stochastic_action_std_bid/ask`/
  `near_bound_rate_stochastic` on the same row) — no filtering was actually
  needed for "deterministic rows only", but every script asserts it anyway.
- **Objective definition** (`final_episode_runner.py:106-111`, exact
  source): `raw_pnl = (cash_T + inv_T·mid_T) − (cash_0 + inv_0·mid_0)`;
  `full_objective = raw_pnl − running_penalty − terminal_penalty`. Verified
  numerically exact (≤1e-14) on a 2,000-row sample. Units: raw price/cash
  units of the midprice process (`INVENTORY_SCALE`/`RETURN_SCALE` only
  rescale policy *observations*, never the reward).
- **🛑 Blocking finding, resolved by explicit user decision**:
  `spread_revenue` and `adverse_selection_loss` do **not** sum with
  `running_penalty`/`terminal_penalty` to `full_objective`, or even to
  `raw_pnl` — `raw_pnl − (spread_revenue − adverse_selection_loss)` has
  mean 4.7, **std 68.6**, correlation 0.08 with `raw_pnl`. Root cause:
  `spread_revenue` only counts revenue at a fill, `adverse_selection_loss`
  only fires when a fill coincides with a regime-1 jump event — neither
  captures ordinary price-diffusion P&L on *held* inventory between trades,
  which dominates `raw_pnl`'s variance. **Decision (user-selected option
  "a")**: fig04 adds a fifth, explicitly-labelled component, "Unattributed
  mark-to-market" = `full_objective − (spread_revenue − adverse_selection_loss
  − running_penalty − terminal_penalty)`, so the five signed components sum
  to the total **by construction** (an identity, not an approximation) —
  verified in-script on every row (≤1e-6 tolerance) before any figure is drawn.
- **`validation_path_level.csv`**: 16,341 rows ≠ 3×5×21×50=15,750.
  Root cause: `hamilton_ppo` learner_seed=1's first 12 checkpoints
  (16,000-192,000) have exact duplicate rows (591 extra = 15,750+591=16,341
  ✓) from an apparently interrupted/rerun evaluation pass. Means unaffected;
  `n_paths`/`se_full_objective` in `validation_seed_summary.csv` for that
  one group are inflated/understated. Not used directly by any figure here
  (fig01 uses the pre-aggregated `validation_seed_summary.csv`'s mean, and
  fig01 uses cross-seed min-max bands, not this SE) but flagged for the record.
- **`action_diagnostics.csv`**: 348 = 3×115 (RL) + 3×1 (benchmarks); 115 =
  23×5 seeds. The 23 (not 21) is a real, non-bug composition: a `source`
  column (`"validation"`/`"holdout"`) — 21 validation-source rows (all
  checkpoints, 50-path set) plus 2 holdout-source rows (200k/1M, 500-path
  set) per (policy, seed). figA3 filters `source=='validation'` explicitly
  to avoid double-plotting 200k/1M.
- **Hamilton observation layout** (`event_driven_hamilton_ppo_wrapper.py:168-171`,
  code not assumption, used by S2): `[tanh(q/inventory_scale), 1 − t/T, belief]`.
- **Value-capture-fraction finding (fig03 stats sidecar)**: `mean(oracle −
  frozen_clone)` = **-0.036** (std 3.61) on the holdout set — the two
  benchmarks are statistically indistinguishable here, so the value-capture
  ratio's denominator is near-degenerate. The computed fractions
  (Hamilton ×102, MLP ×31, LSTM ×83, all with CIs spanning roughly ±370)
  are reported in the JSON exactly as computed but are **not interpretable**
  as "% of the analytical edge captured" for this holdout set — flagged
  here rather than presented as a clean number.

## Reconciliation cross-checks

- **fig03**: 48/48 recomputed per-seed means matched `benchmark_paired_contrasts.csv`
  (9 arch-vs-benchmark rows × 5 seeds) and `architecture_contrasts.csv`
  (3 arch-pair rows) within 1e-6. No CI-method disagreement is expected or
  claimed to be resolved — fig03's per-seed whiskers use a fresh percentile
  bootstrap (its own RNG stream, `stats_helpers.paired_path_bootstrap`),
  not the saved CSVs' bootstrap/normal-theory CIs; only the point estimates
  (means) were cross-checked, as specified.
- **figA2**: LSTM column-order repair (`scripts/fix_training_log_lstm_column_bug.py`)
  confirmed present in `training_log_long.csv` at runtime — spot-checked
  `return_lstm_ppo` seed 0 at t=4,000 against the pre-fix backup
  (`action_std_bid` 0.199→0.221 ≈ exp(-1.5); `actor_grad_norm_mean`
  3.87→0.41, and non-negative as a norm must be). Backup read only for this
  one-time verification, never for plotting.

## Optional compute jobs (S1, S2)

**Both run, on explicit go-ahead.** Before running, `fig07_policy_surface.py`
and `figA1_filter_calibration.py` (their consumers) were confirmed to fail
with a clear, instructive `FileNotFoundError` rather than fabricating data
when run before their upstream job — verified, not assumed.

- **S2** (`build_policy_surface_grid.py`): seconds of compute, as estimated.
  Loaded all 5 `hamilton_ppo` 1,000,000-transition checkpoints via
  `checkpoint_manifest.csv` (resolved to the original Phase 7 files, local
  `models/` dirs confirmed empty as documented). Reused
  `phase4_common.py`'s already-tested `hamilton_ppo_depths`/
  `analytical_belief_weighted_depths`/`build_analytical_controls` helpers
  directly rather than re-deriving the observation layout or action
  transform. Output: `policy_surface_grid.npz`.
- **S1** (`build_filter_calibration_dataset.py`): ~611s estimated from a
  20-path timing smoke-test (matches the brief's own "minutes of CPU"
  estimate), ran in the background. Seed range 300000-304999 (5,000 seeds)
  verified programmatically disjoint from all 20 prior seed ranges in
  `phase7_post_training_common.PRIOR_SEED_RANGES` plus this experiment's own
  validation/holdout ranges. Outputs: `filter_calibration_reliability_bins.csv`,
  `filter_calibration_detection_delays.csv`, `filter_calibration_summary.json`.

## Headline numbers per figure

- **fig01 (learning curves)**: all three architectures cluster in the
  low-30s validation objective across most of training, well below the
  analytical benchmarks (regime-conditioned ≈37.3, belief-weighted ≈38.2,
  frozen clone ≈38.2); Hamilton is the least noisy, MLP the most
  consistently below the others from ~16k transitions onward.
- **fig02 (holdout ECDF)**: the lower-tail panel (F≤0.10) shows the
  analytical benchmarks rising almost vertically around objective ≈-20 to 0,
  while all three RL architectures have materially fatter left tails — MLP
  worst, then LSTM, then Hamilton closest to the benchmarks.
- **fig03 (paired forest, recomputed)**: cross-seed means vs. benchmarks are
  all negative (Hamilton ≈-3.6 to -3.7; MLP ≈-1.0 to -1.1; LSTM ≈-2.9 to
  -3.0) but every 95% CI crosses zero — no architecture is statistically
  distinguishable from any benchmark, or from each other, at 5 seeds. Full
  per-row means/CIs in `fig03_paired_forest_stats.json`.
- **fig04 (component decomposition)**: every architecture shows
  significantly LOWER spread revenue and LOWER running/terminal penalties
  than belief-weighted, but a significantly POSITIVE "unattributed
  mark-to-market" term (largest for MLP, ≈+7) that roughly offsets the
  spread-revenue shortfall — net objective difference ends up statistically
  indistinguishable from zero (matches fig03). Adverse-selection-loss-per-fill
  (ratio of sums) is small and not clearly separated from zero for any architecture.
- **fig05 (risk-return)**: the three analytical benchmarks cluster tightly
  near CVaR₁₀≈-55 to -65, mean≈46.3. Hamilton's 5 seeds sit closest to that
  cluster (CVaR₁₀ -55 to -95); MLP is the most dispersed and worst-tailed
  (CVaR₁₀ down to -280) without a compensating higher mean.
- **fig06 (trajectory mechanism, single episode)**: the Hamilton filter's
  posterior tracks the true simulated regime almost exactly (belief jumps
  to the correct value essentially at every switch). Belief-state PPO's
  inventory path nearly overlaps the belief-weighted analytical policy's;
  raw-return MLP sits persistently short (≈-10) for most of the episode —
  visibly different, more directional inventory behaviour on the identical
  price path. Illustrative only (n=1 episode).
- **fig08 (budget contrast)**: full-objective change 200k→1M is positive
  for every architecture (Hamilton +3.1, MLP +6.9, LSTM +3.7) but every
  cross-seed t-interval (df=4) crosses zero. Every behavioural metric shown
  (fills, spread, spread revenue, adverse-selection loss) moves in a
  consistent, larger-magnitude direction across seeds — see
  `fig08_budget_contrast_stats.json` for per-metric per-seed detail.
- **figA2 (PPO diagnostics)**: Hamilton's v2 diagnostics log completed after
  this report was first drafted and is now included (all three
  architectures) — the script auto-detects completeness and footnotes it as
  a protocol-replication run. Hamilton and MLP track each other very
  closely on explained variance, mean action std, and approx_kl (both
  feedforward, same PPO hyperparameters); LSTM decays its action std more
  slowly (ends ≈0.15 vs ≈0.135 for Hamilton/MLP) and has a visibly lower
  actor grad norm than MLP for most of training, closer to Hamilton's.
  Actor grad norm trends upward for all three, MLP ending highest
  (≈1.0-1.4), Hamilton and LSTM lower (≈0.9-1.0).
  Note: Hamilton's v2 sanity-check gap vs. Phase 7's original (-14 to -25
  points, every seed) turned out LARGER than an earlier, since-superseded
  v1 attempt — root-caused exactly (not just observed) to Phase 7's
  original eval calling `torch.manual_seed()` 21 times/run vs. this
  experiment's own official protocol (which v2 correctly replicates to
  match MLP/LSTM) calling it 1,050 times/run. Not a bug; see
  `DATA_INVENTORY.md` §1 for the full explanation. Does not affect any
  other figure or the official results.
- **figA3 (exploration control)**: near-bound action rate stays at
  essentially 0.00 across all 21 checkpoints, both deterministic and
  stochastic passes, all three architectures — confirms action saturation
  is controlled at log_std_init=-1.5 throughout training.
- **fig07 (policy surface)**: Hamilton's learned bid/ask depth grid shows
  clean, textbook inventory-skew behaviour — bid depth increases and ask
  depth decreases with inventory (discouraging further accumulation,
  encouraging unwinding), gradient running almost purely along the
  inventory axis with comparatively weak belief-dependence. The skew
  becomes visibly steeper as tau→0.1 (near episode end), consistent with
  the terminal inventory penalty dominating late in the episode. The
  belief-weighted analytical policy's contours (white) track the same
  qualitative structure across all three tau values.
- **figA1 (filter calibration)**: Brier score **0.00157** over 1,400,687
  events / 5,000 fresh paths (seeds 300000-304999) — reliability diagram
  hugs the diagonal closely (slight overconfidence in the 0.6-0.7 predicted
  bin, otherwise tight). Detection is near-instantaneous: 98.1% of the
  28,049 detected regime switches are caught within 0 events, 100.0% within
  5 events; one outlier took 167 events (axis clipped at 15 for legibility,
  full value in the stats sidecar and the caption note on the plot).

## House-style notes for anyone extending this set

- Import `fig_style` and `stats_helpers` — never redefine the palette or a
  bootstrap routine locally.
- `fig_style.verify_policy_set(...)` right after loading any new source
  file's policy/architecture column.
- No `ax.set_title` with descriptive text anywhere — panel letters
  (`FS.panel_letter`) only; put the descriptive content in this report and
  the thesis caption, not baked into the figure.
- Legends: check placement against actual data extent before finalising
  (two figures in this set needed a legend-position fix after first render
  — `fig03`, `fig06` — because "loc=lower/upper right" silently overlapped
  data at the edges of a wide value range).
- Every figure's `_stats.json` should be sufficient, on its own, to write
  the thesis caption's numbers without re-opening the source CSVs.
