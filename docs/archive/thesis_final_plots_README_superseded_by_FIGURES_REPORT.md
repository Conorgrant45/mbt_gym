# Thesis final plots

Five self-contained, read-only plotting scripts for the thesis results
chapter, built on the already-saved outputs of the
`final_reduced_exploration_architecture_comparison` experiment
(`results/final_reduced_exploration_architecture_comparison/`). **None of
these scripts train, retrain, or re-evaluate any agent, and none modifies
any source CSV, checkpoint, or existing plot** — each one only reads
already-saved holdout results and writes exactly one new PNG into
`figures/`.

Every script is fully independent (no shared imports between them beyond
the standard scientific-Python stack: `pathlib`, `numpy`, `pandas`,
`matplotlib`). Each has its own `PROJECT_ROOT`, its own copy of the
colour/marker palette, and its own `CONFIGURATION` block at the top that
can be edited without touching any other script. Run any script directly
with `python <script_name>.py` from this folder (or any working directory
— all paths are derived from the hardcoded `PROJECT_ROOT`).

Output figures land in `figures/`, PNG only (300 DPI, `bbox_inches="tight"`),
one file per script.

---

## Script-by-script

### `01_training_budget_slope.py` — training-budget slope plot

- **Source data**: `final_holdout_seed_summary.csv` (per-seed holdout means,
  already aggregated over the 500 shared holdout paths, seeds
  290000–290499).
- **What it shows**: for each of the three architectures, a thin line per
  learner seed connecting that seed's paired (200,000-transition,
  1,000,000-transition) holdout-objective mean, plus a thick cross-seed
  mean line, plus optional horizontal reference lines for the oracle,
  belief-weighted, and frozen-clone benchmarks.
- **Editable config**: title/title-visibility, axis labels, figure size,
  font sizes, thin/mean line widths, marker sizes, legend location, output
  filename/DPI, `SHOW_FIGURE`, `CHECKPOINTS` (which two checkpoints to
  plot), `SHOW_BENCHMARK_LINES`/`BENCHMARKS_TO_SHOW`, and the shared
  colour/marker/label dictionaries at the top.
- **Statistical treatment**: no CI is computed here — this is a raw paired
  visualisation (each seed's own two checkpoint means, connected). The
  script asserts every learner seed has both checkpoints present before
  plotting a "paired" line for it.

### `02_paired_benchmark_forest.py` — paired benchmark forest plot

- **Source data**: `benchmark_paired_contrasts.csv` by default — an
  **already-computed, path-level paired** contrast (RL objective minus
  benchmark objective, differenced per shared `path_seed`, then averaged)
  with both a normal-theory CI and a 10,000-replicate paired bootstrap CI
  (see `final_analyze_contrasts.py` in the repo root for the original
  computation). If `RECOMPUTE_BOOTSTRAP_CI = True` is set, the script
  instead reads `final_holdout_episode_level.csv` directly and recomputes
  the paired bootstrap from scratch (same method, configurable
  `BOOTSTRAP_SEED`/`N_BOOTSTRAP_REPLICATIONS`/`CONFIDENCE_LEVEL`), as an
  independent cross-check.
- **What it shows**: one point + 95% CI whisker per (architecture, learner
  seed) against each of the three benchmarks (oracle, belief-weighted,
  frozen clone), as three side-by-side horizontal-forest panels, at the
  fixed 1,000,000-transition checkpoint (the experiment's pre-registered
  primary checkpoint — never a validation-selected one, precisely to avoid
  the checkpoint-selection "winner's curse" documented elsewhere in this
  project).
- **Editable config**: title/visibility, axis label, figure size, font
  sizes, marker/line/cap sizes, legend location, output filename/DPI,
  `SHOW_FIGURE`, `FIXED_CHECKPOINT`, `BENCHMARKS`, `CONFIDENCE_LEVEL`,
  `RECOMPUTE_BOOTSTRAP_CI`, `BOOTSTRAP_SEED`, `N_BOOTSTRAP_REPLICATIONS`,
  `GROUP_GAP` (vertical spacing between architecture blocks).
- **Statistical treatment**: strictly **paired** throughout — every
  difference is `(RL policy on path_seed p) - (benchmark on the SAME
  path_seed p)`, never an unpaired two-sample comparison of two 500-length
  vectors. No unpaired standard error is used anywhere in this script.

### `03_objective_inventory_scatter.py` — objective vs. risk-exposure scatter

- **Source data**: `final_holdout_seed_summary.csv`.
- **What it shows**: one point per learner seed per architecture (mean
  holdout objective vs. mean |inventory|, at the fixed 1,000,000-transition
  checkpoint), plus one clearly-labelled point each for oracle,
  belief-weighted, and frozen-clone. No confidence ellipses (only 5 seeds
  per architecture — an ellipse would overstate precision).
- **Editable config**: `X_METRIC` at the very top of the file — swap
  between `"mean_mean_abs_inventory"` (default) and
  `"mean_adverse_selection_loss"` to change what risk measure is plotted
  on the x-axis; the axis label updates automatically via
  `X_METRIC_LABELS` (or set `X_LABEL` manually to override). Also: title/
  visibility, y-axis label, figure size, font sizes, marker sizes, legend
  location, output filename/DPI, `SHOW_FIGURE`, `FIXED_CHECKPOINT`,
  per-benchmark label offsets (`BENCHMARK_LABEL_OFFSETS`, needed because
  the three analytical benchmarks sit almost on top of one another).
- **Statistical treatment**: none beyond plotting already-aggregated means;
  no CI/ellipse is drawn by design.

### `04_objective_ecdf.py` — episode-level objective ECDF

- **Source data**: `final_holdout_episode_level.csv` (the raw, one-row-per-
  episode file — required, since an ECDF cannot be approximated from any
  summary-statistic CSV).
- **What it shows**: for each architecture, a **seed-specific ECDF is
  computed for each of the 5 learner seeds separately** (500 episodes
  each), then the pointwise mean of those 5 ECDFs is plotted with a
  configurable band across seeds. Benchmarks (oracle, belief-weighted,
  frozen clone) get a single 500-episode ECDF each (no seed concept, no
  band). A vertical reference line at objective = 0 shows the probability
  of a loss-making episode directly.
- **Editable config**: title/visibility, axis labels, figure size, font
  sizes, line widths, band alpha, legend location, output filename/DPI,
  `SHOW_FIGURE`, `FIXED_CHECKPOINT`, `N_GRID_POINTS`, `X_AXIS_LIMITS`
  (`None` = auto full range; the data is heavy-tailed, so setting e.g.
  `(-150.0, 250.0)` gives a more readable central view for a thesis figure
  at the cost of hiding the extreme tails), `BAND_TYPE`
  (`"minmax"`/`"std"`/`"quantile"`) and its parameters.
- **Statistical treatment**: the 500 holdout paths are **shared** across
  all 5 learner seeds of a given architecture. This script never pools an
  architecture's 5 × 500 = 2,500 rows into one flat ECDF (that would treat
  5 correlated resamples of the same 500 exogenous paths as 2,500
  independent draws). It always computes one ECDF per learner seed first,
  and only combines *those 5 curves* (mean + band across seeds) —
  uncertainty shown in the band reflects seed-to-seed variation only.

### `05_inventory_time_profile.py` — inventory-over-time profile

- **Source data**: `trajectory_subset.csv` (event-by-event `time`,
  `inventory_after`, etc.).
- **Data adequacy — READ BEFORE USING IN THE THESIS**: this file is
  explicitly a small subset, confirmed against
  `final_holdout_evaluation_manifest.json`'s own
  `trajectory_subset_path_seeds`/`trajectory_subset_policies` fields. It
  contains only **5 episodes per policy** (`path_seed` 290000–290004),
  **only learner_seed = 0**, **only the 1,000,000-transition checkpoint**,
  and **only one** of the three analytical benchmarks (`belief_weighted` —
  `oracle` and `frozen_clone` are absent). The script does not fabricate,
  interpolate, or regenerate any of the missing data — it validates
  exactly what is present, plots only that, and prints (to stdout) plus
  annotates (in the figure caption) precisely which requested series were
  skipped and why. **Treat this figure as illustrative only** — with n=5
  episodes and a single learner seed, the shown band is not a statistically
  powered confidence interval, and this is stated directly in the figure's
  own caption and title.
  - What would be needed for a full version: a trajectory export logging
    every event's `(time, inventory_after)` across all 5 learner seeds and
    the full 500-path holdout for every architecture, plus `oracle` and
    `frozen_clone`. No such file currently exists in this repository; this
    script will pick it up automatically if one is added later with the
    same column names, since it re-derives which episodes are available
    from the data rather than hardcoding counts.
- **What it shows**: mean/median |inventory| against normalised time
  (t/T), evaluated on a common time grid using the **most recently
  observed inventory** at or before each grid point (a step function,
  matching the event-driven environment's genuinely piecewise-constant
  inventory) — never linear interpolation between events.
- **Editable config**: title/visibility, axis labels, figure size, font
  sizes, line width, band alpha, legend location, output filename/DPI,
  `SHOW_FIGURE`, `CHECKPOINT_TO_USE`/`LEARNER_SEED_TO_USE` (validated
  against what's actually in the data — raises a clear error listing
  available values if changed to something unavailable),
  `TIME_GRID_POINTS`, `TERMINAL_TIME`, `INITIAL_INVENTORY` (documented
  project-wide episode-start convention, not an assumption specific to
  this script), `CENTRAL_STATISTIC` (`"mean"`/`"median"`), `BAND_QUANTILES`.
- **Statistical treatment**: per-episode step functions are aggregated
  only *within* a fixed (policy, learner_seed, checkpoint) group across its
  available episodes — no seed- or path-pooling beyond what the source
  file already contains.

---

## Were plots 4 and 5 backed by sufficient data?

- **Plot 4 (ECDF)**: **Yes.** `final_holdout_episode_level.csv` contains
  the full 500-episode-per-(policy, learner_seed, checkpoint) raw
  distribution needed for a proper ECDF, for all three architectures, all
  5 learner seeds, both checkpoints, and all three benchmarks. No data gap.
- **Plot 5 (inventory-over-time)**: **Partially.** Genuine trajectory-level
  data exists (`trajectory_subset.csv`) and the script uses it as-is, but
  it is a small, explicitly-labelled subset (5 episodes, 1 learner seed, 1
  checkpoint, missing 2 of 3 benchmarks) rather than a full-coverage
  export. The script runs successfully and produces a real plot from real
  data, but the result should be captioned as illustrative in the thesis,
  not as a statistically powered result — see the script's own docstring
  and in-figure caveat text for the exact wording used.

## Colour/marker consistency

All five scripts use the same colour-blind-friendly (Okabe–Ito-derived)
palette and marker shapes for the three architectures, duplicated verbatim
in each script's configuration block (by design — no shared plotting
module, so each script can be edited and run in isolation):

| Series | Colour | Marker / linestyle |
|---|---|---|
| Hamilton belief-state MLP | `#0072B2` (blue) | `o` |
| Raw-return MLP | `#E69F00` (orange) | `s` |
| Raw-return LSTM | `#009E73` (bluish green) | `^` |
| Oracle (analytical) | `#000000` (black) | solid / `D` |
| Belief-weighted (analytical) | `#999999` (grey) | dashed / `X` |
| Frozen supervised clone | `#CC79A7` (reddish purple) | dotted / `P` |
