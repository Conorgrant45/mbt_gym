"""
fig03_paired_forest.py
-------------------------
Paired forest plot at the fixed 1,000,000-transition holdout checkpoint
(500 shared paths, deterministic evaluation), RECOMPUTED directly from
final_holdout_episode_level.csv (not read from any pre-aggregated contrast
CSV -- those are used only as an independent cross-check on the point
estimates, see below). Answers one question: are the estimated differences
in holdout objective distinguishable from zero.

Six rows, two groups:
  Against belief-weighted analytical control:
    1. Belief-state PPO  - belief-weighted analytical
    2. MLP PPO           - belief-weighted analytical
    3. LSTM PPO          - belief-weighted analytical
  Between learned policies:
    4. Belief-state PPO  - MLP PPO
    5. Belief-state PPO  - LSTM PPO
    6. LSTM PPO          - MLP PPO

Each row shows only a cross-seed mean (filled diamond) and its
hierarchical-bootstrap 95% CI (horizontal line) -- no per-seed dots/whiskers.
Differences are always "first policy - comparator" (row 1-3: RL - benchmark;
row 4-6: first-named architecture - second-named architecture); negative
values favour the comparator named second.

*** Seed-pairing note (rows 4-6, "between learned policies") ***
Learner seed i of one architecture and learner seed i of another are NOT
matched experimental units -- they are independently-initialised,
independently-trained runs that merely share a seed LABEL. There is no
causal or experimental link between "hamilton_ppo seed 2" and
"return_mlp_ppo seed 2" beyond that label. Treating them as matched (as an
earlier version of this script did, following architecture_contrasts.csv's
own "matched-seed" convention) lets each architecture's seed-level noise
spuriously cancel in the bootstrap, understating the true uncertainty.

This version's hierarchical bootstrap instead resamples each architecture's
5 learner seeds INDEPENDENTLY (via stats_helpers.hierarchical_bootstrap's
per-key resampling: arch_seeds={"a": [...], "b": [...]}, two separate keys,
each drawn with its own independent rng.choice call every replicate) while
still resampling the 500 holdout PATHS ONCE per replicate and applying that
same path draw to both architectures -- this is what "pathwise pairing" is
preserved for: path p is the same exogenous market realisation regardless
of which policy is evaluated on it, so path-level (not seed-level) pairing
is the real, experimentally-justified pairing in this design.
Note the POINT ESTIMATE is unaffected by this choice -- for a balanced
design (equal seeds, equal paths per seed on both sides), mean(A) - mean(B)
equals the average of the 5 "matched" per-seed differences algebraically;
only the confidence interval widens to reflect the added, honest
uncertainty of two independent samples rather than one artificially-paired
one. Data ARE sufficient to do this correctly (full per-(architecture,
seed, path) values are available for both architectures in every row); no
methodological shortfall to report.

CONFIGURATION block below is safe to edit.
"""
import time

import numpy as np
import pandas as pd

import fig_style as FS
import stats_helpers as SH

# ======================================================================
# CONFIGURATION
# ======================================================================
FIG_NAME = "fig03_paired_forest"
FIXED_CHECKPOINT = 1_000_000
B_HIER = 10_000
BOOTSTRAP_SEED = 20260101
CROSSCHECK_TOL = 1e-6

X_LABEL = "Mean objective difference"
FOREST_COLOR = "#0072B2"  # thesis-wide policy blue -- one colour for every estimate/CI in this figure

INPUT_EPISODE_CSV = FS.RESULTS_DIR / "final_holdout_episode_level.csv"
INPUT_BENCH_CONTRASTS_CSV = FS.RESULTS_DIR / "benchmark_paired_contrasts.csv"
INPUT_ARCH_CONTRASTS_CSV = FS.RESULTS_DIR / "architecture_contrasts.csv"

POLICY_LABELS = {"hamilton_ppo": "Belief-state PPO", "return_mlp_ppo": "Raw-return MLP PPO",
                  "return_lstm_ppo": "Raw-return LSTM PPO", "belief_weighted": "Belief-weighted analytical policy"}

# (first, comparator, group) -- row order and sign convention exactly as specified.
ROW_SPECS = [
    ("hamilton_ppo", "belief_weighted", "benchmark"),
    ("return_mlp_ppo", "belief_weighted", "benchmark"),
    ("return_lstm_ppo", "belief_weighted", "benchmark"),
    ("hamilton_ppo", "return_mlp_ppo", "pair"),
    ("hamilton_ppo", "return_lstm_ppo", "pair"),
    ("return_lstm_ppo", "return_mlp_ppo", "pair"),
]
GROUP_LABELS = {"benchmark": "Against the belief-weighted analytical policy", "pair": "Between learned policies"}


def _load_value_matrices(df: pd.DataFrame):
    fixed = df[(df["checkpoint_transition"] == FIXED_CHECKPOINT) & (df["deterministic"] == True)]  # noqa: E712
    bench = df[df["policy"].isin(FS.BENCHMARK_POLICIES) & df["checkpoint_transition"].isna()
               & (df["deterministic"] == True)]  # noqa: E712

    path_order = np.array(sorted(fixed["path_seed"].unique()))
    for arch in FS.RL_POLICIES:
        for seed in range(5):
            paths = sorted(fixed[(fixed["policy"] == arch) & (fixed["learner_seed"] == seed)]["path_seed"])
            if paths != list(path_order):
                raise ValueError(f"{arch!r} seed {seed}: path_seed set does not match the common path_order")
    for b in FS.BENCHMARK_POLICIES:
        paths = sorted(bench[bench["policy"] == b]["path_seed"])
        if paths != list(path_order):
            raise ValueError(f"benchmark {b!r}: path_seed set does not match the common path_order")

    arch_vals = {}
    for arch in FS.RL_POLICIES:
        arch_vals[arch] = {}
        for seed in range(5):
            sub = fixed[(fixed["policy"] == arch) & (fixed["learner_seed"] == seed)].set_index("path_seed")
            arch_vals[arch][seed] = sub.loc[path_order, "full_objective"].to_numpy()

    bench_vals = {}
    for b in FS.BENCHMARK_POLICIES:
        sub = bench[bench["policy"] == b].set_index("path_seed")
        bench_vals[b] = sub.loc[path_order, "full_objective"].to_numpy()

    return path_order, arch_vals, bench_vals


def _crosscheck(label, mine, theirs, tol, report):
    diff = abs(mine - theirs)
    ok = diff < tol
    report.append(dict(row=label, mine=float(mine), reference=float(theirs), abs_diff=float(diff),
                        within_tol=bool(ok)))
    if not ok:
        print(f"  [CROSS-CHECK MISMATCH] {label}: mine={mine:.6f} reference={theirs:.6f} diff={diff:.2e}")


def main():
    t0 = time.time()
    df = pd.read_csv(INPUT_EPISODE_CSV)
    FS.verify_policy_set(set(df["policy"].unique()), context="fig03 input")
    path_order, arch_vals, bench_vals = _load_value_matrices(df)
    n_paths = len(path_order)
    print(f"Loaded {n_paths} shared holdout paths.")

    bench_contrasts = pd.read_csv(INPUT_BENCH_CONTRASTS_CSV)
    bench_contrasts = bench_contrasts[bench_contrasts["metric"] == "full_objective"]
    arch_contrasts = pd.read_csv(INPUT_ARCH_CONTRASTS_CSV)
    arch_contrasts = arch_contrasts[arch_contrasts["metric"] == "full_objective"]

    rng_counter = [0]

    def next_seed():
        rng_counter[0] += 1
        return BOOTSTRAP_SEED + rng_counter[0]

    crosscheck_report = []
    rows = []  # each: dict(label, group, cross_mean, cross_ci)

    for first, comparator, group in ROW_SPECS:
        if group == "benchmark":
            arch, bench = first, comparator

            # Point-estimate cross-check against the saved per-seed contrast CSV (mean of the 5
            # per-seed paired means) -- independent of the bootstrap/CI method used here.
            per_seed_means = [float((arch_vals[arch][s] - bench_vals[bench]).mean()) for s in range(5)]
            for s, m in enumerate(per_seed_means):
                ref_row = bench_contrasts[(bench_contrasts["architecture"] == arch)
                                           & (bench_contrasts["learner_seed"] == s)
                                           & (bench_contrasts["benchmark"] == bench)]
                if len(ref_row):
                    _crosscheck(f"{arch} vs {bench} seed {s} (mean)", m,
                                float(ref_row["paired_mean_diff"].iloc[0]), CROSSCHECK_TOL, crosscheck_report)

            def compute_stat(path_idx, seed_choice, arch=arch, bench=bench):
                vals = [(arch_vals[arch][s][path_idx] - bench_vals[bench][path_idx]).mean()
                        for s in seed_choice["a"]]
                return float(np.mean(vals))

            hier = SH.hierarchical_bootstrap(compute_stat, n_paths, {"a": list(range(5))},
                                              B=B_HIER, seed=next_seed())
            label = f"{POLICY_LABELS[arch]} - {POLICY_LABELS[bench]}"

        else:  # group == "pair" -- independent per-architecture seed resampling, see module docstring
            a, b = first, comparator

            # Point estimate: algebraically identical to the average of 5 "matched" per-seed
            # differences in this balanced design (equal seeds/paths both sides) -- still a valid
            # cross-check of the NUMBER even though the CI below deliberately does not assume
            # seed-level matching.
            mean_a = float(np.mean([arch_vals[a][s].mean() for s in range(5)]))
            mean_b = float(np.mean([arch_vals[b][s].mean() for s in range(5)]))
            point_estimate = mean_a - mean_b
            ref_row = arch_contrasts[(arch_contrasts["architecture_a"] == a) & (arch_contrasts["architecture_b"] == b)]
            if len(ref_row):
                _crosscheck(f"{a} vs {b} (cross-seed mean, point estimate only -- CI method differs by design)",
                            point_estimate, float(ref_row["mean_matched_seed_diff"].iloc[0]),
                            CROSSCHECK_TOL, crosscheck_report)

            def compute_stat(path_idx, seed_choice, a=a, b=b):
                a_vals = [arch_vals[a][s][path_idx].mean() for s in seed_choice["a"]]
                b_vals = [arch_vals[b][s][path_idx].mean() for s in seed_choice["b"]]
                return float(np.mean(a_vals) - np.mean(b_vals))

            hier = SH.hierarchical_bootstrap(compute_stat, n_paths, {"a": list(range(5)), "b": list(range(5))},
                                              B=B_HIER, seed=next_seed())
            label = f"{POLICY_LABELS[a]} - {POLICY_LABELS[b]}"

        rows.append(dict(label=label, group=group, cross_mean=hier["point"],
                          cross_ci=(hier["ci_lo"], hier["ci_hi"])))
        print(f"  {label}: mean={hier['point']:.3f} CI=[{hier['ci_lo']:.3f},{hier['ci_hi']:.3f}] "
              f"[{time.time()-t0:.1f}s]")

    n_mismatch = sum(1 for r in crosscheck_report if not r["within_tol"])
    print(f"Cross-check: {len(crosscheck_report)} comparisons, {n_mismatch} exceeded {CROSSCHECK_TOL} tolerance.")

    # ------------------------------------------------------------------
    # Plot -- six rows, two groups, diamond + CI only
    # ------------------------------------------------------------------
    ROW_STEP = 1.0
    GROUP_GAP = 0.9
    TOP_MARGIN = 0.85

    y_positions = []
    y = 0.0
    prev_group = None
    group_header_y = {}
    for i, r in enumerate(rows):
        if prev_group is not None and r["group"] != prev_group:
            y -= GROUP_GAP
        if r["group"] not in group_header_y:
            group_header_y[r["group"]] = y + 0.55
        y_positions.append(y)
        y -= ROW_STEP
        prev_group = r["group"]

    all_ci = [v for r in rows for v in r["cross_ci"]]
    xlim = max(abs(v) for v in all_ci) * 1.10

    FS.apply_style()
    fig, ax = FS.new_figure(width="full", height_in=3.3)

    for r, yy in zip(rows, y_positions):
        lo, hi = r["cross_ci"]
        ax.plot([lo, hi], [yy, yy], color=FOREST_COLOR, linewidth=1.8, zorder=3, solid_capstyle="round")
        ax.plot(r["cross_mean"], yy, marker="D", color=FOREST_COLOR, markersize=7, zorder=4,
                 markeredgecolor="black", markeredgewidth=0.7)

    # Group separator + subtle group headers.
    sep_y = (y_positions[2] + y_positions[3]) / 2.0
    ax.axhline(sep_y, color="0.85", linewidth=0.8, zorder=1)
    for group, hy in group_header_y.items():
        ax.text(-xlim, hy, GROUP_LABELS[group], fontsize=7, color="0.4", style="italic",
                 ha="left", va="bottom")

    ax.axvline(0.0, color="black", linewidth=1.0, zorder=2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
    ax.set_ylim(y_positions[-1] - 0.6, group_header_y[rows[0]["group"]] + TOP_MARGIN - 0.55)
    ax.set_xlim(-xlim, xlim)
    ax.set_xlabel(X_LABEL)

    ax.yaxis.grid(False)
    ax.xaxis.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # No internal title -- the LaTeX caption provides it.

    stats = {
        "figure": FIG_NAME, "checkpoint": FIXED_CHECKPOINT, "B_hierarchical": B_HIER,
        "seed_pairing_method_rows_4_6": "independent per-architecture resampling of 5 learner seeds "
                                          "(arch_seeds={'a': [...], 'b': [...]}, two independently-drawn keys); "
                                          "holdout paths resampled once per replicate and shared across both "
                                          "sides (pathwise pairing preserved).",
        "rows": rows,
        "crosscheck": crosscheck_report,
        "crosscheck_n_mismatches": n_mismatch,
    }
    FS.save_figure(fig, FIG_NAME, stats, bbox_inches="tight")
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
