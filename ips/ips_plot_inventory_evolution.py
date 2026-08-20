"""
ips/ips_plot_inventory_evolution.py
--------------------------------
Internal diagnostic (NOT a thesis artefact): reconstructs the piecewise-
constant inventory path implied by the existing event-level Parquet files
under results/inventory_penalty_sensitivity/event_level/ on a common
201-point time grid over [0, T], for all 3 PPO architectures x 5 learner
seeds x 2 penalty calibrations x 500 holdout paths, and produces a 2x3
seed-level trajectory figure plus two diagnostic CSVs.

Does NOT retrain any model, rerun holdout evaluation, or touch any thesis/
LaTeX file. Reads existing Parquet files only, one file (= one calibration x
architecture x learner_seed, 500 episodes) at a time -- the full ~342MB
event-level dataset is never concatenated in memory; only the small
aggregated 201-point curves and per-episode scalars are retained.

Reuses (read-only): ips_common.py (architecture/seed/calibration/holdout-seed
conventions, EVENT_LEVEL_DIR/RESULTS_DIR paths), envs.make_envs.TERMINAL_TIME.

Reconstruction rule (exactly as specified): Q_0 = 0; after an event at time
t, inventory_after is carried forward until the next event; at each grid
point, use the most recently observed inventory_after (zero before the
first event). This is NOT an event-index average -- episodes have different
event times, so grid points are aligned in CALENDAR time via event_time.

Run from repo root:
    python ips/ips_plot_inventory_evolution.py
"""
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import ips.ips_common as IC
from envs.make_envs import TERMINAL_TIME

N_GRID = 201
GRID = np.linspace(0.0, TERMINAL_TIME, N_GRID)

DIAG_DIR = IC.RESULTS_DIR / "diagnostics"

ARCH_COLORS = {
    "hamilton_ppo": "#0072B2",
    "return_mlp_ppo": "#E69F00",
    "return_lstm_ppo": "#009E73",
}
ARCH_TITLES = {
    "hamilton_ppo": "Belief-state PPO",
    "return_mlp_ppo": "Raw-return MLP PPO",
    "return_lstm_ppo": "Raw-return LSTM PPO",
}
CALIBRATION_LINESTYLE = {"original": "--", "high_penalty": "-"}
SEED_LW, MEAN_LW = 0.8, 2.4
SEED_ALPHA, MEAN_ALPHA = 0.35, 1.0

EVENT_COLS = ["holdout_episode_seed", "event_index", "event_time",
              "elapsed_inter_event_time", "inventory_before", "inventory_after"]

RECONCILE_TOL_ABS = 1e-6  # both derivations trace back to the identical simulated inventory_after values


# ======================================================================
# Core reconstruction
# ======================================================================
def reconstruct_episode_grid(event_time: np.ndarray, inventory_after: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Piecewise-constant reconstruction on `grid` (event_time/inventory_after
    must already be sorted ascending by time). idx = index of the most recent
    event with event_time <= grid[i]; idx < 0 (before the first event) -> 0.0."""
    idx = np.searchsorted(event_time, grid, side="right") - 1
    safe_idx = np.clip(idx, 0, None)
    return np.where(idx < 0, 0.0, inventory_after[safe_idx])


def event_level_path(calibration: str, architecture: str, seed: int) -> Path:
    return IC.EVENT_LEVEL_DIR / calibration / f"{architecture}_seed{seed}.parquet"


# ======================================================================
# Per-(calibration, architecture, seed) file processing
# ======================================================================
def process_seed_file(calibration: str, architecture: str, seed: int) -> dict:
    path = event_level_path(calibration, architecture, seed)
    assert path.exists(), f"Missing event-level file: {path}"
    df = pd.read_parquet(path, columns=EVENT_COLS)

    assert not df[EVENT_COLS].isna().any().any(), f"{path}: missing values in required columns"

    episode_ids = df["holdout_episode_seed"].unique()
    assert len(episode_ids) == 500, f"{path}: expected 500 holdout episodes, got {len(episode_ids)}"
    assert set(episode_ids.tolist()) == set(IC.HOLDOUT_SEEDS), \
        f"{path}: holdout_episode_seed set does not match ips_common.HOLDOUT_SEEDS"

    # Natural (event_index) chronological order -- checked BEFORE the
    # event_time sort below, so this is a genuine data-integrity check, not
    # a tautology of the sort that follows.
    df_index_order = df.sort_values(["holdout_episode_seed", "event_index"])
    for ep_seed, g in df_index_order.groupby("holdout_episode_seed", sort=False):
        et = g["event_time"].to_numpy()
        assert np.all(np.diff(et) >= -1e-12), \
            f"{path}: event_time is not non-decreasing within episode {ep_seed} (event_index order)"
        assert np.isclose(et[-1], TERMINAL_TIME, atol=1e-9), \
            f"{path}: episode {ep_seed} terminal event_time {et[-1]} != TERMINAL_TIME {TERMINAL_TIME}"

    sum_signed_grid = np.zeros(N_GRID)
    sum_abs_grid = np.zeros(N_GRID)
    exact_signed, exact_abs, term_signed, term_abs = [], [], [], []

    # Official reconstruction sort, exactly as specified: by event_time then event_index.
    df_sorted = df.sort_values(["holdout_episode_seed", "event_time", "event_index"])
    for ep_seed, g in df_sorted.groupby("holdout_episode_seed", sort=False):
        et = g["event_time"].to_numpy()
        dtau = g["elapsed_inter_event_time"].to_numpy()
        inv_before = g["inventory_before"].to_numpy()
        inv_after = g["inventory_after"].to_numpy()

        grid_vals = reconstruct_episode_grid(et, inv_after, GRID)
        assert grid_vals[0] == 0.0, f"{path}: episode {ep_seed} reconstructed trajectory does not start at zero"

        sum_signed_grid += grid_vals
        sum_abs_grid += np.abs(grid_vals)

        T_ep = et[-1]
        exact_signed.append(float(np.sum(inv_before * dtau) / T_ep))
        exact_abs.append(float(np.sum(np.abs(inv_before) * dtau) / T_ep))
        term_signed.append(float(inv_after[-1]))
        term_abs.append(float(abs(inv_after[-1])))

    n_ep = len(episode_ids)
    return dict(
        mean_signed_grid=sum_signed_grid / n_ep,
        mean_abs_grid=sum_abs_grid / n_ep,
        exact_time_avg_signed_per_episode=np.array(exact_signed),
        exact_time_avg_abs_per_episode=np.array(exact_abs),
        terminal_signed_per_episode=np.array(term_signed),
        terminal_abs_per_episode=np.array(term_abs),
    )


# ======================================================================
# Orchestration
# ======================================================================
def load_all() -> dict:
    results = {}
    combos_present = set()
    t0 = time.time()
    for calibration in IC.CALIBRATION_NAMES:
        for architecture in IC.ARCHITECTURES:
            for seed in IC.LEARNER_SEEDS:
                r = process_seed_file(calibration, architecture, seed)
                results[(calibration, architecture, seed)] = r
                combos_present.add((calibration, architecture))
            print(f"  processed {calibration}/{architecture}: 5/5 seeds "
                  f"[{time.time()-t0:.0f}s elapsed]")

    expected_combos = {(c, a) for c in IC.CALIBRATION_NAMES for a in IC.ARCHITECTURES}
    assert combos_present == expected_combos, \
        f"missing calibration-architecture combinations: {expected_combos - combos_present}"
    assert len(expected_combos) == 6, "expected exactly 6 calibration-architecture combinations"
    for architecture in IC.ARCHITECTURES:
        n_seeds = sum(1 for (c, a, s) in results if a == architecture)
        assert n_seeds == 2 * len(IC.LEARNER_SEEDS), \
            f"{architecture}: expected {2 * len(IC.LEARNER_SEEDS)} (calibration, seed) files, got {n_seeds}"
    print(f"All 6 calibration-architecture combinations present, 5 seeds each, "
          f"500 holdout episodes each ({time.time()-t0:.0f}s total).")
    return results


def reconcile_with_policy_summary(results: dict):
    policy_summary_path = IC.RESULTS_DIR / "policy_summary.csv"
    assert policy_summary_path.exists(), f"{policy_summary_path} not found -- run ips_aggregate_summary.py first"
    ps = pd.read_csv(policy_summary_path)

    print("\nReconciling cross-seed terminal means against policy_summary.csv ...")
    max_abs_diff = 0.0
    for calibration in IC.CALIBRATION_NAMES:
        for architecture in IC.ARCHITECTURES:
            seed_term_signed = [results[(calibration, architecture, s)]["terminal_signed_per_episode"].mean()
                                 for s in IC.LEARNER_SEEDS]
            seed_term_abs = [results[(calibration, architecture, s)]["terminal_abs_per_episode"].mean()
                              for s in IC.LEARNER_SEEDS]
            cross_seed_signed = float(np.mean(seed_term_signed))
            cross_seed_abs = float(np.mean(seed_term_abs))

            row = ps[(ps["penalty_calibration"] == calibration) & (ps["policy"] == architecture)]
            assert len(row) == 1, f"policy_summary.csv: expected exactly 1 row for ({calibration}, {architecture})"
            ref_signed = float(row["cross_seed_mean_terminal_inventory"].iloc[0])
            ref_abs = float(row["cross_seed_mean_abs_terminal_inventory"].iloc[0])

            diff_signed = abs(cross_seed_signed - ref_signed)
            diff_abs = abs(cross_seed_abs - ref_abs)
            max_abs_diff = max(max_abs_diff, diff_signed, diff_abs)

            print(f"  {calibration:12s} {architecture:16s}: "
                  f"E[Q_T]={cross_seed_signed:+.4f} (ref {ref_signed:+.4f}, diff {diff_signed:.2e})   "
                  f"E[|Q_T|]={cross_seed_abs:.4f} (ref {ref_abs:.4f}, diff {diff_abs:.2e})")

            assert diff_signed < RECONCILE_TOL_ABS, (
                f"RECONCILIATION FAILED: {calibration}/{architecture} E[Q_T] {cross_seed_signed} vs "
                f"policy_summary.csv {ref_signed} (diff {diff_signed} >= tol {RECONCILE_TOL_ABS})"
            )
            assert diff_abs < RECONCILE_TOL_ABS, (
                f"RECONCILIATION FAILED: {calibration}/{architecture} E[|Q_T|] {cross_seed_abs} vs "
                f"policy_summary.csv {ref_abs} (diff {diff_abs} >= tol {RECONCILE_TOL_ABS})"
            )
    print(f"Reconciliation PASSED for all 6 combinations (max abs diff {max_abs_diff:.3e}, tol {RECONCILE_TOL_ABS}).")


# ======================================================================
# Outputs
# ======================================================================
def build_grid_csv(results: dict) -> pd.DataFrame:
    rows = []
    for calibration in IC.CALIBRATION_NAMES:
        for architecture in IC.ARCHITECTURES:
            for seed in IC.LEARNER_SEEDS:
                r = results[(calibration, architecture, seed)]
                for t_idx, t in enumerate(GRID):
                    rows.append(dict(
                        penalty_calibration=calibration, architecture=architecture, learner_seed=seed,
                        time=float(t),
                        mean_signed_inventory=float(r["mean_signed_grid"][t_idx]),
                        mean_abs_inventory=float(r["mean_abs_grid"][t_idx]),
                    ))
    return pd.DataFrame(rows)


def build_bias_summary_csv(results: dict) -> pd.DataFrame:
    rows = []
    for calibration in IC.CALIBRATION_NAMES:
        for architecture in IC.ARCHITECTURES:
            for seed in IC.LEARNER_SEEDS:
                r = results[(calibration, architecture, seed)]
                mean_signed_grid = r["mean_signed_grid"]
                abs_curve = np.abs(mean_signed_grid)
                argmax = int(np.argmax(abs_curve))
                rows.append(dict(
                    penalty_calibration=calibration, architecture=architecture, learner_seed=seed,
                    exact_time_avg_signed_inventory=float(r["exact_time_avg_signed_per_episode"].mean()),
                    exact_time_avg_abs_inventory=float(r["exact_time_avg_abs_per_episode"].mean()),
                    mean_terminal_signed_inventory=float(r["terminal_signed_per_episode"].mean()),
                    mean_terminal_abs_inventory=float(r["terminal_abs_per_episode"].mean()),
                    max_abs_mean_signed_inventory=float(abs_curve[argmax]),
                    time_of_max_abs_mean_signed_inventory=float(GRID[argmax]),
                    frac_grid_points_positive=float(np.mean(mean_signed_grid > 0)),
                    frac_grid_points_negative=float(np.mean(mean_signed_grid < 0)),
                ))
    return pd.DataFrame(rows)


def build_figure(results: dict, png_path: Path, pdf_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    row_fields = [("mean_signed_grid", r"Mean signed inventory  $E[Q_t]$"),
                  ("mean_abs_grid", r"Mean absolute inventory  $E[|Q_t|]$")]

    for row, (field, ylabel) in enumerate(row_fields):
        row_min, row_max = np.inf, -np.inf
        for col, architecture in enumerate(IC.ARCHITECTURES):
            ax = axes[row, col]
            color = ARCH_COLORS[architecture]
            for calibration in IC.CALIBRATION_NAMES:
                ls = CALIBRATION_LINESTYLE[calibration]
                seed_curves = [results[(calibration, architecture, s)][field] for s in IC.LEARNER_SEEDS]
                for sc in seed_curves:
                    ax.plot(GRID, sc, color=color, linestyle=ls, linewidth=SEED_LW, alpha=SEED_ALPHA)
                    row_min, row_max = min(row_min, sc.min()), max(row_max, sc.max())
                mean_curve = np.mean(np.stack(seed_curves), axis=0)
                ax.plot(GRID, mean_curve, color=color, linestyle=ls, linewidth=MEAN_LW, alpha=MEAN_ALPHA)
                row_min, row_max = min(row_min, mean_curve.min()), max(row_max, mean_curve.max())

            if row == 0:
                ax.set_title(ARCH_TITLES[architecture])
                ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":", zorder=0)
            if row == len(row_fields) - 1:
                ax.set_xlabel("Time")
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)

        pad = 0.05 * (row_max - row_min) if row_max > row_min else 1.0
        for col in range(3):
            axes[row, col].set_ylim(row_min - pad, row_max + pad)

    legend_handles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=SEED_LW, alpha=SEED_ALPHA,
               label="Original -- individual seed"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=MEAN_LW, alpha=1.0,
               label="Original -- cross-seed mean"),
        Line2D([0], [0], color="black", linestyle="-", linewidth=SEED_LW, alpha=SEED_ALPHA,
               label="High-penalty -- individual seed"),
        Line2D([0], [0], color="black", linestyle="-", linewidth=MEAN_LW, alpha=1.0,
               label="High-penalty -- cross-seed mean"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Inventory evolution over the episode: original vs. high-penalty calibration "
                 "(500 holdout paths, 5 learner seeds/architecture)")
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main():
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading and validating all 30 event-level files (6 calibration-architecture combinations "
          "x 5 learner seeds, 500 holdout episodes each)...")
    results = load_all()

    reconcile_with_policy_summary(results)

    grid_df = build_grid_csv(results)
    grid_path = DIAG_DIR / "inventory_evolution_grid.csv"
    grid_df.to_csv(grid_path, index=False)
    print(f"\nSaved {grid_path} ({len(grid_df)} rows)")

    bias_df = build_bias_summary_csv(results)
    bias_path = DIAG_DIR / "inventory_directional_bias_summary.csv"
    bias_df.to_csv(bias_path, index=False)
    print(f"Saved {bias_path} ({len(bias_df)} rows)")

    png_path = DIAG_DIR / "inventory_evolution_by_seed.png"
    pdf_path = DIAG_DIR / "inventory_evolution_by_seed.pdf"
    build_figure(results, png_path, pdf_path)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")

    print("\nAll validation checks PASSED.")
    return results, grid_df, bias_df


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nVALIDATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)
