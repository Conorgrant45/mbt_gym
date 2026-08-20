"""
figA2_ppo_diagnostics.py
----------------------------
Per-PPO-update optimisation diagnostics vs. cumulative training transitions,
2x2 panels: explained_variance, mean action std (note: the training log's
"log_std_mean" column actually stores exp(log_std), i.e. mean ACTION std --
verified directly against final_instrumented_ppo.py's construction:
`log_std_mean=float(action_std.mean())`, kept under that name only for
schema parity with earlier phases), approx_kl, actor_grad_norm_mean.

Source (read-only): training_log_long.csv -- return_mlp_ppo and
return_lstm_ppo ONLY. hamilton_ppo is deliberately excluded: as of this
writing, hamilton_training_log_long.csv (the properly-interleaved v2
diagnostics-only re-run, see DATA_INVENTORY.md) is INCOMPLETE (2/5 seeds
present when this script last ran -- checked programmatically below, not
assumed) and, per the brief, only included "if it exists AND is complete".
If a later run completes it, re-running this script will pick it up
automatically and footnote it as a protocol-replication run, not part of
the official training.

Also verifies the LSTM column-order repair (see
scripts/fix_training_log_lstm_column_bug.py) is present in the CURRENTLY
LOADED training_log_long.csv, by comparing one row against the pre-fix
backup -- the backup is read ONLY for this one-time sanity print, never for
plotting.

CONFIGURATION block below is safe to edit.
"""
import numpy as np
import pandas as pd

import fig_style as FS

# ======================================================================
# CONFIGURATION
# ======================================================================
FIG_NAME = "figA2_ppo_diagnostics"
INPUT_CSV = FS.RESULTS_DIR / "training_log_long.csv"
BACKUP_CSV_FOR_VERIFICATION_ONLY = FS.RESULTS_DIR / "training_log_long.csv.bak_before_lstm_column_fix"
HAMILTON_LOG_CSV = FS.RESULTS_DIR / "hamilton_training_log_long.csv"
EXPECTED_ROWS_PER_SEED = 250
N_SEEDS = 5

PANELS = [
    ("explained_variance", "Explained variance"),
    ("log_std_mean", "Mean action std"),
    ("approx_kl", "Approximate KL divergence"),
    ("actor_grad_norm_mean", "Actor grad norm"),
]
ARCHITECTURES = ["return_mlp_ppo", "return_lstm_ppo"]


def _verify_lstm_fix_present(df: pd.DataFrame):
    if not BACKUP_CSV_FOR_VERIFICATION_ONLY.exists():
        print("  (backup file not found -- skipping the before/after spot-check; repair is still assumed "
              "applied per DATA_INVENTORY.md)")
        return
    backup = pd.read_csv(BACKUP_CSV_FOR_VERIFICATION_ONLY)
    row_now = df[(df["architecture"] == "return_lstm_ppo") & (df["learner_seed"] == 0)
                 & (df["num_timesteps"] == 4000)].iloc[0]
    row_before = backup[(backup["architecture"] == "return_lstm_ppo") & (backup["learner_seed"] == 0)
                          & (backup["num_timesteps"] == 4000)].iloc[0]
    # Known-good post-fix property: action_std_bid should start near exp(-1.5)=0.2231 (log_std_init),
    # and actor_grad_norm_mean should be non-negative (a norm can never be negative).
    print(f"  Pre-fix backup, return_lstm_ppo seed0 t=4000: action_std_bid={row_before.get('action_std_bid'):.4f}, "
          f"actor_grad_norm_mean={row_before.get('actor_grad_norm_mean'):.4f}")
    print(f"  Current file,   return_lstm_ppo seed0 t=4000: action_std_bid={row_now['action_std_bid']:.4f}, "
          f"actor_grad_norm_mean={row_now['actor_grad_norm_mean']:.4f}")
    assert row_now["actor_grad_norm_mean"] >= 0, "actor_grad_norm_mean is negative -- fix not applied!"
    assert abs(row_now["action_std_bid"] - np.exp(-1.5)) < 0.05, \
        "action_std_bid not near exp(-1.5) at t=4000 -- fix not applied!"
    print("  LSTM column-order fix CONFIRMED present in the file used for this figure.")


def _hamilton_log_complete() -> bool:
    if not HAMILTON_LOG_CSV.exists():
        return False
    try:
        h = pd.read_csv(HAMILTON_LOG_CSV)
    except Exception:
        return False
    seeds_present = h["learner_seed"].nunique() if "learner_seed" in h.columns else 0
    complete = seeds_present == N_SEEDS and len(h) == N_SEEDS * EXPECTED_ROWS_PER_SEED
    print(f"  hamilton_training_log_long.csv: {seeds_present}/{N_SEEDS} seeds present, "
          f"{len(h) if 'h' in dir() else 0} rows -- {'COMPLETE' if complete else 'INCOMPLETE, excluding hamilton_ppo'}")
    return complete


def main():
    df = pd.read_csv(INPUT_CSV)
    print("Verifying LSTM column-order repair against the pre-fix backup (backup used for verification only):")
    _verify_lstm_fix_present(df)

    architectures = list(ARCHITECTURES)
    footnote = None
    if _hamilton_log_complete():
        h = pd.read_csv(HAMILTON_LOG_CSV)
        h = h[[c for c in df.columns if c in h.columns]]
        df = pd.concat([df, h], ignore_index=True)
        architectures = ["hamilton_ppo"] + architectures
        footnote = "hamilton_ppo included as a protocol-replication diagnostics run (not the official Phase 7 training)."

    for arch in architectures:
        for seed in range(N_SEEDS):
            n = len(df[(df["architecture"] == arch) & (df["learner_seed"] == seed)])
            if n != EXPECTED_ROWS_PER_SEED:
                raise ValueError(f"{arch!r} seed {seed}: expected {EXPECTED_ROWS_PER_SEED} update rows, found {n}")

    fig, axes = FS.new_figure(width="full", height_in=4.4, nrows=2, ncols=2)
    axes_flat = axes.reshape(-1)
    stats = {"figure": FIG_NAME, "architectures_shown": architectures, "footnote": footnote}

    for ax, (col, ylabel) in zip(axes_flat, PANELS):
        for arch in architectures:
            arch_df = df[df["architecture"] == arch]
            for seed in range(N_SEEDS):
                seed_df = arch_df[arch_df["learner_seed"] == seed].sort_values("num_timesteps")
                ax.plot(seed_df["num_timesteps"], seed_df[col], color=FS.COLORS[arch], alpha=0.25,
                         linewidth=0.5, zorder=2)
            cross = arch_df.groupby("num_timesteps")[col].mean().sort_index()
            ax.plot(cross.index, cross.values, color=FS.COLORS[arch], linestyle=FS.LINESTYLES[arch],
                     linewidth=1.4, zorder=3, label=FS.display(arch))
        ax.axvline(200_000, color="0.6", linestyle=":", linewidth=0.7, zorder=1)
        ax.set_ylabel(ylabel, fontsize=8)
        if col == "explained_variance":
            ax.axhline(0.0, color="firebrick", linewidth=0.6, linestyle=":", alpha=0.6, zorder=1)

    axes_flat[0].legend(loc="best", fontsize=6.5, frameon=True)
    for ax in axes_flat[2:]:
        ax.set_xlabel("Cumulative training transitions", fontsize=8)
    FS.panel_letter(axes_flat[0], "a")
    FS.panel_letter(axes_flat[1], "b")
    FS.panel_letter(axes_flat[2], "c")
    FS.panel_letter(axes_flat[3], "d")

    FS.save_figure(fig, FIG_NAME, stats)


if __name__ == "__main__":
    main()
