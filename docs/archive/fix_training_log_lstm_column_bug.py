"""
scripts/fix_training_log_lstm_column_bug.py
--------------------------------------------------
One-off repair for a confirmed column-misalignment bug in
results/final_reduced_exploration_architecture_comparison/training_log_long.csv.

Root cause: final_run_training.py writes each architecture's per-update
diagnostics with `diag_df.to_csv(path, mode="a", header=False, index=False)`.
FinalInstrumentedPPO (feed-forward: return_mlp_ppo) and
FinalInstrumentedRecurrentPPO (return_lstm_ppo, in final_instrumented_ppo.py)
build their per-update dict with DIFFERENT key orders from column 10
("explained_variance" in the MLP dict) onward. Since return_mlp_ppo was
processed first, its key order became the CSV header; every
return_lstm_ppo row was then appended positionally under that header,
silently mislabeling columns 10-29. Columns 1-9 (architecture, learner_seed,
n_updates, num_timesteps, policy_loss, value_loss, entropy_loss, approx_kl,
clip_fraction) are identical in both dicts and are NOT affected.

This bug is confirmed by static analysis of both dict constructions
(phase5_instrumented_ppo.py / final_instrumented_ppo.py) AND cross-checked
empirically against known ground truth: the CSV's "return_mean" column for
LSTM rows starts at 0.2231 = exp(-1.5) = exp(LOG_STD_INIT), i.e. it is
actually action_std_bid, not a return; "advantage_mean" is a constant
0.0003, i.e. the learning rate; "log_std_bid" is strictly positive
(0.29-1.45), i.e. the real actor_grad_norm_mean.

This bug is confined to this ONE diagnostic CSV -- FinalInstrumentedRecurrentPPO.train()
is a byte-for-byte copy of RecurrentPPO's own train() with read-only logging
added, so no optimizer/gradient/loss computation was affected; the trained
checkpoints and every validation/holdout result file are untouched.

This script writes a CORRECTED replacement for training_log_long.csv
(backing up the original first) by re-labelling the return_lstm_ppo rows'
columns 10-29 according to the exact positional mapping derived from
final_instrumented_ppo.py's dict construction. Values are moved, never
recomputed or approximated.

Run from repo root:
    python scripts/fix_training_log_lstm_column_bug.py
"""
import shutil
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results/final_reduced_exploration_architecture_comparison")
LOG_PATH = RESULTS_DIR / "training_log_long.csv"
BACKUP_PATH = RESULTS_DIR / "training_log_long.csv.bak_before_lstm_column_fix"

# CSV column name (as currently mislabeled, per the MLP/header key order) ->
# the TRUE metric name that column actually holds for return_lstm_ppo rows,
# derived from FinalInstrumentedRecurrentPPO's dict construction order in
# final_instrumented_ppo.py (lines 167-193).
LSTM_TRUE_METRIC_AT_CSV_COLUMN = {
    "explained_variance": "total_loss",
    "log_std_mean": "explained_variance",
    "advantage_mean": "learning_rate",
    "advantage_std": "log_std_mean",
    "advantage_min": "log_std_bid",
    "advantage_max": "log_std_ask",
    "return_mean": "action_std_bid",
    "return_std": "action_std_ask",
    "actor_grad_norm_mean": "advantage_mean",
    "critic_grad_norm_mean": "advantage_std",
    "actor_param_update_norm": "advantage_min",
    "critic_param_update_norm": "advantage_max",
    "frac_sampled_action_clipped": "return_mean",
    "frac_sampled_action_near_bound": "return_std",
    "log_std_bid": "actor_grad_norm_mean",
    "log_std_ask": "critic_grad_norm_mean",
    "action_std_bid": "actor_param_update_norm",
    "action_std_ask": "critic_param_update_norm",
    "learning_rate": "frac_sampled_action_clipped",
    "total_loss": "frac_sampled_action_near_bound",
}
UNAFFECTED_COLUMNS = ["architecture", "learner_seed", "n_updates", "num_timesteps",
                       "policy_loss", "value_loss", "entropy_loss", "approx_kl", "clip_fraction"]


def main():
    df = pd.read_csv(LOG_PATH)
    all_metric_cols = [c for c in df.columns if c not in ("architecture", "learner_seed")]
    assert set(LSTM_TRUE_METRIC_AT_CSV_COLUMN.keys()) | set(UNAFFECTED_COLUMNS[2:]) == set(all_metric_cols), (
        "Column set mismatch -- this repair mapping was derived from a specific column layout; "
        "re-derive it before running if the CSV schema has changed."
    )

    is_lstm = df["architecture"] == "return_lstm_ppo"
    n_lstm = int(is_lstm.sum())
    print(f"Rows to repair (return_lstm_ppo): {n_lstm}")
    assert n_lstm > 0, "No return_lstm_ppo rows found -- nothing to repair."

    lstm_rows = df.loc[is_lstm].copy()
    repaired_lstm = pd.DataFrame(index=lstm_rows.index)
    for col in UNAFFECTED_COLUMNS:
        repaired_lstm[col] = lstm_rows[col]
    for mislabeled_col, true_metric in LSTM_TRUE_METRIC_AT_CSV_COLUMN.items():
        repaired_lstm[true_metric] = lstm_rows[mislabeled_col]
    repaired_lstm = repaired_lstm[df.columns]  # restore original column order

    # Spot-check against the empirical ground truth used to discover this bug.
    import numpy as np
    assert np.isclose(repaired_lstm["action_std_bid"].iloc[0], np.exp(-1.5), atol=0.01), \
        "Sanity check failed: repaired action_std_bid should start near exp(-1.5)."
    assert repaired_lstm["learning_rate"].nunique() == 1, \
        "Sanity check failed: repaired learning_rate should be constant."
    assert (repaired_lstm["actor_grad_norm_mean"] >= 0).all(), \
        "Sanity check failed: repaired actor_grad_norm_mean should be non-negative."
    print("Sanity checks passed (action_std_bid ~ exp(-1.5); learning_rate constant; "
          "actor_grad_norm_mean >= 0).")

    df_fixed = df.copy()
    df_fixed.loc[is_lstm, :] = repaired_lstm

    shutil.copy2(LOG_PATH, BACKUP_PATH)
    print(f"Backed up original to {BACKUP_PATH}")
    df_fixed.to_csv(LOG_PATH, index=False)
    print(f"Wrote repaired {LOG_PATH}")


if __name__ == "__main__":
    main()
