"""
final/final_build_action_diagnostics.py
--------------------------------------
Builds action_diagnostics.csv: a single tidy long-format consolidation of
every action-std / near-bound-rate diagnostic (both deterministic and
stochastic) already present in validation_seed_summary.csv (every
checkpoint, for the learning curve) and final_holdout_seed_summary.csv (the
fixed 200k/1m checkpoints, for the final comparison), tagged by `source`.

This performs NO fresh evaluation -- it is a pure projection/reshape of
columns that already exist in those two summary files (mean_bid_action,
mean_ask_action, action_std_bid, action_std_ask,
near_bound_rate_deterministic, stochastic_action_std_bid,
stochastic_action_std_ask, near_bound_rate_stochastic,
stochastic_full_objective), reproducible at any time without retraining or
re-evaluating.

Run from repo root:
    python final/final_build_action_diagnostics.py
"""
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

import pandas as pd

import final.final_common as FC

ACTION_FIELDS = [
    "mean_bid_action", "mean_ask_action", "action_std_bid", "action_std_ask",
    "near_bound_rate_deterministic",
    "stochastic_action_std_bid", "stochastic_action_std_ask",
    "near_bound_rate_stochastic", "stochastic_full_objective",
]


def _extract(df: pd.DataFrame, policy_col: str, source: str) -> pd.DataFrame:
    keep = [policy_col, "learner_seed", "checkpoint_transition", "n_paths"] + [f"mean_{f}" for f in ACTION_FIELDS]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out = out.rename(columns={policy_col: "policy", **{f"mean_{f}": f for f in ACTION_FIELDS}})
    out.insert(0, "source", source)
    return out


def main():
    val_path = FC.RESULTS_DIR / "validation_seed_summary.csv"
    hold_path = FC.RESULTS_DIR / "final_holdout_seed_summary.csv"
    assert val_path.exists(), f"{val_path} not found -- run final_aggregate_validation.py first"
    assert hold_path.exists(), f"{hold_path} not found -- run final_aggregate_holdout.py first"

    val_df = pd.read_csv(val_path)
    hold_df = pd.read_csv(hold_path)

    out = pd.concat([
        _extract(val_df, "architecture", "validation"),
        _extract(hold_df, "policy", "holdout"),
    ], ignore_index=True)
    out = out.sort_values(["source", "policy", "learner_seed", "checkpoint_transition"], na_position="first")

    out_path = FC.RESULTS_DIR / "action_diagnostics.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
