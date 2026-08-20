"""
build_filter_calibration_dataset.py  (OPTIONAL JOB S1 -- NOT RUN AUTOMATICALLY)
--------------------------------------------------------------------------------
Simulates ~5,000 fresh paths (seeds 300000-304999 -- disjoint from every
range in final_common.PRIOR_SEED_RANGES / phase7_post_training_common.PRIOR_SEED_RANGES,
checked programmatically below, not assumed) to calibrate the event-time
Hamilton filter (make_event_time_filter() / EventTimeHamiltonFilter) against
the TRUE simulated hidden regime. No RL model is loaded or evaluated -- a
fixed neutral action ([0.0, 0.0], i.e. mid-of-range depth on both sides) is
used purely to advance the environment/generate the arrival-time price
sequence the filter observes; regime switches and price jumps are exogenous
to the action, so this choice does not affect what is being calibrated.

Retains ONLY (per the brief -- kilobytes of output, not raw paths):
  - binned (posterior, true-regime) reliability counts (~20 equal-width
    bins over [0,1])
  - per-switch detection delay, in EVENTS (not wall-clock time), from a
    true regime switch to the first subsequent event where the posterior
    crosses 0.5 in the direction of the new regime
  - the overall Brier score: mean((belief - 1{true_regime==1})^2) across
    every recorded event, every path

Never touches the validation (280000-280049) or holdout (290000-290499)
seed ranges, or any training seed.

This script is intentionally NOT executed as part of building the other
figures -- it is a separate, explicit-opt-in compute job (see
FIGURES_REPORT.md). Run manually:

    python build_filter_calibration_dataset.py
"""
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

from pathlib import Path

import numpy as np
import pandas as pd

import fig_style as FS
from shared.evaluate_agents_event_driven import make_event_time_filter
from envs.event_driven_regime_env import EventDrivenRegimeSwitchingEnv
import final.final_common as FC
import phase7_post_training.phase7_post_training_common as P7PC

# ======================================================================
# CONFIGURATION
# ======================================================================
N_PATHS = 5_000
SEED_RANGE = list(range(300_000, 305_000))
N_RELIABILITY_BINS = 20
NEUTRAL_ACTION = np.array([0.0, 0.0], dtype=np.float64)

OUTPUT_DIR = FC.RESULTS_DIR  # results/final_reduced_exploration_architecture_comparison -- read-only elsewhere,
                              # this is the one job script permitted to WRITE new files there (a fresh,
                              # clearly-named calibration dataset, never overwriting any existing file)
OUTPUT_RELIABILITY_CSV = OUTPUT_DIR / "filter_calibration_reliability_bins.csv"
OUTPUT_DELAYS_CSV = OUTPUT_DIR / "filter_calibration_detection_delays.csv"
OUTPUT_SUMMARY_JSON = OUTPUT_DIR / "filter_calibration_summary.json"


def _verify_seed_disjointness():
    prior = dict(P7PC.PRIOR_SEED_RANGES)
    prior["final_validation"] = set(FC.VALIDATION_SEEDS)
    prior["final_holdout"] = set(FC.HOLDOUT_SEEDS)
    new_set = set(SEED_RANGE)
    overlaps = {name: sorted(new_set & rng) for name, rng in prior.items() if new_set & rng}
    if overlaps:
        raise ValueError(f"S1 seed range {SEED_RANGE[0]}-{SEED_RANGE[-1]} overlaps prior ranges: {overlaps}")
    print(f"Verified: seed range {SEED_RANGE[0]}-{SEED_RANGE[-1]} ({len(SEED_RANGE)} seeds) is disjoint from "
          f"all {len(prior)} prior seed ranges (including validation/holdout).")


def _simulate_one_path(seed: int) -> pd.DataFrame:
    env = EventDrivenRegimeSwitchingEnv(seed=seed)
    filt = make_event_time_filter()
    raw_state = env.reset()
    filt.reset(initial_price=env.raw_midprice)

    rows = []
    done = False
    while not done:
        raw_state, reward, done, info = env.step(NEUTRAL_ACTION)
        belief = filt.update(info["price_after"], info["elapsed_time"], info["event_type"] == "arrival")
        if info["event_type"] == "arrival":
            rows.append(dict(belief=belief, true_regime=info["regime_at_event"]))
    return pd.DataFrame(rows)


def main():
    _verify_seed_disjointness()

    all_beliefs = []
    all_regimes = []
    delay_records = []

    for i, seed in enumerate(SEED_RANGE[:N_PATHS]):
        ep = _simulate_one_path(seed)
        all_beliefs.append(ep["belief"].to_numpy())
        all_regimes.append(ep["true_regime"].to_numpy())

        regime = ep["true_regime"].to_numpy()
        belief = ep["belief"].to_numpy()
        switch_idx = np.where(np.diff(regime) != 0)[0] + 1
        for si in switch_idx:
            new_regime = regime[si]
            crossed = np.where((belief[si:] >= 0.5) == bool(new_regime))[0]
            if len(crossed):
                delay_records.append(dict(path_seed=seed, switch_event_index=int(si),
                                            new_regime=int(new_regime), detection_delay_events=int(crossed[0])))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{N_PATHS} paths simulated")

    beliefs = np.concatenate(all_beliefs)
    regimes = np.concatenate(all_regimes)
    brier = float(np.mean((beliefs - regimes) ** 2))

    bin_edges = np.linspace(0.0, 1.0, N_RELIABILITY_BINS + 1)
    bin_idx = np.clip(np.digitize(beliefs, bin_edges[1:-1]), 0, N_RELIABILITY_BINS - 1)
    reliability_rows = []
    for b in range(N_RELIABILITY_BINS):
        mask = bin_idx == b
        n = int(mask.sum())
        reliability_rows.append(dict(
            bin_lo=float(bin_edges[b]), bin_hi=float(bin_edges[b + 1]), n=n,
            mean_predicted=float(beliefs[mask].mean()) if n else float("nan"),
            empirical_frequency=float(regimes[mask].mean()) if n else float("nan"),
        ))
    reliability_df = pd.DataFrame(reliability_rows)
    delays_df = pd.DataFrame(delay_records)

    reliability_df.to_csv(OUTPUT_RELIABILITY_CSV, index=False)
    delays_df.to_csv(OUTPUT_DELAYS_CSV, index=False)

    import json
    OUTPUT_SUMMARY_JSON.write_text(json.dumps(dict(
        brier_score=brier, n_paths=N_PATHS, n_events=int(len(beliefs)),
        n_switches_detected=int(len(delays_df)), seed_range=[SEED_RANGE[0], SEED_RANGE[-1]],
        n_reliability_bins=N_RELIABILITY_BINS,
    ), indent=2))

    print(f"Saved {OUTPUT_RELIABILITY_CSV}")
    print(f"Saved {OUTPUT_DELAYS_CSV}")
    print(f"Saved {OUTPUT_SUMMARY_JSON}")
    print(f"Brier score: {brier:.5f}  |  n_events={len(beliefs)}  |  n_switches_detected={len(delays_df)}")


if __name__ == "__main__":
    main()
