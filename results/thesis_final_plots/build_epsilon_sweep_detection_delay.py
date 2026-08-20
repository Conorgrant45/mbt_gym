"""
build_epsilon_sweep_detection_delay.py
------------------------------------------
Sweeps the regime-1 jump size (epsilon, price units) DECREASING from its
calibrated value, and shows how the Hamilton filter's detection-delay
distribution (figA1_filter_calibration.py panel (b)'s statistic, same
definition) degrades as the jump signal shrinks toward the pure-diffusion
limit. No RL, no training -- same "No RL" convention as
build_filter_calibration_dataset.py (job S1): a fixed neutral action just
advances the environment/generates the price sequence the filter observes;
regime switches and price jumps are exogenous, unaffected by the action.

Both the TRUE environment (EventDrivenRegimeSwitchingEnv(epsilon=...)) and
the filter (EventTimeHamiltonFilter(jump_size=epsilon/INITIAL_PRICE, ...))
are given the SAME epsilon at each sweep point -- a well-specified filter
throughout, so what's being measured is detection degrading because the
signal itself is genuinely weaker, not filter misspecification. Both
regimes already share the same diffusion volatility in this project's
calibration (R0_VOLATILITY_PCT == R1_VOLATILITY_PCT, see envs/make_envs.py's
own docstring/comment), so epsilon is already the sole distinguishing
signal -- no extra equalisation needed (unlike the older, fixed-step-env
diagnostic_epsilon_sweep.py this supersedes for the event-driven setting).

Epsilon values: geomspace(0.5, 0.0005, 6) -- same range/count as the
precedent diagnostic_epsilon_sweep.py used for the fixed-step environment.

Seeds: 6 disjoint 500-seed blocks starting at 320000 (320000-320499,
320500-320999, ... 322500-322999) -- verified disjoint from every range in
phase7_post_training_common.PRIOR_SEED_RANGES, this experiment's own
validation/holdout ranges, AND build_filter_calibration_dataset.py's own
300000-304999 block.

Output: one grid of detection-delay histograms (one panel per epsilon
value, log-scale y-axis, no panel (a)/reliability-diagram companion per
request), plus a small CSV of the underlying per-switch delays and a
summary JSON (Brier score and delay stats per epsilon).

Run from repo root:
    python results/thesis_final_plots/build_epsilon_sweep_detection_delay.py
"""
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

import time

import numpy as np
import pandas as pd

import fig_style as FS
from envs.event_driven_regime_env import EventDrivenRegimeSwitchingEnv
from envs.make_envs import EPSILON, INITIAL_PRICE, R0_VOLATILITY_PCT, R1_VOLATILITY_PCT, LAMBDA, TRANSITION_GENERATOR
from beliefs.event_time_hamilton_filter import EventTimeHamiltonFilter
import phase7_post_training.phase7_post_training_common as P7PC
import final.final_common as FC

# ======================================================================
# CONFIGURATION
# ======================================================================
FIG_NAME = "fig_epsilon_sweep_detection_delay"
EPSILON_VALUES = np.geomspace(EPSILON, 0.0005, 6).tolist()  # decreasing: [0.5, ..., 0.0005]
N_PATHS_PER_EPSILON = 400
SEED_BLOCK_START = 320_000
SEED_BLOCK_SIZE = 500
NEUTRAL_ACTION = np.array([0.0, 0.0])

OUTPUT_DELAYS_CSV = FC.RESULTS_DIR / "epsilon_sweep_detection_delays.csv"
OUTPUT_SUMMARY_JSON = FC.RESULTS_DIR / "epsilon_sweep_summary.json"


def _verify_seed_disjointness(all_sweep_seeds: set):
    prior = dict(P7PC.PRIOR_SEED_RANGES)
    prior["final_validation"] = set(FC.VALIDATION_SEEDS)
    prior["final_holdout"] = set(FC.HOLDOUT_SEEDS)
    prior["filter_calibration_S1"] = set(range(300_000, 305_000))
    overlaps = {name: sorted(all_sweep_seeds & rng) for name, rng in prior.items() if all_sweep_seeds & rng}
    if overlaps:
        raise ValueError(f"Epsilon-sweep seed range overlaps prior ranges: {overlaps}")
    print(f"Verified: {len(all_sweep_seeds)} sweep seeds disjoint from all {len(prior)} prior seed ranges.")


def _simulate_one_path(seed: int, epsilon: float) -> pd.DataFrame:
    env = EventDrivenRegimeSwitchingEnv(seed=seed, epsilon=epsilon)
    filt = EventTimeHamiltonFilter(
        transition_generator=TRANSITION_GENERATOR,
        regime_volatilities=[R0_VOLATILITY_PCT, R1_VOLATILITY_PCT],
        jump_size=epsilon / INITIAL_PRICE,
        lambda_bid=LAMBDA, lambda_ask=LAMBDA,
    )
    env.reset()
    filt.reset(initial_price=env.raw_midprice)

    rows = []
    done = False
    while not done:
        _, _, done, info = env.step(NEUTRAL_ACTION)
        belief = filt.update(info["price_after"], info["elapsed_time"], info["event_type"] == "arrival")
        if info["event_type"] == "arrival":
            rows.append(dict(belief=belief, true_regime=info["regime_at_event"]))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    all_seeds = set()
    for i in range(len(EPSILON_VALUES)):
        block_start = SEED_BLOCK_START + i * SEED_BLOCK_SIZE
        all_seeds |= set(range(block_start, block_start + N_PATHS_PER_EPSILON))
    _verify_seed_disjointness(all_seeds)

    delay_rows = []
    summary = {}
    for i, epsilon in enumerate(EPSILON_VALUES):
        block_start = SEED_BLOCK_START + i * SEED_BLOCK_SIZE
        seeds = range(block_start, block_start + N_PATHS_PER_EPSILON)

        all_beliefs, all_regimes = [], []
        n_switches, n_detected = 0, 0
        for seed in seeds:
            ep = _simulate_one_path(seed, epsilon)
            belief, regime = ep["belief"].to_numpy(), ep["true_regime"].to_numpy()
            all_beliefs.append(belief)
            all_regimes.append(regime)

            switch_idx = np.where(np.diff(regime) != 0)[0] + 1
            for si in switch_idx:
                new_regime = regime[si]
                n_switches += 1
                crossed = np.where((belief[si:] >= 0.5) == bool(new_regime))[0]
                if len(crossed):
                    n_detected += 1
                    delay_rows.append(dict(epsilon=epsilon, path_seed=seed, switch_event_index=int(si),
                                             new_regime=int(new_regime), detection_delay_events=int(crossed[0])))

        beliefs_cat = np.concatenate(all_beliefs)
        regimes_cat = np.concatenate(all_regimes)
        brier = float(np.mean((beliefs_cat - regimes_cat) ** 2))
        summary[f"{epsilon:.6f}"] = dict(
            epsilon=epsilon, brier_score=brier, n_events=int(len(beliefs_cat)),
            n_switches=n_switches, n_detected=n_detected,
        )
        print(f"epsilon={epsilon:.6f}: brier={brier:.5f}  switches={n_switches}  "
              f"detected={n_detected} ({100*n_detected/max(n_switches,1):.1f}%)  [{time.time()-t0:.1f}s]")

    delays_df = pd.DataFrame(delay_rows)
    delays_df.to_csv(OUTPUT_DELAYS_CSV, index=False)
    import json
    OUTPUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Saved {OUTPUT_DELAYS_CSV}")
    print(f"Saved {OUTPUT_SUMMARY_JSON}")

    # ------------------------------------------------------------------
    # Plot: grid of detection-delay histograms, one panel per epsilon
    # ------------------------------------------------------------------
    n = len(EPSILON_VALUES)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = FS.new_figure(width="full", height_in=2.3 * nrows, nrows=nrows, ncols=ncols)
    axes_flat = np.array(axes).reshape(-1)

    for ax, epsilon in zip(axes_flat, EPSILON_VALUES):
        d = delays_df[delays_df["epsilon"] == epsilon]["detection_delay_events"]
        if len(d) == 0:
            ax.text(0.5, 0.5, "no detections", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        else:
            max_delay = int(d.max())
            ax.hist(d, bins=np.arange(0, max_delay + 2) - 0.5, color=FS.COLORS["hamilton_ppo"],
                     edgecolor="black", linewidth=0.4)
            ax.set_yscale("log")
        s = summary[f"{epsilon:.6f}"]
        ax.set_title(f"$\\epsilon$={epsilon:.4f}", fontsize=9)
        ax.text(0.97, 0.95, f"Brier={s['brier_score']:.4f}\n{100*s['n_detected']/s['n_switches']:.1f}% detected",
                 transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="dimgray")
        ax.set_xlabel("Detection delay (events)", fontsize=7.5)
        ax.set_ylabel("Count (log)", fontsize=7.5)

    for ax in axes_flat[n:]:
        ax.axis("off")

    FS.save_figure(fig, FIG_NAME, dict(figure=FIG_NAME, epsilon_values=EPSILON_VALUES,
                                          n_paths_per_epsilon=N_PATHS_PER_EPSILON, summary=summary),
                    bbox_inches="tight")
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
