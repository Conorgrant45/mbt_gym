"""
plot_midprice_path.py
-------------------------
Quick illustrative plot: the exogenous midprice process over one episode
(t in [0, T=1.0]), with hidden-regime background shading for context. Not
part of the fig01-fig08/figA1-figA3 results set -- a simple background/
methodology figure showing what the price process actually looks like.

*** VISUAL EXAGGERATION DISCLOSURE (deliberate, not a bug) ***
The calibrated model sets R0_VOLATILITY == R1_VOLATILITY (both 0.01, see
envs/make_envs.py) -- regime 0 is diffusion-identical to regime 1, only
missing the jump component. At the price scale needed to show a regime-1
jump (~0.5 price units), regime-0's genuine diffusion wiggle is real but
visually reads as a flat line, which looks like a rendering artifact. Per
explicit request (approved by supervisor), REGIME_0_VOLATILITY_MULTIPLIER
below inflates ONLY regime 0's diffusion volatility for THIS illustrative
plot, so the calm regime visibly moves. This is NOT the calibrated
simulation used anywhere else in this project (no other script imports
this multiplier) -- it is disclosed here, in the figure's own caption, and
in the saved stats sidecar specifically so it is never mistaken for the
real calibration. Set REGIME_0_VOLATILITY_MULTIPLIER = 1.0 to reproduce the
true calibrated path (matches fig06_trajectory_mechanism.py's regime shading
exactly at multiplier=1.0).

Simulates ONE fresh episode of EventDrivenRegimeSwitchingEnv directly (a
fixed neutral action [0,0] just to advance the environment/generate the
price sequence -- regime switches and price jumps are exogenous, unaffected
by the action, same convention as build_filter_calibration_dataset.py).
Uses path_seed=290000 -- the same episode already shown in
fig06_trajectory_mechanism.py -- purely for narrative continuity with that
figure; not a holdout/validation/training seed reuse (290000 is itself
inside this experiment's holdout range, but this script never touches any
policy or reward, only the exogenous price path, so no leakage concern).

Run from repo root:
    python results/thesis_final_plots/plot_midprice_path.py
"""
import numpy as np
from matplotlib.patches import Patch

import fig_style as FS
from envs.event_driven_regime_env import EventDrivenRegimeSwitchingEnv, DEFAULT_SIGMA
from mbt_gym.gym.index_names import TIME_INDEX

FIG_NAME = "midprice_path"
PATH_SEED = 290_000
NEUTRAL_ACTION = np.array([0.0, 0.0])

# See "VISUAL EXAGGERATION DISCLOSURE" above. 1.0 = true calibrated volatility.
REGIME_0_VOLATILITY_MULTIPLIER = 120.0


def main():
    exaggerated_sigma = dict(DEFAULT_SIGMA)
    exaggerated_sigma[0] = DEFAULT_SIGMA[0] * REGIME_0_VOLATILITY_MULTIPLIER
    env = EventDrivenRegimeSwitchingEnv(seed=PATH_SEED, sigma=exaggerated_sigma)
    env.reset()

    times, prices, regimes = [0.0], [env.raw_midprice], [env.current_regime]
    done = False
    while not done:
        _, _, done, info = env.step(NEUTRAL_ACTION)
        times.append(float(info["raw_state"][TIME_INDEX]))  # cumulative time; info["elapsed_time"] is a delta
        prices.append(info["price_after"])
        regimes.append(info["regime_at_event"])
    times, prices, regimes = np.array(times), np.array(prices), np.array(regimes)

    fig, ax = FS.new_figure(width="full", height_in=2.6)

    # Regime background shading (same convention as fig06_trajectory_mechanism.py).
    change_idx = np.where(np.diff(regimes) != 0)[0]
    seg_starts = np.concatenate([[0], change_idx + 1])
    seg_ends = np.concatenate([change_idx + 1, [len(times) - 1]])
    for s, e in zip(seg_starts, seg_ends):
        ax.axvspan(times[s], times[e], color="0.85" if regimes[s] == 0 else "0.65", linewidth=0, zorder=0)

    ax.step(times, prices, where="post", color=FS.COLORS["hamilton_ppo"], linewidth=1.3, zorder=2)

    ax.set_title("Midprice path")
    ax.set_xlabel("Time")
    ax.set_ylabel("Midprice")
    ax.set_xlim(0.0, 1.0)

    legend_handles = [
        Patch(facecolor="0.85", edgecolor="none", label="Regime 0 (no jumps)"),
        Patch(facecolor="0.65", edgecolor="none", label="Regime 1 (jump regime)"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8, frameon=True)

    FS.save_figure(fig, FIG_NAME, dict(
        figure=FIG_NAME, path_seed=PATH_SEED, n_events=len(times) - 1,
        regime_0_volatility_multiplier=REGIME_0_VOLATILITY_MULTIPLIER,
        note="NOT the calibrated simulation if regime_0_volatility_multiplier != 1.0 -- "
             "regime-0 diffusion volatility was deliberately inflated for visual clarity only.",
    ), bbox_inches="tight")


if __name__ == "__main__":
    main()
