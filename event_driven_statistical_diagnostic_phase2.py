"""
event_driven_statistical_diagnostic_phase2.py
-------------------------------------------------
Phase 2 statistical diagnostics (event-driven agent-integration audit
brief, Section 11). Two parts:

1. A moderate analytic-policy-only event-driven diagnostic (episode event
   count, event-time distribution, regime occupation, terminal-price
   moments, fill count, terminal inventory, raw P&L, full objective),
   compared against the Phase 1 equivalence-diagnostic numbers already on
   disk (results/equivalence_event_driven_vs_fixed_step.csv) to flag any
   change caused by the NEW wrappers/filter added in Phase 2 (Phase 1's own
   diagnostic ran directly against the raw environment with a naive
   constant policy -- this reruns the same policy/seed-range style through
   the Phase-2 evaluator to confirm nothing drifted).

2. Event-time Hamilton filter calibration: classification accuracy, Brier
   score, log loss, mean posterior by true regime, switching-detection lag,
   and posterior calibration bins -- computed using the environment's own
   PRIVILEGED true-regime label purely for OFFLINE evaluation of the
   filter's own quality. This is never fed to any learned agent's
   observation (see envs/event_driven_hamilton_ppo_wrapper.py and
   tests/test_event_driven_agent_integration.py's leakage tests).

Run from repo root:
    python event_driven_statistical_diagnostic_phase2.py --episodes 300
"""
import argparse
import time

import numpy as np
import pandas as pd

import simulate_belief_weighted as SBW
import evaluate_agents_event_driven as EAED
from envs.event_driven_hamilton_ppo_wrapper import make_event_time_filter
from envs.event_driven_regime_env import EventDrivenRegimeSwitchingEnv, stationary_distribution_from_generator
from envs.make_envs import TRANSITION_GENERATOR

SEED_START = 160_000


def run_analytic_diagnostic(n_episodes: int) -> pd.DataFrame:
    controls = {}
    for regime, params in SBW.REGIME_PARAMS.items():
        da, db, qag, qbg = SBW.build_optimal_control(**params)
        controls[regime] = {"delta_ask": da, "delta_bid": db, "q_ask": qag, "q_bid": qbg}

    rows = []
    for agent_type in ("oracle", "belief_weighted", "randomised", "naive"):
        for i in range(n_episodes):
            seed = SEED_START + i
            m = EAED.run_event_analytic_agent_episode(agent_type, controls, seed)
            m["agent_type"] = agent_type
            rows.append(m)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for agent_type, sub in df.groupby("agent_type"):
        for metric in ("full_objective", "raw_pnl", "fills", "terminal_abs_inventory", "episode_length"):
            x = sub[metric].to_numpy(dtype=float)
            rows.append(dict(agent_type=agent_type, metric=metric, mean=x.mean(), std=x.std(ddof=1),
                              se=x.std(ddof=1) / np.sqrt(len(x)), n=len(x)))
    return pd.DataFrame(rows)


def compare_with_phase1(current: pd.DataFrame):
    print("\n" + "=" * 100)
    print("COMPARISON WITH PHASE 1 EQUIVALENCE DIAGNOSTIC (naive policy, results/equivalence_event_driven_vs_fixed_step.csv)")
    print("=" * 100)
    try:
        phase1 = pd.read_csv("results/equivalence_event_driven_vs_fixed_step.csv")
    except FileNotFoundError:
        print("  Phase 1 comparison file not found -- skipping direct comparison.")
        return
    phase1_event = phase1[phase1["env"] == "event_driven"].set_index("metric")
    naive_now = current[(current["agent_type"] == "naive")]
    for metric, phase1_key in (("full_objective", "full_objective"), ("raw_pnl", "raw_pnl"), ("fills", "n_fills")):
        row = naive_now[naive_now["metric"] == metric]
        if len(row) == 0 or phase1_key not in phase1_event.index:
            continue
        now_mean, now_se = row["mean"].values[0], row["se"].values[0]
        p1_mean, p1_se = phase1_event.loc[phase1_key, "mean"], phase1_event.loc[phase1_key, "se"]
        combined_se = np.sqrt(now_se ** 2 + p1_se ** 2)
        z = (now_mean - p1_mean) / combined_se if combined_se > 0 else np.nan
        flag = "FLAG (|z|>3)" if abs(z) > 3 else ""
        print(f"  naive/{metric:20s} Phase1={p1_mean:10.4f} (se={p1_se:.4f})  "
              f"Phase2={now_mean:10.4f} (se={now_se:.4f})  z={z:+.2f}  {flag}")


# ======================================================================
# Event-time Hamilton filter calibration (privileged diagnostic)
# ======================================================================
def run_filter_calibration(n_episodes: int) -> dict:
    beliefs = []
    true_regimes = []
    elapsed_times = []
    switch_lags_events = []
    switch_lags_time = []

    for i in range(n_episodes):
        seed = SEED_START + 500_000 + i
        env = EventDrivenRegimeSwitchingEnv(seed=seed)
        filt = make_event_time_filter()
        raw_state = env.reset()
        filt.reset(initial_price=env.raw_midprice)

        prev_true_regime = env.current_regime
        pending_switch_time = 0.0
        awaiting_detection = False
        target_regime = None

        done = False
        while not done:
            action = np.array([0.0, 0.0])
            raw_state, reward, done, info = env.step(action)
            belief = filt.update(info["price_after"], info["elapsed_time"], info["event_type"] == "arrival")
            true_regime = info["regime_at_event"]

            beliefs.append(belief)
            true_regimes.append(true_regime)
            elapsed_times.append(info["elapsed_time"])

            if awaiting_detection:
                pending_switch_time += info["elapsed_time"]
                if (belief >= 0.5) == bool(target_regime):
                    switch_lags_time.append(pending_switch_time)
                    awaiting_detection = False

            if true_regime != prev_true_regime:
                if awaiting_detection:
                    # A second switch happened before detection completed -- abandon this one.
                    pass
                awaiting_detection = True
                target_regime = true_regime
                pending_switch_time = 0.0
                switch_event_index = 0
            prev_true_regime = true_regime

    beliefs = np.array(beliefs)
    true_regimes = np.array(true_regimes)
    eps = 1e-12
    accuracy = float(np.mean((beliefs >= 0.5) == true_regimes))
    brier = float(np.mean((beliefs - true_regimes) ** 2))
    log_loss = float(-np.mean(true_regimes * np.log(beliefs + eps) + (1 - true_regimes) * np.log(1 - beliefs + eps)))
    mean_belief_regime0 = float(beliefs[true_regimes == 0].mean()) if np.any(true_regimes == 0) else float("nan")
    mean_belief_regime1 = float(beliefs[true_regimes == 1].mean()) if np.any(true_regimes == 1) else float("nan")

    calibration_bins = []
    bin_edges = np.linspace(0, 1, 11)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (beliefs >= lo) & (beliefs < hi if hi < 1.0 else beliefs <= hi)
        if mask.sum() > 0:
            calibration_bins.append(dict(
                bin_lo=lo, bin_hi=hi, n=int(mask.sum()),
                mean_predicted=float(beliefs[mask].mean()),
                empirical_p_regime1=float(true_regimes[mask].mean()),
            ))

    return dict(
        n_observations=len(beliefs), accuracy=accuracy, brier_score=brier, log_loss=log_loss,
        mean_belief_regime0=mean_belief_regime0, mean_belief_regime1=mean_belief_regime1,
        mean_switch_detection_lag_seconds=float(np.mean(switch_lags_time)) if switch_lags_time else float("nan"),
        n_switches_detected=len(switch_lags_time),
        calibration_bins=calibration_bins,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    args = p.parse_args()

    t0 = time.time()
    print(f"Running {args.episodes}-episode analytic-policy event-driven diagnostic...")
    df = run_analytic_diagnostic(args.episodes)
    summary = summarise(df)
    print("\n" + "=" * 100)
    print("ANALYTIC POLICY SUMMARY (event-driven, Phase 2 evaluator)")
    print("=" * 100)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n(elapsed: {time.time()-t0:.1f}s)")

    compare_with_phase1(summary)

    theoretical_pi1 = stationary_distribution_from_generator(TRANSITION_GENERATOR)[1]
    print(f"\nTheoretical stationary P(regime=1) = {theoretical_pi1:.4f}")

    print(f"\nRunning filter calibration diagnostic ({args.episodes} episodes, privileged true-regime labels)...")
    t1 = time.time()
    calib = run_filter_calibration(args.episodes)
    print("\n" + "=" * 100)
    print("EVENT-TIME HAMILTON FILTER CALIBRATION (privileged diagnostic only)")
    print("=" * 100)
    for k, v in calib.items():
        if k != "calibration_bins":
            print(f"  {k} = {v}")
    print("\n  Calibration bins (mean predicted belief vs empirical P(regime=1)):")
    for b in calib["calibration_bins"]:
        print(f"    [{b['bin_lo']:.1f},{b['bin_hi']:.1f}) n={b['n']:6d}  "
              f"mean_predicted={b['mean_predicted']:.4f}  empirical_p_regime1={b['empirical_p_regime1']:.4f}")
    print(f"\n(elapsed: {time.time()-t1:.1f}s)")

    print(f"\nTotal diagnostic time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
