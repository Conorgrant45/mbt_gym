"""
equivalence_diagnostic_event_driven_vs_fixed_step.py
------------------------------------------------------
Phase 1 equivalence diagnostic (event-driven-environment audit brief,
Section 6). Compares the existing fixed-step RegimeSwitchingEnv and the new
EventDrivenRegimeSwitchingEnv under the SAME constant (naive) quoting policy,
over a large Monte Carlo sample of independent episodes each. The two
environments are NOT pathwise identical (different discretisations of the
same continuous-time model), so this compares DISTRIBUTIONAL summary
statistics with confidence intervals, not paired per-episode differences.

Does not modify any production file. Read-only reuse of
compare_four_policies_paired.instrument_env for the fixed-step side's
arrival/fill instrumentation, and simulate_belief_weighted.NAIVE_DEPTH/
normalise_depth for the shared constant policy.

Run from repo root:
    python equivalence_diagnostic_event_driven_vs_fixed_step.py --episodes 500
"""
import argparse
import time

import numpy as np
import pandas as pd

import simulate_belief_weighted as SBW  # READ ONLY
import compare_four_policies_paired as C4P  # READ ONLY (instrument_env)
from envs.make_envs import make_regime_envs, N_STEPS, PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION, STEP_SIZE
from envs.event_driven_regime_env import EventDrivenRegimeSwitchingEnv, stationary_distribution_from_generator
from envs.make_envs import TRANSITION_GENERATOR
from mbt_gym.gym.index_names import ASK_INDEX, BID_INDEX, INVENTORY_INDEX, CASH_INDEX, ASSET_PRICE_INDEX

PHI = PER_STEP_INVENTORY_AVERSION
ALPHA = TERMINAL_INVENTORY_AVERSION
DT = STEP_SIZE

FIXED_STEP_SEED_START = 900_000
EVENT_DRIVEN_SEED_START = 950_000


def run_fixed_step_episode(seed: int) -> dict:
    np.random.seed(seed)
    env = make_regime_envs(switch_within_episode=True, seed=seed)
    logs = C4P.instrument_env(env, full_trace=False)

    action = np.array([[SBW.normalise_depth(SBW.NAIVE_DEPTH), SBW.normalise_depth(SBW.NAIVE_DEPTH)]])

    obs = env.reset()
    cash_0, inv_0, mid_0 = env.raw_cash, env.raw_inventory, env.raw_midprice

    n_buy = n_sell = n_fills = 0
    time_in_regime_1 = 0.0
    running_penalty = 0.0

    done = np.array([False])
    while not np.all(done):
        regime = env.current_regime
        pre_len = len(logs[regime]["arrival_log"])
        obs, reward, done, info = env.step(action)

        arrivals = logs[regime]["arrival_log"][pre_len]
        fills = logs[regime]["fill_log"][pre_len]
        buy_arrival = float(arrivals[0, ASK_INDEX])
        sell_arrival = float(arrivals[0, BID_INDEX])
        ask_fill_event = buy_arrival * float(fills[0, ASK_INDEX])
        bid_fill_event = sell_arrival * float(fills[0, BID_INDEX])

        n_buy += buy_arrival
        n_sell += sell_arrival
        n_fills += ask_fill_event + bid_fill_event
        if regime == 1:
            time_in_regime_1 += DT
        inv_after = float(info["raw_state"][INVENTORY_INDEX])
        running_penalty += PHI * DT * inv_after ** 2

    cash_T = float(info["raw_state"][CASH_INDEX])
    inv_T = float(info["raw_state"][INVENTORY_INDEX])
    mid_T = float(info["raw_state"][ASSET_PRICE_INDEX])
    raw_pnl = (cash_T + inv_T * mid_T) - (cash_0 + inv_0 * mid_0)
    terminal_penalty = ALPHA * inv_T ** 2
    full_objective = raw_pnl - running_penalty - terminal_penalty

    return dict(
        n_buy_arrivals=n_buy, n_sell_arrivals=n_sell, n_fills=n_fills,
        time_in_regime_1=time_in_regime_1, terminal_price=mid_T,
        running_penalty=running_penalty, terminal_inventory=inv_T,
        raw_pnl=raw_pnl, full_objective=full_objective,
    )


def run_event_driven_episode(seed: int) -> dict:
    # Same physical depth on both sides (SBW.NAIVE_DEPTH), normalised with
    # simulate_belief_weighted's own convention -- identical formula to
    # envs.event_driven_regime_env.denormalise_depth's inverse (asserted
    # equal for MAX_DEPTH in tests/test_event_driven_regime_env.py).
    naive_action_value = SBW.normalise_depth(SBW.NAIVE_DEPTH)
    action = np.array([naive_action_value, naive_action_value])

    env = EventDrivenRegimeSwitchingEnv(seed=seed)
    env.reset()

    n_buy = n_sell = n_fills = 0
    time_in_regime_1 = 0.0
    running_penalty = 0.0

    done = False
    while not done:
        obs, reward, done, info = env.step(action)
        if info["event_type"] == "arrival":
            if info["arrival_side"] == "buy":
                n_buy += 1
            else:
                n_sell += 1
            n_fills += int(info["fill_indicator"].sum())
        if info["regime_at_event"] == 1:
            time_in_regime_1 += info["elapsed_time"]
        running_penalty += info["running_penalty_increment"]

    terminal_price = info["price_after"]
    terminal_inventory = info["inventory_after"]
    terminal_penalty = info["terminal_penalty_increment"]
    raw_pnl = (info["cash_after"] + info["inventory_after"] * info["price_after"]) - (
        env.initial_cash + env.initial_inventory * env.initial_price
    )
    full_objective = raw_pnl - running_penalty - terminal_penalty

    return dict(
        n_buy_arrivals=n_buy, n_sell_arrivals=n_sell, n_fills=n_fills,
        time_in_regime_1=time_in_regime_1, terminal_price=terminal_price,
        running_penalty=running_penalty, terminal_inventory=terminal_inventory,
        raw_pnl=raw_pnl, full_objective=full_objective,
    )


def summarise(records: list, label: str) -> pd.DataFrame:
    df = pd.DataFrame(records)
    rows = []
    for col in df.columns:
        x = df[col].to_numpy(dtype=float)
        n = len(x)
        mean = x.mean()
        std = x.std(ddof=1)
        se = std / np.sqrt(n)
        rows.append(dict(env=label, metric=col, n=n, mean=mean, std=std, se=se,
                          ci_lo=mean - 1.96 * se, ci_hi=mean + 1.96 * se))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--output", type=str, default="results/equivalence_event_driven_vs_fixed_step.csv")
    args = p.parse_args()

    t0 = time.time()
    print(f"Running {args.episodes} fixed-step episodes (naive constant policy)...")
    fixed_records = [run_fixed_step_episode(FIXED_STEP_SEED_START + i) for i in range(args.episodes)]
    print(f"  done in {time.time()-t0:.1f}s")

    t1 = time.time()
    print(f"Running {args.episodes} event-driven episodes (naive constant policy)...")
    event_records = [run_event_driven_episode(EVENT_DRIVEN_SEED_START + i) for i in range(args.episodes)]
    print(f"  done in {time.time()-t1:.1f}s")

    fixed_summary = summarise(fixed_records, "fixed_step")
    event_summary = summarise(event_records, "event_driven")
    combined = pd.concat([fixed_summary, event_summary], ignore_index=True)

    pivot = combined.pivot(index="metric", columns="env", values=["mean", "se", "n"])
    print("\n" + "=" * 100)
    print("EQUIVALENCE COMPARISON (naive constant policy, matched-distribution Monte Carlo)")
    print("=" * 100)

    flags = []
    for metric in fixed_summary["metric"]:
        f_row = fixed_summary[fixed_summary["metric"] == metric].iloc[0]
        e_row = event_summary[event_summary["metric"] == metric].iloc[0]
        diff = e_row["mean"] - f_row["mean"]
        combined_se = np.sqrt(f_row["se"] ** 2 + e_row["se"] ** 2)
        z = diff / combined_se if combined_se > 0 else np.nan
        flag = "FLAG (|z|>3)" if abs(z) > 3 else ""
        flags.append(flag)
        print(f"{metric:22s} fixed={f_row['mean']:12.4f} (se={f_row['se']:.4f})  "
              f"event={e_row['mean']:12.4f} (se={e_row['se']:.4f})  "
              f"diff={diff:+.4f}  z={z:+.2f}  {flag}")

    theoretical_pi1 = stationary_distribution_from_generator(TRANSITION_GENERATOR)[1]
    print(f"\nTheoretical stationary P(regime=1) = {theoretical_pi1:.4f}")
    print(f"Fixed-step  mean time_in_regime_1 fraction = {fixed_summary.set_index('metric').loc['time_in_regime_1','mean']:.4f}")
    print(f"Event-driven mean time_in_regime_1 fraction = {event_summary.set_index('metric').loc['time_in_regime_1','mean']:.4f}")

    import os
    os.makedirs("results", exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"\nFull summary saved to {args.output}")
    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
