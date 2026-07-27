"""
compare_four_policies_paired.py
---------------------------------
Standalone diagnostic. Does NOT modify any production file -- imports the
existing oracle control table builder, get_control, normalise_depth,
NAIVE_ACTION and make_filter (Hamilton filter) from
simulate_belief_weighted.py and uses them completely unchanged. Uses the
corrected 4000-step environment (envs/make_envs.py) exactly as currently
configured -- no parameter is altered.

Extends compare_oracle_naive_paired.py's paired methodology from two
policies to four: oracle, belief-weighted, randomised, naive.

Pairing mechanism (see compare_oracle_naive_paired.py for the full
derivation) -- for each episode index:
    np.random.seed(episode_seed)
    env = make_regime_envs(switch_within_episode=True, seed=episode_seed)
is repeated identically before EACH policy's fresh episode, so all four
policies see bit-for-bit identical regime/arrival/jump/diffusion/fill
draws, differing only in which depths they quote (hence which fills
actually realise).

One extra subtlety versus the two-policy script: the existing randomised
policy (simulate_belief_weighted.py's run_agent) samples a hard regime via
np.random.choice(2, p=[1-belief, belief]) -- the GLOBAL numpy RNG, the same
stream RegimeSwitchingEnv's own regime-switching draws use. Calling it
verbatim inside a paired episode would consume extra draws from that same
global stream mid-episode, desynchronising the regime path for randomised
relative to the other three policies. This script instead draws that
sample from a DEDICATED, independent np.random.default_rng(episode_seed +
1000) -- the same pattern already used elsewhere in this codebase (e.g.
plot_final_belief.py's rng_rand = np.random.default_rng(seed + 1000)).
This does not change the policy's behaviour (still Bernoulli(belief)
sampling of a hard regime label) -- it only isolates which random-number
stream feeds that specific draw, which is required for valid pairing.

Run:
    python compare_four_policies_paired.py --episodes 20 --verify-episodes 20
    python compare_four_policies_paired.py --episodes 1000 --verify-episodes 20 --output results/four_policy_paired_1000.csv
"""

import os
import time
import argparse
import numpy as np
import pandas as pd

import simulate_belief_weighted as SBW  # existing policy definitions -- READ ONLY
from envs.make_envs import (
    make_regime_envs, N_STEPS, TERMINAL_TIME, STEP_SIZE,
    PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION, INITIAL_PRICE,
)
from mbt_gym.gym.index_names import ASK_INDEX, BID_INDEX

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POLICIES = ["oracle", "belief", "randomised", "naive"]

PHI = PER_STEP_INVENTORY_AVERSION
ALPHA = TERMINAL_INVENTORY_AVERSION
DT = STEP_SIZE

METRIC_COLS = [
    "raw_pnl", "full_objective", "spread_revenue", "adverse_selection_loss",
    "running_inventory_penalty", "terminal_inventory_penalty",
    "mean_absolute_inventory", "terminal_absolute_inventory", "total_fills",
]


# ======================================================================
# Recording RNG wrappers (identical to compare_oracle_naive_paired.py)
# ======================================================================
class RecordingRNG:
    def __init__(self, real_rng):
        self._real = real_rng
        self.normal_log = []
        self.exponential_log = []
        self.uniform_log = []

    def normal(self, size):
        v = self._real.normal(size=size)
        self.normal_log.append(v.copy())
        return v

    def exponential(self, scale, size):
        v = self._real.exponential(scale=scale, size=size)
        self.exponential_log.append(v.copy())
        return v

    def uniform(self, size):
        v = self._real.uniform(size=size)
        self.uniform_log.append(v.copy())
        return v


class JumpOnlyRNG:
    def __init__(self, real_rng):
        self._real = real_rng
        self.exponential_log = []

    def exponential(self, scale, size):
        v = self._real.exponential(scale=scale, size=size)
        self.exponential_log.append(v.copy())
        return v

    def __getattr__(self, name):
        return getattr(self._real, name)


def instrument_env(env, full_trace: bool):
    logs = {0: {}, 1: {}}
    for regime_idx in (0, 1):
        md = env.envs[regime_idx].model_dynamics
        arrival_log, fill_log = [], []

        orig_get_arrivals = md.arrival_model.get_arrivals
        def arrivals_hook(orig=orig_get_arrivals, log=arrival_log):
            arr = orig()
            log.append(arr.copy())
            return arr
        md.arrival_model.get_arrivals = arrivals_hook

        orig_get_fills = md.fill_probability_model.get_fills
        def fills_hook(depths, orig=orig_get_fills, log=fill_log):
            f = orig(depths)
            log.append(f.copy())
            return f
        md.fill_probability_model.get_fills = fills_hook

        md.midprice_model.rng = RecordingRNG(md.midprice_model.rng) if full_trace else JumpOnlyRNG(md.midprice_model.rng)

        logs[regime_idx] = dict(arrival_log=arrival_log, fill_log=fill_log,
                                 midprice_rng=md.midprice_model.rng)
    return logs


# ======================================================================
# Single paired episode rollout for one policy
# ======================================================================
def run_episode(policy, controls, episode_seed, full_trace: bool):
    np.random.seed(episode_seed)  # pairs the GLOBAL-state regime-switching draws
    env = make_regime_envs(switch_within_episode=True, seed=episode_seed)  # fresh env, pairs the child RNG streams
    logs = instrument_env(env, full_trace)

    filt = SBW.make_filter() if policy in ("belief", "randomised") else None
    if filt is not None:
        filt.reset()
    # Dedicated, independent RNG for randomised's own regime sample -- see
    # module docstring for why this must NOT be the global np.random stream.
    rng_random_policy = np.random.default_rng(episode_seed + 1000) if policy == "randomised" else None

    obs = env.reset()
    cash_0, inv_0, mid_0 = env.raw_cash, env.raw_inventory, env.raw_midprice
    mid = mid_0

    if full_trace:
        trace_regime = np.empty(N_STEPS, dtype=int)
        trace_buy = np.empty(N_STEPS)
        trace_sell = np.empty(N_STEPS)
        trace_jump_up = np.empty(N_STEPS)
        trace_jump_down = np.empty(N_STEPS)
        trace_diffusion_z = np.empty(N_STEPS)

    inv_abs_sum = 0.0
    running_inventory_penalty = 0.0
    spread_revenue = 0.0
    adverse_selection_loss = 0.0
    total_fills = 0.0
    obj_accum_env = 0.0

    step = 0
    done = np.array([False])
    while not np.all(done):
        regime = env.current_regime
        obs_flat = np.array(obs).flatten()
        inventory = obs_flat[1]
        t_idx = env.current_step
        inv_sc = inventory / SBW.INV_UNIT

        if policy == "oracle":
            ctrl = controls[regime]
            ask_depth, bid_depth = SBW.get_control(
                ctrl["delta_ask"], ctrl["delta_bid"], ctrl["q_ask"], ctrl["q_bid"], t_idx, inv_sc
            )
        elif policy == "belief":
            belief = filt.update(mid)
            c0, c1 = controls[0], controls[1]
            a0, b0 = SBW.get_control(c0["delta_ask"], c0["delta_bid"], c0["q_ask"], c0["q_bid"], t_idx, inv_sc)
            a1, b1 = SBW.get_control(c1["delta_ask"], c1["delta_bid"], c1["q_ask"], c1["q_bid"], t_idx, inv_sc)
            ask_depth = (1.0 - belief) * a0 + belief * a1
            bid_depth = (1.0 - belief) * b0 + belief * b1
        elif policy == "randomised":
            belief = filt.update(mid)
            sampled_regime = int(rng_random_policy.choice(2, p=[1.0 - belief, belief]))
            ctrl = controls[sampled_regime]
            ask_depth, bid_depth = SBW.get_control(
                ctrl["delta_ask"], ctrl["delta_bid"], ctrl["q_ask"], ctrl["q_bid"], t_idx, inv_sc
            )
        else:  # naive
            ask_depth = bid_depth = SBW.NAIVE_DEPTH

        action = np.array([[SBW.normalise_depth(bid_depth), SBW.normalise_depth(ask_depth)]])

        pre_len_arr = len(logs[regime]["arrival_log"])
        pre_len_exp = len(logs[regime]["midprice_rng"].exponential_log)
        if full_trace:
            pre_len_norm = len(logs[regime]["midprice_rng"].normal_log)

        obs, reward, done, info = env.step(action)
        mid = info["raw_midprice"]
        obj_accum_env += float(np.sum(reward))

        arrivals = logs[regime]["arrival_log"][pre_len_arr]
        fills = logs[regime]["fill_log"][pre_len_arr]
        buy_arrival = float(arrivals[0, ASK_INDEX])
        sell_arrival = float(arrivals[0, BID_INDEX])
        ask_fill_attempt = float(fills[0, ASK_INDEX])
        bid_fill_attempt = float(fills[0, BID_INDEX])
        ask_fill_event = buy_arrival * ask_fill_attempt
        bid_fill_event = sell_arrival * bid_fill_attempt

        exp_calls = logs[regime]["midprice_rng"].exponential_log[pre_len_exp: pre_len_exp + 2]
        if len(exp_calls) == 2:
            jump_up = float(exp_calls[0][0, 0])
            jump_down = float(exp_calls[1][0, 0])
        else:
            jump_up = jump_down = 0.0

        if full_trace:
            norm_calls = logs[regime]["midprice_rng"].normal_log[pre_len_norm: pre_len_norm + 1]
            diffusion_z = float(norm_calls[0][0, 0]) if len(norm_calls) == 1 else 0.0
            trace_regime[step] = regime
            trace_buy[step] = buy_arrival
            trace_sell[step] = sell_arrival
            trace_jump_up[step] = jump_up
            trace_jump_down[step] = jump_down
            trace_diffusion_z[step] = diffusion_z

        total_fills += ask_fill_event + bid_fill_event
        spread_revenue += ask_depth * ask_fill_event + bid_depth * bid_fill_event
        if regime == 1:
            adverse_selection_loss += jump_up * ask_fill_event + jump_down * bid_fill_event

        inv_after = float(info["raw_state"][1])
        inv_abs_sum += abs(inv_after)
        running_inventory_penalty += PHI * (inv_after ** 2) * DT

        step += 1

    steps_taken = step
    cash_T, inv_T, mid_T = info["raw_state"][0], info["raw_state"][1], info["raw_state"][3]
    raw_pnl = (cash_T + inv_T * mid_T) - (cash_0 + inv_0 * mid_0)
    terminal_inventory_penalty = ALPHA * (inv_T ** 2)
    full_objective = raw_pnl - running_inventory_penalty - terminal_inventory_penalty
    recon_diff = abs(full_objective - obj_accum_env)

    metrics = dict(
        raw_pnl=raw_pnl,
        running_inventory_penalty=running_inventory_penalty,
        terminal_inventory_penalty=terminal_inventory_penalty,
        full_objective=full_objective,
        mean_absolute_inventory=inv_abs_sum / steps_taken,
        terminal_absolute_inventory=abs(inv_T),
        total_fills=total_fills,
        spread_revenue=spread_revenue,
        adverse_selection_loss=adverse_selection_loss,
        steps_taken=steps_taken,
        env_reward_sum=obj_accum_env,
        reconciliation_abs_diff=recon_diff,
    )
    if full_trace:
        trace = dict(regime=trace_regime, buy=trace_buy, sell=trace_sell,
                     jump_up=trace_jump_up, jump_down=trace_jump_down, diffusion_z=trace_diffusion_z)
    else:
        trace = None
    return metrics, trace


# ======================================================================
# Full paired run over n_episodes
# ======================================================================
def run_comparison(n_episodes, n_verify, base_seed, progress_every=50):
    print("\nBuilding oracle optimal control tables (unchanged existing solver)...")
    t0 = time.time()
    controls = {}
    for regime, params in SBW.REGIME_PARAMS.items():
        da, db, qag, qbg = SBW.build_optimal_control(**params)
        controls[regime] = {"delta_ask": da, "delta_bid": db, "q_ask": qag, "q_bid": qbg}
    print(f"  built in {time.time()-t0:.1f}s")

    rows = []
    n_path_mismatches = 0
    n_verified = 0

    t_start = time.time()
    for ep in range(n_episodes):
        seed = base_seed + ep
        full_trace = ep < n_verify

        results = {}
        for policy in POLICIES:
            m, tr = run_episode(policy, controls, seed, full_trace)
            results[policy] = (m, tr)
            assert m["steps_taken"] == N_STEPS, f"episode {ep} ({policy}): steps_taken != {N_STEPS}"

        if full_trace:
            n_verified += 1
            ref_trace = results["oracle"][1]
            same = all(
                all(np.array_equal(results[pol][1][k], ref_trace[k]) for k in ref_trace)
                for pol in POLICIES[1:]
            )
            if not same:
                n_path_mismatches += 1
                for pol in POLICIES[1:]:
                    for k in ref_trace:
                        if not np.array_equal(results[pol][1][k], ref_trace[k]):
                            n_diff = np.sum(results[pol][1][k] != ref_trace[k])
                            print(f"  [episode {ep}] MISMATCH oracle vs {pol} in '{k}': {n_diff} of {N_STEPS} steps differ")
            assert same, f"episode {ep}: exogenous paths are NOT identical across all four policies"

        for policy in POLICIES:
            row = dict(episode=ep, policy=policy)
            row.update(results[policy][0])
            rows.append(row)

        if (ep + 1) % progress_every == 0:
            elapsed = time.time() - t_start
            print(f"  ... {ep+1}/{n_episodes} paired episodes done, {elapsed:.1f}s elapsed "
                  f"({elapsed/(ep+1):.3f}s/episode-group so far)")

    elapsed = time.time() - t_start
    print(f"\nCompleted {n_episodes} paired episodes (4 policies each) in {elapsed:.1f}s "
          f"({elapsed/n_episodes:.4f}s/episode-group average)")
    print(f"Verified (full-trace) episodes: {n_verified} -- exogenous-path mismatches: {n_path_mismatches}")

    df = pd.DataFrame(rows)
    max_recon = df["reconciliation_abs_diff"].max()
    print(f"\nReconciliation check (full_objective == raw_pnl - running - terminal, vs env reward sum):")
    print(f"  max |discrepancy| across all {len(df)} (episode,policy) rows = {max_recon:.3e}")
    bad_steps = (df["steps_taken"] != N_STEPS).sum()
    print(f"  rows with steps_taken != {N_STEPS}: {bad_steps}")

    return df, n_path_mismatches, elapsed


# ======================================================================
# Reporting
# ======================================================================
def print_summary_table(df):
    print("\n" + "=" * 120)
    print("SUMMARY TABLE: per-policy means")
    print("=" * 120)
    means = df.groupby("policy")[METRIC_COLS].mean()
    header = f"{'policy':<12}" + "".join(f"{c:>16}" for c in METRIC_COLS)
    print(header)
    for policy in POLICIES:
        row = means.loc[policy]
        print(f"{policy:<12}" + "".join(f"{row[c]:>16.5f}" for c in METRIC_COLS))


def add_return_pct_column(df):
    """
    Return on turnover: raw_pnl / (total_fills * INITIAL_PRICE) * 100.

    Approximates each fill's notional as INITIAL_PRICE (the price scale the
    env is calibrated at) rather than the exact fill price, since fills
    occur at varying depths around a midprice that itself drifts over the
    episode -- INITIAL_PRICE is a fixed, reproducible reference scale for
    this ratio, not an attempt to reconstruct exact per-fill notional.
    Episodes with zero fills give NaN (no turnover to divide by).
    """
    df = df.copy()
    df["return_pct_turnover"] = np.where(
        df["total_fills"] > 0,
        df["raw_pnl"] / (df["total_fills"] * INITIAL_PRICE) * 100.0,
        np.nan,
    )
    return df


def print_return_pct_table(df):
    print("\n" + "=" * 60)
    print("RETURN ON TURNOVER: raw_pnl / (total_fills * INITIAL_PRICE) x 100%")
    print("=" * 60)
    header = f"{'policy':<12}{'mean %':>12}{'std %':>12}"
    print(header)
    stats = {}
    for policy in POLICIES:
        sub = df[df["policy"] == policy]["return_pct_turnover"].dropna()
        mean_pct = float(sub.mean())
        std_pct = float(sub.std(ddof=1))
        stats[policy] = dict(mean_pct=mean_pct, std_pct=std_pct)
        print(f"{policy:<12}{mean_pct:>11.4f}%{std_pct:>11.4f}%")
    return stats


def print_loss_rate_table(df):
    print("\n" + "=" * 60)
    print("LOSS RATE: % of episodes with negative outcome")
    print("=" * 60)
    header = f"{'policy':<12}{'% raw_pnl<0':>16}{'% full_objective<0':>20}"
    print(header)
    loss_rates = {}
    for policy in POLICIES:
        sub = df[df["policy"] == policy]
        pnl_loss_rate = float((sub["raw_pnl"] < 0).mean()) * 100.0
        obj_loss_rate = float((sub["full_objective"] < 0).mean()) * 100.0
        loss_rates[policy] = dict(pnl_loss_rate=pnl_loss_rate, obj_loss_rate=obj_loss_rate)
        print(f"{policy:<12}{pnl_loss_rate:>15.1f}%{obj_loss_rate:>19.1f}%")
    return loss_rates


def paired_diff_vs_naive(df):
    print("\n" + "=" * 100)
    print("PAIRED DIFFERENCES vs NAIVE (95% CI = mean +/- 1.96*SE)")
    print("=" * 100)
    piv = {c: df.pivot(index="episode", columns="policy", values=c) for c in ["raw_pnl", "full_objective"]}

    header = f"{'policy':<12}{'quantity':<14}{'mean':>12}{'std_error':>12}{'95% CI low':>14}{'95% CI high':>14}{'P(D>0)':>10}"
    print(header)
    all_results = {}
    for policy in ("oracle", "belief", "randomised"):
        all_results[policy] = {}
        for name, col in (("D_pnl", "raw_pnl"), ("D_objective", "full_objective")):
            D = piv[col][policy] - piv[col]["naive"]
            n = len(D)
            mean = D.mean()
            se = D.std(ddof=1) / np.sqrt(n)
            ci_lo, ci_hi = mean - 1.96 * se, mean + 1.96 * se
            p_pos = (D > 0).mean()
            all_results[policy][name] = dict(mean=mean, se=se, ci_lo=ci_lo, ci_hi=ci_hi, p_pos=p_pos)
            print(f"{policy:<12}{name:<14}{mean:>12.5f}{se:>12.5f}{ci_lo:>14.5f}{ci_hi:>14.5f}{p_pos:>10.3f}")
    return all_results


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--verify-episodes", type=int, default=20)
    p.add_argument("--base-seed", type=int, default=12345)
    p.add_argument("--output", type=str, default="results/four_policy_paired_1000.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 78)
    print("Environment parameters (unchanged, read from envs/make_envs.py)")
    print("=" * 78)
    print(f"T={TERMINAL_TIME}  n_steps={N_STEPS}  dt={DT}  PHI={PHI}  ALPHA={ALPHA}")
    assert N_STEPS == 4000 and abs(DT - TERMINAL_TIME / 4000) < 1e-15, \
        "Environment is not the corrected 4000-step grid."
    print(f"policies={POLICIES}")
    print(f"episodes={args.episodes}  verify_episodes={args.verify_episodes}  "
          f"base_seed={args.base_seed}  output={args.output}")

    t_total_start = time.time()
    df, n_mismatches, elapsed = run_comparison(
        n_episodes=args.episodes, n_verify=args.verify_episodes, base_seed=args.base_seed,
    )
    total_runtime = time.time() - t_total_start

    df = add_return_pct_column(df)
    print_summary_table(df)
    return_pct_stats = print_return_pct_table(df)
    loss_rates = print_loss_rate_table(df)
    results = paired_diff_vs_naive(df)

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(SCRIPT_DIR, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nEpisode-level results saved to: {output_path}")

    print("\n" + "=" * 78)
    print(f"TOTAL RUNTIME: {total_runtime:.1f}s   "
          f"AVERAGE PER PAIRED EPISODE-GROUP: {total_runtime/args.episodes:.4f}s")
    print("=" * 78)

    if n_mismatches > 0:
        raise SystemExit(f"{n_mismatches} exogenous-path mismatches detected -- investigate before trusting results.")
