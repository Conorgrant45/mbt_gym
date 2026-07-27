"""
compare_oracle_naive_paired.py
--------------------------------
Standalone diagnostic. Does NOT modify any production file -- imports the
existing oracle control table builder, get_control, normalise_depth and
NAIVE_ACTION from simulate_belief_weighted.py and uses them completely
unchanged. Uses the corrected 4000-step environment (envs/make_envs.py)
exactly as currently configured -- no parameter is altered.

Pairing mechanism
------------------
Two sources of randomness exist in this codebase:
  1. Regime-switching draws (RegimeSwitchingEnv.step/reset) use the LEGACY
     GLOBAL np.random API (np.random.choice), not a dedicated per-env RNG
     -- confirmed by reading envs/regime_env.py directly.
  2. Midprice diffusion/jump, arrival and fill draws each use their OWN
     independent numpy.random.default_rng, seeded via
     np.random.SeedSequence(seed).spawn(6) inside make_regime_envs(seed=...)
     -- these are deterministic given the seed, and do NOT depend on the
     global np.random state at all.

Neither source depends on the policy's action: get_arrivals() takes no
action argument at all; get_fills(depths) draws its uniform BEFORE
comparing against the depth-dependent threshold (the draw itself is
policy-independent, only the resulting fill/no-fill outcome differs); the
midprice model's diffusion/jump draws are likewise unconditional on action.

So pairing is achieved by, for each episode index:
    np.random.seed(episode_seed)              # resets the GLOBAL state
    env = make_regime_envs(..., seed=episode_seed)   # resets the CHILD (default_rng) streams
run oracle's full episode, THEN repeat the identical two seeding calls (a
FRESH environment is built, never a reset/reused one) and run naive's full
episode. Because both policies consume exactly one regime-draw and one
arrival/fill/jump/diffusion draw per step, for exactly 4000 steps, the two
runs draw bit-for-bit identical random numbers.

For the first --verify-episodes episodes this is checked explicitly (not
merely assumed), by instrumenting both runs and diffing the raw
regime/arrival/jump/diffusion sequences. For all later episodes, per-step
trace collection is disabled entirely (metrics are still computed exactly
the same way -- only the extra cross-policy trace arrays and the diffusion
(Brownian) log are skipped) so that 1000 paired episodes run efficiently.

Run:
    python compare_oracle_naive_paired.py --episodes 20 --verify-episodes 20
    python compare_oracle_naive_paired.py --episodes 1000 --verify-episodes 20 --output results/oracle_naive_paired_1000.csv
"""

import os
import time
import argparse
import numpy as np
import pandas as pd

import simulate_belief_weighted as SBW  # existing oracle/naive definitions -- READ ONLY
from envs.make_envs import (
    make_regime_envs, N_STEPS, TERMINAL_TIME, STEP_SIZE,
    PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION, INITIAL_PRICE,
)
from mbt_gym.gym.index_names import ASK_INDEX, BID_INDEX

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PHI = PER_STEP_INVENTORY_AVERSION
ALPHA = TERMINAL_INVENTORY_AVERSION
DT = STEP_SIZE

METRIC_COLS = [
    "raw_pnl", "full_objective", "spread_revenue", "adverse_selection_loss",
    "running_inventory_penalty", "terminal_inventory_penalty",
    "mean_absolute_inventory", "terminal_absolute_inventory", "total_fills",
]


# ======================================================================
# Recording RNG wrappers -- pure passthrough, log every draw. Two variants:
# the full one (verify episodes) logs normal+exponential+uniform for
# cross-policy trace comparison; the light one (all later episodes) only
# logs exponential (needed for adverse_selection_loss) and leaves
# normal/uniform completely unwrapped (zero added overhead there).
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
    """Lightweight variant: only wraps exponential (jump magnitudes), needed
    for adverse_selection_loss. normal/uniform pass straight through with no
    logging and no wrapper-call overhead beyond attribute delegation."""
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
    """
    Wrap both sub-environments' arrival/fill/midprice models to log draws
    needed for metrics (always) and, if full_trace, also the extra
    normal/uniform draws needed for cross-policy path verification.
    """
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

        if full_trace:
            md.midprice_model.rng = RecordingRNG(md.midprice_model.rng)
        else:
            md.midprice_model.rng = JumpOnlyRNG(md.midprice_model.rng)

        logs[regime_idx] = dict(arrival_log=arrival_log, fill_log=fill_log,
                                 midprice_rng=md.midprice_model.rng)
    return logs


# ======================================================================
# Single paired episode rollout for one policy (shared by both modes)
# ======================================================================
def run_episode(policy, controls, episode_seed, full_trace: bool):
    """
    policy: 'oracle' or 'naive'
    full_trace: if True, also builds and returns per-step trace arrays
                (regime, buy, sell, jump_up, jump_down, diffusion_z) for
                cross-policy verification. If False, only the scalar
                episode-level metrics are computed -- no per-step arrays
                are allocated, and diffusion draws are not logged at all.
    Returns (metrics: dict, trace: dict or None)
    """
    np.random.seed(episode_seed)  # pairs the GLOBAL-state regime-switching draws
    env = make_regime_envs(switch_within_episode=True, seed=episode_seed)  # fresh env, pairs the child RNG streams
    logs = instrument_env(env, full_trace)

    obs = env.reset()
    cash_0, inv_0, mid_0 = env.raw_cash, env.raw_inventory, env.raw_midprice

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
    obj_accum_env = 0.0  # cross-check via the environment's own returned reward

    step = 0
    done = np.array([False])
    while not np.all(done):
        regime = env.current_regime
        obs_flat = np.array(obs).flatten()
        inventory = obs_flat[1]

        if policy == "oracle":
            t_idx = env.current_step
            inv_sc = inventory / SBW.INV_UNIT
            ctrl = controls[regime]
            ask_depth, bid_depth = SBW.get_control(
                ctrl["delta_ask"], ctrl["delta_bid"], ctrl["q_ask"], ctrl["q_bid"], t_idx, inv_sc
            )
            action = np.array([[SBW.normalise_depth(bid_depth), SBW.normalise_depth(ask_depth)]])
        else:  # naive
            ask_depth = bid_depth = SBW.NAIVE_DEPTH
            action = SBW.NAIVE_ACTION.copy()

        pre_len_arr = len(logs[regime]["arrival_log"])
        pre_len_exp = len(logs[regime]["midprice_rng"].exponential_log)
        if full_trace:
            pre_len_norm = len(logs[regime]["midprice_rng"].normal_log)

        obs, reward, done, info = env.step(action)
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
            jump_up = jump_down = 0.0  # calm regime midprice model has no jump draws

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

    # Reconciliation: our decomposition must match the environment's own
    # cumulative reward (RunningInventoryPenalty with phi, alpha above).
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

        m_oracle, trace_oracle = run_episode("oracle", controls, seed, full_trace)
        m_naive, trace_naive = run_episode("naive", controls, seed, full_trace)

        if full_trace:
            n_verified += 1
            same = all(np.array_equal(trace_oracle[k], trace_naive[k]) for k in trace_oracle)
            if not same:
                n_path_mismatches += 1
                for k in trace_oracle:
                    if not np.array_equal(trace_oracle[k], trace_naive[k]):
                        n_diff = np.sum(trace_oracle[k] != trace_naive[k])
                        print(f"  [episode {ep}] MISMATCH in '{k}': {n_diff} of {N_STEPS} steps differ")
            assert m_oracle["steps_taken"] == N_STEPS and m_naive["steps_taken"] == N_STEPS
            assert same, f"episode {ep}: oracle/naive exogenous paths are NOT identical"

        # Retained for every episode (verify or not): steps_taken and reconciliation.
        assert m_oracle["steps_taken"] == N_STEPS, f"episode {ep} (oracle): steps_taken != {N_STEPS}"
        assert m_naive["steps_taken"] == N_STEPS, f"episode {ep} (naive): steps_taken != {N_STEPS}"

        for policy, m in (("oracle", m_oracle), ("naive", m_naive)):
            row = dict(episode=ep, policy=policy)
            row.update(m)
            rows.append(row)

        if (ep + 1) % progress_every == 0:
            elapsed = time.time() - t_start
            print(f"  ... {ep+1}/{n_episodes} paired episodes done, {elapsed:.1f}s elapsed "
                  f"({elapsed/(ep+1):.3f}s/pair so far)")

    elapsed = time.time() - t_start
    print(f"\nCompleted {n_episodes} paired episodes in {elapsed:.1f}s "
          f"({elapsed/n_episodes:.4f}s/episode pair average)")
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
    print("\n" + "=" * 100)
    print("SUMMARY TABLE 1: per-policy means")
    print("=" * 100)
    means = df.groupby("policy")[METRIC_COLS].mean()
    header = f"{'policy':<10}" + "".join(f"{c:>16}" for c in METRIC_COLS)
    print(header)
    for policy in ("oracle", "naive"):
        row = means.loc[policy]
        print(f"{policy:<10}" + "".join(f"{row[c]:>16.5f}" for c in METRIC_COLS))


def paired_diff_report(df):
    print("\n" + "=" * 100)
    print("SUMMARY TABLE 2: paired oracle-minus-naive differences (95% CI = mean +/- 1.96*SE)")
    print("=" * 100)
    piv_pnl = df.pivot(index="episode", columns="policy", values="raw_pnl")
    piv_obj = df.pivot(index="episode", columns="policy", values="full_objective")
    piv_spread = df.pivot(index="episode", columns="policy", values="spread_revenue")
    piv_adverse = df.pivot(index="episode", columns="policy", values="adverse_selection_loss")

    D_pnl = piv_pnl["oracle"] - piv_pnl["naive"]
    D_objective = piv_obj["oracle"] - piv_obj["naive"]
    D_spread = piv_spread["oracle"] - piv_spread["naive"]
    D_adverse = piv_adverse["oracle"] - piv_adverse["naive"]

    results = {}
    header = f"{'quantity':<14}{'mean':>12}{'std_error':>12}{'95% CI low':>14}{'95% CI high':>14}{'P(D>0)':>10}"
    print(header)
    for name, D in (("D_pnl", D_pnl), ("D_objective", D_objective),
                     ("D_spread", D_spread), ("D_adverse", D_adverse)):
        n = len(D)
        mean = D.mean()
        se = D.std(ddof=1) / np.sqrt(n)
        ci_lo, ci_hi = mean - 1.96 * se, mean + 1.96 * se
        p_pos = (D > 0).mean()
        results[name] = dict(mean=mean, se=se, ci_lo=ci_lo, ci_hi=ci_hi, p_pos=p_pos)
        print(f"{name:<14}{mean:>12.5f}{se:>12.5f}{ci_lo:>14.5f}{ci_hi:>14.5f}{p_pos:>10.3f}")
    return results


def interpretation(results, df):
    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    d_obj = results["D_objective"]
    sig_obj = d_obj["ci_lo"] > 0 or d_obj["ci_hi"] < 0
    print(f"1. Does the oracle significantly outperform on the full objective? "
          f"{'YES' if (sig_obj and d_obj['mean']>0) else ('NO (significant but worse)' if sig_obj else 'NO (95% CI includes 0)')} "
          f"(mean D_objective={d_obj['mean']:.5f}, 95% CI=[{d_obj['ci_lo']:.5f},{d_obj['ci_hi']:.5f}])")

    d_pnl = results["D_pnl"]
    sig_pnl = d_pnl["ci_lo"] > 0 or d_pnl["ci_hi"] < 0
    print(f"2. Does the oracle significantly outperform on raw PnL? "
          f"{'YES' if (sig_pnl and d_pnl['mean']>0) else ('NO (significant but worse)' if sig_pnl else 'NO (95% CI includes 0)')} "
          f"(mean D_pnl={d_pnl['mean']:.5f}, 95% CI=[{d_pnl['ci_lo']:.5f},{d_pnl['ci_hi']:.5f}])")

    d_adv = results["D_adverse"]
    sig_adv = d_adv["ci_lo"] > 0 or d_adv["ci_hi"] < 0
    reduces = sig_adv and d_adv["mean"] < 0
    print(f"3. Does the oracle reduce adverse-selection losses? "
          f"{'YES' if reduces else ('NO -- higher adverse-selection loss' if (sig_adv and d_adv['mean']>0) else 'NO significant difference')} "
          f"(mean D_adverse={d_adv['mean']:.5f}, 95% CI=[{d_adv['ci_lo']:.5f},{d_adv['ci_hi']:.5f}])")

    d_spread = results["D_spread"]
    print(f"4. Spread revenue sacrificed by the oracle: mean D_spread={d_spread['mean']:.5f} "
          f"(negative = oracle earns LESS spread revenue than naive), "
          f"95% CI=[{d_spread['ci_lo']:.5f},{d_spread['ci_hi']:.5f}]")

    naive_mean_obj = df[df.policy == "naive"]["full_objective"].mean()
    rel_pct = 100 * d_obj["mean"] / abs(naive_mean_obj) if naive_mean_obj != 0 else float("nan")
    print(f"5. Is the net difference economically meaningful? mean D_objective is "
          f"{rel_pct:.3f}% of naive's mean objective ({naive_mean_obj:.5f}) -- "
          f"{'a meaningful' if abs(rel_pct) > 5 else 'a small'} relative difference"
          f"{', but not statistically distinguishable from 0' if not sig_obj else ''}.")

    excludes_zero = d_obj["ci_lo"] > 0 or d_obj["ci_hi"] < 0
    print(f"\nD_objective 95% CI excludes zero? {'YES' if excludes_zero else 'NO'} "
          f"([{d_obj['ci_lo']:.5f}, {d_obj['ci_hi']:.5f}])")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--verify-episodes", type=int, default=20)
    p.add_argument("--base-seed", type=int, default=12345)
    p.add_argument("--output", type=str, default="results/oracle_naive_paired_1000.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 78)
    print("Environment parameters (unchanged, read from envs/make_envs.py)")
    print("=" * 78)
    print(f"T={TERMINAL_TIME}  n_steps={N_STEPS}  dt={DT}  PHI={PHI}  ALPHA={ALPHA}")
    assert N_STEPS == 4000 and abs(DT - TERMINAL_TIME / 4000) < 1e-15, \
        "Environment is not the corrected 4000-step grid."
    print(f"episodes={args.episodes}  verify_episodes={args.verify_episodes}  "
          f"base_seed={args.base_seed}  output={args.output}")

    t_total_start = time.time()
    df, n_mismatches, elapsed = run_comparison(
        n_episodes=args.episodes, n_verify=args.verify_episodes, base_seed=args.base_seed,
    )
    total_runtime = time.time() - t_total_start

    print_summary_table(df)
    results = paired_diff_report(df)
    interpretation(results, df)

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(SCRIPT_DIR, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nEpisode-level results saved to: {output_path}")

    print("\n" + "=" * 78)
    print(f"TOTAL RUNTIME: {total_runtime:.1f}s   "
          f"AVERAGE PER PAIRED EPISODE: {total_runtime/args.episodes:.4f}s")
    print("=" * 78)

    if n_mismatches > 0:
        raise SystemExit(f"{n_mismatches} exogenous-path mismatches detected -- investigate before trusting results.")
