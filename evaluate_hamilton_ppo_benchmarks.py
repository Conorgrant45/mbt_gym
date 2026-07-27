"""
evaluate_hamilton_ppo_benchmarks.py
---------------------------------------
Paired benchmark comparison (item 2): naive / Hamilton belief-weighted /
oracle analytic policies vs. the trained Hamilton PPO, on EXACTLY the
same exogenous seeds.

Reuses the repository's already-validated paired-path machinery rather
than implementing a new pairing system:
    - simulate_belief_weighted.py (SBW): build_optimal_control,
      REGIME_PARAMS, MAX_DEPTH, NAIVE_DEPTH, get_control -- READ ONLY.
    - compare_four_policies_paired.py (C4P): run_episode() is called
      UNCHANGED for the four existing analytic/belief-driven policies;
      instrument_env / RecordingRNG / JumpOnlyRNG are imported and reused
      (not reimplemented) for the new PPO episode function, which mirrors
      C4P.run_episode's exact accounting so results are directly
      comparable.
    - envs/hamilton_ppo_wrapper.py: make_filter() (same Hamilton filter
      configuration the trained PPO was trained with).

Does not modify any of the above files, market dynamics, reward, or the
trained model.

Run from repo root:
    python evaluate_hamilton_ppo_benchmarks.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

import simulate_belief_weighted as SBW  # READ ONLY
import compare_four_policies_paired as C4P  # READ ONLY -- reuses run_episode, instrument_env, RNG wrappers
from envs.make_envs import N_STEPS, PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION, STEP_SIZE, make_regime_envs
from envs.hamilton_ppo_wrapper import make_filter
from mbt_gym.gym.index_names import ASK_INDEX, BID_INDEX

import hamilton_ppo_eval_lib as lib

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
MODEL_PATH = REPO_ROOT / "models" / "hamilton_ppo" / "ppo_hamilton_v1.zip"
INVENTORY_SCALE = 10.0

PHI = PER_STEP_INVENTORY_AVERSION
ALPHA = TERMINAL_INVENTORY_AVERSION
DT = STEP_SIZE

POLICIES = ["oracle", "belief", "randomised", "naive", "ppo"]
N_VERIFY = 10  # first N_VERIFY seeds get full exogenous-path tracing/comparison


def run_episode_ppo(model, episode_seed: int, full_trace: bool):
    """
    Mirrors compare_four_policies_paired.py's run_episode() accounting
    exactly (same reconciliation identity, same instrumentation), for the
    trained Hamilton PPO policy -- which C4P.run_episode itself has no
    branch for.
    """
    np.random.seed(episode_seed)  # pairs the GLOBAL-state regime-switching draws, same as C4P.run_episode
    env = make_regime_envs(switch_within_episode=True, seed=episode_seed)
    logs = C4P.instrument_env(env, full_trace)

    filt = make_filter()
    filt.reset()

    obs = env.reset()
    cash_0, inv_0, mid_0 = env.raw_cash, env.raw_inventory, env.raw_midprice
    mid = mid_0
    belief = filt.update(mid)  # b_0 -- matches HamiltonPPOWrapper.reset()'s first-call behaviour

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
        inv_raw = env.raw_inventory
        t_idx = env.current_step
        q_scaled = float(np.tanh(inv_raw / INVENTORY_SCALE))
        tau = 1.0 - t_idx / N_STEPS
        obs_ppo = np.array([q_scaled, tau, belief], dtype=np.float32)

        action, _ = model.predict(obs_ppo, deterministic=True)
        bid_action, ask_action = float(action[0]), float(action[1])
        ask_depth = (ask_action + 1.0) / 2.0 * SBW.MAX_DEPTH
        bid_depth = (bid_action + 1.0) / 2.0 * SBW.MAX_DEPTH
        action_arr = np.array([[bid_action, ask_action]], dtype=np.float32)  # already normalised [-1,1]

        pre_len_arr = len(logs[regime]["arrival_log"])
        pre_len_exp = len(logs[regime]["midprice_rng"].exponential_log)
        if full_trace:
            pre_len_norm = len(logs[regime]["midprice_rng"].normal_log)

        obs, reward, done, info = env.step(action_arr)
        mid = info["raw_midprice"]
        obj_accum_env += float(np.sum(reward))
        belief = filt.update(mid)  # belief FOR THE NEXT decision -- matches HamiltonPPOWrapper.step()

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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = PPO.load(str(MODEL_PATH))
    print(f"Loaded model from {MODEL_PATH}")

    print("Building oracle optimal control tables (unchanged existing solver)...")
    controls = {}
    for regime, params in SBW.REGIME_PARAMS.items():
        da, db, qag, qbg = SBW.build_optimal_control(**params)
        controls[regime] = {"delta_ask": da, "delta_bid": db, "q_ask": qag, "q_bid": qbg}

    seeds = lib.BENCHMARK_SEEDS
    print(f"Paired benchmark comparison on {len(seeds)} seeds: {seeds[0]}..{seeds[-1]}, policies={POLICIES}")

    rows = []
    n_path_mismatches = 0
    n_verified = 0
    for i, seed in enumerate(seeds):
        full_trace = i < N_VERIFY
        results = {}
        for policy in POLICIES:
            if policy == "ppo":
                m, tr = run_episode_ppo(model, seed, full_trace)
            else:
                m, tr = C4P.run_episode(policy, controls, seed, full_trace)
            results[policy] = (m, tr)
            assert m["steps_taken"] == N_STEPS, f"seed {seed} ({policy}): steps_taken != {N_STEPS}"
            assert m["reconciliation_abs_diff"] < 1e-6, (
                f"seed {seed} ({policy}): objective reconciliation failed "
                f"(diff={m['reconciliation_abs_diff']:.3e})"
            )

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
                            print(f"  [seed {seed}] MISMATCH oracle vs {pol} in '{k}': {n_diff}/{N_STEPS} steps differ")
            assert same, f"seed {seed}: exogenous paths are NOT identical across all five policies"

        for policy in POLICIES:
            row = dict(seed=seed, policy=policy)
            row.update(results[policy][0])
            rows.append(row)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(seeds)} paired seeds done")

    print(f"\nExogenous-path verification: {n_verified} seeds fully traced, {n_path_mismatches} mismatches "
          f"across all 5 policies -- {'PASSED' if n_path_mismatches == 0 else 'FAILED'}")

    df = pd.DataFrame(rows)
    max_recon = df["reconciliation_abs_diff"].max()
    print(f"Objective reconciliation: max |discrepancy| across all {len(df)} (seed,policy) rows = {max_recon:.3e} -- "
          f"{'PASSED' if max_recon < 1e-6 else 'FAILED'}")

    out_path = RESULTS_DIR / "hamilton_ppo_paired_benchmarks_v1.csv"
    df.to_csv(out_path, index=False)
    print(f"\nPer-episode paired benchmark results saved to {out_path}")

    print("\n" + "=" * 110)
    print("SUMMARY: per-policy means (raw units throughout -- no mixing of normalised/raw scales)")
    print("=" * 110)
    metric_cols = ["raw_pnl", "full_objective", "spread_revenue", "adverse_selection_loss",
                   "running_inventory_penalty", "terminal_inventory_penalty",
                   "mean_absolute_inventory", "terminal_absolute_inventory", "total_fills"]
    means = df.groupby("policy")[metric_cols].mean()
    print(means.reindex(POLICIES).to_string(float_format=lambda x: f"{x:.4f}"))

    # --- Paired comparisons vs PPO ---
    print("\n" + "=" * 110)
    print("PAIRED COMPARISONS: (analytic policy) minus (Hamilton PPO)")
    print("=" * 110)
    piv_obj = df.pivot(index="seed", columns="policy", values="full_objective")
    piv_pnl = df.pivot(index="seed", columns="policy", values="raw_pnl")
    piv_inv = df.pivot(index="seed", columns="policy", values="mean_absolute_inventory")
    piv_asl = df.pivot(index="seed", columns="policy", values="adverse_selection_loss")

    comparison_rows = []
    for policy in ("oracle", "belief", "randomised", "naive"):
        D_obj = piv_obj[policy] - piv_obj["ppo"]
        n = len(D_obj)
        mean = float(D_obj.mean())
        se = float(D_obj.std(ddof=1) / np.sqrt(n))
        ci_lo, ci_hi = mean - 1.96 * se, mean + 1.96 * se
        median = float(D_obj.median())
        win_rate = float((D_obj > 0).mean())

        D_pnl = float((piv_pnl[policy] - piv_pnl["ppo"]).mean())
        D_inv = float((piv_inv[policy] - piv_inv["ppo"]).mean())
        D_asl = float((piv_asl[policy] - piv_asl["ppo"]).mean())

        comparison_rows.append(dict(
            policy=policy, n=n,
            mean_objective_diff=mean, se_objective_diff=se,
            ci_lo=ci_lo, ci_hi=ci_hi, median_objective_diff=median, win_rate=win_rate,
            mean_raw_pnl_diff=D_pnl, mean_inventory_risk_diff=D_inv, mean_adverse_selection_loss_diff=D_asl,
        ))
        print(f"\n{policy} vs ppo (full_objective, price units):")
        print(f"  mean diff = {mean:.4f}  SE={se:.4f}  95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]  "
              f"median={median:.4f}  win_rate({policy}>ppo)={win_rate:.3f}")
        print(f"  raw_pnl diff (price units) = {D_pnl:.4f}   "
              f"mean|inv| diff (shares) = {D_inv:.4f}   "
              f"adverse_selection_loss diff (price units) = {D_asl:.4f}")

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = RESULTS_DIR / "hamilton_ppo_paired_benchmarks_summary_v1.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nPaired comparison summary saved to {comparison_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
