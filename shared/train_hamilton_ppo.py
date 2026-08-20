"""
shared/train_hamilton_ppo.py
------------------------
Feed-forward PPO training/validation pipeline for HamiltonPPOWrapper --
milestone 1 of the RL stage. Strengthened version: adds periodic
evaluation on FIXED, held-out seeds (never reused for training), an
untrained-policy baseline, save/reload equivalence checking, an explicit
objective/PnL/penalty decomposition, and full run-configuration logging.

Does NOT implement recurrent PPO. Does NOT change market dynamics, the
Hamilton filter, the reward function, or the PPO observation architecture
(all of that lives in envs/hamilton_ppo_wrapper.py, untouched here).

A run at --total-timesteps 200000 (this script's default) is a LEARNING-
VALIDATION run -- it establishes whether the pipeline actually learns
something better than the untrained policy. It is NOT the final
dissertation training run and should not be reported as one.

Run from repo root:
    python shared/train_hamilton_ppo.py
    python shared/train_hamilton_ppo.py --total-timesteps 200000 --seed 0
"""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gym as legacy_gym
import gymnasium
import stable_baselines3

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.hamilton_ppo_wrapper import HamiltonPPOWrapper, DEFAULT_INVENTORY_SCALE
from envs.make_envs import (
    make_regime_envs, N_STEPS, STEP_SIZE,
    PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION,
)
from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, ASSET_PRICE_INDEX, ASK_INDEX, BID_INDEX

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models" / "hamilton_ppo"
LOGS_DIR = REPO_ROOT / "logs" / "hamilton_ppo"
RESULTS_DIR = REPO_ROOT / "results"

PHI = PER_STEP_INVENTORY_AVERSION
ALPHA = TERMINAL_INVENTORY_AVERSION
DT = STEP_SIZE

# Fixed, held-out evaluation seeds -- never used for training, and reused
# IDENTICALLY across the untrained baseline, every periodic checkpoint,
# and the final/reload evaluations, so every checkpoint's number is
# directly comparable (the same fixed "test set" of episodes throughout).
DEFAULT_EVAL_SEEDS = [90001, 90002, 90003, 90004, 90005]

ACTION_BOUND_TOL = 0.01  # "near a bound" threshold on the [-1,1] normalised action scale
RECONCILIATION_TOL = 1e-6


# ======================================================================
# Instrumentation (observe-only; does not alter dynamics) -- same
# technique as compare_four_policies_paired.py / tests/test_hamilton_ppo_
# wrapper.py's reproducibility tests, reused here for fill counting.
# ======================================================================
def instrument_arrival_fill(base_env):
    logs = {0: {"arrivals": [], "fills": []}, 1: {"arrivals": [], "fills": []}}
    for regime_idx in (0, 1):
        md = base_env.envs[regime_idx].model_dynamics
        log = logs[regime_idx]

        orig_get_arrivals = md.arrival_model.get_arrivals

        def arrivals_hook(orig=orig_get_arrivals, log=log):
            arr = orig()
            log["arrivals"].append(arr.copy())
            return arr

        md.arrival_model.get_arrivals = arrivals_hook

        orig_get_fills = md.fill_probability_model.get_fills

        def fills_hook(depths, orig=orig_get_fills, log=log):
            f = orig(depths)
            log["fills"].append(f.copy())
            return f

        md.fill_probability_model.get_fills = fills_hook
    return logs


# ======================================================================
# Evaluation: single episode with full objective/PnL/penalty decomposition
# ======================================================================
def build_eval_env(seed: int, inventory_scale: float):
    """
    A FRESH base_env + wrapper for this one episode. Never reuse an
    already-stepped wrapper via reset(seed=s) for reproducibility -- see
    tests/test_hamilton_ppo_wrapper.py's TestSeedReset, which established
    empirically that reset(seed=s) reseeds only the regime-transition
    draws (legacy global np.random stream), NOT the midprice/arrival/fill
    model RNGs (fixed once at construction). Building fresh here avoids
    that pitfall entirely.
    """
    base_env = make_regime_envs(switch_within_episode=True, seed=seed)
    logs = instrument_arrival_fill(base_env)
    wrapper = HamiltonPPOWrapper(base_env=base_env, inventory_scale=inventory_scale)
    return wrapper, logs


def run_eval_episode(model, seed: int, inventory_scale: float, deterministic: bool = True):
    wrapper, logs = build_eval_env(seed, inventory_scale)
    obs, info = wrapper.reset(seed=seed)

    cash_0 = wrapper.base_env.raw_cash
    inv_0 = wrapper.base_env.raw_inventory
    mid_0 = wrapper.base_env.raw_midprice

    running_inventory_penalty = 0.0
    inv_abs_sum = 0.0
    total_fills = 0.0
    cumulative_objective = 0.0  # == cumulative environment reward, the quantity PPO actually optimises
    actions_taken = []

    terminated = truncated = False
    n_steps = 0
    info = None
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        actions_taken.append(np.asarray(action, dtype=np.float64).copy())

        regime = wrapper.base_env.current_regime
        pre_len = len(logs[regime]["arrivals"])

        obs, reward, terminated, truncated, info = wrapper.step(action)

        arrivals = logs[regime]["arrivals"][pre_len]
        fills = logs[regime]["fills"][pre_len]
        buy_arrival = float(arrivals[0, ASK_INDEX])
        sell_arrival = float(arrivals[0, BID_INDEX])
        ask_fill_attempt = float(fills[0, ASK_INDEX])
        bid_fill_attempt = float(fills[0, BID_INDEX])
        total_fills += buy_arrival * ask_fill_attempt + sell_arrival * bid_fill_attempt

        inv_after = float(info["raw_state"][INVENTORY_INDEX])
        inv_abs_sum += abs(inv_after)
        running_inventory_penalty += PHI * (inv_after ** 2) * DT

        cumulative_objective += float(reward)
        n_steps += 1

    cash_T = float(info["raw_state"][CASH_INDEX])
    inv_T = float(info["raw_state"][INVENTORY_INDEX])
    mid_T = float(info["raw_state"][ASSET_PRICE_INDEX])

    # Raw mark-to-market PnL: cash + inventory value, START to END. This is
    # NOT the same quantity as cumulative_objective (the PPO reward sum) --
    # it excludes the running/terminal inventory penalties. Never label
    # this "raw PnL" as the training objective, and vice versa.
    raw_pnl = (cash_T + inv_T * mid_T) - (cash_0 + inv_0 * mid_0)
    terminal_inventory_penalty = ALPHA * (inv_T ** 2)
    full_objective = raw_pnl - running_inventory_penalty - terminal_inventory_penalty
    reconciliation_abs_diff = abs(full_objective - cumulative_objective)

    return dict(
        seed=seed,
        steps=n_steps,
        cumulative_objective=cumulative_objective,
        raw_pnl=raw_pnl,
        running_inventory_penalty=running_inventory_penalty,
        terminal_inventory_penalty=terminal_inventory_penalty,
        full_objective=full_objective,
        reconciliation_abs_diff=reconciliation_abs_diff,
        mean_abs_inventory=inv_abs_sum / n_steps,
        terminal_abs_inventory=abs(inv_T),
        total_fills=total_fills,
        actions=np.array(actions_taken),  # shape (n_steps, 2)
    )


def evaluate_policy_detailed(model, eval_seeds, inventory_scale: float,
                              action_low, action_high, deterministic: bool = True):
    """
    Run one episode per seed in eval_seeds (fresh env each time -- see
    build_eval_env) and aggregate into a summary dict. Returns
    (summary, per_episode_records) -- per_episode_records is a list of
    the dicts from run_eval_episode (including the raw actions array, for
    save/reload equivalence comparisons).
    """
    records = [run_eval_episode(model, s, inventory_scale, deterministic) for s in eval_seeds]

    for r in records:
        assert r["reconciliation_abs_diff"] < RECONCILIATION_TOL, (
            f"Objective decomposition failed to reconcile for seed {r['seed']}: "
            f"|full_objective - cumulative_objective| = {r['reconciliation_abs_diff']:.3e} "
            f"(full_objective={r['full_objective']}, cumulative_objective={r['cumulative_objective']})"
        )
        assert r["steps"] == N_STEPS, f"eval episode (seed={r['seed']}) did not run the full {N_STEPS} steps"

    all_actions = np.concatenate([r["actions"] for r in records], axis=0)  # (n_episodes*4000, 2)
    action_mean = all_actions.mean(axis=0)
    action_std = all_actions.std(axis=0)
    near_low = np.abs(all_actions - action_low) < ACTION_BOUND_TOL
    near_high = np.abs(all_actions - action_high) < ACTION_BOUND_TOL
    frac_near_bound_per_dim = np.mean(near_low | near_high, axis=0)
    frac_near_bound_any_dim = float(np.mean(np.any(near_low | near_high, axis=1)))

    objectives = np.array([r["cumulative_objective"] for r in records])
    pnls = np.array([r["raw_pnl"] for r in records])
    running_pens = np.array([r["running_inventory_penalty"] for r in records])
    terminal_pens = np.array([r["terminal_inventory_penalty"] for r in records])
    mean_abs_invs = np.array([r["mean_abs_inventory"] for r in records])
    terminal_abs_invs = np.array([r["terminal_abs_inventory"] for r in records])
    fills = np.array([r["total_fills"] for r in records])

    summary = dict(
        n_episodes=len(records),
        mean_objective=float(objectives.mean()),
        std_objective=float(objectives.std()),
        mean_raw_pnl=float(pnls.mean()),
        std_raw_pnl=float(pnls.std()),
        mean_running_inventory_penalty=float(running_pens.mean()),
        mean_terminal_inventory_penalty=float(terminal_pens.mean()),
        mean_abs_inventory=float(mean_abs_invs.mean()),
        mean_terminal_abs_inventory=float(terminal_abs_invs.mean()),
        mean_fills=float(fills.mean()),
        action_mean_bid=float(action_mean[0]),
        action_mean_ask=float(action_mean[1]),
        action_std_bid=float(action_std[0]),
        action_std_ask=float(action_std[1]),
        frac_near_bound_bid=float(frac_near_bound_per_dim[0]),
        frac_near_bound_ask=float(frac_near_bound_per_dim[1]),
        frac_near_bound_any=frac_near_bound_any_dim,
        max_reconciliation_abs_diff=float(max(r["reconciliation_abs_diff"] for r in records)),
    )
    return summary, records


# ======================================================================
# Periodic evaluation callback
# ======================================================================
class PeriodicEvalCallback(BaseCallback):
    """
    Runs evaluate_policy_detailed on the FIXED eval_seeds every eval_freq
    training timesteps. Fully separate from training: constructs fresh
    environments, never touches the training vec env.
    """

    def __init__(self, eval_seeds, inventory_scale: float, eval_freq: int,
                 action_low, action_high, training_curve: list, verbose: int = 1):
        super().__init__(verbose)
        self.eval_seeds = eval_seeds
        self.inventory_scale = inventory_scale
        self.eval_freq = eval_freq
        self.action_low = action_low
        self.action_high = action_high
        self.training_curve = training_curve  # appended to in place, read by caller after learn()
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            self._last_eval_step = self.num_timesteps
            self._run_eval()
        return True

    def _run_eval(self):
        summary, _ = evaluate_policy_detailed(
            self.model, self.eval_seeds, self.inventory_scale,
            self.action_low, self.action_high, deterministic=True,
        )
        summary["total_timesteps"] = self.num_timesteps
        summary["tag"] = f"timestep_{self.num_timesteps}"
        self.training_curve.append(summary)

        if self.verbose:
            print(f"\n[periodic eval @ {self.num_timesteps} timesteps] "
                  f"mean_objective={summary['mean_objective']:.4f} "
                  f"(std={summary['std_objective']:.4f})  "
                  f"mean_raw_pnl={summary['mean_raw_pnl']:.4f}  "
                  f"mean|inv|={summary['mean_abs_inventory']:.3f}  "
                  f"mean_fills={summary['mean_fills']:.1f}  "
                  f"frac_near_bound={summary['frac_near_bound_any']:.3f}\n")

        for key in ("mean_objective", "std_objective", "mean_raw_pnl", "mean_abs_inventory",
                    "mean_terminal_abs_inventory", "mean_fills", "frac_near_bound_any"):
            self.logger.record(f"eval/{key}", summary[key])
        self.logger.record("eval/action_mean_bid", summary["action_mean_bid"])
        self.logger.record("eval/action_mean_ask", summary["action_mean_ask"])


# ======================================================================
# Config / environment helpers
# ======================================================================
def get_dependency_versions():
    return dict(
        python=sys.version,
        platform=platform.platform(),
        gym=legacy_gym.__version__,
        gymnasium=gymnasium.__version__,
        stable_baselines3=stable_baselines3.__version__,
        torch=torch.__version__,
        numpy=np.__version__,
        pandas=pd.__version__,
    )


def introspect_architecture(model):
    """Confirm (not just declare) the actual built actor/critic layer
    shapes, read directly from the constructed torch modules."""
    def linear_out_features(seq_module):
        return [m.out_features for m in seq_module if isinstance(m, torch.nn.Linear)]

    mlp_extractor = model.policy.mlp_extractor
    return dict(
        actor_hidden_layers=linear_out_features(mlp_extractor.policy_net),
        critic_hidden_layers=linear_out_features(mlp_extractor.value_net),
        actor_output_dim=model.policy.action_net.out_features,
        critic_output_dim=model.policy.value_net.out_features,
        activation_fn=model.policy.activation_fn.__name__,
        total_parameters=sum(p.numel() for p in model.policy.parameters()),
    )


def make_train_env_fn(inventory_scale: float, seed: int):
    def _init():
        env = HamiltonPPOWrapper(inventory_scale=inventory_scale, seed=seed)
        return Monitor(env)
    return _init


def assert_no_nan_params(model: PPO):
    for name, param in model.policy.named_parameters():
        assert torch.isfinite(param).all(), f"Non-finite value found in policy parameter '{name}'"


# ======================================================================
# CLI
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0, help="Training seed (PPO init + training env's regime draws).")
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=4_000,
                    help="Rollout length. Default 4000 = exactly one full episode per rollout "
                         "(see report for the n_steps=2000-vs-4000 comparison).")
    p.add_argument("--batch-size", type=int, default=400,
                    help="Must evenly divide n_steps*n_envs. 4000/400=10 minibatches/epoch.")
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=1.0,
                    help="Undiscounted finite-horizon objective -- only change given a concrete numerical problem.")
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--net-arch", type=int, nargs="+", default=[64, 64])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--inventory-scale", type=float, default=DEFAULT_INVENTORY_SCALE)
    p.add_argument("--eval-seeds", type=int, nargs="+", default=DEFAULT_EVAL_SEEDS)
    p.add_argument("--eval-freq", type=int, default=40_000)
    p.add_argument("--model-path", type=str, default=str(MODELS_DIR / "ppo_hamilton_v1"))
    p.add_argument("--run-tag", type=str, default="v1")
    return p.parse_args()


def main():
    args = parse_args()
    t_run_start = time.time()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dep_versions = get_dependency_versions()

    print("=" * 78)
    print(f"Hamilton PPO training run [{args.run_tag}] -- LEARNING-VALIDATION run, "
          f"NOT a final dissertation result")
    print("=" * 78)
    for k, v in vars(args).items():
        print(f"  {k} = {v}")
    print("=" * 78)

    # --- Build training env ---
    np.random.seed(args.seed)
    vec_env = DummyVecEnv([make_train_env_fn(args.inventory_scale, args.seed)])
    action_low = np.asarray(vec_env.action_space.low, dtype=np.float64)
    action_high = np.asarray(vec_env.action_space.high, dtype=np.float64)

    policy_kwargs = dict(net_arch=list(args.net_arch))
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        device=args.device,
        seed=args.seed,
        verbose=1,
    )

    architecture = introspect_architecture(model)
    print("\nConfirmed actor/critic architecture (introspected from the built model):")
    for k, v in architecture.items():
        print(f"  {k} = {v}")

    n_envs = vec_env.num_envs
    n_minibatches = (args.n_steps * n_envs) // args.batch_size
    rollout_geometry = dict(
        n_steps=args.n_steps,
        n_envs=n_envs,
        buffer_size=args.n_steps * n_envs,
        episode_length=N_STEPS,
        rollouts_span_full_episode=(args.n_steps % N_STEPS == 0),
        episodes_per_rollout=args.n_steps / N_STEPS,
        n_minibatches_per_epoch=n_minibatches,
        n_gradient_updates_per_rollout=n_minibatches * args.n_epochs,
        n_rollouts_total=args.total_timesteps // args.n_steps,
    )
    print("\nRollout geometry:")
    for k, v in rollout_geometry.items():
        print(f"  {k} = {v}")

    # --- Write run config BEFORE training (so it's on disk even if training crashes) ---
    run_config = dict(
        run_tag=args.run_tag,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        cli_args=vars(args),
        architecture=architecture,
        rollout_geometry=rollout_geometry,
        dependency_versions=dep_versions,
        resolved_device=str(model.device),
        observation_scaling=dict(
            inventory_scale=args.inventory_scale,
            transform="tanh(raw_inventory / inventory_scale)",
        ),
        eval_seeds=args.eval_seeds,
    )
    config_path = LOGS_DIR / f"run_config_{args.run_tag}.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2, default=str)
    print(f"\nRun config written to {config_path}")

    # --- Untrained-policy baseline evaluation (BEFORE any training) ---
    print("\nRunning untrained-policy baseline evaluation on fixed eval seeds...")
    training_curve = []
    baseline_summary, _ = evaluate_policy_detailed(
        model, args.eval_seeds, args.inventory_scale, action_low, action_high, deterministic=True,
    )
    baseline_summary["total_timesteps"] = 0
    baseline_summary["tag"] = "baseline_untrained"
    training_curve.append(baseline_summary)
    print(f"  untrained: mean_objective={baseline_summary['mean_objective']:.4f} "
          f"(std={baseline_summary['std_objective']:.4f})  "
          f"mean_raw_pnl={baseline_summary['mean_raw_pnl']:.4f}  "
          f"mean|inv|={baseline_summary['mean_abs_inventory']:.3f}  "
          f"mean_fills={baseline_summary['mean_fills']:.1f}  "
          f"frac_near_bound={baseline_summary['frac_near_bound_any']:.3f}")

    # --- Train, with periodic evaluation ---
    eval_callback = PeriodicEvalCallback(
        eval_seeds=args.eval_seeds,
        inventory_scale=args.inventory_scale,
        eval_freq=args.eval_freq,
        action_low=action_low,
        action_high=action_high,
        training_curve=training_curve,
    )

    print("\nTraining...")
    t_train_start = time.time()
    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback, progress_bar=False)
    train_elapsed = time.time() - t_train_start
    print(f"Training finished without crashing in {train_elapsed:.1f}s "
          f"({args.total_timesteps/train_elapsed:.0f} steps/s).")

    # --- Finite-loss / NaN-parameter checks ---
    logged = dict(model.logger.name_to_value)
    loss_entries = {k: v for k, v in logged.items() if "loss" in k or "kl" in k or "entropy" in k}
    print("\nFinal logged training metrics:")
    for k, v in sorted(loss_entries.items()):
        print(f"  {k} = {v}")
    numerical_instability = []
    for k, v in loss_entries.items():
        if not np.isfinite(v):
            numerical_instability.append(f"{k}={v}")
    assert not numerical_instability, f"Non-finite training metric(s) detected: {numerical_instability}"
    print("All logged losses/metrics are finite.")

    assert_no_nan_params(model)
    print("No NaNs/infs found in policy parameters.")

    # --- Save ---
    model.save(args.model_path)
    print(f"\nModel saved to {args.model_path}.zip")

    # --- Final (post-training) evaluation, same fixed eval seeds ---
    print("\nRunning final post-training evaluation on fixed eval seeds...")
    final_summary, final_records = evaluate_policy_detailed(
        model, args.eval_seeds, args.inventory_scale, action_low, action_high, deterministic=True,
    )
    final_summary["total_timesteps"] = args.total_timesteps
    final_summary["tag"] = "final_post_training"
    training_curve.append(final_summary)
    print(f"  final: mean_objective={final_summary['mean_objective']:.4f} "
          f"(std={final_summary['std_objective']:.4f})  "
          f"mean_raw_pnl={final_summary['mean_raw_pnl']:.4f}  "
          f"mean|inv|={final_summary['mean_abs_inventory']:.3f}  "
          f"mean_fills={final_summary['mean_fills']:.1f}  "
          f"frac_near_bound={final_summary['frac_near_bound_any']:.3f}")

    improvement = final_summary["mean_objective"] - baseline_summary["mean_objective"]
    print(f"\nImprovement over untrained baseline (mean_objective): {improvement:+.4f}")

    # --- Save / reload equivalence ---
    print("\nChecking save/reload equivalence...")
    loaded_model = PPO.load(args.model_path, env=vec_env)
    assert_no_nan_params(loaded_model)

    reload_summary, reload_records = evaluate_policy_detailed(
        loaded_model, args.eval_seeds, args.inventory_scale, action_low, action_high, deterministic=True,
    )

    equivalence_ok = True
    max_action_diff = 0.0
    max_objective_diff = 0.0
    for r_final, r_reload in zip(final_records, reload_records):
        assert r_final["seed"] == r_reload["seed"]
        action_diff = float(np.max(np.abs(r_final["actions"] - r_reload["actions"])))
        objective_diff = abs(r_final["cumulative_objective"] - r_reload["cumulative_objective"])
        max_action_diff = max(max_action_diff, action_diff)
        max_objective_diff = max(max_objective_diff, objective_diff)
        if action_diff > 1e-6 or objective_diff > 1e-6:
            equivalence_ok = False

    print(f"  max |action difference| across all eval episodes: {max_action_diff:.3e}")
    print(f"  max |cumulative_objective difference|:            {max_objective_diff:.3e}")
    if equivalence_ok:
        print("  OK: save/reload equivalence confirmed (deterministic actions and episode results match).")
    else:
        print("  WARNING: save/reload produced different results -- see diffs above.")

    # --- Action saturation summary (final eval) ---
    print("\nAction saturation (final evaluation, fraction of steps within "
          f"{ACTION_BOUND_TOL} of a bound):")
    print(f"  bid: {final_summary['frac_near_bound_bid']:.4f}   "
          f"ask: {final_summary['frac_near_bound_ask']:.4f}   "
          f"either: {final_summary['frac_near_bound_any']:.4f}")

    # --- Save all results ---
    training_curve_df = pd.DataFrame(training_curve)
    training_curve_path = RESULTS_DIR / f"hamilton_ppo_training_curve_{args.run_tag}.csv"
    training_curve_df.to_csv(training_curve_path, index=False)
    print(f"\nTraining curve (baseline + periodic + final) saved to {training_curve_path}")

    per_episode_rows = []
    for tag, records in (("final_post_training", final_records), ("reloaded_model", reload_records)):
        for r in records:
            row = {k: v for k, v in r.items() if k != "actions"}
            row["tag"] = tag
            per_episode_rows.append(row)
    per_episode_df = pd.DataFrame(per_episode_rows)
    per_episode_path = RESULTS_DIR / f"hamilton_ppo_eval_episodes_{args.run_tag}.csv"
    per_episode_df.to_csv(per_episode_path, index=False)
    print(f"Per-episode evaluation results saved to {per_episode_path}")

    total_elapsed = time.time() - t_run_start
    run_summary = dict(
        run_tag=args.run_tag,
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        total_elapsed_seconds=total_elapsed,
        train_elapsed_seconds=train_elapsed,
        architecture=architecture,
        rollout_geometry=rollout_geometry,
        baseline_summary=baseline_summary,
        final_summary=final_summary,
        reload_summary=reload_summary,
        improvement_over_baseline_mean_objective=improvement,
        save_reload_equivalence_ok=equivalence_ok,
        max_action_diff_save_reload=max_action_diff,
        max_objective_diff_save_reload=max_objective_diff,
        numerical_instability=numerical_instability,
        final_training_metrics=loss_entries,
    )
    summary_path = LOGS_DIR / f"run_summary_{args.run_tag}.json"
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=2, default=str)
    print(f"Run summary saved to {summary_path}")

    log_path = LOGS_DIR / f"training_log_{args.run_tag}.txt"
    with open(log_path, "w") as f:
        f.write(f"Hamilton PPO training run [{args.run_tag}]\n")
        f.write("NOTE: this is a learning-validation run, not a final dissertation result.\n")
        f.write("=" * 78 + "\n")
        f.write("Configuration:\n")
        for k, v in vars(args).items():
            f.write(f"  {k} = {v}\n")
        f.write("\nArchitecture:\n")
        for k, v in architecture.items():
            f.write(f"  {k} = {v}\n")
        f.write("\nRollout geometry:\n")
        for k, v in rollout_geometry.items():
            f.write(f"  {k} = {v}\n")
        f.write("\nTraining curve (baseline -> periodic -> final):\n")
        for row in training_curve:
            f.write(f"  [{row['tag']:>22}] t={row['total_timesteps']:>7}  "
                     f"obj={row['mean_objective']:.4f}+-{row['std_objective']:.4f}  "
                     f"pnl={row['mean_raw_pnl']:.4f}  |inv|={row['mean_abs_inventory']:.3f}  "
                     f"fills={row['mean_fills']:.1f}  near_bound={row['frac_near_bound_any']:.3f}\n")
        f.write(f"\nSave/reload equivalence OK: {equivalence_ok} "
                 f"(max action diff={max_action_diff:.3e}, max objective diff={max_objective_diff:.3e})\n")
        f.write(f"Numerical instability: {numerical_instability if numerical_instability else 'none'}\n")
        f.write(f"\nTotal wall-clock time: {total_elapsed:.1f}s (training: {train_elapsed:.1f}s)\n")
    print(f"Plain-text log written to {log_path}")

    print("\n" + "=" * 78)
    print("Run complete: PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
