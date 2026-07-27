"""
diagnose_hamilton_ppo_wrapper.py
-----------------------------------
Random-policy diagnostic for HamiltonPPOWrapper. Runs a small number of
random-policy episodes and reports summary statistics used to (a) sanity-
check the wrapper end-to-end and (b) select inventory_scale empirically
from the RAW inventory distribution under a random policy specifically --
a random policy accumulates inventory very differently from the
optimally-skewing policies characterised in earlier diagnostics elsewhere
in this project (mean|inv| ~2-5 there), so those numbers are not a
substitute for actually running this.

info['true_regime'] is read here ONLY after each env.step() call, purely
for an offline belief-accuracy report -- it is never used to build the
PPO observation (see envs/hamilton_ppo_wrapper.py and
tests/test_hamilton_ppo_wrapper.py's leakage tests for the enforcement of
that boundary).

Run from repo root:
    python diagnose_hamilton_ppo_wrapper.py
    python diagnose_hamilton_ppo_wrapper.py --episodes 20 --inventory-scale 10
"""

import argparse
import numpy as np

from envs.hamilton_ppo_wrapper import HamiltonPPOWrapper
from mbt_gym.gym.index_names import INVENTORY_INDEX

SATURATION_THRESHOLD = 0.95  # |tanh(q/scale)| above this counts as "saturated"


def run_episodes(n_episodes: int, inventory_scale: float, seed_base: int = 1000):
    stats = dict(
        returns=[],
        raw_inventory=[],
        q_scaled=[],
        beliefs=[],
        step_rewards=[],
        episode_cum_rewards=[],
        episode_lengths=[],
        n_correct=0,
        n_predictions=0,
        nan_count=0,
        inf_count=0,
    )

    env = None
    for ep in range(n_episodes):
        seed = seed_base + ep
        env = HamiltonPPOWrapper(inventory_scale=inventory_scale, seed=seed)
        obs, info = env.reset(seed=seed)

        stats["q_scaled"].append(float(obs[0]))
        stats["beliefs"].append(float(obs[2]))
        stats["raw_inventory"].append(env.base_env.raw_inventory)

        terminated = truncated = False
        cum_reward = 0.0
        n_steps = 0
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if env.last_return is not None:
                stats["returns"].append(env.last_return)
            raw_inv = float(info["raw_state"][INVENTORY_INDEX])
            stats["raw_inventory"].append(raw_inv)
            stats["q_scaled"].append(float(obs[0]))
            stats["beliefs"].append(float(obs[2]))
            stats["step_rewards"].append(reward)
            cum_reward += reward
            n_steps += 1

            # Offline-only diagnostic: read true_regime AFTER the action/obs
            # were already produced -- never fed back into the observation.
            predicted_regime = int(obs[2] >= 0.5)
            stats["n_predictions"] += 1
            stats["n_correct"] += int(predicted_regime == info["true_regime"])

            if np.isnan(obs).any():
                stats["nan_count"] += int(np.isnan(obs).sum())
            if np.isinf(obs).any():
                stats["inf_count"] += int(np.isinf(obs).sum())
            if np.isnan(reward):
                stats["nan_count"] += 1
            if np.isinf(reward):
                stats["inf_count"] += 1

        stats["episode_lengths"].append(n_steps)
        stats["episode_cum_rewards"].append(cum_reward)

    return stats, env


def quantiles(x, qs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)):
    x = np.asarray(x)
    return {q: float(np.quantile(x, q)) for q in qs}


def print_quantiles(label, x):
    qs = quantiles(x)
    parts = ", ".join(f"q{int(q*100):02d}={v:.4f}" for q, v in qs.items())
    print(f"  {label}: {parts}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--inventory-scale", type=float, default=10.0)
    parser.add_argument("--seed-base", type=int, default=1000)
    args = parser.parse_args()

    print("=" * 78)
    print(f"Random-policy diagnostic: {args.episodes} episodes, "
          f"inventory_scale={args.inventory_scale}")
    print("=" * 78)

    stats, env = run_episodes(args.episodes, args.inventory_scale, args.seed_base)

    returns = np.asarray(stats["returns"])
    raw_inv = np.asarray(stats["raw_inventory"])
    q_scaled = np.asarray(stats["q_scaled"])
    beliefs = np.asarray(stats["beliefs"])
    step_rewards = np.asarray(stats["step_rewards"])
    episode_lengths = np.asarray(stats["episode_lengths"])
    episode_cum_rewards = np.asarray(stats["episode_cum_rewards"])

    print("\n--- Episode length ---")
    print(f"  n_episodes={args.episodes}  mean={episode_lengths.mean():.1f}  "
          f"min={episode_lengths.min()}  max={episode_lengths.max()}")

    print("\n--- Raw return (per step, unscaled percentage return fed to the filter) ---")
    print(f"  mean={returns.mean():.6e}  std={returns.std():.6e}")
    print_quantiles("return quantiles", returns)

    print("\n--- Raw inventory (shares, includes reset + every step) ---")
    print(f"  mean={raw_inv.mean():.4f}  std={raw_inv.std():.4f}  "
          f"min={raw_inv.min():.4f}  max={raw_inv.max():.4f}")
    print_quantiles("inventory quantiles", raw_inv)
    abs_inv = np.abs(raw_inv)
    print_quantiles("|inventory| quantiles", abs_inv)

    print(f"\n--- Inventory-transform saturation (inventory_scale={args.inventory_scale}) ---")
    sat_rate = float(np.mean(np.abs(q_scaled) > SATURATION_THRESHOLD))
    print(f"  q_scaled = tanh(q / {args.inventory_scale})")
    print(f"  mean|q_scaled|={np.abs(q_scaled).mean():.4f}  "
          f"fraction with |q_scaled|>{SATURATION_THRESHOLD} = {sat_rate:.4%}")
    print_quantiles("q_scaled quantiles", q_scaled)

    print("\n--- Belief ---")
    print(f"  mean={beliefs.mean():.4f}  std={beliefs.std():.4f}  "
          f"min={beliefs.min():.4f}  max={beliefs.max():.4f}")
    accuracy = stats["n_correct"] / stats["n_predictions"] if stats["n_predictions"] else float("nan")
    print(f"  belief accuracy vs true_regime (offline diagnostic only, "
          f"never used in the observation): {accuracy:.4f} "
          f"({stats['n_correct']}/{stats['n_predictions']})")

    print("\n--- Reward ---")
    print(f"  cumulative reward per episode: mean={episode_cum_rewards.mean():.6f}  "
          f"std={episode_cum_rewards.std():.6f}")
    print_quantiles("per-episode cumulative reward quantiles", episode_cum_rewards)
    print_quantiles("per-step reward quantiles", step_rewards)

    print("\n--- NaN / infinity counts ---")
    print(f"  NaN count: {stats['nan_count']}   Inf count: {stats['inf_count']}")

    print("\n--- Observation space ---")
    print(f"  low={env.observation_space.low}  high={env.observation_space.high}  "
          f"shape={env.observation_space.shape}  dtype={env.observation_space.dtype}")

    print("\n--- Action space ---")
    print(f"  low={env.action_space.low}  high={env.action_space.high}  "
          f"shape={env.action_space.shape}  dtype={env.action_space.dtype}")

    print("\nDone.")


if __name__ == "__main__":
    main()
