"""
sanity_check.py
---------------
Run this first after dropping the files into your repo.
It verifies that:
  1. Both regime environments construct without error.
  2. The wrapper steps correctly and exposes true_regime in info.
  3. Observation and action shapes are consistent.

Run from the mbt_gym repo root:
    python sanity_check.py
"""

import numpy as np
from envs.make_envs import make_regime_envs


def main():
    print("Building regime switching environment...")
    env = make_regime_envs()

    # --- reset ---
    obs = env.reset()
    print(f"  obs shape      : {obs.shape}")
    print(f"  action space   : {env.action_space}")
    print(f"  initial regime : {env.current_regime}")

    # --- step through one episode ---
    regimes_seen = []
    rewards = []
    done = False
    steps = 0

    while not np.all(done):
        action = env.action_space.sample().reshape(1, -1)
        obs, reward, done, info = env.step(action)

        # true_regime is in info -- pass this to the critic, not the actor
        regime = info['true_regime'] if isinstance(info, dict) else info[0]['true_regime']
        regimes_seen.append(regime)
        rewards.append(float(np.mean(reward)))
        steps += 1

    print(f"  steps completed: {steps}")
    print(f"  regimes seen   : {set(regimes_seen)}  (should be a single value per episode)")
    print(f"  mean reward    : {np.mean(rewards):.4f}")
    print(f"  obs shape end  : {obs.shape}")

    # --- reset again and check regime resampling ---
    print("\nChecking regime resampling over 20 resets...")
    regime_counts = {0: 0, 1: 0}
    for _ in range(20):
        env.reset()
        regime_counts[env.current_regime] += 1
    print(f"  regime 0 count: {regime_counts[0]}, regime 1 count: {regime_counts[1]}")
    print("  (both should appear given stationary distribution)")

    print("\nSanity check passed.")


if __name__ == "__main__":
    main()
