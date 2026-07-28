"""
phase5_critic_advantage_diagnostic.py
--------------------------------------------
Phase 5, Section 7: for the clone-initialised policy at t=0 (before any PPO
update), collect one complete rollout (n_steps=4,000 transitions, the
production rollout length) WITHOUT updating the model, then:
  - compute Monte Carlo episode returns (gamma=1.0, undiscounted) from the
    raw rewards, respecting episode boundaries;
  - compare the (randomly initialised) critic's value predictions against
    those realised returns;
  - independently recompute GAE(lambda) advantages from the same buffer
    contents and cross-check against stable_baselines3's own
    rollout_buffer.advantages/.returns;
  - inspect whether high-advantage actions are economically sensible
    (do they correspond to states where quoting tighter/skewing inventory
    was actually the right call, per the analytical policy's own surface).

Does not store unnecessary full trajectories beyond this one targeted
4,000-step buffer (freed after this script finishes).

Run from repo root:
    python phase5_critic_advantage_diagnostic.py
"""
import numpy as np
import pandas as pd

import phase5_common as P5
import phase4_common as P4
from phase5_instrumented_ppo import InstrumentedPPO


def collect_one_rollout_without_updating(model: InstrumentedPPO):
    """Monkeypatch model.train() to a no-op for exactly one learn() call so
    collect_rollouts() runs (populating model.rollout_buffer, including
    SB3's own GAE computation) but NO gradient step / parameter update
    happens. Restores the real train() immediately afterward."""
    original_train = model.train
    model.train = lambda: None
    try:
        model.learn(total_timesteps=model.n_steps, reset_num_timesteps=True)
    finally:
        model.train = original_train
    return model.rollout_buffer


def recompute_gae_independently(buffer, last_values: np.ndarray, last_dones: np.ndarray,
                                 gamma: float, gae_lambda: float) -> np.ndarray:
    """Independent reimplementation of
    RolloutBuffer.compute_returns_and_advantage's exact recursion (see
    stable_baselines3/common/buffers.py), read directly from the buffer's
    already-populated rewards/values/episode_starts -- a from-scratch
    cross-check, not a call into SB3's own method."""
    buffer_size = buffer.buffer_size
    rewards = buffer.rewards
    values = buffer.values
    episode_starts = buffer.episode_starts

    advantages = np.zeros_like(rewards)
    last_gae_lam = 0.0
    for step in reversed(range(buffer_size)):
        if step == buffer_size - 1:
            next_non_terminal = 1.0 - last_dones.astype(np.float32)
            next_values = last_values
        else:
            next_non_terminal = 1.0 - episode_starts[step + 1]
            next_values = values[step + 1]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[step] = last_gae_lam
    return advantages


def monte_carlo_returns(rewards: np.ndarray, episode_starts: np.ndarray) -> np.ndarray:
    """Undiscounted (gamma=1.0) return-to-go WITHIN each episode segment
    contained in the buffer (segments defined by episode_starts -- the last
    segment may be truncated at the buffer boundary, which is expected and
    handled the same way SB3's own bootstrap does; this function does NOT
    bootstrap, so the final segment's MC return systematically understates
    the true return-to-episode-end -- reported as such, not silently
    corrected)."""
    T = len(rewards)
    mc = np.zeros(T)
    running = 0.0
    for t in reversed(range(T)):
        if t < T - 1 and episode_starts[t + 1]:
            running = 0.0
        running += rewards[t]
        mc[t] = running
    return mc


def main():
    P5.PHASE5_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    controls = P4.build_analytical_controls()
    clone_net = P5.load_supervised_clone_net()

    model, _ = P5.build_hamilton_ppo(learner_seed=0, env_seed=P5.TRAIN_ENV_SEED, model_cls=InstrumentedPPO)
    P5.copy_supervised_actor_weights(model, clone_net)

    buffer = collect_one_rollout_without_updating(model)

    rewards = buffer.rewards.flatten()
    values = buffer.values.flatten()
    episode_starts = buffer.episode_starts.flatten()
    advantages_sb3 = buffer.advantages.flatten()
    returns_sb3 = buffer.returns.flatten()
    actions = buffer.actions.reshape(-1, 2)
    observations = buffer.observations.reshape(-1, 3)

    last_obs = model._last_obs
    last_episode_starts = model._last_episode_starts
    import torch
    from stable_baselines3.common.utils import obs_as_tensor
    with torch.no_grad():
        last_values = model.policy.predict_values(obs_as_tensor(last_obs, model.device)).cpu().numpy().flatten()

    advantages_recomputed = recompute_gae_independently(
        buffer, last_values, last_episode_starts, gamma=model.gamma, gae_lambda=model.gae_lambda,
    ).flatten()  # buffer-shaped (n_steps, n_envs); flatten to match advantages_sb3's shape before diffing
    gae_max_abs_diff = float(np.max(np.abs(advantages_recomputed - advantages_sb3)))

    mc_returns = monte_carlo_returns(rewards, episode_starts)
    critic_error = values - mc_returns  # positive => critic OVER-estimates realised (truncated) MC return

    q_scaled = observations[:, 0]
    tau = observations[:, 1]
    belief = observations[:, 2]
    analytical_bid, analytical_ask = [], []
    for qs, t, b in zip(q_scaled, tau, belief):
        q = float(np.arctanh(np.clip(qs, -0.999999, 0.999999)) * P4.INVENTORY_SCALE)
        bid_d, ask_d = P4.analytical_belief_weighted_depths(controls, float(t), q, float(b))
        analytical_bid.append(bid_d)
        analytical_ask.append(ask_d)
    analytical_bid = np.array(analytical_bid)
    analytical_ask = np.array(analytical_ask)
    learned_bid_depth = (actions[:, 0] + 1.0) / 2.0 * P4.MAX_DEPTH
    learned_ask_depth = (actions[:, 1] + 1.0) / 2.0 * P4.MAX_DEPTH

    top_frac = 0.1
    n_top = max(1, int(len(advantages_sb3) * top_frac))
    top_idx = np.argsort(-advantages_sb3)[:n_top]
    bottom_idx = np.argsort(advantages_sb3)[:n_top]

    def sensible_fraction(idx):
        bid_err = np.abs(learned_bid_depth[idx] - analytical_bid[idx])
        ask_err = np.abs(learned_ask_depth[idx] - analytical_ask[idx])
        return float(np.mean((bid_err < 1.0) & (ask_err < 1.0)))

    summary = dict(
        n_transitions=len(rewards),
        gae_recompute_max_abs_diff=gae_max_abs_diff,
        gae_recompute_matches=bool(gae_max_abs_diff < 1e-5),
        mean_reward=float(rewards.mean()), std_reward=float(rewards.std()),
        mean_critic_value=float(values.mean()), std_critic_value=float(values.std()),
        mean_mc_return=float(mc_returns.mean()), std_mc_return=float(mc_returns.std()),
        mean_critic_error=float(critic_error.mean()), std_critic_error=float(critic_error.std()),
        max_abs_critic_error=float(np.max(np.abs(critic_error))),
        advantage_mean=float(advantages_sb3.mean()), advantage_std=float(advantages_sb3.std()),
        advantage_min=float(advantages_sb3.min()), advantage_max=float(advantages_sb3.max()),
        return_mean=float(returns_sb3.mean()), return_std=float(returns_sb3.std()),
        frac_top10pct_advantage_actions_near_analytical=sensible_fraction(top_idx),
        frac_bottom10pct_advantage_actions_near_analytical=sensible_fraction(bottom_idx),
    )
    print("Critic / advantage diagnostic summary:")
    for k, v in summary.items():
        print(f"  {k} = {v}")

    df = pd.DataFrame([summary])
    out_path = P5.PHASE5_RESULTS_DIR / "phase5_critic_advantage_diagnostic.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
