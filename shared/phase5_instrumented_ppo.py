"""
shared/phase5_instrumented_ppo.py
--------------------------------
InstrumentedPPO: a drop-in subclass of stable_baselines3.PPO whose train()
method is a byte-for-byte copy of SB3 2.8.0's PPO.train() (see
stable_baselines3/ppo/ppo.py in the installed package -- reproduced here,
not altered, so this remains "the existing PPO configuration unchanged" per
the Phase 5 brief), with READ-ONLY instrumentation added around the existing
computation: no loss term, clipping rule, optimizer, or hyperparameter is
touched.

Added per PPO update (== one call to train(), i.e. one full n_epochs x
n_minibatches pass over one rollout buffer -- since n_steps=4000 matches the
requested checkpoint cadence exactly, "one PPO update" in this phase's
reports means one such call, the natural unit SB3 itself logs at):
  - advantage mean/std/min/max and return mean/std, read from the rollout
    buffer BEFORE the first gradient step of this update (i.e. the inputs
    the update is about to act on, not post-update artefacts);
  - actor-only and critic-only gradient norms, computed after loss.backward()
    but before SB3's own joint clip_grad_norm_ call, per minibatch, then
    averaged over the update (the joint clip itself is untouched);
  - actor and critic parameter-update norms (L2 distance between the
    parameter vector at the start and end of this train() call);
  - sampled-action clipping/saturation rate, from the buffer's raw
    (pre-environment-clip) sampled actions.

self.update_records accumulates one dict per update; never cleared
automatically (the training harness reads it once per run).
"""
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance

import shared.phase5_common as P5


class InstrumentedPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_records = []

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # --- Phase 5 instrumentation: pre-update snapshot -------------------
        buf = self.rollout_buffer
        adv_all = buf.advantages.flatten()
        ret_all = buf.returns.flatten()
        raw_actions_all = buf.actions  # PRE-environment-clip sampled actions
        actor_params = P5.get_actor_params(self)
        critic_params = P5.get_critic_params(self)
        actor_vec_before = P5.flat_param_vector(actor_params)
        critic_vec_before = P5.flat_param_vector(critic_params)

        actor_grad_norms, critic_grad_norms = [], []

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()

                # --- Phase 5 instrumentation: pre-clip grad norms, split by
                # actor/critic parameter subset. SB3's own joint clip below
                # is untouched (same call, same args, same effect). ---
                actor_grad_norms.append(_grad_norm(actor_params))
                critic_grad_norms.append(_grad_norm(critic_params))

                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        # --- Phase 5 instrumentation: post-update snapshot ------------------
        actor_vec_after = P5.flat_param_vector(actor_params)
        critic_vec_after = P5.flat_param_vector(critic_params)
        raw_actions_flat = raw_actions_all.reshape(-1, raw_actions_all.shape[-1])

        self.update_records.append(dict(
            n_updates=self._n_updates,
            num_timesteps=self.num_timesteps,
            policy_loss=float(np.mean(pg_losses)),
            value_loss=float(np.mean(value_losses)),
            entropy_loss=float(np.mean(entropy_losses)),
            approx_kl=float(np.mean(approx_kl_divs)),
            clip_fraction=float(np.mean(clip_fractions)),
            explained_variance=float(explained_var),
            log_std_mean=float(th.exp(self.policy.log_std).mean().item()),
            advantage_mean=float(adv_all.mean()), advantage_std=float(adv_all.std()),
            advantage_min=float(adv_all.min()), advantage_max=float(adv_all.max()),
            return_mean=float(ret_all.mean()), return_std=float(ret_all.std()),
            actor_grad_norm_mean=float(np.mean(actor_grad_norms)),
            critic_grad_norm_mean=float(np.mean(critic_grad_norms)),
            actor_param_update_norm=float(np.linalg.norm(actor_vec_after - actor_vec_before)),
            critic_param_update_norm=float(np.linalg.norm(critic_vec_after - critic_vec_before)),
            frac_sampled_action_clipped=float(np.mean(np.abs(raw_actions_flat) > 1.0)),
            frac_sampled_action_near_bound=float(np.mean(np.any(np.abs(np.clip(raw_actions_flat, -1, 1)) > 0.95, axis=1))),
        ))


def _grad_norm(params: list) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().data.norm(2).item()) ** 2
    return float(np.sqrt(total))
