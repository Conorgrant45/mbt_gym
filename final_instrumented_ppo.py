"""
final_instrumented_ppo.py
------------------------------
Instrumented PPO/RecurrentPPO subclasses for the final architecture-
comparison experiment. Both are byte-for-byte copies of their respective
SB3/sb3-contrib train() methods (SB3 2.8.0 / sb3_contrib 2.8.0, verified
against the installed package source) with READ-ONLY diagnostics added:
no loss term, clipping rule, optimizer, or hyperparameter is touched.

FinalInstrumentedPPO (feed-forward: Hamilton, Return MLP): identical to
phase7_instrumented_ppo.Phase7InstrumentedPPO -- reused directly (that
class is architecture-agnostic; it only depends on model.policy having the
standard ActorCriticPolicy structure, unrelated to which observation
wrapper the environment uses).

FinalInstrumentedRecurrentPPO (Return LSTM): a new subclass of
sb3_contrib.RecurrentPPO, since RecurrentPPO.train() has a materially
different internal structure (masked sequences, LSTM state threading) that
phase7_instrumented_ppo.Phase7InstrumentedPPO cannot be reused for as-is.

Every per-update record has the SAME schema across both classes (feed-
forward and recurrent), enabling a single downstream analysis/plotting
pipeline: policy_loss, value_loss, entropy_loss, approx_kl, clip_fraction,
explained_variance, log_std_mean, log_std_bid, log_std_ask, action_std_bid,
action_std_ask, learning_rate, total_loss, advantage_mean/std/min/max,
return_mean/std, actor_grad_norm_mean, critic_grad_norm_mean,
actor_param_update_norm, critic_param_update_norm,
frac_sampled_action_clipped, frac_sampled_action_near_bound.
"""
import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.utils import explained_variance

from phase7_instrumented_ppo import Phase7InstrumentedPPO as FinalInstrumentedPPO  # feed-forward, reused as-is
import final_common as FC

__all__ = ["FinalInstrumentedPPO", "FinalInstrumentedRecurrentPPO"]


def _grad_norm(params: list) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().data.norm(2).item()) ** 2
    return float(np.sqrt(total))


class FinalInstrumentedRecurrentPPO(RecurrentPPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_records = []

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # --- pre-update snapshot (identical convention to FinalInstrumentedPPO) ---
        buf = self.rollout_buffer
        adv_all = buf.advantages.flatten()
        ret_all = buf.returns.flatten()
        raw_actions_all = buf.actions
        actor_params = FC.get_actor_params(self)
        critic_params = FC.get_critic_params(self)
        actor_vec_before = FC.flat_param_vector(actor_params)
        critic_vec_before = FC.flat_param_vector(critic_params)

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

                mask = rollout_data.mask > 1e-8

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions, rollout_data.lstm_states, rollout_data.episode_starts,
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()

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

        # --- post-update snapshot ---
        actor_vec_after = FC.flat_param_vector(actor_params)
        critic_vec_after = FC.flat_param_vector(critic_params)
        raw_actions_flat = raw_actions_all.reshape(-1, raw_actions_all.shape[-1])
        log_std = self.policy.log_std.detach().cpu().numpy().copy()
        action_std = np.exp(log_std)

        self.update_records.append(dict(
            n_updates=self._n_updates,
            num_timesteps=self.num_timesteps,
            policy_loss=float(np.mean(pg_losses)),
            value_loss=float(np.mean(value_losses)),
            entropy_loss=float(np.mean(entropy_losses)),
            approx_kl=float(np.mean(approx_kl_divs)),
            clip_fraction=float(np.mean(clip_fractions)),
            total_loss=float(loss.item()),
            explained_variance=float(explained_var),
            learning_rate=float(self.policy.optimizer.param_groups[0]["lr"]),
            log_std_mean=float(action_std.mean()),  # NOTE: matches Phase5/7 InstrumentedPPO's naming --
                                                      # this field is actually mean ACTION STD (exp(log_std)),
                                                      # kept for schema parity across architectures; use
                                                      # log_std_bid/log_std_ask for the raw log_std values.
            log_std_bid=float(log_std[0]), log_std_ask=float(log_std[1]),
            action_std_bid=float(action_std[0]), action_std_ask=float(action_std[1]),
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
