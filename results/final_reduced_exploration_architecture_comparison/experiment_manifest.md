# Experiment Manifest: final_reduced_exploration_architecture_comparison

- Created: 2026-08-03T16:14:23
- Git branch: `event-driven-environment`  |  Git commit: `6e2a3d622cb6a68d0c3714298ec7df58bdd19164`

## Architectures

| Architecture | Observation | Recurrent |
|---|---|---|
| `hamilton_ppo` | [tanh(q/inventory_scale), tau, hamilton_belief] -- shape (3,) | no |
| `return_mlp_ppo` | [tanh(q/inventory_scale), tau, tanh(r/return_scale), tanh(delta_tau/elapsed_time_scale)] -- shape (4,) | no |
| `return_lstm_ppo` | identical observation to return_mlp_ppo, shape (4,), consumed by an MlpLstmPolicy with its own recurrent state (not part of the raw observation vector) | yes |

## Training configuration (identical across all three architectures)

- Environment type: `event`
- Training env seed: `70000`
- Total transitions per run: `1,000,000`
- Learner seeds: `[0, 1, 2, 3, 4]`
- log_std_init (reduced exploration, all 3 architectures): `-1.5`
- Net arch: `[64, 64]`
- LSTM hidden size / layers (recurrent only): `64` / `1`
- Inventory scale: `10.0`  |  Return scale: `0.005`
- Action bounds/transform: action in [-1,1]^2 (bid, ask); physical depth = (action+1)/2*MAX_DEPTH, MAX_DEPTH = -log(0.01)/kappa; unchanged from Phases 1-7.

### PPO / RecurrentPPO hyperparameters

| Key | Value |
|---|---|
| `learning_rate` | `0.0003` |
| `n_steps` | `4000` |
| `batch_size` | `400` |
| `n_epochs` | `10` |
| `gamma` | `1.0` |
| `gae_lambda` | `0.95` |
| `clip_range` | `0.2` |
| `ent_coef` | `0.0` |
| `vf_coef` | `0.5` |
| `max_grad_norm` | `0.5` |
| `device` | `cpu` |

### Checkpoint cadence (21 checkpoints)

`[16000, 32000, 48000, 64000, 80000, 96000, 112000, 128000, 144000, 160000, 176000, 192000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]`

Fixed primary-analysis checkpoints: `[200000, 1000000]` (200k for the 200k-vs-1m comparison, 1,000,000 as the PRIMARY final result -- never selected by validation performance).

## Seed ranges

- Validation: `280000` .. `280049` (50 paths)
- Final unseen holdout: `290000` .. `290499` (500 paths)

### Disjointness proof

- Validation range vs. every prior project seed range: disjoint = **True** (checked against 19 prior ranges)
- Holdout range vs. every prior project seed range: disjoint = **True**
- Validation vs. holdout (this experiment, mutually): disjoint = **True**

## Hamilton / Phase 7 reuse-compatibility check

**Compatible: True**

| Field | Match |
|---|---|
| `environment_type_event` | True |
| `training_env_seed` | True |
| `total_transitions` | True |
| `learner_seeds` | True |
| `log_std_init` | True |
| `init_type_random` | True |
| `checkpoint_timesteps` | True |
| `net_arch` | True |
| `ppo_kwargs.learning_rate` | True |
| `ppo_kwargs.n_steps` | True |
| `ppo_kwargs.batch_size` | True |
| `ppo_kwargs.n_epochs` | True |
| `ppo_kwargs.gamma` | True |
| `ppo_kwargs.gae_lambda` | True |
| `ppo_kwargs.clip_range` | True |
| `ppo_kwargs.ent_coef` | True |
| `ppo_kwargs.vf_coef` | True |
| `ppo_kwargs.max_grad_norm` | True |
| `all_checkpoint_files_present_and_hash_verified` | True |

If compatible, Hamilton's 5 existing Phase 7 checkpoints (all 21 per seed) are reused READ-ONLY (never copied/modified/retrained) and RE-EVALUATED on this experiment's own fresh validation/holdout seeds with the richer per-episode schema this experiment requires. If not compatible, hamilton_ppo is trained fresh identically to the other two architectures.

## Dependency versions

```
{
  "python": "3.13.14 (tags/v3.13.14:fd17997, Jun 10 2026, 13:03:48) [MSC v.1944 64 bit (AMD64)]",
  "platform": "Windows-11-10.0.26200-SP0",
  "gym": "0.26.2",
  "gymnasium": "1.2.3",
  "stable_baselines3": "2.8.0",
  "sb3_contrib": "2.8.0",
  "torch": "2.11.0+cpu",
  "numpy": "2.2.4",
  "pandas": "3.0.0"
}
```