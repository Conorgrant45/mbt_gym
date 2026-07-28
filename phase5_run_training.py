"""
phase5_run_training.py
----------------------------
Phase 5, Sections 4-6/8: the controlled experiment.

Group A: random initialisation, existing Hamilton PPO configuration, no
         changes.
Group B: actor initialised from the Phase 4 supervised clone, critic
         randomly initialised, existing PPO configuration unchanged.
Group C: supervised actor loaded, actor AND critic frozen (no updates at
         all) -- a control confirming any change seen in Group B comes from
         PPO updates, not evaluation-side drift.

3 learner seeds x 3 groups, training-environment seed 70000,
64,000 transitions (16 rollouts of n_steps=4000) for A/B. No architecture,
hyperparameter, reward, action-space, or filter change anywhere in this
script.

Run from repo root:
    python phase5_run_training.py
"""
import json
import time

import numpy as np
import pandas as pd
import torch

import phase5_common as P5
from phase5_instrumented_ppo import InstrumentedPPO

CHECKPOINTS_DIR = P5.PHASE5_LOGS_DIR / "checkpoints"


def checkpoint_row(model, controls, coarse_grid, clone_vector, group, learner_seed, timestep, torch_seed):
    q_grid, tau_grid, b_grid = coarse_grid
    ev = P5.evaluate_checkpoint(model, torch_seed=torch_seed)
    surf = P5.action_surface_metrics(model, controls, q_grid, tau_grid, b_grid)
    dist = P5.actor_param_distance(model, clone_vector)
    row = dict(group=group, learner_seed=learner_seed, timestep=timestep,
               actor_distance_from_clone=dist)
    row.update(ev)
    row.update(surf)
    return row


def save_checkpoint_model(model, group, learner_seed, timestep):
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINTS_DIR / f"{group}_seed{learner_seed}_t{timestep:06d}"
    model.save(str(path))
    return str(path) + ".zip"


def run_group_a(learner_seed: int, controls, coarse_grid, clone_vector) -> dict:
    model, _ = P5.build_hamilton_ppo(learner_seed=learner_seed, model_cls=InstrumentedPPO)

    checkpoint_rows = []
    torch.manual_seed(P5.STOCHASTIC_TORCH_SEED_BASE)
    row0 = checkpoint_row(model, controls, coarse_grid, clone_vector, "A_random_init", learner_seed, 0,
                           P5.STOCHASTIC_TORCH_SEED_BASE)
    row0["model_path"] = save_checkpoint_model(model, "A_random_init", learner_seed, 0)
    checkpoint_rows.append(row0)

    done = 0
    while done < P5.TOTAL_TRANSITIONS:
        model.learn(total_timesteps=P5.PPO_KWARGS["n_steps"], reset_num_timesteps=False)
        done = model.num_timesteps
        if done in P5.CHECKPOINT_TIMESTEPS:
            r = checkpoint_row(model, controls, coarse_grid, clone_vector, "A_random_init", learner_seed, done,
                                P5.STOCHASTIC_TORCH_SEED_BASE + done)
            r["model_path"] = save_checkpoint_model(model, "A_random_init", learner_seed, done)
            checkpoint_rows.append(r)
            print(f"[A seed={learner_seed}] t={done}: det_obj={r['det_mean_objective']:.2f} "
                  f"stoch_obj={r['stoch_mean_objective']:.2f} actor_dist_clone={r['actor_distance_from_clone']:.3f}")

    update_rows = [dict(group="A_random_init", learner_seed=learner_seed, **r) for r in model.update_records]
    return dict(checkpoint_rows=checkpoint_rows, update_rows=update_rows)


def run_group_b(learner_seed: int, controls, coarse_grid, clone_net, clone_vector) -> dict:
    model, _ = P5.build_hamilton_ppo(learner_seed=learner_seed, model_cls=InstrumentedPPO)
    transfer_report = P5.copy_supervised_actor_weights(model, clone_net)

    grid_obs_dense = P5.dense_grid_observations()
    verify_t0 = P5.verify_actor_matches_clone(model, clone_net, grid_obs_dense)
    assert verify_t0["passed"], f"Group B seed {learner_seed}: t=0 actor does not match clone: {verify_t0}"

    checkpoint_rows = []
    per_update_clone_rows = []

    torch.manual_seed(P5.STOCHASTIC_TORCH_SEED_BASE)
    row0 = checkpoint_row(model, controls, coarse_grid, clone_vector, "B_clone_init", learner_seed, 0,
                           P5.STOCHASTIC_TORCH_SEED_BASE)
    row0["model_path"] = save_checkpoint_model(model, "B_clone_init", learner_seed, 0)
    row0["actor_clone_verify_max_abs_diff"] = verify_t0["max_abs_diff"]
    checkpoint_rows.append(row0)
    det_obj_t0 = row0["det_mean_objective"]

    done = 0
    update_idx = 0
    while done < P5.TOTAL_TRANSITIONS:
        model.learn(total_timesteps=P5.PPO_KWARGS["n_steps"], reset_num_timesteps=False)
        done = model.num_timesteps
        update_idx += 1

        # Section 6: after EVERY update (clone-initialised runs only) --
        # cheap grid-based action-surface metrics + a deterministic-only
        # rollout evaluation (no stochastic pass here; the 8 designated
        # checkpoints below already cover stochastic performance at coarser
        # granularity -- see phase5_common.coarse_grid's docstring for the
        # documented cost/precision tradeoff).
        q_grid, tau_grid, b_grid = coarse_grid
        surf = P5.action_surface_metrics(model, controls, q_grid, tau_grid, b_grid)
        dist = P5.actor_param_distance(model, clone_vector)
        det_records = [P5.run_event_episode(model, s, deterministic=True) for s in P5.DIAGNOSTIC_VALIDATION_SEEDS]
        det_obj = float(np.mean([r["full_objective"] for r in det_records]))
        per_update_clone_rows.append(dict(
            group="B_clone_init", learner_seed=learner_seed, update_index=update_idx, num_timesteps=done,
            actor_distance_from_clone=dist, deterministic_objective=det_obj,
            deterministic_objective_change_from_t0=det_obj - det_obj_t0,
            **surf,
        ))

        if done in P5.CHECKPOINT_TIMESTEPS:
            r = checkpoint_row(model, controls, coarse_grid, clone_vector, "B_clone_init", learner_seed, done,
                                P5.STOCHASTIC_TORCH_SEED_BASE + done)
            r["model_path"] = save_checkpoint_model(model, "B_clone_init", learner_seed, done)
            checkpoint_rows.append(r)
            print(f"[B seed={learner_seed}] t={done}: det_obj={r['det_mean_objective']:.2f} "
                  f"stoch_obj={r['stoch_mean_objective']:.2f} actor_dist_clone={r['actor_distance_from_clone']:.3f}")

    update_rows = [dict(group="B_clone_init", learner_seed=learner_seed, **r) for r in model.update_records]
    return dict(checkpoint_rows=checkpoint_rows, update_rows=update_rows,
                per_update_clone_rows=per_update_clone_rows, transfer_report=transfer_report)


def run_group_c(learner_seed: int, controls, coarse_grid, clone_net, clone_vector) -> dict:
    model, _ = P5.build_hamilton_ppo(learner_seed=learner_seed)
    P5.copy_supervised_actor_weights(model, clone_net)

    actor_before = P5.flat_param_vector(P5.get_actor_params(model))
    critic_before = P5.flat_param_vector(P5.get_critic_params(model))

    checkpoint_rows = []
    for t in P5.CHECKPOINT_TIMESTEPS:
        r = checkpoint_row(model, controls, coarse_grid, clone_vector, "C_frozen_clone", learner_seed, t,
                            P5.STOCHASTIC_TORCH_SEED_BASE + t)
        checkpoint_rows.append(r)
        print(f"[C seed={learner_seed}] t={t}: det_obj={r['det_mean_objective']:.2f} "
              f"stoch_obj={r['stoch_mean_objective']:.2f}")

    actor_after = P5.flat_param_vector(P5.get_actor_params(model))
    critic_after = P5.flat_param_vector(P5.get_critic_params(model))
    frozen_ok = (np.array_equal(actor_before, actor_after) and np.array_equal(critic_before, critic_after))
    assert frozen_ok, "Group C (frozen control): weights changed despite no training call -- control invalid"

    return dict(checkpoint_rows=checkpoint_rows, frozen_weights_verified=frozen_ok)


def main():
    P5.PHASE5_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    P5.PHASE5_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    import phase4_common as P4
    controls = P4.build_analytical_controls()
    coarse_grid = P5.coarse_grid()

    clone_net = P5.load_supervised_clone_net()
    ref_model, _ = P5.build_hamilton_ppo(learner_seed=0)
    P5.copy_supervised_actor_weights(ref_model, clone_net)
    clone_vector = P5.flat_param_vector(P5.get_actor_params(ref_model))

    all_checkpoint_rows = []
    all_update_rows = []
    all_per_update_clone_rows = []
    transfer_reports = {}
    frozen_verifications = {}

    t0 = time.time()
    for seed in P5.LEARNER_SEEDS:
        print(f"\n{'='*70}\nGroup A (random init), learner_seed={seed}\n{'='*70}")
        res_a = run_group_a(seed, controls, coarse_grid, clone_vector)
        all_checkpoint_rows += res_a["checkpoint_rows"]
        all_update_rows += res_a["update_rows"]

        print(f"\n{'='*70}\nGroup B (clone init), learner_seed={seed}\n{'='*70}")
        res_b = run_group_b(seed, controls, coarse_grid, clone_net, clone_vector)
        all_checkpoint_rows += res_b["checkpoint_rows"]
        all_update_rows += res_b["update_rows"]
        all_per_update_clone_rows += res_b["per_update_clone_rows"]
        transfer_reports[seed] = res_b["transfer_report"]

        print(f"\n{'='*70}\nGroup C (frozen clone control), learner_seed={seed}\n{'='*70}")
        res_c = run_group_c(seed, controls, coarse_grid, clone_net, clone_vector)
        all_checkpoint_rows += res_c["checkpoint_rows"]
        frozen_verifications[seed] = res_c["frozen_weights_verified"]

    elapsed = time.time() - t0
    print(f"\nAll groups/seeds complete in {elapsed:.1f}s")

    checkpoint_df = pd.DataFrame(all_checkpoint_rows)
    checkpoint_path = P5.PHASE5_RESULTS_DIR / "phase5_checkpoint_metrics.csv"
    checkpoint_df.to_csv(checkpoint_path, index=False)
    print(f"Saved {checkpoint_path} ({len(checkpoint_df)} rows)")

    update_df = pd.DataFrame(all_update_rows)
    update_path = P5.PHASE5_RESULTS_DIR / "phase5_update_metrics.csv"
    update_df.to_csv(update_path, index=False)
    print(f"Saved {update_path} ({len(update_df)} rows)")

    drift_df = pd.DataFrame(all_per_update_clone_rows)
    drift_path = P5.PHASE5_RESULTS_DIR / "phase5_action_surface_drift.csv"
    drift_df.to_csv(drift_path, index=False)
    print(f"Saved {drift_path} ({len(drift_df)} rows)")

    manifest = dict(
        train_env_seed=P5.TRAIN_ENV_SEED, learner_seeds=list(P5.LEARNER_SEEDS),
        total_transitions=P5.TOTAL_TRANSITIONS, checkpoint_timesteps=P5.CHECKPOINT_TIMESTEPS,
        ppo_kwargs=P5.PPO_KWARGS, net_arch=P5.NET_ARCH,
        diagnostic_validation_seeds=P5.DIAGNOSTIC_VALIDATION_SEEDS,
        supervised_clone_path=str(P5.SUPERVISED_CLONE_PATH),
        transfer_reports={str(k): v for k, v in transfer_reports.items()},
        frozen_weights_verified={str(k): v for k, v in frozen_verifications.items()},
        elapsed_seconds=elapsed,
    )
    manifest_path = P5.PHASE5_RESULTS_DIR / "phase5_run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
