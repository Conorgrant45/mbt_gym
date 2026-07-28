"""
phase5_reduced_log_std_diagnostic.py
--------------------------------------------
Phase 5, Section 10, conditional variant E: reduced exploration variance.

Triggered because the main experiment (phase5_run_training.py) showed the
t=0 stochastic policy dramatically worse than the deterministic policy
(phase5_initial_exploration.py: mean deterministic objective 29.4 vs mean
stochastic objective -5.8), AND log_std_mean stayed pinned near its initial
value of 0.0 (action std~1.0) across all 16 updates / 64,000 transitions in
BOTH groups A and B (phase5_update_metrics.csv) -- i.e. PPO's own gradient
never meaningfully shrinks the SB3 default exploration variance within this
training budget.

This is a SHORT, explicitly-labelled diagnostic variant, not a
hyperparameter search and not a change to the main experiment's
configuration: ONE reduced log_std_init value (-1.5, action std~0.223,
~4.5x smaller than the SB3 default), clone-initialised actor, 3 learner
seeds, HALF the main experiment's transition budget (32,000, since this is
confirmatory, not a full re-run) -- reported as a separate result, never
substituted for groups A/B/C.

Run from repo root:
    python phase5_reduced_log_std_diagnostic.py
"""
import time

import numpy as np
import pandas as pd
import torch

import phase4_common as P4
import phase5_common as P5
from phase5_instrumented_ppo import InstrumentedPPO
from phase5_run_training import checkpoint_row, save_checkpoint_model

REDUCED_LOG_STD_INIT = -1.5
REDUCED_TOTAL_TRANSITIONS = 32_000
REDUCED_CHECKPOINT_TIMESTEPS = [0, 4_000, 8_000, 16_000, 24_000, 32_000]


def run_group_e(learner_seed: int, controls, coarse_grid, clone_net, clone_vector) -> dict:
    model, _ = P5.build_hamilton_ppo(learner_seed=learner_seed, model_cls=InstrumentedPPO,
                                      log_std_init=REDUCED_LOG_STD_INIT)
    P5.copy_supervised_actor_weights(model, clone_net)

    checkpoint_rows = []
    torch.manual_seed(P5.STOCHASTIC_TORCH_SEED_BASE)
    row0 = checkpoint_row(model, controls, coarse_grid, clone_vector, "E_reduced_log_std", learner_seed, 0,
                           P5.STOCHASTIC_TORCH_SEED_BASE)
    row0["log_std_init"] = REDUCED_LOG_STD_INIT
    checkpoint_rows.append(row0)

    done = 0
    while done < REDUCED_TOTAL_TRANSITIONS:
        model.learn(total_timesteps=P5.PPO_KWARGS["n_steps"], reset_num_timesteps=False)
        done = model.num_timesteps
        if done in REDUCED_CHECKPOINT_TIMESTEPS:
            r = checkpoint_row(model, controls, coarse_grid, clone_vector, "E_reduced_log_std", learner_seed, done,
                                P5.STOCHASTIC_TORCH_SEED_BASE + done)
            r["log_std_init"] = REDUCED_LOG_STD_INIT
            checkpoint_rows.append(r)
            print(f"[E seed={learner_seed}] t={done}: det_obj={r['det_mean_objective']:.2f} "
                  f"stoch_obj={r['stoch_mean_objective']:.2f} stoch_frac_near_bound={r['stoch_frac_near_bound']:.3f}")

    update_rows = [dict(group="E_reduced_log_std", learner_seed=learner_seed, **r) for r in model.update_records]
    return dict(checkpoint_rows=checkpoint_rows, update_rows=update_rows)


def main():
    controls = P4.build_analytical_controls()
    coarse_grid = P5.coarse_grid()
    clone_net = P5.load_supervised_clone_net()

    ref_model, _ = P5.build_hamilton_ppo(learner_seed=0)
    P5.copy_supervised_actor_weights(ref_model, clone_net)
    clone_vector = P5.flat_param_vector(P5.get_actor_params(ref_model))

    all_checkpoint_rows, all_update_rows = [], []
    t0 = time.time()
    for seed in P5.LEARNER_SEEDS:
        print(f"\n{'='*70}\nGroup E (reduced log_std={REDUCED_LOG_STD_INIT}), learner_seed={seed}\n{'='*70}")
        res = run_group_e(seed, controls, coarse_grid, clone_net, clone_vector)
        all_checkpoint_rows += res["checkpoint_rows"]
        all_update_rows += res["update_rows"]
    elapsed = time.time() - t0
    print(f"\nGroup E complete in {elapsed:.1f}s")

    checkpoint_df = pd.DataFrame(all_checkpoint_rows)
    checkpoint_df.to_csv(P5.PHASE5_RESULTS_DIR / "phase5_reduced_log_std_checkpoint_metrics.csv", index=False)
    update_df = pd.DataFrame(all_update_rows)
    update_df.to_csv(P5.PHASE5_RESULTS_DIR / "phase5_reduced_log_std_update_metrics.csv", index=False)
    print("Saved phase5_reduced_log_std_checkpoint_metrics.csv / phase5_reduced_log_std_update_metrics.csv")

    # Compare against Group B's ACTUAL results at the same checkpoints (0-32000).
    b_df = pd.read_csv(P5.PHASE5_RESULTS_DIR / "phase5_checkpoint_metrics.csv")
    b_df = b_df[(b_df["group"] == "B_clone_init") & (b_df["timestep"] <= REDUCED_TOTAL_TRANSITIONS)]
    print("\nComparison at matched checkpoints (0-32000), det/stoch mean objective:")
    for t in REDUCED_CHECKPOINT_TIMESTEPS:
        e_det = checkpoint_df[checkpoint_df["timestep"] == t]["det_mean_objective"].mean()
        e_stoch = checkpoint_df[checkpoint_df["timestep"] == t]["stoch_mean_objective"].mean()
        b_det = b_df[b_df["timestep"] == t]["det_mean_objective"].mean()
        b_stoch = b_df[b_df["timestep"] == t]["stoch_mean_objective"].mean()
        print(f"  t={t:6d}  E(reduced): det={e_det:7.2f} stoch={e_stoch:7.2f}   "
              f"B(default): det={b_det:7.2f} stoch={b_stoch:7.2f}")


if __name__ == "__main__":
    main()
