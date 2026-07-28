"""
phase5_initial_exploration.py
------------------------------------
Phase 5, Section 3: for the clone-initialised (Group B) Hamilton PPO actor,
BEFORE any PPO update, record the initial action mean, log_std, action std,
fraction of sampled actions near the action bounds, and both the
deterministic and stochastic objective on the fixed diagnostic validation
seeds. log_std is read as-is from the existing PPO configuration -- this
script never modifies it.

Run from repo root:
    python phase5_initial_exploration.py
"""
import numpy as np
import pandas as pd
import torch

import phase5_common as P5

N_INIT_SEEDS = P5.LEARNER_SEEDS  # (0, 1, 2) -- same 3 learner seeds as Group B


def main():
    P5.PHASE5_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    clone = P5.load_supervised_clone_net()
    grid_obs = P5.dense_grid_observations()

    rows = []
    for seed in N_INIT_SEEDS:
        model, _ = P5.build_hamilton_ppo(learner_seed=seed)
        P5.copy_supervised_actor_weights(model, clone)

        verify = P5.verify_actor_matches_clone(model, clone, grid_obs)
        assert verify["passed"], f"seed {seed}: clone-to-actor transfer failed verification: {verify}"

        log_std = model.policy.log_std.detach().numpy().copy()
        action_std = np.exp(log_std)

        eval_result = P5.evaluate_checkpoint(model, torch_seed=P5.STOCHASTIC_TORCH_SEED_BASE + seed)

        rows.append(dict(
            learner_seed=seed,
            log_std_bid=float(log_std[0]), log_std_ask=float(log_std[1]),
            action_std_bid=float(action_std[0]), action_std_ask=float(action_std[1]),
            actor_clone_max_abs_diff=verify["max_abs_diff"],
            **eval_result,
        ))
        print(f"seed={seed}: log_std={log_std}, action_std={action_std}, "
              f"det_obj={eval_result['det_mean_objective']:.3f}, "
              f"stoch_obj={eval_result['stoch_mean_objective']:.3f}, "
              f"stoch_frac_near_bound={eval_result['stoch_frac_near_bound']:.3f}")

    df = pd.DataFrame(rows)
    out_path = P5.PHASE5_RESULTS_DIR / "phase5_initial_exploration.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(df.to_string(index=False))

    det_mean = df["det_mean_objective"].mean()
    stoch_mean = df["stoch_mean_objective"].mean()
    print(f"\nMean across seeds: deterministic={det_mean:.3f}  stochastic={stoch_mean:.3f}  "
          f"gap={det_mean - stoch_mean:.3f}")


if __name__ == "__main__":
    main()
