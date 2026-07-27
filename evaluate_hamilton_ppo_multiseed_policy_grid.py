"""
evaluate_hamilton_ppo_multiseed_policy_grid.py
--------------------------------------------------
Policy-surface grid, symmetry error, and economic-direction checks for
each of the 5 independently-trained Hamilton PPO models (seeds 0-4).
Pure forward-pass evaluation -- no environment stepping, no retraining.

Same grid as the single-seed milestone:
    q   in {-20,-15,...,20}, tau in {1.0,0.75,0.5,0.25,0.05}, b in {0,0.1,...,1.0}

Run from repo root:
    python evaluate_hamilton_ppo_multiseed_policy_grid.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from envs.make_envs import KAPPA

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models" / "hamilton_ppo"

INVENTORY_SCALE = 10.0
MAX_DEPTH = -np.log(0.01) / KAPPA
Q_GRID = np.arange(-20, 21, 5)
TAU_GRID = np.array([1.0, 0.75, 0.5, 0.25, 0.05])
B_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)
TRAINING_SEEDS = [0, 1, 2, 3, 4]


def denormalise_depth(action_value: float) -> float:
    return (action_value + 1.0) / 2.0 * MAX_DEPTH


def build_grid_df(model, training_seed: int) -> pd.DataFrame:
    rows = []
    for q in Q_GRID:
        q_scaled = float(np.tanh(q / INVENTORY_SCALE))
        for tau in TAU_GRID:
            for b in B_GRID:
                obs = np.array([q_scaled, tau, b], dtype=np.float32)
                action, _ = model.predict(obs, deterministic=True)
                bid_action, ask_action = float(action[0]), float(action[1])
                rows.append(dict(
                    training_seed=training_seed, q=int(q), tau=float(tau), b=float(b), q_scaled=q_scaled,
                    bid_action=bid_action, ask_action=ask_action,
                    bid_depth=denormalise_depth(bid_action), ask_depth=denormalise_depth(ask_action),
                ))
    return pd.DataFrame(rows)


def symmetry_error(df: pd.DataFrame) -> tuple:
    idx = df.set_index(["q", "tau", "b"])
    errs = []
    for q in Q_GRID:
        for tau in TAU_GRID:
            for b in B_GRID:
                bid_here = idx.loc[(int(q), float(tau), float(b)), "bid_action"]
                ask_mirror = idx.loc[(int(-q), float(tau), float(b)), "ask_action"]
                errs.append(bid_here - ask_mirror)
    errs = np.array(errs)
    return float(np.mean(np.abs(errs))), float(np.max(np.abs(errs)))


def economic_direction_checks(df: pd.DataFrame) -> dict:
    bid_by_q = df.groupby("q")["bid_action"].mean()
    ask_by_q = df.groupby("q")["ask_action"].mean()
    bid_slope_q = float(np.polyfit(bid_by_q.index.astype(float), bid_by_q.values, 1)[0])
    ask_slope_q = float(np.polyfit(ask_by_q.index.astype(float), ask_by_q.values, 1)[0])

    bid_by_b = df.groupby("b")["bid_action"].mean()
    ask_by_b = df.groupby("b")["ask_action"].mean()
    bid_slope_b = float(np.polyfit(bid_by_b.index.astype(float), bid_by_b.values, 1)[0])
    ask_slope_b = float(np.polyfit(ask_by_b.index.astype(float), ask_by_b.values, 1)[0])

    # Terminal behaviour: at q=+20, does ask_action DECREASE (more aggressive) as tau->0?
    sub_pos = df[df["q"] == 20].groupby("tau")["ask_action"].mean().sort_index()  # tau ascending
    ask_terminal_slope_pos_q = float(np.polyfit(sub_pos.index.astype(float), sub_pos.values, 1)[0])
    # at q=-20, does bid_action DECREASE (more aggressive/tighter) as tau->0?
    sub_neg = df[df["q"] == -20].groupby("tau")["bid_action"].mean().sort_index()
    bid_terminal_slope_neg_q = float(np.polyfit(sub_neg.index.astype(float), sub_neg.values, 1)[0])

    return dict(
        bid_slope_wrt_q=bid_slope_q, ask_slope_wrt_q=ask_slope_q,
        bid_direction_correct=(bid_slope_q > 0),  # bid should widen (increase) as q increases
        ask_direction_correct=(ask_slope_q < 0),  # ask should tighten (decrease) as q increases
        bid_slope_wrt_belief=bid_slope_b, ask_slope_wrt_belief=ask_slope_b,
        belief_widens_ask=(ask_slope_b > 0), belief_widens_bid=(bid_slope_b > 0),
        # positive slope wrt tau (ascending tau) means action DECREASES as tau->0 (since tau ascends
        # away from 0) -- i.e. a POSITIVE slope here means the action gets MORE AGGRESSIVE (smaller)
        # as maturity approaches, which is the economically-expected liquidation direction.
        ask_terminal_slope_pos_q=ask_terminal_slope_pos_q,
        ask_more_aggressive_near_terminal_when_long=(ask_terminal_slope_pos_q > 0),
        bid_terminal_slope_neg_q=bid_terminal_slope_neg_q,
        bid_more_aggressive_near_terminal_when_short=(bid_terminal_slope_neg_q > 0),
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_grids = []
    symmetry_rows = []
    direction_rows = []

    for ts in TRAINING_SEEDS:
        model_path = MODELS_DIR / f"ppo_hamilton_seed_{ts}.zip"
        print(f"training_seed={ts}: loading {model_path}")
        model = PPO.load(str(model_path))
        df = build_grid_df(model, ts)
        all_grids.append(df)

        mean_err, max_err = symmetry_error(df)
        symmetry_rows.append(dict(training_seed=ts, mean_symmetry_error=mean_err, max_symmetry_error=max_err))
        print(f"  symmetry: mean={mean_err:.4f}  max={max_err:.4f}")

        checks = economic_direction_checks(df)
        checks["training_seed"] = ts
        direction_rows.append(checks)
        print(f"  ask_slope_wrt_q={checks['ask_slope_wrt_q']:.4f} (correct if <0)  "
              f"bid_slope_wrt_q={checks['bid_slope_wrt_q']:.4f} (correct if >0)  "
              f"ask_terminal_slope(q=+20)={checks['ask_terminal_slope_pos_q']:.4f} "
              f"(correct/aggressive-near-terminal if >0)")

    grid_df = pd.concat(all_grids, ignore_index=True)
    grid_path = RESULTS_DIR / "hamilton_ppo_multiseed_policy_grid.csv"
    grid_df.to_csv(grid_path, index=False)
    print(f"\nPolicy grid (5 seeds x 495 points) saved to {grid_path}")

    symmetry_df = pd.DataFrame(symmetry_rows)
    symmetry_path = RESULTS_DIR / "hamilton_ppo_multiseed_symmetry.csv"
    symmetry_df.to_csv(symmetry_path, index=False)
    print(f"Symmetry-error summary saved to {symmetry_path}")

    direction_df = pd.DataFrame(direction_rows)
    direction_path = RESULTS_DIR / "hamilton_ppo_multiseed_direction_checks.csv"
    direction_df.to_csv(direction_path, index=False)
    print(f"Economic-direction-check summary saved to {direction_path}")

    print("\n" + "=" * 100)
    print("SYMMETRY ERROR BY TRAINING SEED")
    print("=" * 100)
    print(symmetry_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 100)
    print("ECONOMIC-DIRECTION CHECKS BY TRAINING SEED")
    print("=" * 100)
    cols = ["training_seed", "bid_slope_wrt_q", "ask_slope_wrt_q", "bid_direction_correct", "ask_direction_correct",
            "bid_slope_wrt_belief", "ask_slope_wrt_belief", "belief_widens_bid", "belief_widens_ask",
            "ask_terminal_slope_pos_q", "ask_more_aggressive_near_terminal_when_long",
            "bid_terminal_slope_neg_q", "bid_more_aggressive_near_terminal_when_short"]
    print(direction_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nDone.")


if __name__ == "__main__":
    main()
