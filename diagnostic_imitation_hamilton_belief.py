"""
diagnostic_imitation_hamilton_belief.py
-----------------------------------------
RL-architecture audit, Section 8: supervised-imitation representability
diagnostic. Determines whether the [64, 64] Tanh MLP architecture used by
Hamilton PPO can represent the analytical belief-weighted policy's action
surface, using the same observation input (q_scaled, tau, belief) and
action output convention (normalised [-1, 1] bid/ask depth) that
HamiltonPPOWrapper/PPO's policy network actually uses.

Method
------
1. Roll out the analytical belief-weighted policy (reusing
   simulate_belief_weighted.py's own primitives, not reimplementing them)
   for many episodes, recording (q_scaled, tau, belief) -> (bid_action,
   ask_action) at every step -- the same state distribution Hamilton PPO
   would actually operate in, not an arbitrary synthetic grid.
2. Train/test split (disjoint episode seeds).
3. Fit a plain torch MLP (Linear(3,64)-Tanh-Linear(64,64)-Tanh-Linear(64,2),
   matching stable_baselines3's default MlpPolicy net_arch=[64,64] +
   activation_fn=Tanh) by supervised MSE regression against the analytical
   policy's own actions.
4. Report train/test action MSE, max error, error by inventory/belief bin,
   fraction of outputs that would be clipped by the [-1,1] action space,
   and the cloned policy's own closed-loop holdout objective compared
   against the original analytical policy's holdout objective on the same
   fresh seeds.

Run from repo root:
    python diagnostic_imitation_hamilton_belief.py
"""
import time

import numpy as np
import torch
import torch.nn as nn

from envs.make_envs import make_regime_envs, N_STEPS
import simulate_belief_weighted as SBW

N_TRAIN_EPISODES = 60
N_TEST_EPISODES = 20
HOLDOUT_EVAL_EPISODES = 30
SEED_TRAIN_START = 500_000
SEED_HOLDOUT_START = 600_000


def collect_belief_policy_dataset(controls, n_episodes, seed_start):
    X, Y = [], []
    for ep in range(n_episodes):
        seed = seed_start + ep
        np.random.seed(seed)
        env = make_regime_envs(switch_within_episode=True, seed=seed)
        filt = SBW.make_filter()
        filt.reset()
        obs = env.reset()
        mid = env.raw_midprice
        done = np.array([False])
        while not np.all(done):
            obs_flat = np.array(obs).flatten()
            inventory = obs_flat[1]
            t_idx = env.current_step
            inv_sc = inventory / SBW.INV_UNIT
            belief = filt.update(mid)

            c0, c1 = controls[0], controls[1]
            a0, b0 = SBW.get_control(c0["delta_ask"], c0["delta_bid"], c0["q_ask"], c0["q_bid"], t_idx, inv_sc)
            a1, b1 = SBW.get_control(c1["delta_ask"], c1["delta_bid"], c1["q_ask"], c1["q_bid"], t_idx, inv_sc)
            ask_depth = (1.0 - belief) * a0 + belief * a1
            bid_depth = (1.0 - belief) * b0 + belief * b1
            bid_action = SBW.normalise_depth(bid_depth)
            ask_action = SBW.normalise_depth(ask_depth)

            q_scaled = float(np.tanh(inventory / 10.0))
            tau = 1.0 - float(t_idx) / float(N_STEPS)
            X.append([q_scaled, tau, belief])
            Y.append([bid_action, ask_action])

            action = np.array([[bid_action, ask_action]])
            obs, reward, done, info = env.step(action)
            mid = info["raw_midprice"]

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def build_mlp():
    return nn.Sequential(
        nn.Linear(3, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 2),
    )


def evaluate_cloned_policy_objective(net, seed_start, n_episodes):
    net.eval()
    objectives = []
    with torch.no_grad():
        for ep in range(n_episodes):
            seed = seed_start + ep
            np.random.seed(seed)
            env = make_regime_envs(switch_within_episode=True, seed=seed)
            filt = SBW.make_filter()
            filt.reset()
            obs = env.reset()
            mid = env.raw_midprice
            done = np.array([False])
            obj_accum = 0.0
            while not np.all(done):
                obs_flat = np.array(obs).flatten()
                inventory = obs_flat[1]
                t_idx = env.current_step
                belief = filt.update(mid)
                q_scaled = float(np.tanh(inventory / 10.0))
                tau = 1.0 - float(t_idx) / float(N_STEPS)
                x = torch.tensor([[q_scaled, tau, belief]], dtype=torch.float32)
                y = net(x).numpy()[0]
                y_clipped = np.clip(y, -1.0, 1.0)
                action = np.array([[float(y_clipped[0]), float(y_clipped[1])]])
                obs, reward, done, info = env.step(action)
                mid = info["raw_midprice"]
                obj_accum += float(np.sum(reward))
            objectives.append(obj_accum)
    return np.array(objectives)


def evaluate_analytical_policy_objective(controls, seed_start, n_episodes):
    objectives = []
    for ep in range(n_episodes):
        seed = seed_start + ep
        np.random.seed(seed)
        env = make_regime_envs(switch_within_episode=True, seed=seed)
        filt = SBW.make_filter()
        filt.reset()
        obs = env.reset()
        mid = env.raw_midprice
        done = np.array([False])
        obj_accum = 0.0
        while not np.all(done):
            obs_flat = np.array(obs).flatten()
            inventory = obs_flat[1]
            t_idx = env.current_step
            inv_sc = inventory / SBW.INV_UNIT
            belief = filt.update(mid)
            c0, c1 = controls[0], controls[1]
            a0, b0 = SBW.get_control(c0["delta_ask"], c0["delta_bid"], c0["q_ask"], c0["q_bid"], t_idx, inv_sc)
            a1, b1 = SBW.get_control(c1["delta_ask"], c1["delta_bid"], c1["q_ask"], c1["q_bid"], t_idx, inv_sc)
            ask_depth = (1.0 - belief) * a0 + belief * a1
            bid_depth = (1.0 - belief) * b0 + belief * b1
            action = np.array([[SBW.normalise_depth(bid_depth), SBW.normalise_depth(ask_depth)]])
            obs, reward, done, info = env.step(action)
            mid = info["raw_midprice"]
            obj_accum += float(np.sum(reward))
        objectives.append(obj_accum)
    return np.array(objectives)


def main():
    t0 = time.time()
    print("Building optimal controls...")
    controls = {}
    for regime, params in SBW.REGIME_PARAMS.items():
        da, db, qag, qbg = SBW.build_optimal_control(**params)
        controls[regime] = {"delta_ask": da, "delta_bid": db, "q_ask": qag, "q_bid": qbg}

    print(f"Collecting {N_TRAIN_EPISODES} training episodes under the belief-weighted policy...")
    X_train, Y_train = collect_belief_policy_dataset(controls, N_TRAIN_EPISODES, SEED_TRAIN_START)
    print(f"Collecting {N_TEST_EPISODES} held-out (unseen-seed) episodes for test MSE...")
    X_test, Y_test = collect_belief_policy_dataset(controls, N_TEST_EPISODES, SEED_TRAIN_START + N_TRAIN_EPISODES)
    print(f"  train samples={len(X_train)}, test samples={len(X_test)}  ({time.time()-t0:.1f}s)")

    net = build_mlp()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xtr = torch.tensor(X_train)
    Ytr = torch.tensor(Y_train)
    Xte = torch.tensor(X_test)
    Yte = torch.tensor(Y_test)

    n = len(Xtr)
    batch_size = 4096
    n_epochs = 60
    print(f"\nTraining MLP ([3]->64->64->[2], Tanh) for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], Ytr[idx]
            pred = net(xb)
            loss = torch.mean((pred - yb) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                test_mse = torch.mean((net(Xte) - Yte) ** 2).item()
            print(f"  epoch {epoch:3d}  train_mse={total_loss/n:.6f}  test_mse={test_mse:.6f}")

    with torch.no_grad():
        train_pred = net(Xtr).numpy()
        test_pred = net(Xte).numpy()
    train_mse = float(np.mean((train_pred - Y_train) ** 2))
    test_mse = float(np.mean((test_pred - Y_test) ** 2))
    train_max_err = float(np.max(np.abs(train_pred - Y_train)))
    test_max_err = float(np.max(np.abs(test_pred - Y_test)))
    frac_clipped = float(np.mean(np.abs(test_pred) > 1.0))

    print("\n" + "=" * 78)
    print("SUPERVISED IMITATION RESULTS")
    print("=" * 78)
    print(f"train action MSE : {train_mse:.6f}")
    print(f"test  action MSE : {test_mse:.6f}")
    print(f"train max abs err: {train_max_err:.4f}")
    print(f"test  max abs err: {test_max_err:.4f}")
    print(f"test fraction of outputs outside [-1,1] (would be clipped): {frac_clipped:.4%}")

    belief_col = X_test[:, 2]
    q_col = np.abs(X_test[:, 0])
    err_per_sample = np.mean((test_pred - Y_test) ** 2, axis=1)
    print("\nTest MSE by belief tercile:")
    for lo, hi in [(0.0, 0.333), (0.333, 0.667), (0.667, 1.0)]:
        mask = (belief_col >= lo) & (belief_col < hi if hi < 1.0 else belief_col <= hi)
        if mask.sum() > 0:
            print(f"  belief in [{lo:.2f},{hi:.2f}): n={mask.sum():6d}  mse={err_per_sample[mask].mean():.6f}")
    print("Test MSE by abs(q_scaled) tercile:")
    q_terciles = np.quantile(q_col, [0.333, 0.667])
    bounds = [(0, q_terciles[0], "low"), (q_terciles[0], q_terciles[1], "mid"), (q_terciles[1], 1.01, "high")]
    for lo, hi, label in bounds:
        mask = (q_col >= lo) & (q_col < hi)
        if mask.sum() > 0:
            print(f"  |q_scaled| {label:5s} [{lo:.3f},{hi:.3f}): n={mask.sum():6d}  mse={err_per_sample[mask].mean():.6f}")

    print(f"\nEvaluating cloned policy and analytical policy on {HOLDOUT_EVAL_EPISODES} FRESH holdout seeds...")
    clone_obj = evaluate_cloned_policy_objective(net, SEED_HOLDOUT_START, HOLDOUT_EVAL_EPISODES)
    analytic_obj = evaluate_analytical_policy_objective(controls, SEED_HOLDOUT_START, HOLDOUT_EVAL_EPISODES)

    print(f"  analytical belief-weighted policy: mean={analytic_obj.mean():.4f}  std={analytic_obj.std():.4f}")
    print(f"  cloned MLP (closed-loop)         : mean={clone_obj.mean():.4f}  std={clone_obj.std():.4f}")
    print(f"  mean paired difference (clone - analytic): {(clone_obj - analytic_obj).mean():.4f}")
    print(f"  fraction of matched seeds where clone beats analytic: {(clone_obj > analytic_obj).mean():.2%}")

    print(f"\nTotal diagnostic time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
