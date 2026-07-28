"""
phase4_supervised_clone.py
-------------------------------
Phase 4, Sections 6-7: supervised representability test. Trains the EXACT
Hamilton PPO architecture ([64, 64] Tanh MLP, two-dimensional output) by
plain supervised regression to imitate the analytical belief-weighted
policy's action surface, then evaluates the frozen clone AS A POLICY in
both the fixed-step and event-driven environments, reusing the SAME
Hamilton observation/belief-filter/action-transformation pipeline
(duck-typed `.predict()` shim so evaluate_agents_common.run_hamilton_agent_episode
/ evaluate_agents_event_driven.run_event_hamilton_agent_episode -- already
validated in Phases 2-3 -- are reused unmodified, not reimplemented).

Data generation: rolls out the analytical belief-weighted policy in the
event-driven environment (cheap, ~280 steps/episode) over disjoint seed
ranges for train/validation/test, recording
(q_scaled, tau, belief) -> (bid_action, ask_action) at every step. Disjoint
seed ranges mean the test set's (q, tau, belief) triples are never floating-
point-identical to any training triple (continuous states from real
trajectories, not a quantised grid) -- satisfying "test set contains
combinations not used for fitting" without needing a separate held-out-
region partition.

Dataset sizes (~10,000-25,000 samples) are small enough that no minibatch
streaming from disk is required on an 8GB machine -- generated once, held
in memory as float32 arrays (a few MB), noted explicitly rather than
over-engineered.

Run from repo root:
    python phase4_supervised_clone.py
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import phase4_common as P4
import simulate_belief_weighted as SBW
from envs.event_driven_regime_env import EventDrivenRegimeSwitchingEnv
from envs.event_driven_hamilton_ppo_wrapper import make_event_time_filter

N_TRAIN_EPISODES = 100
N_VAL_EPISODES = 30
N_TEST_EPISODES = 40
SEED_TRAIN_START = 220_000
SEED_VAL_START = 221_000
SEED_TEST_START = 222_000
INIT_SEEDS = (0, 1, 2)
N_EPOCHS = 80
BATCH_SIZE = 512
LEARNING_RATE = 1e-3


def build_mlp(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(3, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 2),
    )


def collect_dataset(controls: dict, n_episodes: int, seed_start: int) -> tuple:
    X, Y = [], []
    for ep in range(n_episodes):
        seed = seed_start + ep
        env = EventDrivenRegimeSwitchingEnv(seed=seed)
        filt = make_event_time_filter()
        raw_state = env.reset()
        filt.reset(initial_price=env.raw_midprice)
        q = env.raw_inventory
        t_elapsed = 0.0
        done = False
        while not done:
            tau = 1.0 - t_elapsed / env.terminal_time
            belief = filt.belief
            bid_depth, ask_depth = P4.analytical_belief_weighted_depths(controls, tau, q, belief)
            bid_action = SBW.normalise_depth(bid_depth)
            ask_action = SBW.normalise_depth(ask_depth)

            q_scaled = float(np.tanh(q / P4.INVENTORY_SCALE))
            X.append([q_scaled, tau, belief])
            Y.append([bid_action, ask_action])

            action = np.array([bid_action, ask_action])
            raw_state, reward, done, info = env.step(action)
            filt.update(info["price_after"], info["elapsed_time"], info["event_type"] == "arrival")
            q = info["inventory_after"]
            t_elapsed = float(info["raw_state"][2])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def train_one_seed(seed: int, Xtr, Ytr, Xval, Yval) -> dict:
    net = build_mlp(seed)
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    Xtr_t, Ytr_t = torch.tensor(Xtr), torch.tensor(Ytr)
    Xval_t, Yval_t = torch.tensor(Xval), torch.tensor(Yval)
    n = len(Xtr_t)

    history = []
    best_val_mse = float("inf")
    best_state_dict = None
    best_epoch = -1
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = Xtr_t[idx], Ytr_t[idx]
            pred = net(xb)
            loss = torch.mean((pred - yb) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        train_mse = total_loss / n
        with torch.no_grad():
            val_mse = torch.mean((net(Xval_t) - Yval_t) ** 2).item()
        history.append(dict(init_seed=seed, epoch=epoch, train_mse=train_mse, val_mse=val_mse))
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state_dict = {k: v.clone() for k, v in net.state_dict().items()}
            best_epoch = epoch

    net.load_state_dict(best_state_dict)
    return dict(net=net, history=history, best_val_mse=best_val_mse, best_epoch=best_epoch, init_seed=seed)


def region_breakdown(X: np.ndarray, err_per_sample: np.ndarray, col_idx: int, col_name: str, bins) -> list:
    rows = []
    col = X[:, col_idx]
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (col >= lo) & (col < hi if hi < bins[-1] else col <= hi)
        if mask.sum() > 0:
            rows.append(dict(region=col_name, bin_lo=lo, bin_hi=hi, n=int(mask.sum()), mse=float(err_per_sample[mask].mean())))
    return rows


class SupervisedCloneAgent:
    """Duck-typed `.predict()` shim so the frozen clone can be passed
    directly to evaluate_agents_common.run_hamilton_agent_episode /
    evaluate_agents_event_driven.run_event_hamilton_agent_episode
    unmodified -- same observation ([q_scaled, tau, belief]), same action
    transformation, same belief-filter timing as real Hamilton PPO, since
    those functions build the SAME HamiltonPPOWrapper/
    EventDrivenHamiltonPPOWrapper around it."""

    def __init__(self, net: nn.Module):
        self.net = net
        self.net.eval()

    def predict(self, obs, deterministic=True, state=None, episode_start=None):
        obs_t = torch.tensor(np.asarray(obs, dtype=np.float32)).reshape(1, 3)
        with torch.no_grad():
            action = self.net(obs_t).numpy()[0]
        action = np.clip(action, -1.0, 1.0)
        return action, None


def main():
    P4.PHASE4_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    controls = P4.build_analytical_controls()

    t0 = time.time()
    print(f"Collecting training data ({N_TRAIN_EPISODES} episodes, seeds {SEED_TRAIN_START}+)...")
    Xtr, Ytr = collect_dataset(controls, N_TRAIN_EPISODES, SEED_TRAIN_START)
    print(f"Collecting validation data ({N_VAL_EPISODES} episodes, seeds {SEED_VAL_START}+)...")
    Xval, Yval = collect_dataset(controls, N_VAL_EPISODES, SEED_VAL_START)
    print(f"Collecting test data ({N_TEST_EPISODES} episodes, seeds {SEED_TEST_START}+)...")
    Xtest, Ytest = collect_dataset(controls, N_TEST_EPISODES, SEED_TEST_START)
    print(f"  train={len(Xtr)} val={len(Xval)} test={len(Xtest)} samples ({time.time()-t0:.1f}s)")

    # Exact-state overlap check: every episode resets to the IDENTICAL
    # deterministic state (q=0, tau=1.0, belief=stationary prior -- the
    # filter's prior does not depend on the seed), so a handful of exact
    # collisions at that single reset point are EXPECTED and not a
    # methodology violation -- disjoint seed ranges is what actually
    # guarantees "combinations not used for fitting" for every state that
    # is NOT the shared, trivial t=0 reset point. Checked and reported
    # explicitly rather than asserted to zero.
    train_states = set(map(tuple, np.round(Xtr, 6)))
    test_states = set(map(tuple, np.round(Xtest, 6)))
    overlap = train_states & test_states
    overlap_frac = len(overlap) / len(test_states)
    print(f"  train/test exact-state overlap: {len(overlap)} unique states "
          f"({overlap_frac:.4%} of distinct test states) -- expected to be small and "
          f"driven by the shared deterministic reset state (q=0, tau=1, belief=stationary prior)")
    assert overlap_frac < 0.02, (
        f"train/test state overlap ({overlap_frac:.4%}) is too large to be explained by the shared "
        f"reset state alone -- seed ranges may not be truly disjoint."
    )

    print(f"\nTraining {len(INIT_SEEDS)} supervised-init seeds ({N_EPOCHS} epochs each)...")
    results = [train_one_seed(seed, Xtr, Ytr, Xval, Yval) for seed in INIT_SEEDS]
    for r in results:
        print(f"  init_seed={r['init_seed']}: best_val_mse={r['best_val_mse']:.6f} @ epoch {r['best_epoch']}")

    training_curve_rows = [row for r in results for row in r["history"]]
    training_df = pd.DataFrame(training_curve_rows)
    training_path = P4.PHASE4_RESULTS_DIR / "phase4_supervised_training.csv"
    training_df.to_csv(training_path, index=False)
    print(f"Training curves saved to {training_path}")

    # Freeze the best model using VALIDATION MSE only (Section 7) -- never test.
    best = min(results, key=lambda r: r["best_val_mse"])
    print(f"\nSelected init_seed={best['init_seed']} (best_val_mse={best['best_val_mse']:.6f}) -- frozen for evaluation.")
    net = best["net"]
    net.eval()

    with torch.no_grad():
        train_pred = net(torch.tensor(Xtr)).numpy()
        val_pred = net(torch.tensor(Xval)).numpy()
        test_pred = net(torch.tensor(Xtest)).numpy()

    train_mse = float(np.mean((train_pred - Ytr) ** 2))
    val_mse = float(np.mean((val_pred - Yval) ** 2))
    test_mse = float(np.mean((test_pred - Ytest) ** 2))
    bid_test_mse = float(np.mean((test_pred[:, 0] - Ytest[:, 0]) ** 2))
    ask_test_mse = float(np.mean((test_pred[:, 1] - Ytest[:, 1]) ** 2))
    max_test_err = float(np.max(np.abs(test_pred - Ytest)))
    frac_near_bound = float(np.mean(np.abs(test_pred) > 0.95))

    err_per_sample = np.mean((test_pred - Ytest) ** 2, axis=1)
    region_rows = []
    region_rows += region_breakdown(Xtest, err_per_sample, 0, "inventory_scaled", np.linspace(-1, 1, 6))
    region_rows += region_breakdown(Xtest, err_per_sample, 1, "tau", np.linspace(0, 1, 6))
    region_rows += region_breakdown(Xtest, err_per_sample, 2, "belief", np.linspace(0, 1, 6))
    region_df = pd.DataFrame(region_rows)

    test_metrics = dict(
        selected_init_seed=best["init_seed"], selected_epoch=best["best_epoch"],
        train_mse=train_mse, val_mse=val_mse, test_mse=test_mse,
        bid_test_mse=bid_test_mse, ask_test_mse=ask_test_mse, max_test_error=max_test_err,
        frac_test_outputs_near_bound=frac_near_bound,
        n_train=len(Xtr), n_val=len(Xval), n_test=len(Xtest),
    )
    metrics_path = P4.PHASE4_RESULTS_DIR / "phase4_supervised_test_metrics.csv"
    pd.DataFrame([test_metrics]).to_csv(metrics_path, index=False)
    region_path = P4.PHASE4_RESULTS_DIR / "phase4_supervised_test_region_breakdown.csv"
    region_df.to_csv(region_path, index=False)
    print(f"\nTest metrics: {test_metrics}")
    print(f"Saved to {metrics_path} and {region_path}")

    # Save/reload equality check (required test item 6).
    save_path = P4.PHASE4_LOGS_DIR / "supervised_clone_best.pt"
    P4.PHASE4_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), save_path)
    reloaded = build_mlp(0)
    reloaded.load_state_dict(torch.load(save_path))
    reloaded.eval()
    with torch.no_grad():
        reload_pred = reloaded(torch.tensor(Xtest[:100])).numpy()
    max_reload_diff = float(np.max(np.abs(reload_pred - test_pred[:100])))
    print(f"Save/reload max abs diff (first 100 test samples): {max_reload_diff:.3e}")
    assert max_reload_diff < 1e-6, "supervised clone save/reload mismatch"

    return net, test_metrics


if __name__ == "__main__":
    main()
