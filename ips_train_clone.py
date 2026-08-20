"""
ips_train_clone.py
----------------------------
Trains the frozen supervised clone for the HIGH-PENALTY calibration:
the same [64, 64] Tanh MLP architecture, trained by plain supervised
regression to imitate the HIGH-PENALTY analytical belief-weighted policy's
action surface (phi=0.10, alpha=0.010), exactly mirroring
phase4_supervised_clone.py's method for the original calibration.

Reuses UNMODIFIED: phase4_supervised_clone.build_mlp, .collect_dataset,
.train_one_seed, .region_breakdown, .SupervisedCloneAgent -- every one of
these is already parameterised by a `controls` dict / explicit seeds, so no
copy-paste re-derivation of the training loop was necessary or done. The
data-collection environment itself does not depend on phi/alpha (see
ips_common.py's module docstring for why); only the `controls` table
(which chooses the target actions) needs to be the HIGH-penalty one, which
is why this script exists as a thin driver rather than a modification of
phase4_supervised_clone.py.

The ORIGINAL-calibration clone is never retrained here -- it already exists
at ips_common.original_clone_path() (== phase5_common.SUPERVISED_CLONE_PATH)
and is reused read-only by ips_evaluate_holdout.py.

Run from repo root:
    python ips_train_clone.py
"""
import time

import numpy as np
import pandas as pd
import torch

import ips_common as IC
from phase4_supervised_clone import build_mlp, collect_dataset, train_one_seed, region_breakdown, SupervisedCloneAgent

INIT_SEEDS = (0, 1, 2)  # identical to phase4_supervised_clone.INIT_SEEDS


def main():
    IC.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IC.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    disjoint = IC.verify_new_seeds_disjoint()
    assert disjoint["disjoint_from_prior"], f"Clone seed blocks collide with prior ranges: {disjoint['overlaps']}"
    assert disjoint["disjoint_from_holdout"], f"Clone seed blocks collide with holdout: {disjoint['holdout_overlap']}"
    print(f"Clone seed blocks verified disjoint from every prior range and from the holdout set.")

    t0 = time.time()
    print(f"Building HIGH-PENALTY analytical controls (phi={IC.PHI_HIGH}, alpha={IC.ALPHA_HIGH})...")
    controls = IC.build_analytical_controls(IC.PHI_HIGH, IC.ALPHA_HIGH)

    print(f"Collecting training data ({IC.CLONE_N_TRAIN_EPISODES} episodes, seeds {IC.CLONE_SEED_TRAIN_START}+)...")
    Xtr, Ytr = collect_dataset(controls, IC.CLONE_N_TRAIN_EPISODES, IC.CLONE_SEED_TRAIN_START)
    print(f"Collecting validation data ({IC.CLONE_N_VAL_EPISODES} episodes, seeds {IC.CLONE_SEED_VAL_START}+)...")
    Xval, Yval = collect_dataset(controls, IC.CLONE_N_VAL_EPISODES, IC.CLONE_SEED_VAL_START)
    print(f"Collecting test data ({IC.CLONE_N_TEST_EPISODES} episodes, seeds {IC.CLONE_SEED_TEST_START}+)...")
    Xtest, Ytest = collect_dataset(controls, IC.CLONE_N_TEST_EPISODES, IC.CLONE_SEED_TEST_START)
    print(f"  train={len(Xtr)} val={len(Xval)} test={len(Xtest)} samples ({time.time()-t0:.1f}s)")

    train_states = set(map(tuple, np.round(Xtr, 6)))
    test_states = set(map(tuple, np.round(Xtest, 6)))
    overlap = train_states & test_states
    overlap_frac = len(overlap) / len(test_states)
    print(f"  train/test exact-state overlap: {len(overlap)} unique states "
          f"({overlap_frac:.4%} of distinct test states) -- expected small, driven by the shared "
          f"deterministic reset state (q=0, tau=1, belief=stationary prior)")
    assert overlap_frac < 0.02, (
        f"train/test state overlap ({overlap_frac:.4%}) too large to be explained by the shared reset state alone"
    )

    print(f"\nTraining {len(INIT_SEEDS)} supervised-init seeds...")
    results = [train_one_seed(seed, Xtr, Ytr, Xval, Yval) for seed in INIT_SEEDS]
    for r in results:
        print(f"  init_seed={r['init_seed']}: best_val_mse={r['best_val_mse']:.6f} @ epoch {r['best_epoch']}")

    training_curve_rows = [row for r in results for row in r["history"]]
    training_df = pd.DataFrame(training_curve_rows)
    training_path = IC.RESULTS_DIR / "clone_high_penalty_supervised_training.csv"
    training_df.to_csv(training_path, index=False)
    print(f"Training curves saved to {training_path}")

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
        calibration="high_penalty", phi=IC.PHI_HIGH, alpha=IC.ALPHA_HIGH,
        selected_init_seed=best["init_seed"], selected_epoch=best["best_epoch"],
        train_mse=train_mse, val_mse=val_mse, test_mse=test_mse,
        bid_test_mse=bid_test_mse, ask_test_mse=ask_test_mse, max_test_error=max_test_err,
        frac_test_outputs_near_bound=frac_near_bound,
        n_train=len(Xtr), n_val=len(Xval), n_test=len(Xtest),
    )
    metrics_path = IC.RESULTS_DIR / "clone_high_penalty_supervised_test_metrics.csv"
    pd.DataFrame([test_metrics]).to_csv(metrics_path, index=False)
    region_path = IC.RESULTS_DIR / "clone_high_penalty_supervised_test_region_breakdown.csv"
    region_df.to_csv(region_path, index=False)
    print(f"\nTest metrics: {test_metrics}")
    print(f"Saved to {metrics_path} and {region_path}")

    save_path = IC.clone_checkpoint_path()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), save_path)
    reloaded = build_mlp(0)
    reloaded.load_state_dict(torch.load(save_path))
    reloaded.eval()
    with torch.no_grad():
        reload_pred = reloaded(torch.tensor(Xtest[:100])).numpy()
    max_reload_diff = float(np.max(np.abs(reload_pred - test_pred[:100])))
    print(f"Save/reload max abs diff (first 100 test samples): {max_reload_diff:.3e}")
    assert max_reload_diff < 1e-6, "supervised clone save/reload mismatch"
    print(f"Saved frozen high-penalty clone to {save_path}")

    elapsed = time.time() - t0
    print(f"\nClone training complete in {elapsed:.1f}s")
    return net, test_metrics


if __name__ == "__main__":
    main()
