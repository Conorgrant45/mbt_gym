"""
tests/test_inventory_penalty_sensitivity.py
------------------------------------------------------
Focused validation tests for the inventory-penalty sensitivity experiment
(ips_*.py). Covers exactly the items the brief's "Validation before the
full run" section requires:

  - the running reward subtracts phi * integral(Q_t^2, dt);
  - the terminal reward subtracts alpha * Q_T^2;
  - the event-driven implementation accumulates the running penalty using
    the REALISED inter-event duration (not a fixed-dt approximation);
  - changing phi/alpha affects BOTH the training reward AND the
    analytical-control calculation;
  - the higher-penalty analytical controls are genuinely recomputed
    (produce different, non-trivial values), not loaded from the original
    calibration;
  - output paths cannot collide with the original experiment;
  - seed ranges are disjoint from every prior range, and the holdout range
    is IDENTICAL (not merely disjoint) to the original experiment's own
    holdout range, per the brief's explicit pairing requirement;
  - event-level rows are sufficient to reconstruct the episode-level
    summary (reward/penalty/inventory-integral reconciliation).

Run from repo root:
    pytest tests/test_inventory_penalty_sensitivity.py -v
"""
import numpy as np
import pytest

import ips_common as IC
import ips_episode_runner as ER
import final_common as FC
import phase4_common as P4
import simulate_belief_weighted as SBW
from envs.event_driven_regime_env import EventDrivenRegimeSwitchingEnv


# ======================================================================
# Penalty calibration values
# ======================================================================
def test_penalty_calibrations_scale_by_exactly_ten():
    assert IC.PHI_HIGH == pytest.approx(IC.PHI_ORIGINAL * 10.0)
    assert IC.ALPHA_HIGH == pytest.approx(IC.ALPHA_ORIGINAL * 10.0)
    assert IC.PHI_ORIGINAL == 0.01 and IC.ALPHA_ORIGINAL == 0.001
    assert IC.PHI_HIGH == 0.10 and IC.ALPHA_HIGH == 0.010


def test_original_calibration_matches_envs_make_envs_module_defaults():
    from envs.make_envs import PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION
    assert IC.PHI_ORIGINAL == PER_STEP_INVENTORY_AVERSION
    assert IC.ALPHA_ORIGINAL == TERMINAL_INVENTORY_AVERSION


# ======================================================================
# Output paths / seed ranges cannot collide with the original experiment
# ======================================================================
def test_output_dirs_disjoint_from_original_experiment():
    assert IC.RESULTS_DIR != FC.RESULTS_DIR
    assert IC.MODELS_DIR != FC.MODELS_DIR
    assert IC.LOGS_DIR != FC.LOGS_DIR
    assert not str(IC.RESULTS_DIR).startswith(str(FC.RESULTS_DIR) + "\\")
    assert not str(FC.RESULTS_DIR).startswith(str(IC.RESULTS_DIR) + "\\")


def test_checkpoint_paths_disjoint_from_original_experiment():
    for arch in IC.ARCHITECTURES:
        for seed in IC.LEARNER_SEEDS:
            new_path = IC.final_checkpoint_path(arch, seed)
            old_path = IC.original_checkpoint_path(arch, seed)
            assert new_path != old_path


def test_clone_checkpoint_path_disjoint_from_original_clone():
    assert IC.clone_checkpoint_path() != IC.original_clone_path()


def test_new_clone_seed_blocks_disjoint_from_every_prior_range_and_holdout():
    result = IC.verify_new_seeds_disjoint()
    assert result["disjoint_from_prior"], f"overlaps: {result['overlaps']}"
    assert result["disjoint_from_holdout"], f"overlap: {result['holdout_overlap']}"


def test_holdout_seeds_are_IDENTICAL_to_original_experiment_not_merely_disjoint():
    # Task instruction #7 explicitly requires the SAME 500 holdout paths for
    # both calibrations (a paired comparison), unlike every other seed range
    # in this project which is required to be merely DISJOINT from priors.
    assert IC.HOLDOUT_SEEDS == list(FC.HOLDOUT_SEEDS)
    assert len(IC.HOLDOUT_SEEDS) == 500


# ======================================================================
# Running reward: phi * integral(Q_t^2, dt), accumulated using the REALISED
# inter-event duration (not an approximation).
# ======================================================================
def test_running_penalty_uses_realised_inter_event_duration_and_pre_fill_inventory():
    phi, alpha = 0.10, 0.010
    env = EventDrivenRegimeSwitchingEnv(seed=12345, phi=phi, alpha=alpha)
    env.reset()

    # Zero quoted depth -> fill probability exp(-kappa*0) = 1.0 exactly, so
    # the first observable arrival always fills (guaranteed nonzero inventory
    # for the interval preceding the SECOND event).
    zero_depth_action = np.array([-1.0, -1.0])
    _, _, done1, info1 = env.step(zero_depth_action)
    assert info1["event_type"] == "arrival"
    fill_bid, fill_ask = info1["fill_indicator"]
    assert (fill_bid or fill_ask), "zero-depth quote should fill with probability 1 on the first arrival"
    assert info1["running_penalty_increment"] == pytest.approx(0.0), (
        "first interval's inventory-before is 0, so its running penalty must be exactly 0 regardless of phi"
    )
    inv_after_first = info1["inventory_after"]
    assert abs(inv_after_first) == 1

    assert not done1
    _, _, _, info2 = env.step(zero_depth_action)
    expected = phi * (inv_after_first ** 2) * info2["elapsed_time"]
    assert info2["running_penalty_increment"] == pytest.approx(expected, rel=1e-10), (
        "running penalty for the second interval must equal phi * (inventory held during that interval)^2 "
        "* the REALISED elapsed_time of that interval -- not any fixed-dt approximation"
    )


def test_terminal_penalty_is_alpha_times_terminal_inventory_squared():
    phi, alpha = 0.10, 0.010
    env = EventDrivenRegimeSwitchingEnv(seed=777, phi=phi, alpha=alpha)
    env.reset()
    done = False
    info = None
    action = np.array([-1.0, -1.0])  # zero depth: guarantees some nonzero terminal inventory
    while not done:
        _, _, done, info = env.step(action)
    terminal_inventory = info["inventory_after"]
    assert info["terminal_penalty_increment"] == pytest.approx(alpha * terminal_inventory ** 2, rel=1e-10)


def test_reward_and_penalties_scale_by_exactly_ten_holding_dynamics_fixed():
    """The strongest single check: run the SAME seed/action sequence under
    both calibrations and verify (a) the underlying dynamics (inventory,
    cash, price path) are IDENTICAL -- phi/alpha must not leak into
    anything except the penalty subtraction -- and (b) every nonzero
    running/terminal penalty scales by EXACTLY 10x."""
    seed = 555
    action = np.array([0.0, 0.0])

    def rollout(phi, alpha):
        env = EventDrivenRegimeSwitchingEnv(seed=seed, phi=phi, alpha=alpha)
        env.reset()
        out = []
        done = False
        for _ in range(5000):  # generous cap; a real episode is ~280 events, never actually reached
            _, reward, done, info = env.step(action)
            out.append((reward, info))
            if done:
                break
        assert done, "episode did not reach termination within the generous step cap -- fixture is broken"
        return out

    orig = rollout(IC.PHI_ORIGINAL, IC.ALPHA_ORIGINAL)
    high = rollout(IC.PHI_HIGH, IC.ALPHA_HIGH)
    assert len(orig) == len(high) and len(orig) > 0

    saw_nonzero_running = False
    saw_nonzero_terminal = False
    for (r_o, i_o), (r_h, i_h) in zip(orig, high):
        assert i_o["inventory_after"] == i_h["inventory_after"]
        assert i_o["cash_after"] == pytest.approx(i_h["cash_after"])
        assert i_o["price_after"] == pytest.approx(i_h["price_after"])

        if i_o["running_penalty_increment"] > 0:
            saw_nonzero_running = True
            ratio = i_h["running_penalty_increment"] / i_o["running_penalty_increment"]
            assert ratio == pytest.approx(10.0, rel=1e-9)
        if i_o["terminal_penalty_increment"] > 0:
            saw_nonzero_terminal = True
            ratio = i_h["terminal_penalty_increment"] / i_o["terminal_penalty_increment"]
            assert ratio == pytest.approx(10.0, rel=1e-9)

        expected_r_h = (r_o - (i_h["running_penalty_increment"] - i_o["running_penalty_increment"])
                         - (i_h["terminal_penalty_increment"] - i_o["terminal_penalty_increment"]))
        assert r_h == pytest.approx(expected_r_h, abs=1e-9)

    assert saw_nonzero_running, "test did not exercise any nonzero running penalty -- strengthen the fixture"
    assert saw_nonzero_terminal, "test did not exercise the terminal penalty -- strengthen the fixture"


# ======================================================================
# Analytical controls: phi/alpha change -> the control table changes, and
# is genuinely recomputed (not loaded from the original calibration).
# ======================================================================
def test_analytical_controls_change_with_phi_and_alpha():
    controls_orig = IC.build_analytical_controls(IC.PHI_ORIGINAL, IC.ALPHA_ORIGINAL)
    controls_high = IC.build_analytical_controls(IC.PHI_HIGH, IC.ALPHA_HIGH)
    for regime in (0, 1):
        assert not np.allclose(controls_orig[regime]["delta_ask"], controls_high[regime]["delta_ask"]), (
            "higher inventory penalties must change the analytical ask-depth surface"
        )
        assert not np.allclose(controls_orig[regime]["delta_bid"], controls_high[regime]["delta_bid"]), (
            "higher inventory penalties must change the analytical bid-depth surface"
        )
    # Directional sanity check restricted to where it is analytically unambiguous:
    # near the terminal time, w_T = exp(-kappa*alpha*q^2) so for q>0 the ask/bid
    # skew (bid_depth - ask_depth, i.e. how hard the quote leans on reducing a
    # long position) must be LARGER under the higher terminal penalty. Checked
    # only near t=T and at a moderate q, not as a mean over the whole (t,q)
    # grid -- at extreme |q| the linear penalty term dominates in ways that are
    # not simply "wider on average" (inventory-skewing can lower one side's
    # depth even as risk aversion rises), so no whole-grid mean claim is made.
    last_t = -1
    q_target = 10
    for regime in (0, 1):
        ask_idx_o = int(np.searchsorted(controls_orig[regime]["q_ask"], q_target))
        bid_idx_o = int(np.searchsorted(controls_orig[regime]["q_bid"], q_target))
        ask_idx_h = int(np.searchsorted(controls_high[regime]["q_ask"], q_target))
        bid_idx_h = int(np.searchsorted(controls_high[regime]["q_bid"], q_target))
        skew_orig = (controls_orig[regime]["delta_bid"][last_t, bid_idx_o]
                     - controls_orig[regime]["delta_ask"][last_t, ask_idx_o])
        skew_high = (controls_high[regime]["delta_bid"][last_t, bid_idx_h]
                     - controls_high[regime]["delta_ask"][last_t, ask_idx_h])
        assert skew_high > skew_orig, (
            f"regime {regime}: near-terminal inventory-reduction skew at q={q_target} must increase under the "
            f"higher terminal penalty alpha (got {skew_orig} -> {skew_high})"
        )


def test_original_calibration_recomputation_matches_phase4_common_reference():
    """IC.build_analytical_controls(PHI_ORIGINAL, ALPHA_ORIGINAL) must be
    numerically identical to the pre-existing phase4_common.build_analytical_controls()
    (same phi/alpha/kappa/lambda/epsilon) -- proving this experiment's own
    recomputation path is consistent with the established reference
    implementation, not a silently-diverging reimplementation."""
    controls_ips = IC.build_analytical_controls(IC.PHI_ORIGINAL, IC.ALPHA_ORIGINAL)
    controls_p4 = P4.build_analytical_controls()
    for regime in (0, 1):
        assert np.allclose(controls_ips[regime]["delta_ask"], controls_p4[regime]["delta_ask"])
        assert np.allclose(controls_ips[regime]["delta_bid"], controls_p4[regime]["delta_bid"])


def test_analytical_controls_are_a_pure_function_not_cached_across_calibrations():
    """Calling build_analytical_controls twice with different phi/alpha in
    either order must never return a stale/cached table from the other
    calibration."""
    a = IC.build_analytical_controls(IC.PHI_ORIGINAL, IC.ALPHA_ORIGINAL)
    b = IC.build_analytical_controls(IC.PHI_HIGH, IC.ALPHA_HIGH)
    c = IC.build_analytical_controls(IC.PHI_ORIGINAL, IC.ALPHA_ORIGINAL)
    assert np.allclose(a[0]["delta_ask"], c[0]["delta_ask"])
    assert not np.allclose(a[0]["delta_ask"], b[0]["delta_ask"])


# ======================================================================
# Environment/wrapper construction threads phi/alpha through correctly
# ======================================================================
@pytest.mark.parametrize("architecture", ["hamilton_ppo", "return_mlp_ppo", "return_lstm_ppo"])
def test_training_vec_env_uses_the_requested_phi_alpha(architecture):
    vec_env = IC.build_training_vec_env(architecture, IC.PHI_HIGH, IC.ALPHA_HIGH, env_seed=999)
    wrapper = vec_env.envs[0].env  # unwrap Monitor
    assert wrapper.base_env.phi == pytest.approx(IC.PHI_HIGH)
    assert wrapper.base_env.alpha == pytest.approx(IC.ALPHA_HIGH)


@pytest.mark.parametrize("architecture", ["hamilton_ppo", "return_mlp_ppo", "return_lstm_ppo"])
def test_eval_wrapper_uses_the_requested_phi_alpha(architecture):
    wrapper = IC.build_eval_wrapper(architecture, 999, IC.PHI_ORIGINAL, IC.ALPHA_ORIGINAL)
    assert wrapper.base_env.phi == pytest.approx(IC.PHI_ORIGINAL)
    assert wrapper.base_env.alpha == pytest.approx(IC.ALPHA_ORIGINAL)


# ======================================================================
# Event-level rows reconstruct the episode-level summary
# ======================================================================
def test_event_rows_reconcile_with_analytic_episode_summary():
    controls = IC.build_analytical_controls(IC.PHI_HIGH, IC.ALPHA_HIGH)
    summary, event_rows = ER.run_analytic_episode("belief_weighted", controls, seed=290_000,
                                                     phi=IC.PHI_HIGH, alpha=IC.ALPHA_HIGH)
    assert len(event_rows) == summary["n_decision_events"]

    sum_running = sum(r["running_penalty_contribution"] for r in event_rows)
    assert sum_running == pytest.approx(summary["running_penalty"], rel=1e-9)

    sum_terminal = sum(r["terminal_penalty_contribution"] for r in event_rows)
    assert sum_terminal == pytest.approx(summary["terminal_penalty"], rel=1e-9)
    assert event_rows[-1]["terminal_penalty_contribution"] == pytest.approx(summary["terminal_penalty"])
    assert all(r["terminal_penalty_contribution"] == 0.0 for r in event_rows[:-1])

    sum_wealth_change = sum(r["wealth_change"] for r in event_rows)
    assert sum_wealth_change == pytest.approx(summary["raw_pnl"], rel=1e-9)

    sum_reward = sum(r["realised_stage_reward"] for r in event_rows)
    assert sum_reward == pytest.approx(summary["full_objective"], rel=1e-9)

    assert event_rows[-1]["inventory_after"] == pytest.approx(summary["terminal_inventory"])

    # Independent reconstruction of integral(Q^2, dt) from the event rows'
    # (inventory_before, elapsed_inter_event_time) pairs, cross-checked
    # against BOTH the accumulator's own field AND running_penalty/phi.
    reconstructed_integral_q2 = sum(r["inventory_before"] ** 2 * r["elapsed_inter_event_time"] for r in event_rows)
    assert reconstructed_integral_q2 == pytest.approx(summary["integrated_squared_inventory"], rel=1e-9)
    assert reconstructed_integral_q2 == pytest.approx(summary["running_penalty"] / IC.PHI_HIGH, rel=1e-9)

    reconstructed_time_avg_abs = (sum(abs(r["inventory_before"]) * r["elapsed_inter_event_time"] for r in event_rows)
                                   / summary["episode_duration"])
    assert reconstructed_time_avg_abs == pytest.approx(summary["time_avg_abs_inventory"], rel=1e-9)


def test_event_rows_reconcile_with_ppo_episode_summary_including_belief():
    from stable_baselines3 import PPO
    model, _ = IC.build_model("hamilton_ppo", learner_seed=0, phi=IC.PHI_HIGH, alpha=IC.ALPHA_HIGH)
    summary, event_rows = ER.run_ppo_episode(model, "hamilton_ppo", 290_000, IC.PHI_HIGH, IC.ALPHA_HIGH,
                                              deterministic=True)
    assert len(event_rows) == summary["n_decision_events"]
    assert all(0.0 <= r["filtered_regime_belief"] <= 1.0 for r in event_rows), (
        "hamilton_ppo event rows must carry a well-formed filtered belief in [0, 1]"
    )
    sum_running = sum(r["running_penalty_contribution"] for r in event_rows)
    assert sum_running == pytest.approx(summary["running_penalty"], rel=1e-9)


def test_reference_policy_event_rows_have_nan_belief_for_oracle():
    controls = IC.build_analytical_controls(IC.PHI_ORIGINAL, IC.ALPHA_ORIGINAL)
    _, event_rows = ER.run_analytic_episode("oracle", controls, seed=290_001,
                                              phi=IC.PHI_ORIGINAL, alpha=IC.ALPHA_ORIGINAL)
    assert all(np.isnan(r["filtered_regime_belief"]) for r in event_rows), (
        "oracle observes the TRUE regime, not a filtered belief -- must be recorded as NaN, never fabricated"
    )


# ======================================================================
# evaluate_one() row-labelling correctness
# ======================================================================
def test_evaluate_one_labels_rows_with_policy_architecture_and_calibration():
    controls = IC.build_analytical_controls(IC.PHI_ORIGINAL, IC.ALPHA_ORIGINAL)
    summary, event_rows = ER.evaluate_one("oracle", "original", 290_002, IC.PHI_ORIGINAL, IC.ALPHA_ORIGINAL,
                                            controls=controls)
    assert summary["policy"] == "oracle"
    assert summary["penalty_calibration"] == "original"
    assert summary["architecture"] == "analytical_oracle"
    assert summary["holdout_episode_seed"] == 290_002
    assert all(r["policy"] == "oracle" and r["penalty_calibration"] == "original" for r in event_rows)
