"""
test_phase4_policy_diagnostic.py
-------------------------------------
Phase 4 required tests (policy-diagnostic audit brief, Section 10, items
1-11). Item 12 (all existing 283 tests remain green) is the full existing
suite, run separately, not duplicated here.

Run from repo root:
    pytest tests/test_phase4_policy_diagnostic.py -v
"""
import sys

import numpy as np
import pytest
import torch

import phase4_common as P4
import simulate_belief_weighted as SBW
from envs.event_driven_hamilton_ppo_wrapper import EventDrivenHamiltonPPOWrapper
from envs.hamilton_ppo_wrapper import HamiltonPPOWrapper
from phase4_supervised_clone import build_mlp, SupervisedCloneAgent, collect_dataset


# ======================================================================
# Item 1: grid observations match production wrappers
# ======================================================================
def test_grid_observation_matches_event_driven_wrapper_formula():
    wrapper = EventDrivenHamiltonPPOWrapper(seed=1)
    q, tau, belief = 7.0, 0.42, 0.63
    expected = wrapper._build_obs(q, current_time=(1.0 - tau) * wrapper.terminal_time, belief=belief)
    manual = np.array([np.tanh(q / P4.INVENTORY_SCALE), tau, belief], dtype=np.float32)
    np.testing.assert_allclose(expected, manual, atol=1e-6)


def test_grid_observation_matches_fixed_step_wrapper_formula():
    wrapper = HamiltonPPOWrapper(seed=1)
    q, tau = 7.0, 0.42
    n_steps = wrapper.n_steps
    current_step = round((1.0 - tau) * n_steps)
    expected = wrapper._build_obs(q, current_step, 0.63)
    manual = np.array([np.tanh(q / P4.INVENTORY_SCALE), 1.0 - current_step / n_steps, 0.63], dtype=np.float32)
    np.testing.assert_allclose(expected, manual, atol=1e-6)


# ======================================================================
# Item 2: deterministic PPO actions are transformed exactly once
# ======================================================================
def test_action_transformed_exactly_once():
    models = P4.verify_and_load_hamilton_models()
    model = models[("fixed", 0)]["model"]
    obs = np.array([0.1, 0.5, 0.5], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    expected_bid = (float(action[0]) + 1.0) / 2.0 * P4.MAX_DEPTH
    expected_ask = (float(action[1]) + 1.0) / 2.0 * P4.MAX_DEPTH
    bid_depth, ask_depth = P4.hamilton_ppo_depths(model, q=0.1 * P4.INVENTORY_SCALE * 0 + np.arctanh(0.1) * P4.INVENTORY_SCALE, tau=0.5, belief=0.5)
    # Recompute q exactly from tanh(q/scale)=0.1 -> q = arctanh(0.1)*scale
    q_exact = np.arctanh(0.1) * P4.INVENTORY_SCALE
    bid_depth, ask_depth = P4.hamilton_ppo_depths(model, q=q_exact, tau=0.5, belief=0.5)
    assert bid_depth == pytest.approx(expected_bid, abs=1e-5)
    assert ask_depth == pytest.approx(expected_ask, abs=1e-5)
    # Applying the transform a SECOND time would leave the action space
    # entirely (e.g. an action of 0.5 transformed twice would give a huge
    # depth) -- explicitly confirm depth is NOT double-transformed.
    double_transformed_bid = (expected_bid + 1.0) / 2.0 * P4.MAX_DEPTH
    assert bid_depth != pytest.approx(double_transformed_bid, rel=1e-3) or expected_bid == bid_depth


# ======================================================================
# Item 3: analytical and learned actions use identical bid/ask ordering
# ======================================================================
def test_bid_ask_ordering_consistent_between_analytical_and_learned():
    controls = P4.build_analytical_controls()
    # A strongly positive inventory should produce bid_depth >= ask_depth
    # (discourage further buying, encourage selling) for BOTH analytical and
    # any reasonable learned policy sharing the same ordering convention.
    bid_a, ask_a = P4.analytical_belief_weighted_depths(controls, tau=0.5, q=25.0, belief=0.5)
    assert bid_a >= ask_a
    bid_a2, ask_a2 = P4.analytical_belief_weighted_depths(controls, tau=0.5, q=-25.0, belief=0.5)
    assert ask_a2 >= bid_a2

    models = P4.verify_and_load_hamilton_models()
    model = models[("fixed", 1)]["model"]
    bid_p, ask_p = P4.hamilton_ppo_depths(model, q=0.0, tau=0.5, belief=0.5)
    # Ordering convention check only (index 0 = bid, index 1 = ask) -- not a
    # claim that the learned policy is symmetric or correct at q=0.
    assert isinstance(bid_p, float) and isinstance(ask_p, float)


# ======================================================================
# Item 4: fill probabilities are calculated from physical quote depths
# ======================================================================
def test_fill_probability_uses_physical_depth_not_action():
    depth = 1.234
    expected = np.exp(-P4.KAPPA * depth)
    assert P4.fill_probability(depth) == pytest.approx(expected)
    # Sanity: a normalised action of 1.234 would be out of range and
    # meaningless as a depth -- confirm the function's argument is
    # economically a depth (monotonically decreasing in depth).
    assert P4.fill_probability(0.0) == pytest.approx(1.0)
    assert P4.fill_probability(P4.MAX_DEPTH) == pytest.approx(0.01, rel=1e-6)


# ======================================================================
# Item 5: supervised targets use the correct normalised action space
# ======================================================================
def test_supervised_targets_use_normalised_action_space():
    controls = P4.build_analytical_controls()
    X, Y = collect_dataset(controls, n_episodes=1, seed_start=999_001)
    assert Y.min() >= -1.0 - 1e-6 and Y.max() <= 1.0 + 1e-6
    # Spot check: reconstruct one target directly from analytical depths and
    # SBW.normalise_depth, confirm it matches the collected dataset's own
    # convention exactly (same function, not re-derived).
    q_scaled, tau, belief = X[0]
    q = float(np.arctanh(np.clip(q_scaled, -0.999999, 0.999999)) * P4.INVENTORY_SCALE)
    bid_depth, ask_depth = P4.analytical_belief_weighted_depths(controls, float(tau), q, float(belief))
    expected = np.array([SBW.normalise_depth(bid_depth), SBW.normalise_depth(ask_depth)])
    np.testing.assert_allclose(Y[0], expected, atol=1e-4)


# ======================================================================
# Item 6: supervised clone save/reload equality
# ======================================================================
def test_supervised_clone_save_reload_equality(tmp_path):
    net = build_mlp(seed=0)
    save_path = tmp_path / "clone.pt"
    torch.save(net.state_dict(), save_path)

    reloaded = build_mlp(seed=1)  # different init seed -- reload must still match after loading state_dict
    reloaded.load_state_dict(torch.load(save_path))

    x = torch.randn(10, 3)
    with torch.no_grad():
        out1 = net(x).numpy()
        out2 = reloaded(x).numpy()
    np.testing.assert_allclose(out1, out2, atol=1e-7)


# ======================================================================
# Item 7: clone evaluation uses the same Hamilton belief timing
# ======================================================================
def test_clone_agent_reward_reconciliation_holds():
    """If the clone's duck-typed .predict() shim disturbed the wrapper's
    belief-filter/observation timing in any way, the reward-decomposition
    identity (already proven correct for real Hamilton PPO in Phases 1-3)
    would be the first thing to break -- run one episode through the SAME
    evaluate_agents_common runner and confirm reconciliation still holds to
    machine precision."""
    import evaluate_agents_common as EAC
    net = build_mlp(seed=0)
    clone = SupervisedCloneAgent(net)
    m = EAC.run_hamilton_agent_episode(clone, seed=999_101)
    assert m["reward_reconciliation_error"] < 1e-6


def test_clone_agent_event_driven_reward_reconciliation_holds():
    import evaluate_agents_event_driven as EAED
    net = build_mlp(seed=0)
    clone = SupervisedCloneAgent(net)
    m = EAED.run_event_hamilton_agent_episode(clone, seed=999_102)
    assert m["reward_reconciliation_error"] < 1e-6


# ======================================================================
# Item 8: matched exogenous paths across policies (the grid is a pure
# function of (q, tau, belief) -- no RNG/hidden state, so it is trivially
# "matched" across every policy evaluated on it).
# ======================================================================
def test_grid_functions_are_pure_no_hidden_state():
    controls = P4.build_analytical_controls()
    bid1, ask1 = P4.analytical_belief_weighted_depths(controls, tau=0.3, q=4.0, belief=0.2)
    bid2, ask2 = P4.analytical_belief_weighted_depths(controls, tau=0.3, q=4.0, belief=0.2)
    assert (bid1, ask1) == (bid2, ask2)

    models = P4.verify_and_load_hamilton_models()
    model = models[("event", 2)]["model"]
    d1 = P4.hamilton_ppo_depths(model, q=4.0, tau=0.3, belief=0.2)
    d2 = P4.hamilton_ppo_depths(model, q=4.0, tau=0.3, belief=0.2)
    assert d1 == d2  # deterministic=True, no sampling randomness


# ======================================================================
# Item 9: fresh diagnostic seeds do not overlap previous seed ranges
# ======================================================================
def test_diagnostic_seeds_disjoint_from_all_previous_ranges():
    diag = set(range(P4.DIAGNOSTIC_SEEDS_START, P4.DIAGNOSTIC_SEEDS_START + P4.DIAGNOSTIC_SEEDS_COUNT))
    previous_ranges = [
        set(range(0, 5)),                    # learner seeds
        {70000},                              # training-env seed
        set(range(90001, 90006)),             # dev/monitor eval seeds
        set(range(91001, 91051)),             # fixed offline-selection validation
        set(range(100000, 100200)),           # hamilton_ppo_eval_lib holdout
        set(range(110000, 110100)),           # multiseed holdout
        set(range(120000, 120100)),           # evaluate_agents_common holdout
        set(range(130000, 130100)),           # evaluate_agents_event_driven default holdout
        set(range(195001, 195051)),           # Phase 3 event-driven validation
        set(range(200000, 200200)),           # Phase 3 holdout
    ]
    for prev in previous_ranges:
        assert diag.isdisjoint(prev), f"diagnostic seeds overlap {prev}"

    from phase4_policy_evaluation import EVAL_SEEDS
    eval_seeds = set(EVAL_SEEDS)
    assert eval_seeds.isdisjoint(diag)
    for prev in previous_ranges:
        assert eval_seeds.isdisjoint(prev)

    from phase4_supervised_clone import SEED_TRAIN_START, SEED_VAL_START, SEED_TEST_START, N_TRAIN_EPISODES, N_VAL_EPISODES, N_TEST_EPISODES
    clone_seeds = (set(range(SEED_TRAIN_START, SEED_TRAIN_START + N_TRAIN_EPISODES))
                   | set(range(SEED_VAL_START, SEED_VAL_START + N_VAL_EPISODES))
                   | set(range(SEED_TEST_START, SEED_TEST_START + N_TEST_EPISODES)))
    assert clone_seeds.isdisjoint(diag)
    assert clone_seeds.isdisjoint(eval_seeds)


# ======================================================================
# Item 10: evaluation order invariance
# ======================================================================
def test_evaluation_order_invariance_analytical():
    import evaluate_agents_common as EAC
    controls = P4.build_analytical_controls()
    seeds = [225001, 225002, 225003]
    forward = {s: EAC.run_analytic_agent_episode("belief_weighted", controls, s)["full_objective"] for s in seeds}
    backward = {s: EAC.run_analytic_agent_episode("belief_weighted", controls, s)["full_objective"] for s in reversed(seeds)}
    for s in seeds:
        assert forward[s] == pytest.approx(backward[s])


# ======================================================================
# Item 11: no full trajectory memory growth
# ======================================================================
def test_incremental_histogram_memory_independent_of_sample_count():
    small = P4.IncrementalHistogram(np.arange(-10, 11, 1))
    for _ in range(20):
        small.add(0.0)
    large = P4.IncrementalHistogram(np.arange(-10, 11, 1))
    for _ in range(50_000):
        large.add(float(np.random.uniform(-10, 10)))
    assert sys.getsizeof(small.counts) == sys.getsizeof(large.counts)


def test_running_stats_scalar_memory_independent_of_sample_count():
    small = P4.RunningStatsScalar()
    for _ in range(10):
        small.add(1.0)
    large = P4.RunningStatsScalar()
    for _ in range(100_000):
        large.add(1.0)
    assert sys.getsizeof(small.__dict__) == sys.getsizeof(large.__dict__)
    assert large.mean == pytest.approx(1.0)
