"""
test_adverse_selection_identity.py
-----------------------------------
Isolated diagnostic / pytest file verifying that the simulation's
event-level cash, inventory, midprice and reward updates match the
adverse-selection markout identity assumed by the DPP/HJB derivation:

    wealth = cash + inventory * midprice
    an adverse fill earns  spread - adverse_jump

This does NOT modify any production file. Every test builds its own
isolated TradingEnvironment (a single sub-environment, not the
RegimeSwitchingEnv wrapper -- so there is no regime switching to disable),
with:
    - volatility = 0            (Brownian noise disabled)
    - per_step_inventory_aversion = 0, terminal_inventory_aversion = 0
                                    (running & terminal penalties disabled)
    - arrival_model.get_arrivals and fill_probability_model.get_fills
      monkey-patched per test case to return forced, deterministic arrays
      (no reliance on the true Bernoulli/exponential randomness)
    - the midprice model's jump-magnitude RNG draw monkey-patched to
      return forced, deterministic jump sizes (no reliance on the true
      Exponential(mean=eps) randomness)

Run standalone for the full diagnostic report:
    python test_adverse_selection_identity.py

Run under pytest for CI-style pass/fail:
    pytest test_adverse_selection_identity.py -v
"""

import numpy as np

from mbt_gym.gym.TradingEnvironment import TradingEnvironment
from mbt_gym.gym.ModelDynamics import LimitOrderModelDynamics
from mbt_gym.stochastic_processes.arrival_models import PoissonArrivalModel
from mbt_gym.stochastic_processes.fill_probability_models import ExponentialFillFunction
from mbt_gym.rewards.RewardFunctions import RunningInventoryPenalty
from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, TIME_INDEX, ASSET_PRICE_INDEX

from envs.arrival_jump_midprice import ArrivalJumpMidpriceModel

ASK_DEPTH = 0.7
BID_DEPTH = 0.8
KAPPA = 1.5          # irrelevant to correctness here -- fills are forced, not probabilistic
N_STEPS = 100
TERMINAL_TIME = 1.0

# Collects (case, check, expected, simulated, abs_error, passed) for the final report
REPORT_ROWS = []


def make_test_env():
    """Fresh, isolated single-regime TradingEnvironment with noise/penalties disabled."""
    step_size = TERMINAL_TIME / N_STEPS
    env = TradingEnvironment(
        terminal_time=TERMINAL_TIME,
        n_steps=N_STEPS,
        reward_function=RunningInventoryPenalty(
            per_step_inventory_aversion=0.0,
            terminal_inventory_aversion=0.0,
        ),
        model_dynamics=LimitOrderModelDynamics(
            midprice_model=ArrivalJumpMidpriceModel(
                volatility=0.0,           # Brownian noise disabled
                jump_size=0.45,           # mean is irrelevant -- draws are forced below
                step_size=step_size,
                terminal_time=TERMINAL_TIME,
                initial_price=100.0,
                num_trajectories=1,
                seed=0,
            ),
            arrival_model=PoissonArrivalModel(
                intensity=np.array([[140.0, 140.0]]),
                step_size=step_size,
                num_trajectories=1,
                seed=1,
            ),
            fill_probability_model=ExponentialFillFunction(
                fill_exponent=KAPPA,
                step_size=step_size,
                num_trajectories=1,
                seed=2,
            ),
            num_trajectories=1,
        ),
        num_trajectories=1,
        normalise_action_space=False,
        normalise_observation_space=False,
        normalise_rewards=False,
    )
    env.reset()
    return env


def set_state(env, cash, inventory, midprice):
    """Force the exact starting state (cash, inventory, time=0, midprice)."""
    env.model_dynamics.state = np.array([[cash, inventory, 0.0, midprice]], dtype=float)
    env.model_dynamics.midprice_model.current_state = np.array([[midprice]], dtype=float)


class _ForcedRNG:
    """
    Stand-in for numpy.random.Generator, used only to force the midprice
    model's jump-magnitude draws. numpy.random.Generator is a C-extension
    type whose bound methods cannot be monkey-patched in place (attribute
    assignment on an instance method raises AttributeError: read-only) --
    so the whole rng object is replaced with this mock instead.

    ArrivalJumpMidpriceModel.update() calls rng.exponential() for the
    ASK-side magnitude first, then the BID-side magnitude second, every
    call -- call_sequence is consumed in that exact order.
    """
    def __init__(self, call_sequence):
        self._sequence = call_sequence
        self._i = 0

    def exponential(self, scale, size):
        val = self._sequence[self._i % len(self._sequence)]
        self._i += 1
        return np.full(size, val)

    def normal(self, size):
        return np.zeros(size)  # unused when volatility=0, provided for safety


def force_step(env, buy_arrival, sell_arrival, ask_fill, bid_fill, jump_up, jump_down,
               ask_depth=ASK_DEPTH, bid_depth=BID_DEPTH):
    """
    Force deterministic arrivals, fills and jump magnitudes for exactly one step,
    then execute it through the real environment update path.

    arrivals array layout: [:, BID_INDEX]=sell_arrival, [:, ASK_INDEX]=buy_arrival
    fills    array layout: [:, BID_INDEX]=bid_fill,      [:, ASK_INDEX]=ask_fill
    """
    forced_arrivals = np.array([[float(sell_arrival), float(buy_arrival)]])
    forced_fills = np.array([[float(bid_fill), float(ask_fill)]])

    env.model_dynamics.arrival_model.get_arrivals = lambda: forced_arrivals
    env.model_dynamics.fill_probability_model.get_fills = lambda depths: forced_fills
    env.model_dynamics.midprice_model.rng = _ForcedRNG([jump_up, jump_down])

    action = np.array([[bid_depth, ask_depth]])  # [BID_INDEX, ASK_INDEX] layout
    obs, reward, done, info = env.step(action)
    reward = float(np.asarray(reward).reshape(-1)[0])
    return reward, done, info


def wealth(cash, inventory, midprice):
    return cash + inventory * midprice


def _scalar(x):
    return float(np.asarray(x).reshape(-1)[0])


def check(case, label, expected, simulated, exact=False, atol=1e-9):
    expected, simulated = _scalar(expected), _scalar(simulated)
    abs_err = abs(simulated - expected)
    passed = (simulated == expected) if exact else np.isclose(simulated, expected, atol=atol, rtol=0)
    REPORT_ROWS.append((case, label, expected, simulated, abs_err, passed))
    return passed


# ======================================================================
# Case 1: Ask fill with upward adverse jump
# ======================================================================
def test_case_1():
    env = make_test_env()
    set_state(env, cash=0.0, inventory=0, midprice=100.0)
    cash_0, inv_0, mid_0 = 0.0, 0, 100.0
    wealth_before = wealth(cash_0, inv_0, mid_0)

    reward, done, info = force_step(
        env, buy_arrival=1, sell_arrival=0, ask_fill=1, bid_fill=0,
        jump_up=0.5, jump_down=0.0,
    )
    s = env.model_dynamics.state[0]
    cash_after, inv_after, mid_after = s[CASH_INDEX], s[INVENTORY_INDEX], s[ASSET_PRICE_INDEX]
    wealth_after = wealth(cash_after, inv_after, mid_after)
    wealth_change = wealth_after - wealth_before

    assert check("1", "cash_after", 100.7, cash_after, exact=False)
    assert check("1", "inventory_after", -1, inv_after, exact=True)
    assert check("1", "midprice_after", 100.5, mid_after, exact=False)
    assert check("1", "wealth_change", ASK_DEPTH - 0.5, wealth_change, exact=False)
    assert check("1", "reward == wealth_change", wealth_change, reward, exact=False)


# ======================================================================
# Case 2: Ask-side arrival and jump WITHOUT a fill
# ======================================================================
def test_case_2():
    env = make_test_env()
    set_state(env, cash=0.0, inventory=0, midprice=100.0)
    cash_0, inv_0, mid_0 = 0.0, 0, 100.0
    wealth_before = wealth(cash_0, inv_0, mid_0)

    reward, done, info = force_step(
        env, buy_arrival=1, sell_arrival=0, ask_fill=0, bid_fill=0,
        jump_up=0.5, jump_down=0.0,
    )
    s = env.model_dynamics.state[0]
    cash_after, inv_after, mid_after = s[CASH_INDEX], s[INVENTORY_INDEX], s[ASSET_PRICE_INDEX]
    wealth_after = wealth(cash_after, inv_after, mid_after)

    assert check("2", "cash_change", 0.0, cash_after - cash_0, exact=False)
    assert check("2", "inventory_change", 0, inv_after - inv_0, exact=True)
    assert check("2", "wealth_change", 0.0, wealth_after - wealth_before, exact=False)
    assert check("2", "reward == wealth_change", wealth_after - wealth_before, reward, exact=False)


# ======================================================================
# Case 3: Bid fill with downward adverse jump
# ======================================================================
def test_case_3():
    env = make_test_env()
    set_state(env, cash=0.0, inventory=0, midprice=100.0)
    wealth_before = wealth(0.0, 0, 100.0)

    reward, done, info = force_step(
        env, buy_arrival=0, sell_arrival=1, ask_fill=0, bid_fill=1,
        jump_up=0.0, jump_down=0.4,
    )
    s = env.model_dynamics.state[0]
    cash_after, inv_after, mid_after = s[CASH_INDEX], s[INVENTORY_INDEX], s[ASSET_PRICE_INDEX]
    wealth_after = wealth(cash_after, inv_after, mid_after)
    wealth_change = wealth_after - wealth_before

    assert check("3", "cash_after", -99.2, cash_after, exact=False)
    assert check("3", "inventory_after", 1, inv_after, exact=True)
    assert check("3", "midprice_after", 99.6, mid_after, exact=False)
    assert check("3", "wealth_change", BID_DEPTH - 0.4, wealth_change, exact=False)
    assert check("3", "reward == wealth_change", wealth_change, reward, exact=False)


# ======================================================================
# Case 4: Bid-side arrival and jump WITHOUT a fill
# ======================================================================
def test_case_4():
    env = make_test_env()
    set_state(env, cash=0.0, inventory=0, midprice=100.0)
    wealth_before = wealth(0.0, 0, 100.0)

    reward, done, info = force_step(
        env, buy_arrival=0, sell_arrival=1, ask_fill=0, bid_fill=0,
        jump_up=0.0, jump_down=0.4,
    )
    s = env.model_dynamics.state[0]
    cash_after, inv_after, mid_after = s[CASH_INDEX], s[INVENTORY_INDEX], s[ASSET_PRICE_INDEX]
    wealth_after = wealth(cash_after, inv_after, mid_after)

    assert check("4", "cash_change", 0.0, cash_after - 0.0, exact=False)
    assert check("4", "inventory_change", 0, inv_after - 0, exact=True)
    assert check("4", "wealth_change", 0.0, wealth_after - wealth_before, exact=False)
    assert check("4", "reward == wealth_change", wealth_after - wealth_before, reward, exact=False)


# ======================================================================
# Case 5: Calm-regime fills without jumps
# ======================================================================
def test_case_5():
    # 5a: ask fill, no jump
    env = make_test_env()
    set_state(env, cash=0.0, inventory=0, midprice=100.0)
    wealth_before = wealth(0.0, 0, 100.0)
    reward, done, info = force_step(
        env, buy_arrival=1, sell_arrival=0, ask_fill=1, bid_fill=0,
        jump_up=0.0, jump_down=0.0,
    )
    s = env.model_dynamics.state[0]
    wealth_after = wealth(s[CASH_INDEX], s[INVENTORY_INDEX], s[ASSET_PRICE_INDEX])
    wealth_change = wealth_after - wealth_before
    assert check("5a", "wealth_change == ask_depth", ASK_DEPTH, wealth_change, exact=False)
    assert check("5a", "reward == wealth_change", wealth_change, reward, exact=False)

    # 5b: bid fill, no jump
    env = make_test_env()
    set_state(env, cash=0.0, inventory=0, midprice=100.0)
    wealth_before = wealth(0.0, 0, 100.0)
    reward, done, info = force_step(
        env, buy_arrival=0, sell_arrival=1, ask_fill=0, bid_fill=1,
        jump_up=0.0, jump_down=0.0,
    )
    s = env.model_dynamics.state[0]
    wealth_after = wealth(s[CASH_INDEX], s[INVENTORY_INDEX], s[ASSET_PRICE_INDEX])
    wealth_change = wealth_after - wealth_before
    assert check("5b", "wealth_change == bid_depth", BID_DEPTH, wealth_change, exact=False)
    assert check("5b", "reward == wealth_change", wealth_change, reward, exact=False)


# ======================================================================
# Case 6: Existing inventory revaluation without a fill
# ======================================================================
def test_case_6():
    # 6a: upward jump, inventory=2, no fill
    env = make_test_env()
    set_state(env, cash=0.0, inventory=2, midprice=100.0)
    wealth_before = wealth(0.0, 2, 100.0)
    reward, done, info = force_step(
        env, buy_arrival=1, sell_arrival=0, ask_fill=0, bid_fill=0,
        jump_up=0.5, jump_down=0.0,
    )
    s = env.model_dynamics.state[0]
    wealth_after = wealth(s[CASH_INDEX], s[INVENTORY_INDEX], s[ASSET_PRICE_INDEX])
    wealth_change = wealth_after - wealth_before
    assert check("6a", "wealth_change == 2*0.5", 2 * 0.5, wealth_change, exact=False)
    assert check("6a", "reward == wealth_change", wealth_change, reward, exact=False)

    # 6b: downward jump, inventory=2, no fill (fresh reset)
    env = make_test_env()
    set_state(env, cash=0.0, inventory=2, midprice=100.0)
    wealth_before = wealth(0.0, 2, 100.0)
    reward, done, info = force_step(
        env, buy_arrival=0, sell_arrival=1, ask_fill=0, bid_fill=0,
        jump_up=0.0, jump_down=0.4,
    )
    s = env.model_dynamics.state[0]
    wealth_after = wealth(s[CASH_INDEX], s[INVENTORY_INDEX], s[ASSET_PRICE_INDEX])
    wealth_change = wealth_after - wealth_before
    assert check("6b", "wealth_change == -2*0.4", -2 * 0.4, wealth_change, exact=False)
    assert check("6b", "reward == wealth_change", wealth_change, reward, exact=False)


# ======================================================================
# Case 7: Both sides fill in the same step
# ======================================================================
def test_case_7():
    env = make_test_env()
    set_state(env, cash=0.0, inventory=0, midprice=100.0)
    cash_0, inv_0, mid_0 = 0.0, 0, 100.0
    wealth_before = wealth(cash_0, inv_0, mid_0)

    reward, done, info = force_step(
        env, buy_arrival=1, sell_arrival=1, ask_fill=1, bid_fill=1,
        jump_up=0.5, jump_down=0.4,
    )
    s = env.model_dynamics.state[0]
    cash_after, inv_after, mid_after = s[CASH_INDEX], s[INVENTORY_INDEX], s[ASSET_PRICE_INDEX]
    wealth_after = wealth(cash_after, inv_after, mid_after)
    wealth_change = wealth_after - wealth_before

    print("\n=== Case 7: both sides fill (manual derivation vs simulated) ===")
    print(f"  cash:      before={cash_0:.4f}  after={cash_after:.4f}")
    print(f"  inventory: before={inv_0}       after={int(inv_after)}")
    print(f"  price:     before={mid_0:.4f}  after={mid_after:.4f}  (net jump = +0.5-0.4 = +0.1)")
    print(f"  wealth:    before={wealth_before:.4f}  after={wealth_after:.4f}")
    print(f"  reward = {_scalar(reward):.6f}")
    print(f"  ask_depth+bid_depth = {ASK_DEPTH+BID_DEPTH:.4f}")
    print(f"  wealth_change == ask_depth+bid_depth ? {np.isclose(wealth_change, ASK_DEPTH+BID_DEPTH)}"
          f"  (wealth_change={wealth_change:.6f})")

    # Inventory returns to exactly zero (fill_multiplier cancellation), so the
    # net jump has NO effect on wealth -- wealth_change collapses to the pure
    # cash round-trip, which nets the (identical, pre-jump) midprice out
    # exactly: (mid+ask_depth) - (mid-bid_depth) = ask_depth+bid_depth. No
    # extra jump-related cross term survives.
    assert check("7", "inventory returns to 0", 0, inv_after, exact=True)
    assert check("7", "wealth_change == ask_depth+bid_depth", ASK_DEPTH + BID_DEPTH, wealth_change, exact=False)
    assert check("7", "reward == wealth_change", wealth_change, reward, exact=False)


# ======================================================================
# Reporting
# ======================================================================
def print_report():
    print("\n" + "=" * 92)
    print(f"{'Case':<6}{'Check':<32}{'Expected':>14}{'Simulated':>14}{'AbsErr':>12}{'Result':>10}")
    print("=" * 92)
    n_pass = n_fail = 0
    for case, label, expected, simulated, abs_err, passed in REPORT_ROWS:
        status = "PASS" if passed else "FAIL"
        if passed:
            n_pass += 1
        else:
            n_fail += 1
        print(f"{case:<6}{label:<32}{expected:>14.6f}{simulated:>14.6f}{abs_err:>12.2e}{status:>10}")
    print("=" * 92)
    print(f"{n_pass} passed, {n_fail} failed out of {n_pass+n_fail} checks")


def print_event_ordering():
    print("\n" + "=" * 92)
    print("Event ordering (from TradingEnvironment._update_state / _update_agent_state / "
          "_update_market_state, mbt_gym/gym/TradingEnvironment.py:198-216):")
    print("=" * 92)
    print("""
  1. Arrival generation:   arrivals = model_dynamics.get_arrivals_and_fills(action)
                           calls arrival_model.get_arrivals() FIRST
  2. Fill determination:   ... then fill_probability_model.get_fills(depths) SECOND,
                           within the same get_arrivals_and_fills() call
  3. Cash & inventory update: _update_agent_state() -> model_dynamics.update_state():
       - inventory line executes first, cash line second (both reference
         self.midprice, i.e. midprice_model.current_state, which is UNCHANGED
         at this point -- both use the PRE-jump midprice)
  4. Midprice (+ jump) update: _update_market_state() runs AFTER agent-state
       update -- ArrivalJumpMidpriceModel.update() applies diffusion+jump here,
       and the result is copied back into model_dynamics.state[:,ASSET_PRICE_INDEX]
  5. Reward calculation:   TradingEnvironment.step() calls reward_function.calculate(
       current_state, action, next_state, ...) using current_state (captured
       BEFORE _update_state ran) and next_state (captured AFTER both agent-state
       and market-state updates) -- so next_state's price already reflects the jump.

  Consistency with the HJB's "spread - adverse jump" assumption:
    Because the fill executes at old_midprice +/- depth (step 3, pre-jump) and
    the jump is only applied afterward (step 4), an adverse fill's markout is
    exactly (old_midprice +/- depth) - (new_midprice) = +/-depth -/+ jump,
    i.e. spread - adverse_jump for an ask fill under an upward jump (Case 1:
    0.7 - 0.5 = 0.2) and spread - adverse_jump for a bid fill under a downward
    jump (Case 3: 0.8 - 0.4 = 0.4). This matches the HJB assumption exactly --
    confirmed empirically above, not merely by code inspection.
""")


if __name__ == "__main__":
    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    test_case_5()
    test_case_6()
    test_case_7()
    print_event_ordering()
    print_report()
