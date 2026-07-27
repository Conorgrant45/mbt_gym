"""
event_driven_regime_env.py
---------------------------
Phase 1 of the event-driven regime-switching environment. Replaces the
4,000-step fixed time grid (envs/regime_env.py's RegimeSwitchingEnv) with
decisions made only at observable market-order arrival times, while
preserving the exact same economic model: same T=1, same arrival/fill/jump/
regime-transition dynamics, same reward decomposition, same bid/ask action
convention and sign conventions.

Does NOT modify, wrap, or replace RegimeSwitchingEnv, TradingEnvironment or
any mbt_gym class -- this is a fully independent, parallel implementation.
It reuses ONLY the shared economic constants from envs/make_envs.py (KAPPA,
R0_VOLATILITY, R1_VOLATILITY, EPSILON, LAMBDA, TERMINAL_TIME, INITIAL_PRICE,
PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION, and the new
TRANSITION_GENERATOR export), so the two environments cannot silently drift
apart on parameters. State (cash, inventory, time, price, hidden regime) is
simulated directly here rather than delegating to mbt_gym's ModelDynamics/
TradingEnvironment classes, since those assume a fixed dt loop and cannot
represent "wait until the next Poisson event".

======================================================================
Ordering, confirmed from the EXISTING fixed-step implementation
(mbt_gym/mbt_gym/gym/TradingEnvironment.py::_update_state,
mbt_gym/mbt_gym/gym/ModelDynamics.py::LimitOrderModelDynamics.update_state,
envs/arrival_jump_midprice.py, mbt_gym/mbt_gym/rewards/RewardFunctions.py)
-- NOT inferred, read directly from the source:

    1. arrivals, fills = get_arrivals_and_fills(action)   -- drawn from the
       state BEFORE this step's market update; fills use the depths chosen
       by the action and are independent Bernoulli(exp(-kappa*depth)) draws,
       independent of arrivals.
    2. Agent state (cash, inventory) is updated FIRST, using the CURRENT
       (pre-diffusion, pre-jump) midprice: bid fill executes at
       (midprice - bid_depth), ask fill executes at (midprice + ask_depth).
    3. Market state (midprice) is updated SECOND: Brownian diffusion, then
       (regime 1 only) the arrival-triggered permanent jump. This means the
       fill price never includes this step's own jump -- the jump is what
       makes that same fill's mark-to-market value adverse in the reward,
       computed next.
    4. Reward = mark_to_market(next_state) - mark_to_market(next_state's
       cash/inventory replaced by... ) -- concretely (RunningInventoryPenalty
       .calculate, PnL.calculate):
           pnl = (cash_next + inv_next * price_next) - (cash_cur + inv_cur * price_cur)
           reward = pnl - dt * phi * inv_next**2 - alpha * is_terminal * inv_next**2
       i.e. the running penalty uses the (already fill-updated) NEXT inventory,
       scaled by the step's own dt, and the terminal penalty (if this is the
       last step) uses next inventory too, applied exactly once.

This environment reproduces that ordering exactly, generalised from a fixed
dt to a variable elapsed interval Delta-tau between observable events:
fills are still computed and applied at the pre-jump price, the jump is
still applied strictly after the fill, and the running penalty for an
interval of constant inventory Q is exactly phi * Q**2 * Delta-tau (no
approximation is needed here, unlike the fixed grid, because inventory is
provably constant between arrival events -- only an arrival's own fill can
change it, and by the time that fill is processed the interval's penalty has
already been charged against the OLD, correct, constant inventory).

======================================================================
Competing-event simulation
======================================================================
One RL step = the next OBSERVABLE market-order arrival (buy or sell), or the
terminal horizon if no arrival occurs first. Hidden regime transitions are
simulated internally and never produce an RL step/observation -- otherwise
the agent would learn that a regime switch had occurred, which is exactly
the leakage the belief-state design (Phase 2) exists to avoid.

From the current hidden regime z, four competing events race:
    - buy MO arrival:  Exponential(rate=lambda_ask)   [ASK_INDEX; buy MOs hit
                        the agent's resting ask -- ArrivalModel's own
                        docstring convention, unchanged here]
    - sell MO arrival: Exponential(rate=lambda_bid)   [BID_INDEX]
    - regime switch:   Exponential(rate=-Q[z, z])     [Q = continuous-time
                        generator; for this 2-state chain, the only
                        possible destination is 1 - z]
    - horizon:         deterministic, terminal_time - current_time

The earliest of the four determines what happens next. If it is a regime
switch, the environment evolves internally (no RL step) and loops again from
the new regime, with the SAME outstanding bid/ask depths (the market maker's
resting quotes do not know a hidden regime changed). If it is an arrival or
the horizon, control returns to the agent with one environment transition.

Brownian motion accumulates additively over disjoint time intervals, so
diffusing separately over every internal sub-interval (with that
sub-interval's own regime-specific sigma) and diffusing ONCE over the whole
elapsed span with the VARIANCE-WEIGHTED SUM

    integrated_variance = sum_j sigma_{z_j}^2 * Delta-tau_j
    brownian_increment  = sqrt(integrated_variance) * Z,      Z ~ N(0, 1)

are statistically identical (a sum of independent Gaussians is Gaussian with
summed variances) -- so this environment draws Z only once per RL step,
after all internal regime-switch sub-intervals have been resolved, exactly
as prescribed.

======================================================================
RNG stream (single Generator, fixed per-iteration draw order)
======================================================================
One np.random.Generator drives every random draw in this environment (unlike
the fixed-step model's several independently-seeded sub-process RNGs -- this
is a new, independent environment and does not need to replicate that
architecture to be reproducible). Per competing-event iteration:
    1. t_buy    ~ Exponential(rate=lambda_ask)
    2. t_sell   ~ Exponential(rate=lambda_bid)
    3. t_switch ~ Exponential(rate=-Q[z, z])   (only if -Q[z, z] > 0)
If the winning event is an arrival, two (buy) or one (sell, no extra) further
draws follow once all sub-intervals are resolved:
    4. Z              ~ N(0, 1)                          (once per RL step)
    5. fill_uniform   ~ Uniform(0, 1)                    (arriving side only)
    6. jump_magnitude ~ Exponential(mean=epsilon)        (regime 1 only)
This fixed order is exercised directly by
tests/test_event_driven_regime_env.py::test_identical_seeds_reproduce_identical_event_paths.
"""

import gym
import numpy as np
from gym import spaces

from envs.make_envs import (
    TERMINAL_TIME,
    KAPPA,
    R0_VOLATILITY,
    R1_VOLATILITY,
    EPSILON,
    LAMBDA,
    INITIAL_PRICE,
    PER_STEP_INVENTORY_AVERSION,
    TERMINAL_INVENTORY_AVERSION,
    TRANSITION_GENERATOR,
)
from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, TIME_INDEX, ASSET_PRICE_INDEX, BID_INDEX, ASK_INDEX

# Must equal simulate_belief_weighted.MAX_DEPTH and the fixed-step env's own
# ExponentialFillFunction(fill_exponent=KAPPA).max_depth -- see
# tests/test_event_driven_regime_env.py::test_max_depth_matches_fixed_step_convention.
MAX_DEPTH = -np.log(0.01) / KAPPA

# Absolute price-unit diffusion volatility per regime (NOT the Hamilton-filter
# percentage-return-scaled _PCT/_STEP variants in envs/make_envs.py -- those
# are filter-specific and irrelevant to the environment's own price dynamics).
DEFAULT_SIGMA = {0: R0_VOLATILITY, 1: R1_VOLATILITY}

# Both currently equal (LAMBDA) -- named separately since the model
# conceptually allows asymmetric buy/sell arrival intensities.
DEFAULT_LAMBDA_ASK = LAMBDA  # buy-MO arrival rate (can fill the agent's ask)
DEFAULT_LAMBDA_BID = LAMBDA  # sell-MO arrival rate (can fill the agent's bid)

MAX_INVENTORY = 10_000  # matches TradingEnvironment's own default


def stationary_distribution_from_generator(Q: np.ndarray) -> np.ndarray:
    """Stationary distribution pi of a continuous-time generator Q (pi @ Q = 0,
    sum(pi) = 1). Solved directly from Q -- NOT derived from TRANSITION_MATRIX's
    own discrete-time stationary distribution -- though the two are
    mathematically identical (any time-discretisation P(dt) = expm(Q*dt) of a
    generator shares its stationary distribution); see
    tests/test_event_driven_regime_env.py::test_stationary_distribution_matches_discrete_chain.
    """
    n = Q.shape[0]
    A = np.vstack([Q.T, np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    return pi


def denormalise_depth(action_value: float, max_depth: float) -> float:
    """action in [-1, 1] -> physical depth in [0, max_depth]. Identical linear
    formula to TradingEnvironment.normalise_action's own inverse transform
    (gradient = max_depth / 2, intercept = 0) and to
    simulate_belief_weighted.normalise_depth's inverse -- see module docstring
    for the derivation/cross-check."""
    return (float(action_value) + 1.0) / 2.0 * max_depth


class EventDrivenRegimeSwitchingEnv(gym.Env):
    """
    Event-driven counterpart of RegimeSwitchingEnv. One step() call advances
    the environment from the current observable-event time to the NEXT
    observable market-order arrival, or to the terminal horizon, absorbing
    any number of hidden regime transitions along the way without producing
    an intermediate observation. See module docstring for full ordering.

    Single trajectory only (no num_trajectories batching in Phase 1 -- the
    equivalence diagnostics and required tests run many independent env
    instances instead, matching how this project's analytical-policy Monte
    Carlo comparisons already work).

    The returned observation is the raw (un-normalised) state
    [cash, inventory, time, price] -- the SAME ordering as mbt_gym's
    CASH_INDEX/INVENTORY_INDEX/TIME_INDEX/ASSET_PRICE_INDEX convention, and
    the same convention RegimeSwitchingEnv's raw_state/raw_cash/
    raw_inventory/raw_midprice properties already use, so a Phase 2 wrapper
    can be built the same way HamiltonPPOWrapper/ReturnPPOWrapper wrap
    RegimeSwitchingEnv. The hidden regime is NEVER part of this observation.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        transition_generator=TRANSITION_GENERATOR,
        sigma: dict = None,
        epsilon: float = EPSILON,
        lambda_bid: float = DEFAULT_LAMBDA_BID,
        lambda_ask: float = DEFAULT_LAMBDA_ASK,
        kappa: float = KAPPA,
        phi: float = PER_STEP_INVENTORY_AVERSION,
        alpha: float = TERMINAL_INVENTORY_AVERSION,
        terminal_time: float = TERMINAL_TIME,
        initial_price: float = INITIAL_PRICE,
        initial_cash: float = 0.0,
        initial_inventory: float = 0.0,
        max_inventory: float = MAX_INVENTORY,
        initial_regime: int = None,
        seed: int = None,
    ):
        super().__init__()
        self.Q = np.array(transition_generator, dtype=np.float64)
        assert self.Q.shape == (2, 2), "Phase 1 only supports a 2-state chain."
        assert np.allclose(self.Q.sum(axis=1), 0.0, atol=1e-8), "Generator rows must sum to zero."

        self.sigma = dict(sigma) if sigma is not None else dict(DEFAULT_SIGMA)
        self.epsilon = float(epsilon)
        self.lambda_bid = float(lambda_bid)
        self.lambda_ask = float(lambda_ask)
        self.kappa = float(kappa)
        self.phi = float(phi)
        self.alpha = float(alpha)
        self.terminal_time = float(terminal_time)
        self.initial_price = float(initial_price)
        self.initial_cash = float(initial_cash)
        self.initial_inventory = float(initial_inventory)
        self.max_inventory = float(max_inventory)
        self.initial_regime = initial_regime
        self.max_depth = -np.log(0.01) / self.kappa

        self._pi_stat = stationary_distribution_from_generator(self.Q)
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-1e12, -self.max_inventory, 0.0, 0.0], dtype=np.float32),
            high=np.array([1e12, self.max_inventory, self.terminal_time, np.inf], dtype=np.float32),
        )

        self.cash = None
        self.inventory = None
        self.time = None
        self.price = None
        self.regime = None

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------
    def reset(self):
        self.cash = self.initial_cash
        self.inventory = self.initial_inventory
        self.time = 0.0
        self.price = self.initial_price
        self.regime = (
            int(self.initial_regime)
            if self.initial_regime is not None
            else int(self.rng.choice(len(self._pi_stat), p=self._pi_stat))
        )
        return self.raw_state.copy()

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        bid_depth = denormalise_depth(action_arr[BID_INDEX], self.max_depth)
        ask_depth = denormalise_depth(action_arr[ASK_INDEX], self.max_depth)

        cash_before, inv_before, price_before = self.cash, self.inventory, self.price
        regime_before_interval = self.regime

        event = self._advance_to_next_event(bid_depth, ask_depth)

        terminal_penalty = 0.0
        done = event["event_type"] == "terminal"
        if done:
            terminal_penalty = self.alpha * (self.inventory ** 2)

        raw_pnl_step = (self.cash + self.inventory * self.price) - (cash_before + inv_before * price_before)
        reward = raw_pnl_step - event["running_penalty"] - terminal_penalty

        info = dict(
            elapsed_time=event["elapsed_time"],
            event_type=event["event_type"],
            arrival_side=event["arrival_side"],
            number_internal_regime_switches=event["n_internal_switches"],
            regime_before_interval=regime_before_interval,
            regime_at_event=event["regime_at_event"],
            true_regime=event["regime_at_event"],
            integrated_variance=event["integrated_variance"],
            brownian_increment=event["brownian_increment"],
            jump_increment=event["jump_increment"],
            fill_indicator=np.array([event["fill_bid"], event["fill_ask"]], dtype=np.int64),
            running_penalty_increment=event["running_penalty"],
            terminal_penalty_increment=terminal_penalty,
            raw_state=self.raw_state.copy(),
            cash_before=cash_before,
            inventory_before=inv_before,
            price_before=price_before,
            cash_after=self.cash,
            inventory_after=self.inventory,
            price_after=self.price,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
        )
        return self.raw_state.copy(), float(reward), done, info

    def render(self, mode="human"):
        return None

    def close(self):
        pass

    def seed(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Core competing-event simulation
    # ------------------------------------------------------------------
    def _advance_to_next_event(self, bid_depth: float, ask_depth: float) -> dict:
        integrated_variance = 0.0
        n_internal_switches = 0
        running_penalty = 0.0
        elapsed_time = 0.0

        while True:
            z = self.regime
            time_left = self.terminal_time - self.time

            t_buy = self.rng.exponential(1.0 / self.lambda_ask)
            t_sell = self.rng.exponential(1.0 / self.lambda_bid)
            q_out = -self.Q[z, z]
            t_switch = self.rng.exponential(1.0 / q_out) if q_out > 0 else np.inf

            candidates = {"terminal": time_left, "switch": t_switch, "buy": t_buy, "sell": t_sell}
            winner = min(candidates, key=candidates.get)
            dtau = candidates[winner]

            integrated_variance += self.sigma[z] ** 2 * dtau
            running_penalty += self.phi * (self.inventory ** 2) * dtau
            elapsed_time += dtau
            self.time += dtau

            if winner == "terminal":
                self.time = self.terminal_time  # exact horizon, no float drift
                event_type, arrival_side = "terminal", None
                break
            elif winner == "switch":
                self.regime = 1 - z  # 2-state chain: the only other state
                n_internal_switches += 1
                continue
            else:
                event_type, arrival_side = "arrival", winner  # "buy" or "sell"
                break

        regime_at_event = self.regime

        z_std_normal = self.rng.normal()
        brownian_increment = np.sqrt(integrated_variance) * z_std_normal
        price_pre_jump = self.price + brownian_increment

        jump_increment = 0.0
        fill_bid = 0
        fill_ask = 0
        if event_type == "arrival":
            if arrival_side == "buy":
                # Buy MO arrival can fill the agent's resting ASK.
                already_at_min = self.inventory <= -self.max_inventory
                fill_prob = np.exp(-self.kappa * ask_depth)
                fill_ask = int((not already_at_min) and (self.rng.uniform() < fill_prob))
                if regime_at_event == 1:
                    jump_increment = float(self.rng.exponential(self.epsilon))
            else:
                # Sell MO arrival can fill the agent's resting BID.
                already_at_max = self.inventory >= self.max_inventory
                fill_prob = np.exp(-self.kappa * bid_depth)
                fill_bid = int((not already_at_max) and (self.rng.uniform() < fill_prob))
                if regime_at_event == 1:
                    jump_increment = -float(self.rng.exponential(self.epsilon))

            if fill_bid:
                self.inventory += 1
                self.cash -= price_pre_jump - bid_depth
            if fill_ask:
                self.inventory -= 1
                self.cash += price_pre_jump + ask_depth

        self.price = price_pre_jump + jump_increment

        return dict(
            event_type=event_type,
            arrival_side=arrival_side,
            n_internal_switches=n_internal_switches,
            integrated_variance=integrated_variance,
            brownian_increment=brownian_increment,
            jump_increment=jump_increment,
            fill_bid=fill_bid,
            fill_ask=fill_ask,
            running_penalty=running_penalty,
            elapsed_time=elapsed_time,
            regime_at_event=regime_at_event,
        )

    # ------------------------------------------------------------------
    # Raw-state accessors (same convention as RegimeSwitchingEnv)
    # ------------------------------------------------------------------
    @property
    def raw_state(self) -> np.ndarray:
        state = np.zeros(4, dtype=np.float64)
        state[CASH_INDEX] = self.cash
        state[INVENTORY_INDEX] = self.inventory
        state[TIME_INDEX] = self.time
        state[ASSET_PRICE_INDEX] = self.price
        return state

    @property
    def raw_cash(self) -> float:
        return float(self.cash)

    @property
    def raw_inventory(self) -> float:
        return float(self.inventory)

    @property
    def raw_midprice(self) -> float:
        return float(self.price)

    @property
    def current_regime(self) -> int:
        return int(self.regime)
