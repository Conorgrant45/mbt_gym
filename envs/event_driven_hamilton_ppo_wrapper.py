"""
event_driven_hamilton_ppo_wrapper.py
---------------------------------------
Gymnasium-compatible PPO observation wrapper around
EventDrivenRegimeSwitchingEnv, exposing the event-time Hamilton-filter
belief b_n in place of the hidden true regime. Structurally modelled on
envs/hamilton_ppo_wrapper.py's HamiltonPPOWrapper (same legacy gym.Env ->
Gymnasium translation, same observation composition, same
reset/step chronology) -- does not modify that file or the fixed-step
environment it wraps.

Observation (shape (3,), float32):
    x_n = [q_n_scaled, tau_n, b_n]
        q_n_scaled = tanh(q_n / inventory_scale)   -- q_n is RAW inventory
                                                        (shares), read from
                                                        info['inventory_after']
                                                        / base_env.raw_inventory,
                                                        never reconstructed
                                                        from a normalised obs.
        tau_n      = 1 - current_time / terminal_time  -- fraction of the
                                                        HORIZON (not step
                                                        count -- there is no
                                                        fixed step count in
                                                        the event-driven
                                                        environment) remaining.
        b_n        = P(Z_{t_n}=1 | information through event n) --
                                                        EventTimeHamiltonFilter
                                                        belief, already
                                                        including both the
                                                        CTMC prediction over
                                                        the elapsed
                                                        inter-event interval
                                                        and the likelihood
                                                        update from the most
                                                        recent observable
                                                        return (see
                                                        beliefs/event_time_
                                                        hamilton_filter.py).

The true regime (info['true_regime']/info['regime_at_event']) is never read
when constructing the observation -- see
tests/test_event_driven_agent_integration.py's explicit leakage tests.

Chronology (no look-ahead), mirroring HamiltonPPOWrapper's:
    reset():
        1. base_env.reset()
        2. filt.reset(initial_price=S_0)  -- prior belief = stationary dist.,
           S_0 cached as the reference price for the FIRST update()'s return
           (no "pass-through first call" special case is needed here, unlike
           the fixed-step filter -- see EventTimeHamiltonFilter.reset()'s
           docstring: every observable event, including the first, has a
           well-defined delta_tau and return in this environment).
        3. return [q_0_scaled, 1.0, b_0]

    step(a_n):
        1. raw_state, reward, done, info = base_env.step(a_n) -- a_n was
           chosen by the caller from the PREVIOUS observation, which already
           encoded belief b_n; nothing here uses information from AFTER this
           call before it returns.
        2. b_{n+1} = filt.update(info['price_after'], info['elapsed_time'],
           info['event_type']=='arrival') -- exactly ONE filter update per
           observable event (arrival or terminal), NEVER from an internal
           hidden regime switch (those never reach step()/info at all --
           EventDrivenRegimeSwitchingEnv absorbs them internally, see
           envs/event_driven_regime_env.py).
        3. q_{n+1} = info['inventory_after']
        4. tau_{n+1} = 1 - info['raw_state'][TIME_INDEX] / terminal_time
        5. return [q_{n+1}_scaled, tau_{n+1}, b_{n+1}], reward, terminated,
           truncated, info

Reward is passed through unchanged.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.make_envs import (
    TERMINAL_TIME,
    R0_VOLATILITY_PCT,
    R1_VOLATILITY_PCT,
    EPSILON_PCT,
    LAMBDA,
    TRANSITION_GENERATOR,
)
from envs.event_driven_regime_env import EventDrivenRegimeSwitchingEnv
from beliefs.event_time_hamilton_filter import EventTimeHamiltonFilter
from mbt_gym.gym.index_names import INVENTORY_INDEX, TIME_INDEX, ASSET_PRICE_INDEX

# Same default as envs/hamilton_ppo_wrapper.py's DEFAULT_INVENTORY_SCALE --
# imported (not redefined) so both environments' Hamilton wrappers share one
# empirically-justified value unless explicitly overridden.
from envs.hamilton_ppo_wrapper import DEFAULT_INVENTORY_SCALE


def make_event_time_filter() -> EventTimeHamiltonFilter:
    """Event-time-filter analogue of envs.hamilton_ppo_wrapper.make_filter():
    built directly from envs.make_envs constants, not re-derived. Regime
    intensities/volatilities are the SAME for both regimes by construction
    here (matching the production calibration) -- see
    EventTimeHamiltonFilter's constructor guards for what happens if that
    ever stops being true."""
    return EventTimeHamiltonFilter(
        transition_generator=TRANSITION_GENERATOR,
        regime_volatilities=[R0_VOLATILITY_PCT, R1_VOLATILITY_PCT],
        jump_size=EPSILON_PCT,
        lambda_bid=LAMBDA,
        lambda_ask=LAMBDA,
    )


class EventDrivenHamiltonPPOWrapper(gym.Env):
    """
    Gymnasium Env wrapping EventDrivenRegimeSwitchingEnv, replacing the raw
    (cash, inventory, time, price) observation with [q_scaled, tau, belief]
    and driving an EventTimeHamiltonFilter from the observable-event price
    sequence. See module docstring for the full reset/step chronology.

    Parameters
    ----------
    base_env : EventDrivenRegimeSwitchingEnv, optional
        Pre-built environment to wrap. If None, one is built via
        EventDrivenRegimeSwitchingEnv(seed=seed).
    inventory_scale : float
        Denominator in tanh(q / inventory_scale). Shared default with the
        fixed-step HamiltonPPOWrapper.
    seed : int, optional
        Forwarded to EventDrivenRegimeSwitchingEnv(seed=...) when base_env is
        None.
    """

    metadata = {"render_modes": []}
    ENVIRONMENT_TYPE = "event"

    def __init__(self, base_env=None, inventory_scale: float = DEFAULT_INVENTORY_SCALE, seed: int = None):
        super().__init__()

        if inventory_scale <= 0:
            raise ValueError(f"inventory_scale must be positive, got {inventory_scale}")

        self.base_env = base_env if base_env is not None else EventDrivenRegimeSwitchingEnv(seed=seed)
        self.inventory_scale = float(inventory_scale)
        self.terminal_time = self.base_env.terminal_time

        self.filt = make_event_time_filter()

        # base_env.action_space is already a gymnasium.spaces.Box normalised
        # to [-1, 1] (see EventDrivenRegimeSwitchingEnv) -- re-expressed here
        # with identical bounds/dtype/ordering ([bid_action, ask_action]).
        # Actions pass straight through to base_env.step() with no extra
        # transform -- the base env already denormalises internally.
        base_low = np.asarray(self.base_env.action_space.low, dtype=np.float32)
        base_high = np.asarray(self.base_env.action_space.high, dtype=np.float32)
        self.action_space = spaces.Box(low=base_low, high=base_high, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _scaled_inventory(self, raw_inventory: float) -> float:
        return float(np.tanh(raw_inventory / self.inventory_scale))

    def _build_obs(self, raw_inventory: float, current_time: float, belief: float) -> np.ndarray:
        q_scaled = self._scaled_inventory(raw_inventory)
        tau = 1.0 - (float(current_time) / float(self.terminal_time))
        return np.array([q_scaled, tau, belief], dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------
    def reset(self, *, seed: int = None, options: dict = None):
        raw_state = self.base_env.reset()
        s0 = float(raw_state[ASSET_PRICE_INDEX])
        self.filt.reset(initial_price=s0)

        q0 = float(raw_state[INVENTORY_INDEX])
        obs = self._build_obs(q0, current_time=0.0, belief=self.filt.belief)
        return obs, {}

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        raw_state, reward, done, info = self.base_env.step(action_arr)

        belief = self.filt.update(
            info["price_after"], info["elapsed_time"], info["event_type"] == "arrival"
        )
        q_next = float(info["inventory_after"])
        current_time = float(info["raw_state"][TIME_INDEX])

        next_obs = self._build_obs(q_next, current_time, belief)
        terminated = bool(done)
        truncated = False

        return next_obs, float(reward), terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Introspection (not part of the PPO observation)
    # ------------------------------------------------------------------
    @property
    def belief(self) -> float:
        return self.filt.belief
