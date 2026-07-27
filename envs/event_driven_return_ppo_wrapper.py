"""
event_driven_return_ppo_wrapper.py
-------------------------------------
Gymnasium-compatible PPO observation wrapper around
EventDrivenRegimeSwitchingEnv, exposing the raw one-event midprice return AND
the elapsed time since the previous observable event, in place of the
Hamilton-filter belief. Used by BOTH model-free agents (event-driven
return_mlp_ppo and return_lstm_ppo) -- structurally modelled on
envs/return_ppo_wrapper.py's ReturnPPOWrapper (same legacy gym.Env ->
Gymnasium translation, reset/step handling, action-space conversion, seed
handling all follow the same pattern) and on this module's sibling
EventDrivenHamiltonPPOWrapper.

Observation (shape (4,), float32):
    x_n = [q_n_scaled, tau_n, r_n_scaled, delta_tau_n_scaled]
        q_n_scaled       = tanh(q_n / inventory_scale)   -- identical
                                                              definition/
                                                              default scale
                                                              to the
                                                              event-driven
                                                              Hamilton
                                                              wrapper.
        tau_n            = 1 - current_time / terminal_time  -- identical
                                                              time convention
                                                              to the
                                                              event-driven
                                                              Hamilton
                                                              wrapper.
        r_n_scaled       = tanh(r_n / return_scale)      -- r_n is the raw
                                                              percentage
                                                              return
                                                              observed over
                                                              the elapsed
                                                              interval since
                                                              the previous
                                                              observable
                                                              event
                                                              (info['price_after']
                                                              vs the cached
                                                              previous
                                                              price).
        delta_tau_n_scaled = tanh(delta_tau_n / elapsed_time_scale) -- the
                                                              elapsed time
                                                              itself, since a
                                                              raw return
                                                              alone confounds
                                                              a large return
                                                              over a long
                                                              wait with a
                                                              large return
                                                              over a short
                                                              one (see
                                                              ELAPSED_TIME_
                                                              SCALE_REASONING
                                                              below).

No HamiltonFilter/EventTimeHamiltonFilter object is instantiated anywhere in
this module. The true regime, internal-regime-switch counts and integrated
variance (all privileged info-dict fields) are never read when constructing
the observation -- see tests/test_event_driven_agent_integration.py's
leakage tests.

Chronology (no look-ahead), identical structure to
EventDrivenHamiltonPPOWrapper:
    reset():
        1. base_env.reset()
        2. S_0 = raw price at reset; cache it, do NOT compute a return.
        3. return [q_0_scaled, 1.0, 0.0, 0.0] -- r_0_scaled AND
           delta_tau_0_scaled are exactly 0.0 by construction (no prior
           event exists yet), not an approximation of some "true" reset
           value.

    step(a_n):
        1. raw_state, reward, done, info = base_env.step(a_n) -- a_n was
           chosen from the PREVIOUS observation; nothing here uses
           information from after this call before it returns.
        2. r_{n+1} = (info['price_after'] - S_n) / S_n, using the wrapper's
           own cached S_n (the same one-line formula the environment/filter
           already use internally -- not a second implementation of
           anything privileged).
        3. delta_tau_{n+1} = info['elapsed_time'] (an OBSERVABLE quantity --
           the agent, in a real market, would know how long it waited for
           the next arrival; this is not privileged information).
        4. q_{n+1} = info['inventory_after']
        5. tau_{n+1} = 1 - info['raw_state'][TIME_INDEX] / terminal_time
        6. return [q_{n+1}_scaled, tau_{n+1}, tanh(r_{n+1}/return_scale),
           tanh(delta_tau_{n+1}/elapsed_time_scale)], reward, terminated,
           truncated, info

Reward, termination and truncation are passed through unchanged.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.make_envs import EPSILON_PCT, LAMBDA
from envs.event_driven_regime_env import EventDrivenRegimeSwitchingEnv
from envs.hamilton_ppo_wrapper import DEFAULT_INVENTORY_SCALE
from mbt_gym.gym.index_names import INVENTORY_INDEX, TIME_INDEX, ASSET_PRICE_INDEX

# Same return-scale reasoning as envs/return_ppo_wrapper.py's
# DEFAULT_RETURN_SCALE (imported implicitly by reusing EPSILON_PCT directly,
# not re-derived): the single-arrival jump impact dominates the return
# distribution's economically meaningful variation, so tanh(r / EPSILON_PCT)
# gives a hard [-1, 1] bound with good resolution on the single-jump event.
DEFAULT_RETURN_SCALE = EPSILON_PCT

# ----------------------------------------------------------------------
# ELAPSED_TIME_SCALE_REASONING
#
# delta_tau_n is the time between consecutive OBSERVABLE arrivals -- an
# Exponential(lambda_bid + lambda_ask) random variable when arrival
# intensity is regime-independent (the production case; see
# beliefs/event_time_hamilton_filter.py's module docstring). Its natural
# unit is therefore the MEAN inter-arrival gap, 1/(lambda_bid+lambda_ask) =
# 1/280 ~ 0.00357 -- tanh(delta_tau/scale) with scale set to this mean puts
# a "typical" gap at tanh(1) ~ 0.76, giving good resolution across the bulk
# of the (highly right-skewed) exponential distribution while still
# hard-bounding the rare long waits (delta_tau is formally unbounded above)
# into [-1, 1] without ever needing to be clipped. Mirrors the same
# design philosophy as DEFAULT_INVENTORY_SCALE/DEFAULT_RETURN_SCALE
# (empirically/structurally justified scale, not an arbitrary constant).
# ----------------------------------------------------------------------
DEFAULT_ELAPSED_TIME_SCALE = 1.0 / (2.0 * LAMBDA)


class EventDrivenReturnPPOWrapper(gym.Env):
    """
    Gymnasium Env wrapping EventDrivenRegimeSwitchingEnv, replacing the raw
    (cash, inventory, time, price) observation with
    [q_scaled, tau, r_scaled, delta_tau_scaled]. No belief, no filter, no
    regime. Used identically by both event-driven return_mlp_ppo
    (feed-forward) and return_lstm_ppo (recurrent) -- see module docstring
    for the full reset/step chronology.

    Parameters
    ----------
    base_env : EventDrivenRegimeSwitchingEnv, optional
        Pre-built environment to wrap. If None, one is built via
        EventDrivenRegimeSwitchingEnv(seed=seed).
    inventory_scale : float
        Denominator in tanh(q / inventory_scale). Shares
        DEFAULT_INVENTORY_SCALE with the event-driven Hamilton wrapper
        (imported, not redefined).
    return_scale : float
        Denominator in tanh(r / return_scale). See module-level constant.
    elapsed_time_scale : float
        Denominator in tanh(delta_tau / elapsed_time_scale). See
        ELAPSED_TIME_SCALE_REASONING above.
    seed : int, optional
        Forwarded to EventDrivenRegimeSwitchingEnv(seed=...) when base_env
        is None.
    """

    metadata = {"render_modes": []}
    ENVIRONMENT_TYPE = "event"

    def __init__(
        self,
        base_env=None,
        inventory_scale: float = DEFAULT_INVENTORY_SCALE,
        return_scale: float = DEFAULT_RETURN_SCALE,
        elapsed_time_scale: float = DEFAULT_ELAPSED_TIME_SCALE,
        seed: int = None,
    ):
        super().__init__()

        if inventory_scale <= 0:
            raise ValueError(f"inventory_scale must be positive, got {inventory_scale}")
        if return_scale <= 0:
            raise ValueError(f"return_scale must be positive, got {return_scale}")
        if elapsed_time_scale <= 0:
            raise ValueError(f"elapsed_time_scale must be positive, got {elapsed_time_scale}")

        self.base_env = base_env if base_env is not None else EventDrivenRegimeSwitchingEnv(seed=seed)
        self.inventory_scale = float(inventory_scale)
        self.return_scale = float(return_scale)
        self.elapsed_time_scale = float(elapsed_time_scale)
        self.terminal_time = self.base_env.terminal_time

        base_low = np.asarray(self.base_env.action_space.low, dtype=np.float32)
        base_high = np.asarray(self.base_env.action_space.high, dtype=np.float32)
        self.action_space = spaces.Box(low=base_low, high=base_high, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._prev_price = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _scaled_inventory(self, raw_inventory: float) -> float:
        return float(np.tanh(raw_inventory / self.inventory_scale))

    def _scaled_return(self, raw_return: float) -> float:
        return float(np.tanh(raw_return / self.return_scale))

    def _scaled_elapsed_time(self, delta_tau: float) -> float:
        return float(np.tanh(delta_tau / self.elapsed_time_scale))

    def _build_obs(self, raw_inventory: float, current_time: float,
                    scaled_return: float, scaled_elapsed_time: float) -> np.ndarray:
        q_scaled = self._scaled_inventory(raw_inventory)
        tau = 1.0 - (float(current_time) / float(self.terminal_time))
        return np.array([q_scaled, tau, scaled_return, scaled_elapsed_time], dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------
    def reset(self, *, seed: int = None, options: dict = None):
        raw_state = self.base_env.reset()
        self._prev_price = float(raw_state[ASSET_PRICE_INDEX])

        q0 = float(raw_state[INVENTORY_INDEX])
        obs = self._build_obs(q0, current_time=0.0, scaled_return=0.0, scaled_elapsed_time=0.0)
        return obs, {}

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        raw_state, reward, done, info = self.base_env.step(action_arr)

        price_next = float(info["price_after"])
        raw_return = (price_next - self._prev_price) / self._prev_price
        self._prev_price = price_next

        delta_tau = float(info["elapsed_time"])
        q_next = float(info["inventory_after"])
        current_time = float(info["raw_state"][TIME_INDEX])

        next_obs = self._build_obs(
            q_next, current_time,
            self._scaled_return(raw_return), self._scaled_elapsed_time(delta_tau),
        )
        terminated = bool(done)
        truncated = False

        return next_obs, float(reward), terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        pass
