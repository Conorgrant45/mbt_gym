"""
return_ppo_wrapper.py
------------------------
Gymnasium-compatible PPO observation wrapper around the existing
RegimeSwitchingEnv, exposing the raw one-step midprice return in place of
the Hamilton-filter belief. Used by BOTH model-free agents (feed-forward
return_mlp_ppo and recurrent return_lstm_ppo) so that recurrence is the
only architectural difference between them -- see module docstring of
envs/hamilton_ppo_wrapper.py for the sibling belief-state wrapper this one
is structurally modelled on (legacy gym.Env -> Gymnasium translation,
reset/step handling, action-space conversion, seed handling, raw-state
access all follow the same pattern).

Observation (shape (3,), float32):
    x_t = [q_t_scaled, tau_t, r_t_scaled]
        q_t_scaled = tanh(q_t / inventory_scale)   -- identical definition
                                                        and default scale
                                                        (DEFAULT_INVENTORY_
                                                        SCALE, imported from
                                                        hamilton_ppo_wrapper,
                                                        not re-derived) to
                                                        the Hamilton wrapper.
        tau_t      = 1 - current_step / n_steps     -- identical time
                                                        convention to the
                                                        Hamilton wrapper.
        r_t_scaled = tanh(r_t / return_scale)        -- r_t is the raw
                                                        percentage return
                                                        (S_t - S_{t-1})/S_{t-1}
                                                        observed at this
                                                        step; see
                                                        RETURN_SCALE_
                                                        REASONING below.

No HamiltonFilter object is instantiated anywhere in this module. The true
regime (info['true_regime']) is never read when constructing the
observation -- see tests/test_return_ppo_wrapper.py's leakage tests.

Chronology (no look-ahead), identical structure to HamiltonPPOWrapper:
    reset():
        1. base_env.reset()
        2. S_0 = base_env.raw_midprice; cache it, do NOT compute a return
        3. return [q_0_scaled, 1.0, 0.0]   -- r_0_scaled is exactly 0.0,
           by construction (no prior price exists yet to form a return
           from), not an approximation of some "true" reset return.

    step(a_t):
        1. obs, reward, done, info = base_env.step(a_t)  -- a_t was chosen
           from the PREVIOUS observation; nothing here uses S_{t+1} before
           this call.
        2. S_{t+1} = info['raw_midprice']
        3. r_{t+1} = (S_{t+1} - S_t) / S_t                -- computed from
           the wrapper's own cached S_t, the exact same one-line formula
           HamiltonFilter.update() uses internally (not reimplementing the
           filter -- this wrapper never constructs one).
        4. q_{t+1} = info['raw_state'][INVENTORY_INDEX]    (raw inventory)
        5. tau_{t+1} = 1 - current_step / n_steps
        6. return [q_{t+1}_scaled, tau_{t+1}, tanh(r_{t+1}/return_scale)],
           reward, terminated, truncated, info

Reward, termination and truncation are passed through unchanged -- exactly
as in HamiltonPPOWrapper.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.make_envs import make_regime_envs, N_STEPS, EPSILON_PCT
from envs.hamilton_ppo_wrapper import DEFAULT_INVENTORY_SCALE
from mbt_gym.gym.index_names import INVENTORY_INDEX

# ----------------------------------------------------------------------
# RETURN_SCALE_REASONING
#
# Chosen empirically via diagnose_hamilton_ppo_wrapper.py --episodes 20
# (80,000 random-policy steps, current calibration: R0_VOLATILITY=
# R1_VOLATILITY=0.01, EPSILON=0.5, LAMBDA=140, dt=0.00025 -- see
# envs/make_envs.py). That diagnostic already reports the raw one-step
# percentage return fed to the Hamilton filter (identical formula used
# here), since HamiltonPPOWrapper computes and exposes the same quantity
# via env.last_return for introspection:
#
#   mean  = -4.09e-06   std = 1.626e-03
#   q01   = -4.6e-03    q99 = +4.6e-03
#   q0.001 = -1.74e-02  q0.999 = +1.65e-02
#   max |r| observed (80,000 steps) = 4.36e-02
#
# The return distribution is dominated by the arrival-triggered jump
# process, not the diffusion term: EPSILON_PCT = EPSILON / INITIAL_PRICE =
# 0.5/100 = 0.005 (the mean impact of a SINGLE market-order arrival) is
# already ~3x the empirical std and lands close to the q99 quantile,
# whereas the pure-diffusion contribution alone (R0_SIGMA_STEP ~ 1.6e-6)
# is negligible by comparison -- almost all of the return signal's
# variation comes from whether a jump occurred, not from Brownian noise.
#
# tanh(r / EPSILON_PCT) is chosen (not a raw linear scaling) for two
# reasons, both mirroring the inventory-scaling design already used
# in hamilton_ppo_wrapper.py:
#   1. It guarantees a hard-bounded [-1, 1] observation component no
#      matter how large a realised return is (the jump magnitude is
#      Exponential-distributed, hence formally unbounded above -- a
#      linear scale could not offer a finite Box bound that is never
#      violated).
#   2. At scale=EPSILON_PCT, only 0.81% of the 80,000 diagnostic steps
#      saturate (|tanh(r/scale)| > 0.95) -- these are the genuinely rare
#      events where 2+ jumps land in the same 0.00025 dt step. Smaller
#      candidate scales (0.001, 0.0016 ~ 1 std) saturate 2.4-3.4% of
#      steps; larger scales (0.01, 0.02) saturate less but compress the
#      single-jump signal (the dominant, economically meaningful event)
#      much closer to 0, losing resolution on exactly the events this
#      observation exists to surface.
#
# EPSILON_PCT is imported directly from envs.make_envs (not re-derived
# or hard-coded), so this stays in sync with the canonical calibration
# automatically if EPSILON or INITIAL_PRICE are ever revisited.
# ----------------------------------------------------------------------
DEFAULT_RETURN_SCALE = EPSILON_PCT


class ReturnPPOWrapper(gym.Env):
    """
    Gymnasium Env wrapping RegimeSwitchingEnv, replacing the raw
    (cash, inventory, time, midprice, ...) observation with
    [q_scaled, tau, scaled_return]. No belief, no filter, no regime.
    Used identically by both return_mlp_ppo (feed-forward) and
    return_lstm_ppo (recurrent) -- see module docstring for the full
    reset/step chronology.

    Parameters
    ----------
    base_env : RegimeSwitchingEnv, optional
        Pre-built environment to wrap. If None, one is built via
        make_regime_envs(switch_within_episode=True, seed=seed) --
        identical construction convention to HamiltonPPOWrapper.
    inventory_scale : float
        Denominator in tanh(q / inventory_scale). Shares
        DEFAULT_INVENTORY_SCALE with HamiltonPPOWrapper (imported, not
        redefined) so both wrappers use the same empirically-justified
        value unless explicitly overridden.
    return_scale : float
        Denominator in tanh(r / return_scale). See RETURN_SCALE_REASONING
        above for how DEFAULT_RETURN_SCALE was chosen.
    seed : int, optional
        Forwarded to make_regime_envs() when base_env is None -- seeds
        every underlying stochastic process (midprice/arrival/fill AND
        regime-transition draws, all isolated local np.random.Generator
        instances as of the RNG-isolation fix; see make_regime_envs's and
        RegimeSwitchingEnv's docstrings). reset(seed=...) on an
        ALREADY-CONSTRUCTED wrapper does NOT reseed anything (seed is
        accepted for Gymnasium API compatibility but is otherwise inert
        here) -- reproducibility always requires constructing a fresh
        ReturnPPOWrapper(seed=s), never reset()-ing an existing one.
    """

    metadata = {"render_modes": []}

    def __init__(self, base_env=None, inventory_scale: float = DEFAULT_INVENTORY_SCALE,
                 return_scale: float = DEFAULT_RETURN_SCALE, seed: int = None):
        super().__init__()

        if inventory_scale <= 0:
            raise ValueError(f"inventory_scale must be positive, got {inventory_scale}")
        if return_scale <= 0:
            raise ValueError(f"return_scale must be positive, got {return_scale}")

        self.base_env = base_env if base_env is not None else make_regime_envs(
            switch_within_episode=True, seed=seed,
        )
        self.inventory_scale = float(inventory_scale)
        self.return_scale = float(return_scale)
        self.n_steps = N_STEPS

        # base_env.action_space is a legacy gym.spaces.Box, already
        # normalised to [-1,1] by TradingEnvironment (normalise_action_space
        # =True is the default) -- re-expressed as a gymnasium.spaces.Box
        # with IDENTICAL bounds/dtype/ordering ([bid_depth, ask_depth]).
        # Actions are passed straight through to base_env.step() with no
        # extra denormalisation, exactly as in HamiltonPPOWrapper -- the
        # base env already does that internally.
        base_low = np.asarray(self.base_env.action_space.low, dtype=np.float32)
        base_high = np.asarray(self.base_env.action_space.high, dtype=np.float32)
        self.action_space = spaces.Box(low=base_low, high=base_high, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._prev_midprice = None
        self._last_return = None  # test/diagnostic introspection only

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _scaled_inventory(self, raw_inventory: float) -> float:
        return float(np.tanh(raw_inventory / self.inventory_scale))

    def _scaled_return(self, raw_return: float) -> float:
        return float(np.tanh(raw_return / self.return_scale))

    def _build_obs(self, raw_inventory: float, current_step: int, scaled_return: float) -> np.ndarray:
        q_scaled = self._scaled_inventory(raw_inventory)
        tau = 1.0 - (float(current_step) / float(self.n_steps))
        return np.array([q_scaled, tau, scaled_return], dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------
    def reset(self, *, seed: int = None, options: dict = None):
        # seed is accepted for Gymnasium API compatibility but is otherwise
        # inert: as of the RNG-isolation fix, every stochastic component of
        # base_env (midprice/arrival/fill/regime) is seeded ONCE, at
        # construction, via an isolated local Generator -- there is no
        # process-global state left to reseed here, and reseeding one on an
        # already-stepped instance would not be genuine reproducibility
        # anyway (see class docstring).
        self.base_env.reset()

        s0 = self.base_env.raw_midprice
        self._prev_midprice = s0
        self._last_return = None  # no prior price exists yet -- not a return of 0 from a computation

        q0 = self.base_env.raw_inventory
        obs = self._build_obs(q0, self.base_env.current_step, 0.0)  # r_0_scaled is exactly 0.0 by construction
        return obs, {}

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32).reshape(1, -1)
        obs, reward, done, info = self.base_env.step(action_arr)

        s_next = info["raw_midprice"]
        raw_return = (s_next - self._prev_midprice) / self._prev_midprice
        self._last_return = raw_return
        self._prev_midprice = s_next

        q_next = float(info["raw_state"][INVENTORY_INDEX])
        current_step = info["current_step"]

        next_obs = self._build_obs(q_next, current_step, self._scaled_return(raw_return))
        reward_scalar = float(np.sum(reward))
        terminated = bool(np.all(done))
        truncated = False

        return next_obs, reward_scalar, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Introspection (not part of the PPO observation)
    # ------------------------------------------------------------------
    @property
    def last_return(self):
        """Raw (unscaled) percentage return computed at the most recent
        step(), or None (at reset, before any return has been observed).
        Test/diagnostic use only -- never fed into the PPO observation
        directly (the observation carries tanh(last_return/return_scale))."""
        return self._last_return
