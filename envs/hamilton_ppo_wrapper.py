"""
hamilton_ppo_wrapper.py
------------------------
Gymnasium-compatible PPO observation wrapper around the existing
RegimeSwitchingEnv, exposing the Hamilton-filter belief b_t in place of
the hidden true regime.

Why this module exists (not a change to regime_env.py or make_envs.py):
RegimeSwitchingEnv and every mbt_gym class it wraps subclass the LEGACY
`gym.Env` API (gym==0.26.2 here): reset() takes no arguments and returns
just `obs`; step() returns a 4-tuple `(obs, reward, done, info)`, not
Gymnasium's 5-tuple `(obs, reward, terminated, truncated, info)`.
stable_baselines3>=2.0 (2.8.0 installed here) requires the Gymnasium API
and gymnasium.spaces, not gym.spaces. Rather than adding the `shimmy`
bridge package as a new dependency, this module absorbs that legacy/
Gymnasium boundary translation itself -- it is the one place in the
project whose entire purpose is to sit at that boundary.

Observation (shape (3,), float32):
    x_t = [q_t_scaled, tau_t, b_t]
        q_t_scaled = tanh(q_t / inventory_scale)     -- q_t is RAW inventory
                                                          (shares), read via
                                                          env.raw_inventory /
                                                          info['raw_state'],
                                                          never reconstructed
                                                          from a normalised obs.
        tau_t      = 1 - current_step / n_steps       -- fraction of episode
                                                          remaining.
        b_t        = P(Z_t=1 | r_1:t)                 -- HamiltonFilter belief,
                                                          reusing the existing
                                                          filter class unchanged.

The true regime (info['true_regime']) and realised volatility are never
read when constructing the observation -- see
tests/test_hamilton_ppo_wrapper.py for explicit leakage tests.

Chronology (no look-ahead):
    reset():
        1. base_env.reset()
        2. filt.reset()                    -- prior belief = stationary dist.
        3. filt.update(S_0)                -- caches S_0 only; does NOT touch
                                               the posterior, since
                                               HamiltonFilter.update()'s first
                                               call after reset() short-
                                               circuits on `_prev_midprice is
                                               None` (see beliefs/
                                               hamilton_filter.py) -- belief
                                               returned is the unchanged prior.
        4. return [q_0_scaled, 1.0, b_0]

    step(a_t):
        1. obs, reward, done, info = base_env.step(a_t)   -- a_t was chosen by
           the caller from the PREVIOUS observation (reset's or the prior
           step's), which already encoded belief b_t; nothing here uses
           r_{t+1} before this call.
        2. S_{t+1} = info['raw_midprice']
        3. b_{t+1} = filt.update(S_{t+1})  -- the filter internally computes
           r_{t+1} = (S_{t+1} - S_t) / S_t from its own cached S_t and
           updates its posterior exactly once. The wrapper additionally
           records this same return itself (self.last_return) purely for
           test/diagnostic introspection -- this is the same one-line
           arithmetic formula given in the task spec, not a second
           implementation of the filter's likelihood/recursion.
        4. q_{t+1} = info['raw_state'][INVENTORY_INDEX]      (raw inventory)
        5. tau_{t+1} = 1 - current_step / n_steps
        6. return [q_{t+1}_scaled, tau_{t+1}, b_{t+1}], reward, terminated, truncated, info

Reward is passed through unchanged (RunningInventoryPenalty's cumulative
per-step reward, summed over the single trajectory axis, is not itself
altered -- see the module docstring's chronology above).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.make_envs import (
    make_regime_envs,
    N_STEPS,
    TRANSITION_MATRIX,
    R0_SIGMA_STEP,
    R1_SIGMA_STEP,
    STEP_SIZE,
    LAMBDA,
    EPSILON_PCT,
)
from mbt_gym.gym.index_names import INVENTORY_INDEX
from beliefs.hamilton_filter import HamiltonFilter

# Empirically selected: diagnose_hamilton_ppo_wrapper.py, run over 20
# random-policy episodes (80,000 steps), shows raw |inventory| quantiles
# q95=13, q99=16 shares. At inventory_scale=10, 0.00% of steps saturate
# the tanh transform (|q_scaled|>0.95), vs. 3.32% at scale=5; scale=10
# also gives visibly better resolution (mean|q_scaled|=0.357) than the
# more conservative scale=15 (0.276) or scale=20 (0.218), which compress
# most observations toward zero. See the diagnostic's saturation-rate
# output for the full comparison.
DEFAULT_INVENTORY_SCALE = 10.0


def make_filter() -> HamiltonFilter:
    """
    Build the same Hamilton filter configuration used throughout the
    project (simulate_belief_weighted.py's make_filter(), reproduced here
    directly against envs.make_envs constants rather than importing that
    script as a dependency of core wrapper infrastructure). r=0: verified
    elsewhere in this project to give an identical marginal belief to any
    r>0 for this model, at lower cost. Return-only emission (RV
    augmentation stays disabled, matching the project default).
    """
    return HamiltonFilter(
        transition_matrix=TRANSITION_MATRIX,
        regime_volatilities=[R0_SIGMA_STEP, R1_SIGMA_STEP],
        r=0,
        jump_size=EPSILON_PCT,
        jump_intensity=LAMBDA,
        step_size=STEP_SIZE,
    )


class HamiltonPPOWrapper(gym.Env):
    """
    Gymnasium Env wrapping RegimeSwitchingEnv, replacing the raw
    (cash, inventory, time, midprice, ...) observation with
    [q_scaled, tau, belief] and driving a HamiltonFilter from raw midprice
    returns. See module docstring for the full reset/step chronology.

    Parameters
    ----------
    base_env : RegimeSwitchingEnv, optional
        Pre-built environment to wrap. If None, one is built via
        make_regime_envs(switch_within_episode=True, seed=seed) -- episode-
        level-only switching (switch_within_episode=False) is not the
        intended regime-detection setting for this wrapper.
    inventory_scale : float
        Denominator in tanh(q / inventory_scale). Must be set from the
        empirical inventory diagnostics (diagnose_hamilton_ppo_wrapper.py),
        not assumed.
    seed : int, optional
        Forwarded to make_regime_envs() when base_env is None -- seeds
        every underlying stochastic process (midprice/arrival/fill AND
        regime-transition draws, all isolated local np.random.Generator
        instances as of the RNG-isolation fix; see make_regime_envs's and
        RegimeSwitchingEnv's docstrings). reset(seed=...) on an
        ALREADY-CONSTRUCTED wrapper does NOT reseed anything (seed is
        accepted for Gymnasium API compatibility but is otherwise inert
        here) -- reproducibility always requires constructing a fresh
        HamiltonPPOWrapper(seed=s), never reset()-ing an existing one (see
        tests/test_hamilton_ppo_wrapper.py's TestSeedReset).
    """

    metadata = {"render_modes": []}

    def __init__(self, base_env=None, inventory_scale: float = DEFAULT_INVENTORY_SCALE,
                 seed: int = None):
        super().__init__()

        if inventory_scale <= 0:
            raise ValueError(f"inventory_scale must be positive, got {inventory_scale}")

        self.base_env = base_env if base_env is not None else make_regime_envs(
            switch_within_episode=True, seed=seed,
        )
        self.inventory_scale = float(inventory_scale)
        self.n_steps = N_STEPS

        self.filt = make_filter()

        # base_env.action_space is a legacy gym.spaces.Box, already
        # normalised to [-1,1] by TradingEnvironment (normalise_action_space
        # =True is the default) -- re-expressed as a gymnasium.spaces.Box
        # with IDENTICAL bounds/dtype/ordering ([bid_depth, ask_depth]).
        # Actions are passed straight through to base_env.step() with no
        # extra denormalisation -- the base env already does that internally
        # (TradingEnvironment.step(): action = self.normalise_action(action,
        # inverse=True)).
        base_low = np.asarray(self.base_env.action_space.low, dtype=np.float32)
        base_high = np.asarray(self.base_env.action_space.high, dtype=np.float32)
        self.action_space = spaces.Box(low=base_low, high=base_high, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
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

    def _build_obs(self, raw_inventory: float, current_step: int, belief: float) -> np.ndarray:
        q_scaled = self._scaled_inventory(raw_inventory)
        tau = 1.0 - (float(current_step) / float(self.n_steps))
        return np.array([q_scaled, tau, belief], dtype=np.float32)

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
        # anyway (see class docstring / TestSeedReset).

        self.base_env.reset()
        self.filt.reset()

        s0 = self.base_env.raw_midprice
        belief0 = self.filt.update(s0)  # caches S_0 only -- does not update the posterior (see module docstring)
        self._prev_midprice = s0
        self._last_return = None

        q0 = self.base_env.raw_inventory
        obs = self._build_obs(q0, self.base_env.current_step, belief0)
        return obs, {}

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32).reshape(1, -1)
        obs, reward, done, info = self.base_env.step(action_arr)

        s_next = info["raw_midprice"]
        self._last_return = (s_next - self._prev_midprice) / self._prev_midprice
        self._prev_midprice = s_next

        belief = self.filt.update(s_next)  # exactly one filter update per env step
        q_next = float(info["raw_state"][INVENTORY_INDEX])
        current_step = info["current_step"]

        next_obs = self._build_obs(q_next, current_step, belief)
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
    def belief(self) -> float:
        return self.filt.belief

    @property
    def last_return(self):
        """Percentage return computed at the most recent step(), or None
        (at reset, before any return has been observed). Test/diagnostic use
        only -- never fed into the PPO observation directly (b_t already
        summarises it via the filter)."""
        return self._last_return
