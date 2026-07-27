"""
regime_env.py
-------------
A thin wrapper around two mbt_gym TradingEnvironment instances that
switches between them according to a hidden two-state Markov chain.

The true regime is exposed only via info['true_regime'], which is passed
to the critic during training and withheld from the actor. Neither
underlying mbt_gym environment is modified.

Usage
-----
    from mbt_gym.gym.TradingEnvironment import TradingEnvironment
    from mbt_gym.gym.ModelDynamics import LimitOrderModelDynamics
    from mbt_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
    from mbt_gym.stochastic_processes.arrival_models import (
        PoissonArrivalModel, HawkesArrivalModel
    )
    from mbt_gym.stochastic_processes.fill_probability_models import ExponentialFillFunction
    from mbt_gym.rewards.RewardFunctions import RunningInventoryPenalty
    from envs.regime_env import RegimeSwitchingEnv, make_regime_envs
    import numpy as np

    env_r0, env_r1 = make_regime_envs()
    P = [[0.95, 0.05],
         [0.10, 0.90]]
    env = RegimeSwitchingEnv(env_r0, env_r1, P)

    obs = env.reset()
    obs, reward, done, info = env.step(env.action_space.sample())
    print(info['true_regime'])  # 0 or 1 -- available to critic, not actor
"""

import numpy as np
import gym

from mbt_gym.gym.index_names import ASSET_PRICE_INDEX, CASH_INDEX, INVENTORY_INDEX, TIME_INDEX


class RegimeSwitchingEnv(gym.Env):
    """
    Wraps two mbt_gym TradingEnvironment instances and switches between them
    according to a hidden two-state Markov chain.

    Parameters
    ----------
    env_r0 : TradingEnvironment
        Environment for regime 0 (e.g. calm, liquid).
    env_r1 : TradingEnvironment
        Environment for regime 1 (e.g. stressed, illiquid).
    transition_matrix : array-like, shape (2, 2)
        Row-stochastic Markov transition matrix. P[i, j] is the probability
        of transitioning from regime i to regime j at the end of each episode.
        For within-episode switching set switch_within_episode=True.
    switch_within_episode : bool
        If True, the regime can switch at every step according to P.
        If False (default), regime is fixed for the duration of each episode
        and resampled at reset(). Episode-level switching is simpler to train
        on and is the recommended starting point.
    initial_regime : int or None
        If set, always start in this regime. If None (default), the initial
        regime is sampled from the stationary distribution of P.
    regime_rng : np.random.Generator, optional
        Independent local RNG for regime-transition draws (both the
        initial-regime sample in reset() and the within-episode transition
        in step()). If None, a fresh OS-entropy-seeded Generator is created.

        RNG-ISOLATION FIX (audited and fixed after a reproducibility
        failure): this class previously drew regime transitions from the
        LEGACY GLOBAL np.random API (np.random.choice(...)), the same
        process-wide singleton that HamiltonPPOWrapper/ReturnPPOWrapper's
        reset(seed=...) used to reseed via np.random.seed(seed) (see their
        module docstrings, now updated). Since PeriodicEvalCallback runs
        full evaluation episodes on a SEPARATE wrapper instance interleaved
        with ongoing training, every evaluation episode's reset(seed=...)
        call reseeded and then consumed draws from that SAME global stream
        -- silently corrupting the TRAINING environment's own subsequent
        regime-transition path from that point on. Two training runs with
        different eval_freq values (hence different numbers/timings of
        evaluation calls within the same total_timesteps) would therefore
        diverge in their regime path, rewards, gradients, and final trained
        weights, despite using identical seeds and hyperparameters --
        confirmed empirically (see tests/test_rng_isolation.py) before this
        fix, and confirmed absent after it. Passing an explicit,
        already-constructed Generator here (see make_regime_envs, which
        derives it from the same top-level seed via SeedSequence.spawn(),
        exactly like the midprice/arrival/fill model RNGs already did)
        means regime draws are now fully isolated: nothing outside this
        instance's own step()/reset() calls can ever advance or reseed it.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        env_r0,
        env_r1,
        transition_matrix,
        switch_within_episode: bool = False,
        initial_regime: int = None,
        regime_rng: np.random.Generator = None,
    ):
        super().__init__()
        self._regime_rng = regime_rng if regime_rng is not None else np.random.default_rng()

        self.envs = [env_r0, env_r1]
        self.P = np.array(transition_matrix, dtype=np.float64)
        self.switch_within_episode = switch_within_episode
        self.initial_regime = initial_regime
        self.regime = 0

        self._validate()

        # Single global episode clock, replacing the two sub-environments'
        # own independent internal clocks (each of which previously only
        # advanced while that sub-environment happened to be active -- see
        # step()/_sync_all_state() below). Both sub-environments must share
        # the same step size and episode length for a shared clock to be
        # meaningful; this is asserted in _validate().
        self.dt = env_r0.step_size
        self.n_steps = env_r0.n_steps
        self.current_step = 0

        self.action_space = env_r0.action_space

        # Use the larger observation space so both regimes can be padded to match.
        obs_size = max(
            env_r0.observation_space.shape[0],
            env_r1.observation_space.shape[0],
        )
        import gym.spaces as spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self._obs_sizes = [
            env_r0.observation_space.shape[0],
            env_r1.observation_space.shape[0],
        ]

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self):
        if self.initial_regime is not None:
            self.regime = self.initial_regime
        else:
            self.regime = self._sample_initial_regime()

        self.current_step = 0

        obs = self.envs[self.regime].reset()
        # Reset the inactive sub-environment too (so its own internal
        # processes -- arrival/fill RNG state, midprice model buffers -- are
        # freshly initialised), then force both sub-environments' cash,
        # inventory, midprice and time to agree with the active one's,
        # so the episode starts from one consistent global state regardless
        # of which regime happens to be sampled first.
        self.envs[1 - self.regime].reset()
        self._sync_all_state()

        return self._pad_obs(obs)

    def step(self, action):
        # Execute the active regime's dynamics exactly as before (arrivals,
        # fills, price, cash, inventory, reward are all unchanged).
        obs, reward, done_subenv, info = self.envs[self.regime].step(action)
        obs = self._pad_obs(obs)

        # Advance the single global clock exactly once per wrapper step,
        # regardless of which sub-environment was active. Previously each
        # sub-environment's own internal clock only advanced while it
        # happened to be active, so the episode's true length in
        # wrapper-level steps depended on how time happened to split
        # between the two regimes -- it could run for far more or fewer
        # than n_steps calls to this method.
        self.current_step += 1

        # Synchronise cash, inventory, midprice and time across BOTH
        # sub-environments after every step (not only when the regime
        # actually switches) -- see _sync_all_state's docstring. This also
        # pins the just-stepped active sub-environment's own time state to
        # the global value, guarding against float drift between its
        # internally-accumulated clock and current_step * dt.
        self._sync_all_state()

        # TradingEnvironment normalises observations to roughly [-1, 1] by
        # default -- and its normalisation bounds (e.g. max_cash) differ
        # between the regime-0 and regime-1 sub-envs, so cash/inventory/price
        # from `obs` are neither real units nor comparable across a regime
        # switch. Capture the real state here, before the regime transitions
        # below, so consumers never have to guess which sub-env was active
        # or reconstruct raw values from a normalised observation.
        raw_state = self.envs[self.regime].model_dynamics.state[0].copy()

        # Episode terminates only when the wrapper's global clock reaches
        # n_steps -- not when either individual sub-environment's own clock
        # (previously unsynchronised, now kept in lockstep by
        # _sync_all_state every step) reaches its own terminal condition.
        # Since both sub-environments share terminal_time/n_steps/step_size
        # (asserted in _validate()) and are kept clock-synced above, this
        # is exactly equivalent to -- and replaces reliance on -- each
        # sub-environment's own internal _get_dones() check; computing it
        # here from current_step makes the termination condition explicit
        # and independently auditable.
        done = self.current_step >= self.n_steps
        dones = np.full_like(done_subenv, done)

        # Attach ground-truth regime label, raw state, and the wrapper's
        # global step counter. Pass info['true_regime'] to the critic; do
        # NOT pass it to the actor. Use info['current_step'] (or the
        # current_step property) -- not a reconstruction from the
        # normalised time observation -- whenever indexing a time-dependent
        # control table, e.g. the oracle's delta_ask/delta_bid[t_idx, ...].
        if isinstance(info, list):
            # mbt_gym returns a list of dicts when num_trajectories > 1
            for d in info:
                d['true_regime']   = self.regime
                d['raw_state']     = raw_state
                d['raw_midprice']  = float(raw_state[ASSET_PRICE_INDEX])
                d['current_step']  = self.current_step
        else:
            info['true_regime']  = self.regime
            info['raw_state']    = raw_state
            info['raw_midprice'] = float(raw_state[ASSET_PRICE_INDEX])
            info['current_step'] = self.current_step

        # Transition the hidden regime. Both sub-environments are already
        # fully state-synced by _sync_all_state above (every step, not only
        # here), so whichever regime becomes active next continues from the
        # current global state and global time -- never from a stale value
        # left over from whenever it was last active.
        if self.switch_within_episode or done:
            next_regime = self._regime_rng.choice(2, p=self.P[self.regime])
            self.regime = next_regime

        return obs, reward, dones, info

    def render(self, mode="human"):
        return self.envs[self.regime].render(mode=mode)

    def seed(self, seed=None):
        for env in self.envs:
            env.seed(seed)
        self._regime_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pad_obs(self, obs: np.ndarray) -> np.ndarray:
        """Pad observation with zeros to match the unified observation space size."""
        target = self.observation_space.shape[0]
        current = obs.shape[-1]
        if current == target:
            return obs
        pad_width = target - current
        if obs.ndim == 1:
            return np.concatenate([obs, np.zeros(pad_width, dtype=obs.dtype)])
        else:
            # vectorised: shape (n_trajectories, obs_dim)
            pad = np.zeros((obs.shape[0], pad_width), dtype=obs.dtype)
            return np.concatenate([obs, pad], axis=-1)

    def _sync_all_state(self):
        """
        Copy cash, inventory, midprice AND the global step/time from the
        currently-active sub-environment into the currently-inactive one,
        and pin the active sub-environment's own time state to the exact
        global value too.

        Called after every step() (and once at the end of reset()) -- not
        only when the regime actually switches. Each sub-environment is an
        independent TradingEnvironment with its own model_dynamics.state
        and its own midprice_model.current_state, which would otherwise
        only ever update while that sub-environment is the active one.
        Syncing unconditionally, every step, means whichever sub-environment
        becomes active next -- on this step or many steps from now -- always
        continues from the current global state and global time, never from
        a stale value left over from whenever it was last active. It also
        means both sub-environments' own internal terminal-time checks
        (_get_dones(), which each independently compares its own
        state[:,TIME_INDEX] against its own terminal_time) stay exactly
        aligned with the wrapper's global current_step, so the terminal
        inventory-aversion penalty (applied inside whichever sub-environment
        is actually stepped, via its reward function's is_terminal_step
        flag) fires at the correct global terminal step and only once --
        the inactive sub-environment's reward function is never invoked
        while it is inactive, so there is no risk of double-counting.
        """
        source_regime = self.regime
        target_regime = 1 - self.regime
        source_state = self.envs[source_regime].model_dynamics.state
        target_dynamics = self.envs[target_regime].model_dynamics

        t = self.current_step * self.dt

        target_dynamics.state[:, CASH_INDEX]        = source_state[:, CASH_INDEX]
        target_dynamics.state[:, INVENTORY_INDEX]   = source_state[:, INVENTORY_INDEX]
        target_dynamics.state[:, ASSET_PRICE_INDEX] = source_state[:, ASSET_PRICE_INDEX]
        target_dynamics.state[:, TIME_INDEX]        = t
        target_dynamics.midprice_model.current_state[:, 0] = source_state[:, ASSET_PRICE_INDEX]

        # Pin the active sub-environment's own time to the same global
        # value, guarding against float drift between its own
        # internally-accumulated clock (state[:,TIME_INDEX] += step_size
        # each call) and the wrapper's multiplicatively-computed global time.
        source_state[:, TIME_INDEX] = t

    def _sample_initial_regime(self) -> int:
        """Sample from the stationary distribution of P."""
        pi = self._stationary_distribution()
        return int(self._regime_rng.choice(2, p=pi))

    def _stationary_distribution(self) -> np.ndarray:
        """Compute the stationary distribution of the transition matrix."""
        # Solve pi @ (P - I) = 0 with sum(pi) = 1
        A = (self.P.T - np.eye(2))
        A[-1, :] = 1.0
        b = np.zeros(2)
        b[-1] = 1.0
        pi = np.linalg.solve(A, b)
        return pi

    def _validate(self):
        assert self.P.shape == (2, 2), "Transition matrix must be 2x2."
        assert np.allclose(self.P.sum(axis=1), 1.0), (
            "Each row of the transition matrix must sum to 1."
        )
        # Observation spaces may differ (e.g. Hawkes adds intensity dims).
        # This is handled by _pad_obs -- no assertion needed here.

        # A single global episode clock (self.dt, self.n_steps) is only
        # meaningful if both sub-environments agree on step size and
        # episode length -- otherwise "current_step * dt" would not
        # correspond to the same point in both sub-environments' own time.
        env_r0, env_r1 = self.envs
        assert env_r0.terminal_time == env_r1.terminal_time, (
            "Both regime sub-environments must share the same terminal_time."
        )
        assert env_r0.n_steps == env_r1.n_steps, (
            "Both regime sub-environments must share the same n_steps."
        )
        assert env_r0.step_size == env_r1.step_size, (
            "Both regime sub-environments must share the same step_size."
        )

    @property
    def current_regime(self) -> int:
        """The current hidden regime (0 or 1)."""
        return self.regime

    @property
    def current_step(self) -> int:
        """
        The wrapper's global step counter (0 at reset, incremented once per
        step() call, terminal at n_steps). This is the authoritative time
        index -- use this (or info['current_step']) rather than either
        sub-environment's own internal clock or a value reconstructed from
        the normalised time observation, e.g. for indexing a time-dependent
        control table such as the oracle's delta_ask/delta_bid[t_idx, ...].
        """
        return self._current_step

    @current_step.setter
    def current_step(self, value: int):
        self._current_step = value

    @property
    def raw_state(self) -> np.ndarray:
        """
        Un-normalised current state row of the active sub-environment
        (cash, inventory, time, midprice, ...).

        `obs` returned by reset()/step() is normalised to roughly [-1, 1]
        by the underlying TradingEnvironment, AND the normalisation bounds
        (e.g. max_cash, derived from each regime's own price range) differ
        between the two sub-envs -- so cash/inventory/price read off `obs`
        are neither real units nor safely comparable across a regime
        switch. Use this property (right after reset()) or
        info['raw_state'] (after step()) whenever you need actual cash,
        inventory or price values -- e.g. for PnL bookkeeping or feeding
        HamiltonFilter.
        """
        return self.envs[self.regime].model_dynamics.state[0].copy()

    @property
    def raw_midprice(self) -> float:
        """Un-normalised current midprice of the active sub-environment. See raw_state."""
        return float(self.raw_state[ASSET_PRICE_INDEX])

    @property
    def raw_cash(self) -> float:
        """Un-normalised current cash of the active sub-environment. See raw_state."""
        return float(self.raw_state[CASH_INDEX])

    @property
    def raw_inventory(self) -> float:
        """Un-normalised current inventory of the active sub-environment. See raw_state."""
        return float(self.raw_state[INVENTORY_INDEX])
