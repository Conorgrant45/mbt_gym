"""
arrival_jump_midprice.py
------------------------
A custom midprice model that jumps on every market order arrival,
not only on the agent's limit order fills.

This implements the model from the mbt_gym paper (Jerome et al. 2023)
equation (4):

    S_t = S_0 + sigma * W_t + xi+ * M+_t - xi- * M-_t

where M+_t and M-_t are the buy/sell market order arrival processes,
and xi+/xi- are permanent price impact parameters.

The mbt_gym built-in BrownianMotionJumpMidpriceModel erroneously
conditions the jump on fills (fills * arrivals) rather than raw
arrivals, contrary to its own paper specification. This class
corrects that by using arrivals alone as the jump trigger.

This is consistent with the dissertation's HJB derivation where:

    dS_t = sigma dW_t + epsilon+ dM+_t - epsilon- dM-_t

and M+_t, M-_t are raw market order arrival processes with
intensities lambda+, lambda-. The adverse selection interpretation
is that informed market order flow permanently impacts the midprice
on every arrival, regardless of whether the market maker is filled.

The per-arrival impact xi+/xi- is itself random: each arrival draws an
independent jump magnitude from Exponential(mean=jump_size), rather
than applying jump_size as a fixed, deterministic move. The HJB
derivation's r_pm = lambda * exp(-1 - kappa*epsilon) only requires
epsilon = E[jump magnitude], so the optimal control is unaffected by
this randomisation -- only the Hamilton filter's emission model (which
needs the full jump-size distribution, not just its mean) must be
updated to match.

Usage
-----
    from envs.arrival_jump_midprice import ArrivalJumpMidpriceModel
    model = ArrivalJumpMidpriceModel(
        volatility=0.005,
        jump_size=0.3,
        step_size=0.005,
        terminal_time=1.0,
    )
"""

import numpy as np
from math import sqrt

from mbt_gym.stochastic_processes.midprice_models import MidpriceModel
from mbt_gym.gym.index_names import BID_INDEX, ASK_INDEX


class ArrivalJumpMidpriceModel(MidpriceModel):
    """
    Brownian motion midprice with permanent price impact on every
    market order arrival.

    Update rule:
        S_{t+dt} = S_t
                 + sigma * sqrt(dt) * Z                [Brownian diffusion]
                 + Exponential(jump_size) * arrivals_ask  [buy MO impact: price up]
                 - Exponential(jump_size) * arrivals_bid  [sell MO impact: price down]

    where Z ~ N(0,1), arrivals_{ask/bid} are binary indicators of whether
    a buy/sell market order arrived in this time step, and each arrival
    draws an independent Exponential random variable with mean jump_size
    (i.e. jump_size is the mean impact per arrival, not a fixed constant).

    Parameters
    ----------
    volatility : float
        Diffusion volatility sigma.
    jump_size : float
        Mean permanent price impact per market order arrival (xi+ = xi- =
        jump_size in expectation, for symmetric impact). The realised
        impact on each arrival is drawn from Exponential(mean=jump_size).
    jump_size_ask : float, optional
        Asymmetric ask-side impact. Overrides jump_size if provided.
    jump_size_bid : float, optional
        Asymmetric bid-side impact. Overrides jump_size if provided.
    initial_price : float
        Initial midprice S_0.
    terminal_time : float
        Episode length T.
    step_size : float
        Time step dt = T / n_steps.
    num_trajectories : int
        Number of parallel trajectories for vectorised training.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        volatility: float = 0.005,
        jump_size: float = 0.3,
        jump_size_ask: float = None,
        jump_size_bid: float = None,
        initial_price: float = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.005,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        self.volatility     = volatility
        self.jump_size_ask  = jump_size_ask if jump_size_ask is not None else jump_size
        self.jump_size_bid  = jump_size_bid if jump_size_bid is not None else jump_size
        self.terminal_time  = terminal_time

        # Compute price range for observation space bounds
        max_diffusion = 4.0 * volatility * terminal_time
        max_jump      = (self.jump_size_ask + self.jump_size_bid) * terminal_time * 200
        max_move      = max_diffusion + max_jump

        super().__init__(
            min_value=np.array([[initial_price - max_move]]),
            max_value=np.array([[initial_price + max_move]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(
        self,
        arrivals: np.ndarray,
        fills: np.ndarray,
        actions: np.ndarray,
        state: np.ndarray = None,
    ) -> np.ndarray:
        """
        Update the midprice.

        Parameters
        ----------
        arrivals : np.ndarray, shape (num_trajectories, 2)
            Binary indicator of market order arrivals: [bid_arrival, ask_arrival].
            arrivals[:, BID_INDEX] = 1 if a sell MO arrived (price moves down).
            arrivals[:, ASK_INDEX] = 1 if a buy MO arrived (price moves up).
        fills : np.ndarray
            Not used -- jumps occur on arrivals, not fills. Kept for API
            compatibility with mbt_gym MidpriceModel interface.
        actions : np.ndarray
            Not used for midprice update.
        state : np.ndarray, optional
            Unused.
        """
        # Brownian diffusion component
        diffusion = (
            self.volatility
            * sqrt(self.step_size)
            * self.rng.normal(size=(self.num_trajectories, 1))
        )

        # Jump component: fires on every market order arrival, not just fills.
        # Each arrival's impact magnitude is itself random -- drawn fresh every
        # step regardless of whether that side actually arrived (mirroring how
        # the diffusion term above is drawn unconditionally), then masked by
        # the arrival indicator.
        # Buy MO arrival (ask_arrival): price jumps up by Exponential(jump_size_ask)
        # Sell MO arrival (bid_arrival): price jumps down by Exponential(jump_size_bid)
        jump_ask_magnitude = self.rng.exponential(
            scale=self.jump_size_ask, size=(self.num_trajectories, 1)
        )
        jump_bid_magnitude = self.rng.exponential(
            scale=self.jump_size_bid, size=(self.num_trajectories, 1)
        )
        jump = (
            jump_ask_magnitude * arrivals[:, ASK_INDEX].reshape(-1, 1)
            - jump_bid_magnitude * arrivals[:, BID_INDEX].reshape(-1, 1)
        )

        self.current_state = self.current_state + diffusion + jump
        return self.current_state
