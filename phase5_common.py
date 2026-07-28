"""
phase5_common.py
--------------------
Shared utilities for Phase 5 (PPO optimisation diagnostic: why does PPO not
reach the policy the [64,64] Tanh Hamilton architecture can already
represent, per Phase 4's REPRESENTATION SUFFICIENT verdict).

Read-only reuse of phase4_common.py (grid/analytical-control machinery),
evaluate_agents_event_driven.py (event-time control lookup), and
train_agents.py's PPO hyperparameter DEFAULTS (reproduced verbatim here, not
re-derived, so this phase can build many short custom-instrumented runs
without going through train_agents.py's CLI/checkpoint-manifest machinery,
while remaining numerically IDENTICAL to a real training run's
configuration). No economic model, Hamilton filter, observation, action
space, or network architecture is changed anywhere in this module.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import phase4_common as P4
import simulate_belief_weighted as SBW
from envs.event_driven_hamilton_ppo_wrapper import EventDrivenHamiltonPPOWrapper, DEFAULT_INVENTORY_SCALE
from envs.event_driven_regime_env import EventDrivenRegimeSwitchingEnv

REPO_ROOT = Path(__file__).resolve().parent
PHASE5_RESULTS_DIR = REPO_ROOT / "results" / "phase5_ppo_optimisation"
PHASE5_LOGS_DIR = REPO_ROOT / "logs" / "phase5_ppo_optimisation"
PLOTS_DIR = PHASE5_RESULTS_DIR / "plots"

SUPERVISED_CLONE_PATH = REPO_ROOT / "logs" / "phase4_policy_diagnostic" / "supervised_clone_best.pt"

# --- Experimental design (task brief Section 4) ---
TRAIN_ENV_SEED = 70_000
LEARNER_SEEDS = (0, 1, 2)
TOTAL_TRANSITIONS = 64_000
CHECKPOINT_TIMESTEPS = [0, 4_000, 8_000, 16_000, 24_000, 32_000, 48_000, 64_000]
GROUPS = ("A_random_init", "B_clone_init", "C_frozen_clone")

# --- train_agents.py's PPO defaults, reproduced VERBATIM (not re-derived) --
# so this phase's runs use an identical configuration to the real Hamilton
# PPO training runs audited in Phases 3-4. Nothing here is tuned.
PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=4_000,
    batch_size=400,
    n_epochs=10,
    gamma=1.0,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    device="cpu",
)
NET_ARCH = [64, 64]

# Fresh diagnostic seeds, disjoint from every previously-used range in this
# project (Phase 1-4's 0-4/70000/90001-90005/91001-91050/100000-100199/
# 110000-110099/120000-120099/130000-130099/195001-195050/200000-200199/
# 210000-210049/220000-222999/225001-225050).
DIAGNOSTIC_VALIDATION_SEEDS = list(range(230_000, 230_020))  # 20 seeds
CRITIC_GAE_DIAGNOSTIC_ENV_SEED = 231_000

ACTOR_CLONE_DISTANCE_TOL = 1e-4  # deterministic-output tolerance, Section 2
STOCHASTIC_TORCH_SEED_BASE = 900_000  # reproducible (not fresh-random) stochastic sampling


def build_hamilton_ppo(learner_seed: int, env_seed: int = TRAIN_ENV_SEED, model_cls=PPO,
                        log_std_init: float = 0.0) -> tuple:
    """Builds a Hamilton PPO model with IDENTICAL hyperparameters to
    train_agents.py's real event-driven training runs (PPO_KWARGS/NET_ARCH
    above), on the event-driven environment. `model_cls` defaults to plain
    PPO; pass InstrumentedPPO (phase5_instrumented_ppo.py) to get per-update
    diagnostics -- InstrumentedPPO.train() is a byte-for-byte copy of PPO's
    own train() with read-only additions, so this is the SAME configuration
    either way. `log_std_init` defaults to SB3's own ActorCriticPolicy
    default (0.0, i.e. action std=1.0) -- the untouched "existing PPO
    configuration" used throughout groups A/B/C; only the Section 10
    conditional diagnostic E passes a reduced value, and that is reported
    as a separate, explicitly-labelled diagnostic variant, never silently
    substituted for the main experiment's setting. Returns (model, vec_env)."""
    def _init():
        env = EventDrivenHamiltonPPOWrapper(inventory_scale=DEFAULT_INVENTORY_SCALE, seed=env_seed)
        return Monitor(env)

    vec_env = DummyVecEnv([_init])
    model = model_cls(
        "MlpPolicy", vec_env,
        policy_kwargs=dict(net_arch=list(NET_ARCH), log_std_init=log_std_init),
        seed=learner_seed,
        verbose=0,
        **PPO_KWARGS,
    )
    return model, vec_env


def build_supervised_clone_net() -> nn.Module:
    """Exact architecture used by phase4_supervised_clone.build_mlp -- NOT
    imported directly (that module pulls in torch-training-loop-only
    dependencies) but structurally identical and verified to load the same
    state_dict without error."""
    return nn.Sequential(
        nn.Linear(3, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 2),
    )


def load_supervised_clone_net() -> nn.Module:
    net = build_supervised_clone_net()
    net.load_state_dict(torch.load(SUPERVISED_CLONE_PATH))
    net.eval()
    return net


# ======================================================================
# Section 2: supervised-clone -> PPO actor weight transfer
# ======================================================================
def map_supervised_linear_layers(supervised_net: nn.Module) -> list:
    return [m for m in supervised_net if isinstance(m, nn.Linear)]


def map_ppo_actor_linear_layers(model: PPO) -> list:
    """The PPO actor's Linear layers, IN FORWARD-PASS ORDER: the two hidden
    layers inside mlp_extractor.policy_net, followed by action_net (the
    final layer producing the Gaussian mean) -- log_std is a separate
    state-independent nn.Parameter, not part of this list, and is NEVER
    touched by the clone-weight transfer (Section 2: copy into the actor
    only)."""
    policy_net_linears = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, nn.Linear)]
    return policy_net_linears + [model.policy.action_net]


def copy_supervised_actor_weights(model: PPO, supervised_net: nn.Module) -> dict:
    """Copies supervised_net's weights into model.policy's ACTOR path only
    (mlp_extractor.policy_net's two hidden Linear layers + action_net).
    Leaves mlp_extractor.value_net, value_net, and log_std entirely
    untouched (critic stays randomly initialised; exploration variance is
    unchanged; see Section 3 -- this phase does not alter log_std_init).

    Raises AssertionError (fails clearly, per Section 2) if the layer counts
    or any individual weight/bias shape do not match exactly.
    """
    supervised_linears = map_supervised_linear_layers(supervised_net)
    ppo_actor_linears = map_ppo_actor_linear_layers(model)
    assert len(supervised_linears) == len(ppo_actor_linears) == 3, (
        f"Expected 3 Linear layers on both sides (2 hidden + 1 output), "
        f"got supervised={len(supervised_linears)} ppo_actor={len(ppo_actor_linears)}"
    )
    mapping_report = []
    with torch.no_grad():
        for i, (src, dst) in enumerate(zip(supervised_linears, ppo_actor_linears)):
            assert src.weight.shape == dst.weight.shape, (
                f"Layer {i}: weight shape mismatch {src.weight.shape} vs {dst.weight.shape}"
            )
            assert src.bias.shape == dst.bias.shape, (
                f"Layer {i}: bias shape mismatch {src.bias.shape} vs {dst.bias.shape}"
            )
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)
            mapping_report.append(dict(layer_index=i, weight_shape=list(src.weight.shape),
                                        bias_shape=list(src.bias.shape)))
    return dict(n_layers_copied=len(mapping_report), layers=mapping_report,
                critic_untouched=True, log_std_untouched=True)


def get_actor_params(model: PPO) -> list:
    params = list(model.policy.mlp_extractor.policy_net.parameters())
    params += list(model.policy.action_net.parameters())
    params += [model.policy.log_std]
    return params


def get_critic_params(model: PPO) -> list:
    params = list(model.policy.mlp_extractor.value_net.parameters())
    params += list(model.policy.value_net.parameters())
    return params


def flat_param_vector(params: list) -> np.ndarray:
    with torch.no_grad():
        return torch.cat([p.detach().flatten() for p in params]).cpu().numpy()


def actor_param_distance(model: PPO, reference_actor_vector: np.ndarray) -> float:
    current = flat_param_vector(get_actor_params(model))
    return float(np.linalg.norm(current - reference_actor_vector))


def verify_actor_matches_clone(model: PPO, supervised_net: nn.Module, grid_obs: np.ndarray,
                                tol: float = ACTOR_CLONE_DISTANCE_TOL) -> dict:
    """Section 2: prove, on a dense grid of observations, that the PPO
    actor's DETERMINISTIC output equals the supervised clone's output
    within floating-point tolerance. grid_obs: (n, 3) float32 array of
    [q_scaled, tau, belief] observations."""
    supervised_net.eval()
    with torch.no_grad():
        clone_out = supervised_net(torch.tensor(grid_obs, dtype=torch.float32)).numpy()
    clone_out = np.clip(clone_out, -1.0, 1.0)

    ppo_out = np.zeros_like(clone_out)
    for i, obs in enumerate(grid_obs):
        action, _ = model.predict(obs, deterministic=True)
        ppo_out[i] = action

    abs_diff = np.abs(ppo_out - clone_out)
    max_diff = float(abs_diff.max())
    return dict(max_abs_diff=max_diff, mean_abs_diff=float(abs_diff.mean()),
                n_states=len(grid_obs), tolerance=tol, passed=bool(max_diff < tol))


def coarse_grid():
    """A coarser (q, tau, belief) grid than Phase 4's full 9,471-state grid
    (270 states here), used for the action-surface-drift diagnostics that
    must be recomputed repeatedly -- at every one of 8 checkpoints x 9 runs,
    and after every one of 16 updates x 3 clone-initialised runs. A single
    model.predict() call costs ~1-2ms; at the full grid size that is
    10-20s per evaluation, which multiplied by ~120 repeated evaluations in
    this phase would dominate wall-clock time on an 8GB CPU-only machine
    for no diagnostic benefit (the action surface varies smoothly in
    (q, tau, belief), per Phase 4's own finite-difference sensitivity
    results) -- 270 points still resolves the qualitative surface shape.
    The dense 9,471-state grid (dense_grid_observations/phase4_common's
    default_grid) is still used for the one-off exact-equality proof in
    Section 2 and the t=0/final-checkpoint comparisons, where precision
    matters most and the cost is paid only once or twice per run."""
    q_grid = np.arange(-40, 41, 10, dtype=float)       # 9 points
    tau_grid = np.round(np.arange(0.0, 1.01, 0.2), 2)  # 6 points
    b_grid = np.round(np.arange(0.0, 1.01, 0.25), 2)   # 5 points
    return q_grid, tau_grid, b_grid


def dense_grid_observations() -> np.ndarray:
    """Section 2's dense state grid, reusing Phase 4's exact (q, tau,
    belief) grid (9,471 states) rather than re-deriving a new one --
    forward passes only, no rollouts, so the full grid is cheap here."""
    q_grid, tau_grid, b_grid = P4.default_grid()
    obs = []
    for tau in tau_grid:
        for b in b_grid:
            for q in q_grid:
                obs.append([np.tanh(q / P4.INVENTORY_SCALE), tau, b])
    return np.array(obs, dtype=np.float32)


# ======================================================================
# Action-surface metrics vs analytical (reused grid/control machinery)
# ======================================================================
def action_surface_metrics(model, controls: dict, q_grid, tau_grid, b_grid) -> dict:
    """Single-model version of phase4_action_surface_diagnostic's metrics
    -- same formulas, applied to one (possibly mid-training) checkpoint."""
    bid_errs, ask_errs = [], []
    spread_errs, skew_errs = [], []
    wrong_skew = 0
    n = 0
    bid_analytical_list, ask_analytical_list = [], []
    bid_learned_list, ask_learned_list = [], []
    for tau in tau_grid:
        for b in b_grid:
            for q in q_grid:
                a_bid, a_ask = P4.analytical_belief_weighted_depths(controls, float(tau), float(q), float(b))
                l_bid, l_ask = P4.hamilton_ppo_depths(model, float(q), float(tau), float(b))
                bid_errs.append((l_bid - a_bid) ** 2)
                ask_errs.append((l_ask - a_ask) ** 2)
                spread_errs.append((l_bid + l_ask) - (a_bid + a_ask))
                a_skew = a_ask - a_bid
                l_skew = l_ask - l_bid
                skew_errs.append(l_skew - a_skew)
                if abs(q) >= 2.0:
                    expected_sign = np.sign(q)
                    if np.sign(l_skew) != expected_sign and l_skew != 0:
                        wrong_skew += 1
                bid_analytical_list.append(a_bid)
                ask_analytical_list.append(a_ask)
                bid_learned_list.append(l_bid)
                ask_learned_list.append(l_ask)
                n += 1
    bid_a = np.array(bid_analytical_list)
    ask_a = np.array(ask_analytical_list)
    bid_l = np.array(bid_learned_list)
    ask_l = np.array(ask_learned_list)
    corr_bid = float(np.corrcoef(bid_a, bid_l)[0, 1]) if np.std(bid_l) > 0 else float("nan")
    corr_ask = float(np.corrcoef(ask_a, ask_l)[0, 1]) if np.std(ask_l) > 0 else float("nan")
    return dict(
        bid_action_depth_mse=float(np.mean(bid_errs)), ask_action_depth_mse=float(np.mean(ask_errs)),
        mean_quoted_spread_error_signed=float(np.mean(spread_errs)),
        mean_inventory_skew_error_signed=float(np.mean(skew_errs)),
        frac_wrong_inventory_skew=float(wrong_skew / n),
        corr_bid=corr_bid, corr_ask=corr_ask,
        frac_wider_than_analytical=float(np.mean(np.array(spread_errs) > 0)),
        n_states=n,
    )


# ======================================================================
# Episode runners parametrised by `deterministic` (Section 3/5 need BOTH --
# evaluate_agents_event_driven.run_event_hamilton_agent_episode hardcodes
# deterministic=True, so a separate runner is used here rather than
# modifying that already-validated module).
# ======================================================================
def run_event_episode(model, seed: int, deterministic: bool) -> dict:
    base_env = EventDrivenRegimeSwitchingEnv(seed=seed)
    wrapper = EventDrivenHamiltonPPOWrapper(base_env=base_env, inventory_scale=DEFAULT_INVENTORY_SCALE)
    obs, _ = wrapper.reset()
    cash_0, inv_0, mid_0 = wrapper.base_env.raw_cash, wrapper.base_env.raw_inventory, wrapper.base_env.raw_midprice

    running_penalty = total_fills = 0.0
    inv_sum = inv_abs_sum = 0.0
    cumulative_objective = 0.0
    n_steps = 0
    bid_actions, ask_actions = [], []
    n_near_bound = 0
    terminated = truncated = False
    info = None
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float64)
        bid_actions.append(float(action[0]))
        ask_actions.append(float(action[1]))
        if np.any(np.abs(action) > 0.95):
            n_near_bound += 1

        obs, reward, terminated, truncated, info = wrapper.step(action)

        fill_bid, fill_ask = info["fill_indicator"]
        total_fills += int(fill_bid) + int(fill_ask)
        running_penalty += info["running_penalty_increment"]
        inv_after = info["inventory_after"]
        inv_sum += inv_after
        inv_abs_sum += abs(inv_after)
        cumulative_objective += float(reward)
        n_steps += 1

    cash_T, inv_T, mid_T = info["cash_after"], info["inventory_after"], info["price_after"]
    raw_pnl = (cash_T + inv_T * mid_T) - (cash_0 + inv_0 * mid_0)
    terminal_penalty = info["terminal_penalty_increment"]
    full_objective = raw_pnl - running_penalty - terminal_penalty

    bid_actions = np.array(bid_actions)
    ask_actions = np.array(ask_actions)
    mean_spread = float(((ask_actions.mean() + 1.0) / 2.0 - (bid_actions.mean() - 1.0) / 2.0) * SBW.MAX_DEPTH) \
        if n_steps else float("nan")

    return dict(
        seed=seed, deterministic=deterministic, n_steps=n_steps,
        full_objective=full_objective, raw_pnl=raw_pnl,
        running_penalty=running_penalty, terminal_penalty=terminal_penalty,
        fills=total_fills, mean_abs_inventory=(inv_abs_sum / n_steps) if n_steps else float("nan"),
        mean_signed_inventory=(inv_sum / n_steps) if n_steps else float("nan"),
        terminal_signed_inventory=inv_T,
        mean_quoted_spread=mean_spread,
        bid_action_mean=float(bid_actions.mean()) if n_steps else float("nan"),
        ask_action_mean=float(ask_actions.mean()) if n_steps else float("nan"),
        bid_action_std=float(bid_actions.std()) if n_steps else float("nan"),
        ask_action_std=float(ask_actions.std()) if n_steps else float("nan"),
        frac_near_bound=(n_near_bound / n_steps) if n_steps else float("nan"),
    )


def evaluate_checkpoint(model, seeds=DIAGNOSTIC_VALIDATION_SEEDS, torch_seed: int = None) -> dict:
    """Section 5: deterministic + stochastic mean objective and behavioural
    metrics, on the SAME fixed diagnostic validation seeds at every
    checkpoint (diagnostic only -- never reused as a final holdout set)."""
    det_records = [run_event_episode(model, s, deterministic=True) for s in seeds]

    if torch_seed is not None:
        torch.manual_seed(torch_seed)
    stoch_records = [run_event_episode(model, s, deterministic=False) for s in seeds]

    def agg(records, prefix):
        obj = np.array([r["full_objective"] for r in records])
        return {
            f"{prefix}_mean_objective": float(obj.mean()),
            f"{prefix}_se_objective": float(obj.std(ddof=1) / np.sqrt(len(obj))) if len(obj) > 1 else 0.0,
            f"{prefix}_mean_raw_pnl": float(np.mean([r["raw_pnl"] for r in records])),
            f"{prefix}_mean_quoted_spread": float(np.mean([r["mean_quoted_spread"] for r in records])),
            f"{prefix}_mean_fills": float(np.mean([r["fills"] for r in records])),
            f"{prefix}_mean_abs_inventory": float(np.mean([r["mean_abs_inventory"] for r in records])),
            f"{prefix}_mean_signed_inventory": float(np.mean([r["mean_signed_inventory"] for r in records])),
            f"{prefix}_loss_rate": float(np.mean(obj < 0)),
            f"{prefix}_frac_near_bound": float(np.mean([r["frac_near_bound"] for r in records])),
        }

    result = {}
    result.update(agg(det_records, "det"))
    result.update(agg(stoch_records, "stoch"))
    result["n_eval_seeds"] = len(seeds)
    return result
