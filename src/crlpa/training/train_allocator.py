from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crlpa.envs.allocation import AllocationEnv
from crlpa.models.cvar_actor_critic import CVaRActorCritic
from crlpa.utils.seeds import set_global_seed


@dataclass
class TrainConfig:
    n_episodes: int = 200
    turnover_penalty: float = 0.0
    entropy_decay: float = 0.1  # final entropy coef as a fraction of the initial
    eval_every: int = 10
    seed: int = 42


def _rollout(env: AllocationEnv, agent: CVaRActorCritic, turnover_penalty: float = 0.0):
    obs = env.observation()
    caches, rewards, costs = [], [], []
    done = False
    while not done:
        weights, cache = agent.act(obs)
        _, reward, done, info = env.step(weights)
        caches.append(cache)
        rewards.append(reward - turnover_penalty * info["turnover"])
        # The breach indicator (0/1) is well scaled for the dual update and the
        # safety critic; the raw cost magnitude (~1e-3) is too small to drive them.
        cost = info["cvar_breach"] if agent.config.cost_mode == "breach" else info["cvar_cost"]
        costs.append(cost)
        obs = env.observation()
    return caches, rewards, costs


def evaluate(env: AllocationEnv, agent: CVaRActorCritic) -> dict[str, float]:
    """Deterministic pass used for validation-best model selection."""
    env.reset()
    obs = env.observation()
    rewards, breaches, turnovers = [], [], []
    done = False
    while not done:
        _, reward, done, info = env.step(agent.predict(obs))
        rewards.append(reward)
        breaches.append(info["cvar_breach"])
        turnovers.append(info["turnover"])
        obs = env.observation()
    arr = np.array(rewards)
    return {
        "mean_reward": float(arr.mean()),
        "total_return": float(np.prod(1 + arr) - 1),
        "cvar_breach_rate": float(np.mean(breaches)),
        "avg_turnover": float(np.mean(turnovers)),
    }


def train(
    train_env: AllocationEnv,
    agent: CVaRActorCritic,
    config: TrainConfig | None = None,
    val_env: AllocationEnv | None = None,
) -> tuple[CVaRActorCritic, pd.DataFrame, dict]:
    """Episodic actor-critic training with Lagrangian CVaR control.

    Returns the agent, a per-episode log, and the best validation checkpoint
    (selected by validation mean reward among constraint-satisfying evaluations
    when a ``val_env`` is supplied).
    """
    config = config or TrainConfig()
    set_global_seed(config.seed)
    init_entropy = agent.config.entropy_coef
    history: list[dict[str, float]] = []
    best_score = -np.inf
    best_state = agent.state_dict()

    for episode in range(config.n_episodes):
        train_env.reset()
        caches, rewards, costs = _rollout(train_env, agent, config.turnover_penalty)
        frac = episode / max(1, config.n_episodes - 1)
        entropy_coef = init_entropy * (1 - frac * (1 - config.entropy_decay))
        stats = agent.update(caches, rewards, costs, entropy_coef=entropy_coef)

        record = {
            "episode": episode,
            "ep_return": float(np.prod(1 + np.array(rewards)) - 1),
            "mean_reward": float(np.mean(rewards)),
            "mean_cvar_cost": float(np.mean(costs)),
            "breach_rate": float(np.mean(np.array(costs) > 0)),
            "entropy_coef": entropy_coef,
            **stats,
        }

        if val_env is not None and episode % config.eval_every == 0:
            val = evaluate(val_env, agent)
            record.update({f"val_{k}": v for k, v in val.items()})
            score = val["mean_reward"] - (val["cvar_breach_rate"] if agent.config.constrained else 0)
            if score > best_score:
                best_score = score
                best_state = agent.state_dict()

        history.append(record)

    if val_env is None:
        best_state = agent.state_dict()
    return agent, pd.DataFrame(history), best_state
