from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from crlpa.data.synthetic import make_synthetic_returns  # noqa: E402
from crlpa.envs.allocation import AllocationEnv  # noqa: E402
from crlpa.envs.constraints import PortfolioConstraints  # noqa: E402
from crlpa.models.cvar_actor_critic import ActorCriticConfig, CVaRActorCritic  # noqa: E402
from crlpa.training.train_allocator import TrainConfig, train  # noqa: E402


def _env(seed: int = 7, n: int = 120) -> AllocationEnv:
    returns = make_synthetic_returns(n_steps=n, seed=seed)
    return AllocationEnv(returns, constraints=PortfolioConstraints(max_weight=0.6, turnover_cap=0.5),
                         cvar_limit=0.02)


def test_agent_outputs_valid_weights():
    env = _env()
    env.reset()
    agent = CVaRActorCritic(env.obs_dim, env.action_dim)
    weights, cache = agent.act(env.observation())
    assert weights.shape == (env.action_dim,)
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()
    assert cache.value.requires_grad


def test_training_runs_and_logs():
    env = _env()
    agent = CVaRActorCritic(env.obs_dim, env.action_dim, ActorCriticConfig(constrained=True))
    agent, history, best = train(env, agent, TrainConfig(n_episodes=5, seed=7), val_env=env)
    assert len(history) == 5
    assert {"ep_return", "lagrange", "mean_cvar_cost", "breach_rate"} <= set(history.columns)
    assert "actor" in best


def test_lagrange_rises_under_breaches():
    # A tight CVaR limit guarantees breaches, so the multiplier should grow.
    returns = make_synthetic_returns(n_steps=120, seed=13)
    env = AllocationEnv(returns, cvar_limit=0.0)  # any tail loss breaches
    agent = CVaRActorCritic(env.obs_dim, env.action_dim, ActorCriticConfig(constrained=True))
    train(env, agent, TrainConfig(n_episodes=15, seed=13))
    assert agent.lagrange.value > 0.0


def test_unconstrained_keeps_lagrange_zero():
    env = _env()
    agent = CVaRActorCritic(env.obs_dim, env.action_dim, ActorCriticConfig(constrained=False))
    train(env, agent, TrainConfig(n_episodes=10, seed=7))
    assert agent.lagrange.value == 0.0


def test_deterministic_predict_is_stable():
    env = _env()
    env.reset()
    agent = CVaRActorCritic(env.obs_dim, env.action_dim)
    obs = env.observation()
    assert np.allclose(agent.predict(obs), agent.predict(obs))
