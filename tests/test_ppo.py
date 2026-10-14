from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from crlpa.data.synthetic import make_synthetic_returns  # noqa: E402
from crlpa.envs.allocation import AllocationEnv  # noqa: E402
from crlpa.envs.constraints import PortfolioConstraints  # noqa: E402
from crlpa.evaluation.backtest import run_policy  # noqa: E402
from crlpa.training.ppo import PPOAllocator, PPOConfig, train_ppo  # noqa: E402


def _env(n=140, seed=7):
    return AllocationEnv(
        make_synthetic_returns(n_steps=n, seed=seed),
        constraints=PortfolioConstraints(max_weight=0.6, turnover_cap=0.5),
        cvar_limit=0.02,
    )


def test_ppo_trains_and_predicts_valid_weights():
    env = _env()
    agent = PPOAllocator(env.obs_dim, env.action_dim, PPOConfig(n_iterations=4, rollout_len=60, epochs=3, seed=7))
    agent, history = train_ppo(env, agent)
    assert len(history) == 4
    w = agent.predict(env.observation())
    assert np.isclose(w.sum(), 1.0) and (w >= 0).all()
    res = run_policy(_env(), lambda e: agent.predict(e.observation()))
    assert np.allclose(res.weights.sum(axis=1), 1.0)


def test_ppo_constrained_raises_lagrange_under_breaches():
    env = AllocationEnv(make_synthetic_returns(n_steps=140, seed=13), cvar_limit=0.0)
    agent = PPOAllocator(env.obs_dim, env.action_dim,
                         PPOConfig(n_iterations=8, rollout_len=80, epochs=3, constrained=True, seed=13))
    train_ppo(env, agent)
    assert agent.lagrange.value > 0.0


def test_ppo_unconstrained_keeps_lagrange_zero():
    env = _env()
    agent = PPOAllocator(env.obs_dim, env.action_dim, PPOConfig(n_iterations=5, rollout_len=60, constrained=False, seed=7))
    train_ppo(env, agent)
    assert agent.lagrange.value == 0.0
