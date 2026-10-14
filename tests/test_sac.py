from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from crlpa.data.synthetic import make_synthetic_returns  # noqa: E402
from crlpa.envs.allocation import AllocationEnv  # noqa: E402
from crlpa.envs.constraints import PortfolioConstraints  # noqa: E402
from crlpa.evaluation.backtest import run_policy  # noqa: E402
from crlpa.training.sac import SACAllocator, SACConfig, train_sac  # noqa: E402


def _env(n=160, seed=7):
    return AllocationEnv(
        make_synthetic_returns(n_steps=n, seed=seed),
        constraints=PortfolioConstraints(max_weight=0.6, turnover_cap=0.5),
        cvar_limit=0.02,
    )


def test_sac_trains_and_predicts_valid_weights():
    env = _env()
    agent = SACAllocator(env.obs_dim, env.action_dim,
                         SACConfig(n_iterations=6, rollout_len=60, warmup_steps=60,
                                   batch_size=32, seed=7))
    agent, history = train_sac(env, agent)
    assert len(history) == 6
    w = agent.predict(env.observation())
    assert np.isclose(w.sum(), 1.0) and (w >= 0).all()
    res = run_policy(_env(), lambda e: agent.predict(e.observation()))
    assert np.allclose(res.weights.sum(axis=1), 1.0)


def test_sac_constrained_raises_lagrange_under_breaches():
    env = AllocationEnv(make_synthetic_returns(n_steps=160, seed=13), cvar_limit=0.0)
    agent = SACAllocator(env.obs_dim, env.action_dim,
                         SACConfig(n_iterations=10, rollout_len=80, warmup_steps=40,
                                   batch_size=32, constrained=True, seed=13))
    train_sac(env, agent)
    assert agent.lagrange.value > 0.0


def test_sac_unconstrained_keeps_lagrange_zero():
    env = _env()
    agent = SACAllocator(env.obs_dim, env.action_dim,
                         SACConfig(n_iterations=6, rollout_len=60, warmup_steps=40,
                                   batch_size=32, constrained=False, seed=7))
    train_sac(env, agent)
    assert agent.lagrange.value == 0.0
