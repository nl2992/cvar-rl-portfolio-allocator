from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from crlpa.data.synthetic import make_synthetic_returns  # noqa: E402
from crlpa.envs.allocation import AllocationEnv  # noqa: E402
from crlpa.evaluation.backtest import run_policy  # noqa: E402
from crlpa.training.differentiable import (  # noqa: E402
    DiffAllocator,
    DiffConfig,
    diff_policy,
    ensemble_diff_policy,
    train_differentiable,
)


def test_diff_allocator_outputs_simplex():
    actor = DiffAllocator(obs_dim=3 * 5 + 2, action_dim=5)
    w = actor.predict(np.zeros(3 * 5 + 2, dtype=np.float32))
    assert w.shape == (5,)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()


def test_anchor_recovers_anchor_at_init():
    # With zero net output, softmax(log_anchor) must reproduce the anchor.
    actor = DiffAllocator(obs_dim=3 * 4 + 2, action_dim=4)
    for p in actor.net.parameters():
        torch.nn.init.zeros_(p)
    anchor = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    w = actor.predict(np.zeros(3 * 4 + 2, dtype=np.float32), np.log(anchor))
    assert np.allclose(w, anchor, atol=1e-5)


def test_training_runs_and_policy_is_valid():
    returns = make_synthetic_returns(n_steps=160, seed=7)
    train, val = returns.iloc[:120], returns.iloc[120:]
    actor, history = train_differentiable(
        train, cost_bps=5.0, cvar_alpha=0.95, cvar_limit=0.03, cvar_window=52, lookback=26,
        config=DiffConfig(n_updates=60, horizon=52, constrained=True, seed=7),
        val_returns=val,
    )
    assert {"sharpe", "cvar", "lagrange"} <= set(history.columns)
    res = run_policy(AllocationEnv(returns), diff_policy(actor, lookback=26))
    weights_ok = np.allclose(res.weights.sum(axis=1), 1.0)
    assert weights_ok


def test_ensemble_policy_produces_valid_simplex():
    returns = make_synthetic_returns(n_steps=120, seed=7)
    actors = [DiffAllocator(obs_dim=3 * returns.shape[1] + 2, action_dim=returns.shape[1]) for _ in range(3)]
    res = run_policy(AllocationEnv(returns), ensemble_diff_policy(actors, lookback=26))
    assert np.allclose(res.weights.sum(axis=1), 1.0)
    assert (res.weights.to_numpy() >= -1e-9).all()


def test_turnover_penalty_reduces_turnover():
    # No val selection (returns the trained actor) so the penalty's effect is visible.
    returns = make_synthetic_returns(n_steps=180, seed=7)
    common = dict(cost_bps=5.0, cvar_alpha=0.95, cvar_limit=0.03, cvar_window=52, lookback=26)
    # anchor=None so the network (not the rebalancing anchor) controls all turnover.
    base, _ = train_differentiable(returns, config=DiffConfig(n_updates=200, horizon=52, anchor=None, turnover_penalty=0.0, seed=7), **common)
    pen, _ = train_differentiable(returns, config=DiffConfig(n_updates=200, horizon=52, anchor=None, turnover_penalty=200.0, seed=7), **common)
    t_base = run_policy(AllocationEnv(returns), diff_policy(base, 26, anchor=None)).metrics["avg_turnover"]
    t_pen = run_policy(AllocationEnv(returns), diff_policy(pen, 26, anchor=None)).metrics["avg_turnover"]
    assert t_pen < t_base
