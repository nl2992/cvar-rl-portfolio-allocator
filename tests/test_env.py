from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crlpa.data.synthetic import make_synthetic_returns
from crlpa.envs.allocation import AllocationEnv
from crlpa.envs.constraints import PortfolioConstraints


def _const_returns(value: float = 0.01, n: int = 10, k: int = 3) -> pd.DataFrame:
    return pd.DataFrame(np.full((n, k), value), columns=[f"a{i}" for i in range(k)])


def test_weights_sum_to_one_after_projection():
    env = AllocationEnv(_const_returns(), constraints=PortfolioConstraints(max_weight=0.6))
    env.reset()
    state, _, _, _ = env.step(np.array([10.0, 1.0, 1.0]))
    assert np.isclose(state.weights.sum(), 1.0)
    assert state.weights.max() <= 0.6 + 1e-9


def test_no_trade_path_matches_buy_and_hold():
    returns = make_synthetic_returns(n_steps=30, seed=7)
    env = AllocationEnv(returns, transaction_cost_bps=5.0)
    env.reset()
    w = np.full(returns.shape[1], 1 / returns.shape[1])
    # First step incurs cost moving from the equal-weight start to itself: zero turnover.
    _, reward, _, info = env.step(w)
    expected = float(w @ returns.iloc[0].to_numpy())
    assert info["turnover"] == pytest.approx(0.0, abs=1e-12)
    assert info["costs"] == pytest.approx(0.0, abs=1e-12)
    assert reward == pytest.approx(expected, abs=1e-12)


def test_transaction_costs_reduce_wealth():
    returns = _const_returns(value=0.0, n=5, k=3)  # zero asset returns
    env = AllocationEnv(returns, transaction_cost_bps=50.0)
    env.reset()
    _, reward, _, info = env.step(np.array([1.0, 0.0, 0.0]))  # concentrate -> turnover
    assert info["turnover"] > 0
    assert reward < 0  # only costs, no return
    assert info["wealth"] < 1.0


def test_observation_has_no_lookahead():
    returns = make_synthetic_returns(n_steps=40, seed=13)
    env = AllocationEnv(returns, lookback=10)
    env.reset()
    # At step 0 there is no history, so momentum/vol features are zero.
    obs0 = env.observation()
    assert np.allclose(obs0[: 2 * env.n_assets], 0.0)
    assert obs0.shape == (env.obs_dim,)


def test_rollout_length_matches_series():
    returns = make_synthetic_returns(n_steps=25, seed=7)
    env = AllocationEnv(returns)
    env.reset()
    steps = 0
    done = False
    while not done:
        _, _, done, _ = env.step(np.full(returns.shape[1], 1 / returns.shape[1]))
        steps += 1
    assert steps == 25
