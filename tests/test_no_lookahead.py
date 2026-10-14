from __future__ import annotations

import numpy as np
import pandas as pd

from crlpa.envs.allocation import AllocationEnv
from crlpa.evaluation.backtest import rolling_weight_policy, run_policy
from crlpa.policies import baselines as b


def test_observation_only_uses_past_returns():
    """Mutating future returns must not change the current observation."""
    base = pd.DataFrame(
        np.linspace(0.001, 0.01, 60 * 4).reshape(60, 4), columns=list("abcd")
    )
    env_a = AllocationEnv(base.copy(), lookback=12)
    env_b_returns = base.copy()
    env_b_returns.iloc[30:] = 999.0  # corrupt the entire future
    env_b = AllocationEnv(env_b_returns, lookback=12)

    env_a.reset()
    env_b.reset()
    # advance both to step 20 with identical actions
    w = np.full(4, 0.25)
    for _ in range(20):
        env_a.step(w)
        env_b.step(w)
    assert np.allclose(env_a.observation(), env_b.observation())


def test_rolling_policy_does_not_peek_at_current_step():
    """The rolling policy at step t must use only returns[:t]."""
    returns = pd.DataFrame(
        np.random.default_rng(1).normal(0.0005, 0.01, (80, 3)), columns=list("xyz")
    )
    captured = {}

    def weight_fn(history: pd.DataFrame) -> np.ndarray:
        captured["max_index"] = history.index.max()
        captured["env_step"] = env.state.step
        return b.equal_weight(history.shape[1])

    env = AllocationEnv(returns)
    policy = rolling_weight_policy(weight_fn, min_history=5)
    run_policy(env, policy)
    # history passed to the policy ends strictly before the current decision step
    assert captured["max_index"] == captured["env_step"] - 1
