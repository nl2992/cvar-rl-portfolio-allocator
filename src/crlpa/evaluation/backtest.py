from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from crlpa.envs.allocation import AllocationEnv
from crlpa.evaluation.metrics import summarise

# A policy maps the current environment to a proposed weight vector. It must use
# only information observable before the current step (env.returns[:env.state.step]).
Policy = Callable[[AllocationEnv], np.ndarray]


@dataclass
class BacktestResult:
    returns: pd.Series
    weights: pd.DataFrame
    info: pd.DataFrame
    metrics: dict[str, float]

    @property
    def wealth(self) -> pd.Series:
        return (1 + self.returns).cumprod().rename("wealth")


def static_policy(weights: np.ndarray) -> Policy:
    """A policy that always proposes the same weights."""
    w = np.asarray(weights, dtype=float)
    return lambda env: w


def rolling_weight_policy(
    weight_fn: Callable[[pd.DataFrame], np.ndarray],
    min_history: int = 26,
) -> Policy:
    """Recompute weights each step from history strictly before the current step."""

    def policy(env: AllocationEnv) -> np.ndarray:
        t = env.state.step
        if t < min_history:
            return np.full(env.n_assets, 1.0 / env.n_assets)
        history = env.returns.iloc[:t]
        return weight_fn(history)

    return policy


def run_policy(env: AllocationEnv, policy: Policy, periods_per_year: int = 52) -> BacktestResult:
    """Roll a policy through the environment and collect series + metrics."""
    env.reset()
    rewards: list[float] = []
    weight_rows: list[np.ndarray] = []
    info_rows: list[dict[str, float]] = []
    done = False
    while not done:
        action = np.asarray(policy(env), dtype=float)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        weight_rows.append(state.weights.copy())
        info_rows.append(info)

    returns = pd.Series(rewards, name="portfolio_return")
    weights = pd.DataFrame(weight_rows, columns=list(env.returns.columns))
    info = pd.DataFrame(info_rows)

    metrics = summarise(returns, periods_per_year=periods_per_year)
    metrics.update(
        {
            "avg_turnover": float(info["turnover"].mean()),
            "total_costs": float(info["costs"].sum()),
            "cvar_breach_rate": float(info["cvar_breach"].mean()),
            "constraint_violations": float((info["constraint_violation"] > 1e-8).sum()),
            "final_wealth": float(info["wealth"].iloc[-1]),
        }
    )
    return BacktestResult(returns=returns, weights=weights, info=info, metrics=metrics)
