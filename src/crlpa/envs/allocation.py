from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crlpa.envs.constraints import PortfolioConstraints, project_weights, violations
from crlpa.evaluation.metrics import cvar


@dataclass
class AllocationState:
    step: int
    weights: np.ndarray
    wealth: float
    peak_wealth: float = 1.0
    drawdown: float = 0.0
    cvar_estimate: float = 0.0
    returns_history: deque = field(default_factory=lambda: deque(maxlen=64))


class AllocationEnv:
    """Gym-like long-only multi-asset allocation environment.

    At decision step ``t`` the agent proposes post-trade weights using only
    information observable strictly before ``t`` (returns ``[:t]``). The proposed
    weights are projected onto the admissible set, transaction costs are charged
    on the resulting trades, and next-period returns ``returns[t]`` are applied.

    A rolling historical CVaR of realised net portfolio returns is maintained as
    both a state feature and a constraint signal. The CVaR constraint cost is
    ``max(0, cvar_estimate - cvar_limit)``.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        transaction_cost_bps: float = 2.0,
        constraints: PortfolioConstraints | None = None,
        cvar_alpha: float = 0.95,
        cvar_limit: float = 0.035,
        cvar_window: int = 52,
        lookback: int = 26,
        exog: np.ndarray | None = None,
    ) -> None:
        if returns.empty:
            raise ValueError("returns cannot be empty")
        self.returns = returns.reset_index(drop=True)
        self.transaction_cost = transaction_cost_bps / 10_000.0
        self.n_assets = returns.shape[1]
        if exog is not None:
            exog = np.asarray(exog, dtype=np.float64)
            if exog.shape[0] != len(self.returns):
                raise ValueError("exog must have one row per return step")
        self.exog = exog
        self.constraints = constraints
        if constraints is not None:
            constraints.feasible_max_weight(self.n_assets)
        self.cvar_alpha = cvar_alpha
        self.cvar_limit = cvar_limit
        self.cvar_window = cvar_window
        self.lookback = lookback
        self.state = self._initial_state()

    # observation layout: [momentum(n), volatility(n), weights(n), drawdown, cvar, exog(k)]
    @property
    def exog_dim(self) -> int:
        return 0 if self.exog is None else int(self.exog.shape[1])

    @property
    def obs_dim(self) -> int:
        return 3 * self.n_assets + 2 + self.exog_dim

    @property
    def action_dim(self) -> int:
        return self.n_assets

    def _initial_state(self) -> AllocationState:
        return AllocationState(
            step=0,
            weights=np.full(self.n_assets, 1.0 / self.n_assets),
            wealth=1.0,
            peak_wealth=1.0,
            drawdown=0.0,
            cvar_estimate=0.0,
            returns_history=deque(maxlen=self.cvar_window),
        )

    def reset(self) -> AllocationState:
        self.state = self._initial_state()
        return self.state

    def observation(self) -> np.ndarray:
        """Feature vector available to the policy at the current step (no look-ahead)."""
        t = self.state.step
        hist = self.returns.iloc[max(0, t - self.lookback) : t].to_numpy()
        if hist.shape[0] < 2:
            momentum = np.zeros(self.n_assets)
            volatility = np.zeros(self.n_assets)
        else:
            momentum = hist.mean(axis=0)
            volatility = hist.std(axis=0)
        parts = [
            momentum,
            volatility,
            self.state.weights,
            np.array([self.state.drawdown, self.state.cvar_estimate]),
        ]
        if self.exog is not None:
            parts.append(self.exog[min(t, len(self.exog) - 1)])
        return np.concatenate(parts).astype(np.float64)

    def step(self, action: np.ndarray) -> tuple[AllocationState, float, bool, dict[str, float]]:
        weights = self._admissible(action)
        old = self.state.weights
        turnover = float(np.abs(weights - old).sum())
        costs = turnover * self.transaction_cost
        gross_return = float(weights @ self.returns.iloc[self.state.step].to_numpy())
        portfolio_return = gross_return - costs

        wealth = self.state.wealth * (1 + portfolio_return)
        peak = max(self.state.peak_wealth, wealth)
        drawdown = float(1.0 - wealth / peak)

        history = self.state.returns_history
        history.append(portfolio_return)
        cvar_estimate = (
            cvar(np.array(history), alpha=self.cvar_alpha) if len(history) >= 5 else 0.0
        )
        cvar_cost = max(0.0, cvar_estimate - self.cvar_limit)

        viol = (
            violations(weights, old, self.constraints) if self.constraints is not None else {}
        )

        done = self.state.step >= len(self.returns) - 1
        self.state = AllocationState(
            step=self.state.step + 1,
            weights=weights,
            wealth=wealth,
            peak_wealth=peak,
            drawdown=drawdown,
            cvar_estimate=cvar_estimate,
            returns_history=history,
        )
        info = {
            "costs": costs,
            "wealth": wealth,
            "turnover": turnover,
            "gross_return": gross_return,
            "drawdown": drawdown,
            "cvar_estimate": cvar_estimate,
            "cvar_cost": cvar_cost,
            "cvar_breach": float(cvar_estimate > self.cvar_limit),
            "constraint_violation": float(sum(viol.values())),
        }
        return self.state, portfolio_return, done, info

    def _admissible(self, action: np.ndarray) -> np.ndarray:
        weights = np.asarray(action, dtype=float)
        if weights.shape != (self.n_assets,):
            raise ValueError(f"expected action shape {(self.n_assets,)}")
        if self.constraints is not None:
            return project_weights(weights, self.state.weights, self.constraints)
        return self._normalise(weights)

    def _normalise(self, action: np.ndarray) -> np.ndarray:
        weights = np.clip(np.asarray(action, dtype=float), 0.0, None)
        total = weights.sum()
        if total <= 0:
            return np.full(self.n_assets, 1.0 / self.n_assets)
        return weights / total
