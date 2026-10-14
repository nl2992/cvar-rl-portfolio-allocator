from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

_TOL = 1e-9


def _clean_window(returns: pd.DataFrame, lookback: int | None) -> np.ndarray:
    window = returns.tail(lookback) if lookback else returns
    return window.to_numpy(dtype=float)


def _simplex_constraints(n: int, max_weight: float):
    bounds = [(0.0, max_weight)] * n
    budget = {"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}
    return bounds, budget


# --- closed-form / heuristic baselines -------------------------------------


def cash(n_assets: int, cash_index: int = -1) -> np.ndarray:
    """All weight on the cash asset."""
    w = np.zeros(n_assets)
    w[cash_index] = 1.0
    return w


def equal_weight(n_assets: int) -> np.ndarray:
    return np.full(n_assets, 1.0 / n_assets)


def inverse_volatility(returns: pd.DataFrame, lookback: int = 52) -> np.ndarray:
    window = returns.tail(lookback)
    vols = window.std().replace(0, np.nan).fillna(window.std().mean())
    inv = 1.0 / vols.to_numpy()
    return inv / inv.sum()


# --- optimiser baselines ----------------------------------------------------


def min_variance(
    returns: pd.DataFrame, lookback: int = 104, max_weight: float = 1.0
) -> np.ndarray:
    """Long-only global minimum-variance portfolio."""
    data = _clean_window(returns, lookback)
    cov = np.cov(data, rowvar=False)
    n = cov.shape[0]
    bounds, budget = _simplex_constraints(n, max_weight)

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    result = minimize(
        objective,
        equal_weight(n),
        method="SLSQP",
        bounds=bounds,
        constraints=[budget],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return _finalise(result.x, n)


def min_variance_shrunk(
    returns: pd.DataFrame, lookback: int = 104, max_weight: float = 1.0
) -> np.ndarray:
    """Long-only minimum-variance with Ledoit-Wolf shrinkage covariance.

    Shrinking the sample covariance toward a scaled identity is the standard fix for
    estimation error in high-dimensional universes, so this is the *strong* optimiser
    baseline the learned allocator must beat (raw sample covariance is fragile when
    the asset count approaches the lookback).
    """
    from sklearn.covariance import LedoitWolf

    data = _clean_window(returns, lookback)
    cov = LedoitWolf().fit(data).covariance_
    n = cov.shape[0]
    bounds, budget = _simplex_constraints(n, max_weight)

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    result = minimize(
        objective,
        equal_weight(n),
        method="SLSQP",
        bounds=bounds,
        constraints=[budget],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return _finalise(result.x, n)


def mean_variance(
    returns: pd.DataFrame,
    lookback: int = 104,
    risk_aversion: float = 8.0,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Long-only mean-variance portfolio: max mu'w - (gamma/2) w'Sigma w."""
    data = _clean_window(returns, lookback)
    mu = data.mean(axis=0)
    cov = np.cov(data, rowvar=False)
    n = cov.shape[0]
    bounds, budget = _simplex_constraints(n, max_weight)

    def objective(w: np.ndarray) -> float:
        return float(-(mu @ w) + 0.5 * risk_aversion * (w @ cov @ w))

    result = minimize(
        objective,
        equal_weight(n),
        method="SLSQP",
        bounds=bounds,
        constraints=[budget],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return _finalise(result.x, n)


def risk_parity(
    returns: pd.DataFrame, lookback: int = 104, max_weight: float = 1.0
) -> np.ndarray:
    """Long-only equal-risk-contribution portfolio."""
    data = _clean_window(returns, lookback)
    cov = np.cov(data, rowvar=False)
    n = cov.shape[0]
    bounds, budget = _simplex_constraints(n, max_weight)
    target = 1.0 / n

    def objective(w: np.ndarray) -> float:
        port_var = float(w @ cov @ w)
        if port_var <= _TOL:
            return 0.0
        marginal = cov @ w
        contrib = w * marginal / port_var
        return float(np.sum((contrib - target) ** 2))

    result = minimize(
        objective,
        equal_weight(n),
        method="SLSQP",
        bounds=bounds,
        constraints=[budget],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return _finalise(result.x, n)


def cvar_optimizer(
    returns: pd.DataFrame,
    lookback: int = 104,
    alpha: float = 0.95,
    target_return: float | None = None,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Minimum-CVaR portfolio via the Rockafellar-Uryasev linear program.

    Variables are ``[w (n), var (1), z (T)]`` where ``z_t >= -r_t'w - var``.
    Objective: minimise ``var + 1/((1-alpha) T) * sum(z_t)``. An optional minimum
    expected-return constraint can be supplied; if infeasible we fall back to the
    unconstrained minimum-CVaR solution.
    """
    data = _clean_window(returns, lookback)
    n_periods, n = data.shape
    beta = 1.0 - alpha

    c = np.concatenate([np.zeros(n), [1.0], np.full(n_periods, 1.0 / (beta * n_periods))])

    # z_t >= -r_t . w - var  ->  -r_t . w - var - z_t <= 0
    a_tail = np.hstack([-data, -np.ones((n_periods, 1)), -np.eye(n_periods)])
    b_tail = np.zeros(n_periods)

    a_ub = a_tail
    b_ub = b_tail
    if target_return is not None:
        mu = data.mean(axis=0)
        row = np.concatenate([-mu, [0.0], np.zeros(n_periods)])
        a_ub = np.vstack([a_ub, row])
        b_ub = np.append(b_ub, -target_return)

    a_eq = np.concatenate([np.ones(n), [0.0], np.zeros(n_periods)]).reshape(1, -1)
    b_eq = np.array([1.0])

    bounds = [(0.0, max_weight)] * n + [(None, None)] + [(0.0, None)] * n_periods
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success and target_return is not None:
        return cvar_optimizer(returns, lookback, alpha, None, max_weight)
    if not result.success:
        return equal_weight(n)
    return _finalise(result.x[:n], n)


def _finalise(weights: np.ndarray, n: int) -> np.ndarray:
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = w.sum()
    if total <= _TOL:
        return equal_weight(n)
    return w / total
