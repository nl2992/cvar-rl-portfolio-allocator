from __future__ import annotations

import numpy as np
import pandas as pd

ArrayLike = pd.Series | np.ndarray


def _as_array(returns: ArrayLike) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    if values.size == 0:
        raise ValueError("returns cannot be empty")
    return values


def cvar(returns: ArrayLike, alpha: float = 0.95) -> float:
    """Conditional value-at-risk (expected shortfall) reported as a positive loss.

    ``alpha`` is the confidence level; the tail is the worst ``1 - alpha`` mass.
    """
    values = _as_array(returns)
    cutoff = np.quantile(values, 1 - alpha)
    tail = values[values <= cutoff]
    if tail.size == 0:
        tail = values[values <= np.min(values)]
    return float(-tail.mean())


def expected_shortfall(returns: ArrayLike, alpha: float = 0.95) -> float:
    """Alias for :func:`cvar`."""
    return cvar(returns, alpha=alpha)


def value_at_risk(returns: ArrayLike, alpha: float = 0.95) -> float:
    """Historical VaR reported as a positive loss at confidence ``alpha``."""
    values = _as_array(returns)
    return float(-np.quantile(values, 1 - alpha))


def max_drawdown(returns: ArrayLike) -> float:
    values = _as_array(returns)
    wealth = np.cumprod(1 + values)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1
    return float(-drawdown.min())


def annualised_return(returns: ArrayLike, periods_per_year: int = 52) -> float:
    values = _as_array(returns)
    growth = np.prod(1 + values)
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / values.size) - 1)


def annualised_volatility(returns: ArrayLike, periods_per_year: int = 52) -> float:
    values = _as_array(returns)
    return float(values.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(returns: ArrayLike, periods_per_year: int = 52, risk_free: float = 0.0) -> float:
    values = _as_array(returns) - risk_free / periods_per_year
    std = values.std(ddof=1)
    if std == 0:
        return 0.0
    return float(values.mean() / std * np.sqrt(periods_per_year))


def sortino(returns: ArrayLike, periods_per_year: int = 52, risk_free: float = 0.0) -> float:
    values = _as_array(returns) - risk_free / periods_per_year
    downside = values[values < 0]
    if downside.size == 0:
        return float("inf") if values.mean() > 0 else 0.0
    downside_dev = np.sqrt(np.mean(downside**2))
    if downside_dev == 0:
        return 0.0
    return float(values.mean() / downside_dev * np.sqrt(periods_per_year))


def calmar(returns: ArrayLike, periods_per_year: int = 52) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return float(annualised_return(returns, periods_per_year) / mdd)


def hit_rate(returns: ArrayLike) -> float:
    values = _as_array(returns)
    return float((values > 0).mean())


def summarise(
    returns: ArrayLike,
    periods_per_year: int = 52,
    risk_free: float = 0.0,
    cvar_alphas: tuple[float, ...] = (0.95, 0.99),
) -> dict[str, float]:
    """Core performance/risk metrics for a single return series."""
    out = {
        "ann_return": annualised_return(returns, periods_per_year),
        "ann_vol": annualised_volatility(returns, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year, risk_free),
        "sortino": sortino(returns, periods_per_year, risk_free),
        "calmar": calmar(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": hit_rate(returns),
    }
    for alpha in cvar_alphas:
        out[f"cvar_{int(round(alpha * 100))}"] = cvar(returns, alpha=alpha)
    return out
