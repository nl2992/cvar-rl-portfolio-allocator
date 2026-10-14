from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_ASSETS: tuple[str, ...] = ("equity_index", "rates", "credit", "commodity", "fx")

# Per-regime annualised-ish weekly drift/vol multipliers for each asset class.
# Regimes: 0 calm bull, 1 equity selloff, 2 inflation/rates shock, 3 recovery.
_REGIME_DRIFT = {
    0: np.array([0.0030, 0.0004, 0.0012, 0.0008, 0.0001]),
    1: np.array([-0.0090, 0.0020, -0.0050, -0.0030, 0.0015]),
    2: np.array([-0.0030, -0.0040, -0.0020, 0.0040, 0.0020]),
    3: np.array([0.0035, 0.0002, 0.0025, 0.0012, -0.0005]),
}
_REGIME_VOL = {
    0: np.array([0.018, 0.006, 0.010, 0.020, 0.008]),
    1: np.array([0.045, 0.010, 0.030, 0.035, 0.012]),
    2: np.array([0.030, 0.015, 0.022, 0.030, 0.014]),
    3: np.array([0.024, 0.007, 0.014, 0.022, 0.009]),
}


def _base_correlation(n_assets: int) -> np.ndarray:
    corr = np.full((n_assets, n_assets), 0.2)
    np.fill_diagonal(corr, 1.0)
    if n_assets >= 2:  # equities and rates tend to diversify
        corr[0, 1] = corr[1, 0] = -0.3
    return corr


def make_synthetic_panel(
    n_steps: int = 520,
    assets: tuple[str, ...] = DEFAULT_ASSETS,
    seed: int = 11,
) -> tuple[pd.DataFrame, pd.Series]:
    """Regime-switching weekly returns with positive long-run drift.

    Returns a ``(returns, regimes)`` pair. The Markov chain spends most time in a
    calm bull regime but visits equity-selloff, rates-shock, and recovery regimes,
    giving the CVaR constraint something economically meaningful to control while
    leaving a positive drift for allocators to harvest.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(assets)
    corr = _base_correlation(n_assets)

    # Persistent regimes via a simple Markov transition matrix.
    transition = np.array(
        [
            [0.94, 0.03, 0.02, 0.01],
            [0.10, 0.80, 0.05, 0.05],
            [0.08, 0.04, 0.83, 0.05],
            [0.20, 0.02, 0.03, 0.75],
        ]
    )
    regimes = np.empty(n_steps, dtype=int)
    state = 0
    for t in range(n_steps):
        regimes[t] = state
        state = rng.choice(4, p=transition[state])

    returns = np.empty((n_steps, n_assets))
    for t in range(n_steps):
        drift = _REGIME_DRIFT[regimes[t]][:n_assets]
        vol = _REGIME_VOL[regimes[t]][:n_assets]
        cov = np.outer(vol, vol) * corr
        returns[t] = rng.multivariate_normal(drift, cov)

    columns = list(assets)
    return (
        pd.DataFrame(returns, columns=columns),
        pd.Series(regimes, name="regime"),
    )


def make_synthetic_returns(
    n_steps: int = 520,
    assets: tuple[str, ...] = DEFAULT_ASSETS,
    seed: int = 11,
) -> pd.DataFrame:
    """Convenience wrapper returning only the returns panel (see :func:`make_synthetic_panel`)."""
    returns, _ = make_synthetic_panel(n_steps=n_steps, assets=assets, seed=seed)
    return returns
