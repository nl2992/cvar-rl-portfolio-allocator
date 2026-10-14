from __future__ import annotations

import numpy as np


def rolling_market_beta(
    returns: np.ndarray,
    market_col: int = 0,
    window: int = 52,
) -> np.ndarray:
    """Rolling beta of each asset to the market column, using only returns[:t].

    Row ``t`` holds betas estimated from ``returns[max(0, t-window):t]`` (a strictly
    backward window), so the feature is free of look-ahead. Early rows (insufficient
    history) are zero.
    """
    t_total, n = returns.shape
    betas = np.zeros((t_total, n), dtype=np.float32)
    for t in range(t_total):
        hist = returns[max(0, t - window) : t]
        if hist.shape[0] < 5:
            continue
        market = hist[:, market_col]
        var = market.var()
        if var <= 1e-12:
            continue
        cov = ((hist - hist.mean(axis=0)) * (market - market.mean())[:, None]).mean(axis=0)
        betas[t] = cov / var
    return betas
