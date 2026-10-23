from __future__ import annotations

import numpy as np
import pandas as pd


def label_regimes(
    returns: pd.DataFrame,
    market_col: str | int = 0,
    vol_window: int = 13,
    selloff_threshold: float = -0.05,
    vol_quantile: float = 0.75,
) -> pd.Series:
    """Label each week by market regime using only contemporaneous market state.

    Regimes (priority order): ``selloff`` if the trailing ``vol_window`` cumulative
    market return is below ``selloff_threshold``; otherwise ``high_vol`` if trailing
    market volatility is in the top ``vol_quantile`` of the sample; otherwise
    ``calm``. The market series defaults to the first column (broad equity).
    """
    market = returns.iloc[:, market_col] if isinstance(market_col, int) else returns[market_col]
    trailing_ret = (1 + market).rolling(vol_window).apply(np.prod, raw=True) - 1
    trailing_vol = market.rolling(vol_window).std()
    vol_cut = trailing_vol.quantile(vol_quantile)

    labels = np.full(len(market), "calm", dtype=object)
    labels[(trailing_vol >= vol_cut).to_numpy()] = "high_vol"
    labels[(trailing_ret <= selloff_threshold).to_numpy()] = "selloff"
    return pd.Series(labels, index=returns.index, name="regime")


def metrics_by_regime(
    returns: pd.Series,
    regimes: pd.Series,
    summarise_fn,
    min_obs: int = 10,
) -> pd.DataFrame:
    """Compute a metrics summary per regime label for a strategy return series."""
    returns = returns.reset_index(drop=True)
    regimes = regimes.reset_index(drop=True)
    rows = {}
    for label in sorted(regimes.unique()):
        mask = (regimes == label).to_numpy()
        if mask.sum() < min_obs:
            continue
        rows[label] = summarise_fn(returns[mask])
        rows[label]["n_weeks"] = int(mask.sum())
    return pd.DataFrame(rows).T
