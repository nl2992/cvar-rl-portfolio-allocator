from __future__ import annotations

import pandas as pd

from crlpa.data.load_prices import load_prices

# ETF proxy universe from the project scope. BIL (1-3m T-bill ETF) is the cash leg
# and supplies a tradable risk-free return.
ETF_UNIVERSE: dict[str, str] = {
    "SPY": "equity",
    "TLT": "rates",
    "HYG": "credit",
    "DBC": "commodity",
    "GLD": "gold",
    "UUP": "fx",
    "BIL": "cash",
}


def weekly_returns_from_prices(prices: pd.DataFrame, rule: str = "W-FRI") -> pd.DataFrame:
    """Resample daily adjusted-close prices to weekly total returns.

    Prices are resampled to the last observation of each week and converted to
    simple returns, so each row is the realised return over the following week.
    The first (NaN) row is dropped.
    """
    weekly_px = prices.resample(rule).last().dropna(how="any")
    returns = weekly_px.pct_change().dropna(how="any")
    return returns


def build_etf_panel(
    tickers: list[str],
    start: str,
    end: str,
    rule: str = "W-FRI",
) -> pd.DataFrame:
    """Date-indexed weekly return panel (columns in requested ticker order)."""
    prices = load_prices(tickers, start=start, end=end)
    returns = weekly_returns_from_prices(prices, rule=rule)
    ordered = [t for t in tickers if t in returns.columns]
    return returns[ordered]


def build_etf_dataset(
    tickers: list[str],
    start: str,
    end: str,
    rule: str = "W-FRI",
) -> pd.DataFrame:
    """Fetch, align, and resample an ETF universe to a weekly return panel (index reset)."""
    return build_etf_panel(tickers, start, end, rule).reset_index(drop=True)
