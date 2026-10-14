from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone

import pandas as pd

_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (research; crlpa data loader)"}


def _to_epoch(date: str) -> int:
    return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def fetch_adjusted_close(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    retries: int = 4,
    pause: float = 1.5,
) -> pd.Series:
    """Fetch an adjusted-close series for one ticker from the Yahoo chart API.

    Adjusted close folds in dividends/splits, giving a total-return price proxy.
    Retries with backoff to ride out the API's frequent rate limiting.
    """
    params = (
        f"?period1={_to_epoch(start)}&period2={_to_epoch(end)}"
        f"&interval={interval}&events=div%2Csplit"
    )
    url = _CHART_URL.format(ticker=ticker) + params
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            result = payload["chart"]["result"][0]
            timestamps = result["timestamp"]
            adj = result["indicators"]["adjclose"][0]["adjclose"]
            index = pd.to_datetime(timestamps, unit="s", utc=True).tz_localize(None).normalize()
            series = pd.Series(adj, index=index, name=ticker, dtype="float64")
            return series[~series.index.duplicated(keep="last")].dropna()
        except Exception as exc:  # network/rate-limit/parse errors -> retry
            last_err = exc
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"failed to fetch {ticker} after {retries} attempts: {last_err}")


def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    min_obs: int = 252,
) -> pd.DataFrame:
    """Load aligned adjusted-close prices for several tickers.

    Tickers with fewer than ``min_obs`` observations are dropped (insufficient
    history). The panel is aligned to the common trading calendar and small gaps
    are forward-filled (corporate actions / sparse non-trading days).
    """
    series: list[pd.Series] = []
    for ticker in tickers:
        s = fetch_adjusted_close(ticker, start, end)
        if len(s) < min_obs:
            print(f"  dropping {ticker}: only {len(s)} obs (< {min_obs})")
            continue
        series.append(s)
        time.sleep(0.5)  # be polite to the API
    if not series:
        raise RuntimeError("no tickers returned sufficient history")
    prices = pd.concat(series, axis=1).sort_index()
    prices = prices.ffill().dropna(how="any")
    return prices
