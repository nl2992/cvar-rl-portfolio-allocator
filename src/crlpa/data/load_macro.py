from __future__ import annotations

import io
import time
import urllib.request

import pandas as pd

_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}&coed={end}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (research; crlpa macro loader)"}

# Default macro state: term spread, high-yield credit spread (OAS), equity vol.
DEFAULT_MACRO = {
    "T10Y2Y": "term_spread",
    "BAMLH0A0HYM2": "credit_spread",
    "VIXCLS": "vix",
}


def _fetch_chunk(series_id: str, start: str, end: str, retries: int = 4) -> pd.Series:
    url = _FRED_CSV.format(series=series_id, start=start, end=end)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(raw))
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")  # '.' -> NaN
            return df.set_index("date")["value"].dropna()
        except Exception as exc:  # network / parse errors -> retry
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch FRED {series_id} [{start}:{end}]: {last_err}")


def fetch_fred_series(
    series_id: str, start: str, end: str, retries: int = 4, chunk_years: int = 4
) -> pd.Series:
    """Fetch a FRED series as a date-indexed float Series, in chunks (no API key).

    FRED throttles large single-shot historical CSV downloads, so the range is
    split into ``chunk_years``-year requests and concatenated.
    """
    bounds = pd.date_range(start=start, end=end, freq=f"{chunk_years}YS").tolist()
    edges = [pd.Timestamp(start), *bounds, pd.Timestamp(end)]
    edges = sorted(set(e for e in edges if pd.Timestamp(start) <= e <= pd.Timestamp(end)))
    pieces = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        pieces.append(_fetch_chunk(series_id, lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d"), retries))
        time.sleep(0.2)
    series = pd.concat(pieces) if pieces else _fetch_chunk(series_id, start, end, retries)
    return series[~series.index.duplicated(keep="last")].sort_index().rename(series_id)


def load_macro_yahoo(start: str = "2008-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Macro state from Yahoo index proxies (fallback when FRED is unavailable).

    term_spread = 10y yield (^TNX) minus 13-week T-bill (^IRX); vix = ^VIX. Both
    are market-observable same-day, so no release lag beyond the 1-week shift the
    feature builder already applies.
    """
    from crlpa.data.load_prices import fetch_adjusted_close

    tnx = fetch_adjusted_close("^TNX", start, end)
    irx = fetch_adjusted_close("^IRX", start, end)
    vix = fetch_adjusted_close("^VIX", start, end)
    df = pd.concat([tnx.rename("tnx"), irx.rename("irx"), vix.rename("vix")], axis=1)
    df = df.sort_index().ffill()
    out = pd.DataFrame({"term_spread": df["tnx"] - df["irx"], "vix": df["vix"]})
    return out.dropna()


def load_macro(
    series: dict[str, str] | None = None,
    start: str = "2007-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """Load several FRED series into a daily, date-indexed, renamed DataFrame."""
    series = series or DEFAULT_MACRO
    frames = []
    for series_id, name in series.items():
        s = fetch_fred_series(series_id, start, end).rename(name)
        frames.append(s)
        time.sleep(0.3)
    return pd.concat(frames, axis=1).sort_index()
