from __future__ import annotations

import numpy as np
import pandas as pd

from crlpa.data.build_dataset import ETF_UNIVERSE, weekly_returns_from_prices


def test_weekly_returns_from_constant_growth_prices():
    # Daily prices growing 0.1%/day for two assets over ~6 weeks.
    idx = pd.bdate_range("2020-01-01", periods=30)
    px = pd.DataFrame(
        {
            "A": 100 * (1.001 ** np.arange(30)),
            "B": 50 * (1.002 ** np.arange(30)),
        },
        index=idx,
    )
    weekly = weekly_returns_from_prices(px, rule="W-FRI")
    assert list(weekly.columns) == ["A", "B"]
    assert (weekly["A"] > 0).all()  # monotonic growth -> positive weekly returns
    assert (weekly["B"] > weekly["A"]).all()  # B grows faster
    assert not weekly.isna().any().any()


def test_weekly_returns_drops_leading_nan_row():
    idx = pd.bdate_range("2021-01-01", periods=20)
    px = pd.DataFrame({"X": np.linspace(100, 110, 20)}, index=idx)
    weekly = weekly_returns_from_prices(px)
    # pct_change's first NaN row must be dropped
    assert weekly.index[0] != px.resample("W-FRI").last().index[0]
    assert not weekly.isna().any().any()


def test_etf_universe_has_cash_leg():
    assert "BIL" in ETF_UNIVERSE
    assert ETF_UNIVERSE["BIL"] == "cash"
    assert {"equity", "rates", "credit", "commodity", "fx"} <= set(ETF_UNIVERSE.values())
