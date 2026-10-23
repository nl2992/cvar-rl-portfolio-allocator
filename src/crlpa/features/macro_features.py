from __future__ import annotations

import numpy as np
import pandas as pd


def _expanding_zscore(df: pd.DataFrame, min_periods: int = 26) -> pd.DataFrame:
    """Standardise each column with expanding mean/std (no look-ahead)."""
    mean = df.expanding(min_periods=min_periods).mean()
    std = df.expanding(min_periods=min_periods).std()
    z = (df - mean) / std.replace(0, np.nan)
    return z.fillna(0.0)


def build_macro_features(
    weekly_dates: pd.DatetimeIndex,
    macro_daily: pd.DataFrame,
    lag_weeks: int = 1,
    add_changes: bool = True,
) -> pd.DataFrame:
    """Weekly macro feature panel aligned to ``weekly_dates`` (no look-ahead).

    Macro series are resampled to the weekly decision grid, **lagged by
    ``lag_weeks``** to respect release availability, optionally augmented with
    weekly changes, expanding-z-scored, and reindexed onto the return calendar.
    """
    weekly = macro_daily.resample("W-FRI").last().ffill()
    if add_changes:
        changes = weekly.diff().add_suffix("_chg")
        weekly = pd.concat([weekly, changes], axis=1)
    weekly = weekly.shift(lag_weeks)  # only information available before the decision
    weekly = weekly.reindex(weekly.index.union(weekly_dates)).ffill().reindex(weekly_dates)
    return _expanding_zscore(weekly).astype(np.float32)
