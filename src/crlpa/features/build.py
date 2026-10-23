from __future__ import annotations

import numpy as np
import pandas as pd

from crlpa.features.factor_features import rolling_market_beta
from crlpa.features.macro_features import build_macro_features


def build_exog(
    panel: pd.DataFrame,
    macro_daily: pd.DataFrame | None = None,
    market_col: int = 0,
    beta_window: int = 52,
    lag_weeks: int = 1,
    include_betas: bool = True,
) -> np.ndarray:
    """Build a per-step exogenous feature matrix aligned to a dated return panel.

    Combines lagged macro covariates (if provided) and rolling market betas into a
    ``[T, k]`` array, one row per return step, all using only information available
    before each decision (see the component builders for look-ahead controls).
    """
    blocks: list[np.ndarray] = []
    if macro_daily is not None:
        macro = build_macro_features(panel.index, macro_daily, lag_weeks=lag_weeks)
        blocks.append(macro.to_numpy(dtype=np.float32))
    if include_betas:
        blocks.append(rolling_market_beta(panel.to_numpy(), market_col, beta_window))
    if not blocks:
        return np.zeros((len(panel), 0), dtype=np.float32)
    exog = np.concatenate(blocks, axis=1).astype(np.float32)
    return np.nan_to_num(exog, nan=0.0, posinf=0.0, neginf=0.0)
