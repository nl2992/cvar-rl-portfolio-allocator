from __future__ import annotations

import numpy as np
import pandas as pd

from crlpa.data.synthetic import make_synthetic_returns
from crlpa.envs.allocation import AllocationEnv
from crlpa.features.build import build_exog
from crlpa.features.factor_features import rolling_market_beta
from crlpa.features.macro_features import build_macro_features


def test_rolling_beta_market_self_beta_is_one():
    returns = make_synthetic_returns(n_steps=120, seed=7).to_numpy()
    betas = rolling_market_beta(returns, market_col=0, window=52)
    assert betas.shape == returns.shape
    # the market's beta to itself is 1 once enough history exists
    assert np.allclose(betas[80:, 0], 1.0, atol=1e-5)


def test_rolling_beta_no_lookahead():
    base = make_synthetic_returns(n_steps=80, seed=7).to_numpy()
    corrupted = base.copy()
    corrupted[50:] = 5.0  # corrupt the future
    b1 = rolling_market_beta(base, window=20)
    b2 = rolling_market_beta(corrupted, window=20)
    assert np.allclose(b1[:50], b2[:50])  # past betas unaffected by future


def test_macro_features_aligned_and_lagged():
    dates = pd.date_range("2015-01-02", periods=60, freq="W-FRI")
    macro_daily = pd.DataFrame(
        {"vix": np.linspace(10, 30, 300)},
        index=pd.date_range("2014-12-01", periods=300, freq="B"),
    )
    feats = build_macro_features(dates, macro_daily, lag_weeks=1)
    assert len(feats) == len(dates)
    assert not feats.isna().any().any()


def test_env_exog_extends_observation():
    returns = make_synthetic_returns(n_steps=40, seed=7)
    exog = np.ones((len(returns), 3), dtype=np.float32)
    env = AllocationEnv(returns, exog=exog)
    assert env.exog_dim == 3
    assert env.obs_dim == 3 * env.n_assets + 2 + 3
    env.reset()
    assert env.observation().shape == (env.obs_dim,)


def test_env_exog_observation_no_lookahead():
    # The observation at step t must depend only on exog[t], not future exog rows.
    returns = make_synthetic_returns(n_steps=40, seed=7)
    exog_a = np.arange(len(returns) * 2, dtype=np.float32).reshape(len(returns), 2)
    exog_b = exog_a.copy()
    exog_b[20:] = -999.0  # corrupt the future
    env_a = AllocationEnv(returns, exog=exog_a)
    env_b = AllocationEnv(returns, exog=exog_b)
    env_a.reset()
    env_b.reset()
    w = np.full(returns.shape[1], 1 / returns.shape[1])
    for _ in range(10):
        env_a.step(w)
        env_b.step(w)
    assert np.allclose(env_a.observation(), env_b.observation())


def test_build_exog_shapes():
    panel = make_synthetic_returns(n_steps=60, seed=7)
    panel.index = pd.date_range("2015-01-02", periods=60, freq="W-FRI")
    macro = pd.DataFrame(
        {"vix": np.linspace(10, 30, 200)},
        index=pd.date_range("2014-12-01", periods=200, freq="B"),
    )
    exog = build_exog(panel, macro, beta_window=20)
    assert exog.shape[0] == len(panel)
    assert exog.shape[1] == panel.shape[1] + 2  # betas (n) + vix level & change
    assert np.isfinite(exog).all()
