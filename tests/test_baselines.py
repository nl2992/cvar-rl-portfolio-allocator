from __future__ import annotations

import numpy as np
import pytest

from crlpa.data.synthetic import make_synthetic_returns
from crlpa.policies import baselines as b

OPTIMISERS = ["min_variance", "mean_variance", "risk_parity", "cvar_optimizer"]


@pytest.fixture(scope="module")
def returns():
    return make_synthetic_returns(n_steps=150, seed=7)


@pytest.mark.parametrize("name", OPTIMISERS)
def test_optimisers_produce_valid_simplex(returns, name):
    w = getattr(b, name)(returns, max_weight=0.6)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-9).all()
    assert w.max() <= 0.6 + 1e-6


def test_equal_and_inverse_vol_simplex(returns):
    assert np.isclose(b.equal_weight(5).sum(), 1.0)
    iv = b.inverse_volatility(returns)
    assert np.isclose(iv.sum(), 1.0)
    assert (iv >= 0).all()


def test_min_variance_below_equal_weight_variance(returns):
    cov = np.cov(returns.to_numpy(), rowvar=False)
    eq = b.equal_weight(returns.shape[1])
    mv = b.min_variance(returns, max_weight=1.0)
    assert (mv @ cov @ mv) <= (eq @ cov @ eq) + 1e-9


def test_cvar_optimizer_respects_target_return(returns):
    target = 0.0005
    w = b.cvar_optimizer(returns, target_return=target, max_weight=1.0)
    realised_mean = float(returns.to_numpy().mean(axis=0) @ w)
    assert realised_mean >= target - 1e-4
