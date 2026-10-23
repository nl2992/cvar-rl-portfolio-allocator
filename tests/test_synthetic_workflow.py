from __future__ import annotations

import numpy as np

from crlpa.data.synthetic import make_synthetic_returns
from crlpa.evaluation.metrics import cvar, max_drawdown
from crlpa.policies.baselines import inverse_volatility


def test_inverse_volatility_weights_sum_to_one() -> None:
    weights = inverse_volatility(make_synthetic_returns(n_steps=60))
    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)


def test_risk_metrics_are_positive_loss_numbers() -> None:
    returns = np.array([0.01, -0.02, -0.08, 0.03])
    assert cvar(returns, alpha=0.75) == 0.08
    assert max_drawdown(returns) > 0

