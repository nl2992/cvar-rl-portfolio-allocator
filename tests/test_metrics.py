from __future__ import annotations

import numpy as np

from crlpa.evaluation.metrics import (
    annualised_return,
    calmar,
    cvar,
    max_drawdown,
    sharpe,
    sortino,
    summarise,
    value_at_risk,
)


def test_cvar_is_tail_mean_loss():
    returns = np.array([0.02, 0.01, -0.03, -0.08, 0.0])
    # worst 40% tail at alpha=0.6 -> {-0.08, -0.03}; mean loss = 0.055
    assert np.isclose(cvar(returns, alpha=0.6), 0.055)


def test_var_le_cvar():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.02, 500)
    assert value_at_risk(r) <= cvar(r) + 1e-9


def test_max_drawdown_positive():
    assert max_drawdown(np.array([0.1, -0.5, 0.1])) > 0


def test_sharpe_zero_variance():
    assert sharpe(np.array([0.01, 0.01, 0.01])) == 0.0


def test_sortino_only_penalises_downside():
    r = np.array([0.02, 0.02, -0.01, 0.03])
    assert sortino(r) > sharpe(r)  # downside dev < total dev when upside is large


def test_calmar_sign_matches_return():
    up = np.array([0.01, 0.02, -0.005, 0.015])
    assert np.sign(calmar(up)) == np.sign(annualised_return(up))


def test_summarise_keys():
    out = summarise(np.array([0.01, -0.02, 0.03, -0.01]))
    for key in ["ann_return", "ann_vol", "sharpe", "sortino", "calmar", "max_drawdown",
                "hit_rate", "cvar_95", "cvar_99"]:
        assert key in out
