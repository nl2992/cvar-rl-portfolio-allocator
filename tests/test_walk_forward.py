from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from crlpa.data.synthetic import make_synthetic_returns  # noqa: E402
from crlpa.evaluation.metrics import summarise  # noqa: E402
from crlpa.evaluation.regimes import label_regimes, metrics_by_regime  # noqa: E402
from crlpa.evaluation.walk_forward import WalkForwardConfig, walk_forward  # noqa: E402
from crlpa.utils.config import Config  # noqa: E402


def _cfg() -> Config:
    return Config({
        "environment": {"transaction_cost_bps": 5.0, "lookback": 8},
        "constraints": {"long_only": True, "max_weight": 0.6, "turnover_cap": 0.5},
        "risk": {"cvar_alpha": 0.95, "cvar_limit": 0.03, "cvar_window": 12},
        "backtest": {"periods_per_year": 52},
    })


def test_label_regimes_covers_series():
    returns = make_synthetic_returns(n_steps=120, seed=7)
    regimes = label_regimes(returns)
    assert len(regimes) == len(returns)
    assert set(regimes.unique()) <= {"calm", "high_vol", "selloff"}


def test_metrics_by_regime_partitions():
    returns = make_synthetic_returns(n_steps=120, seed=7)
    port = returns.mean(axis=1)
    regimes = label_regimes(returns)
    table = metrics_by_regime(port, regimes, summarise, min_obs=2)
    assert table["n_weeks"].sum() <= len(port)
    assert "cvar_95" in table.columns


def test_walk_forward_runs_and_is_out_of_sample():
    returns = make_synthetic_returns(n_steps=180, seed=7)
    wf = WalkForwardConfig(train_window=60, val_window=20, test_window=20, step=20,
                           seeds=(7,), n_updates=30, include_baselines=("inverse_vol",))
    result = walk_forward(returns, _cfg(), wf)
    assert "rl_cvar_constrained" in result.oos_returns
    assert "rl_unconstrained" in result.oos_returns
    # concatenated OOS length == number of folds * test_window
    n_folds = len(result.fold_bounds)
    assert n_folds >= 2
    assert len(result.oos_returns["rl_cvar_constrained"]) == n_folds * wf.test_window
    # OOS folds must lie strictly after each fold's validation end
    for (va_end, te_end) in result.fold_bounds:
        assert te_end - va_end == wf.test_window
