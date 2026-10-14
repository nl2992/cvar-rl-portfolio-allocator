from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from crlpa.data.synthetic import make_synthetic_returns  # noqa: E402
from crlpa.evaluation.stress import override_cfg, stress_split, train_eval_variant  # noqa: E402
from crlpa.utils.config import Config  # noqa: E402


def _cfg() -> Config:
    return Config({
        "environment": {"transaction_cost_bps": 5.0, "lookback": 8},
        "constraints": {"long_only": True, "max_weight": 0.6, "turnover_cap": 0.5},
        "risk": {"cvar_alpha": 0.95, "cvar_limit": 0.03, "cvar_window": 12},
        "backtest": {"periods_per_year": 52},
    })


def test_override_cfg_is_deep_and_isolated():
    cfg = _cfg()
    new = override_cfg(cfg, **{"environment.transaction_cost_bps": 15.0})
    assert new["environment"]["transaction_cost_bps"] == 15.0
    assert cfg["environment"]["transaction_cost_bps"] == 5.0  # original untouched


def test_stress_split_partitions_in_order():
    returns = make_synthetic_returns(n_steps=120, seed=7)
    tr, va, te = stress_split(returns, train_end=80, val_end=100)
    assert (len(tr), len(va), len(te)) == (80, 20, 20)


def test_train_eval_variant_returns_metrics():
    returns = make_synthetic_returns(n_steps=140, seed=7)
    tr, va, te = stress_split(returns, train_end=90, val_end=115)
    metrics = train_eval_variant(tr, va, te, _cfg(), constrained=True, limit=0.02,
                                 seeds=(7,), updates=30)
    assert {"sharpe", "cvar_99", "max_drawdown"} <= set(metrics)
