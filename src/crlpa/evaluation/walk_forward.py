from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from crlpa.evaluation.backtest import rolling_weight_policy, run_policy
from crlpa.experiment import make_env
from crlpa.policies import baselines as b
from crlpa.training.differentiable import DiffConfig, ensemble_diff_policy, train_differentiable


@dataclass
class WalkForwardConfig:
    train_window: int = 312
    val_window: int = 52
    test_window: int = 52
    step: int = 52
    seeds: tuple[int, ...] = (7, 13, 23)
    n_updates: int = 1500
    constrained_limit: float = 0.012
    unconstrained_limit: float = 0.03
    objective: str = "return"
    risk_aversion: float = 0.0
    max_weight: float = 0.4
    include_baselines: tuple[str, ...] = ("min_variance", "inverse_vol")


@dataclass
class WalkForwardResult:
    oos_returns: dict[str, pd.Series]
    fold_metrics: dict[str, pd.DataFrame]
    fold_bounds: list[tuple[int, int]] = field(default_factory=list)


def _train_ensemble(train, val, cfg, wf: WalkForwardConfig, constrained: bool, limit: float):
    lookback = int(cfg.get_path("environment.lookback", 26))
    cost_bps = float(cfg.get_path("environment.transaction_cost_bps", 5.0))
    alpha = float(cfg.get_path("risk.cvar_alpha", 0.95))
    cvar_window = int(cfg.get_path("risk.cvar_window", 52))
    actors = []
    for seed in wf.seeds:
        actor, _ = train_differentiable(
            train, cost_bps, alpha, limit, cvar_window, lookback,
            config=DiffConfig(n_updates=wf.n_updates, horizon=min(104, len(train) - 1),
                              objective=wf.objective, risk_aversion=wf.risk_aversion,
                              constrained=constrained, lagrange_lr=5.0, seed=seed),
            val_returns=val,
        )
        actors.append(actor)
    return ensemble_diff_policy(actors, lookback)


def walk_forward(returns: pd.DataFrame, cfg, wf: WalkForwardConfig | None = None) -> WalkForwardResult:
    """Rolling refit/evaluate of the allocators, concatenating out-of-sample folds.

    Each fold trains on a ``train_window``, selects on a ``val_window``, and is
    evaluated on the next ``test_window`` weeks; the window rolls forward by
    ``step``. Test environments are built on the full history up to the fold end so
    rolling features have warmup, while only the held-out fold slice is scored —
    keeping the evaluation out-of-sample and free of look-ahead.
    """
    wf = wf or WalkForwardConfig()
    ppy = int(cfg.get_path("backtest.periods_per_year", 52))
    variants = ["rl_unconstrained", "rl_cvar_constrained", *wf.include_baselines]
    oos: dict[str, list[pd.Series]] = {v: [] for v in variants}
    folds: dict[str, list[dict]] = {v: [] for v in variants}
    bounds: list[tuple[int, int]] = []

    start = 0
    n = len(returns)
    while True:
        tr_end = start + wf.train_window
        va_end = tr_end + wf.val_window
        te_end = va_end + wf.test_window
        if te_end > n:
            break
        bounds.append((va_end, te_end))
        train = returns.iloc[start:tr_end].reset_index(drop=True)
        val = returns.iloc[tr_end:va_end].reset_index(drop=True)
        prefix = returns.iloc[:te_end].reset_index(drop=True)  # full history for warmup

        policies = {
            "rl_unconstrained": _train_ensemble(train, val, cfg, wf, False, wf.unconstrained_limit),
            "rl_cvar_constrained": _train_ensemble(train, val, cfg, wf, True, wf.constrained_limit),
        }
        for name in wf.include_baselines:
            fn = {
                "min_variance": lambda h: b.min_variance(h, max_weight=wf.max_weight),
                "inverse_vol": lambda h: b.inverse_volatility(h),
                "equal_weight": lambda h: b.equal_weight(h.shape[1]),
                "cvar_optimizer": lambda h: b.cvar_optimizer(h, max_weight=wf.max_weight),
            }[name]
            policies[name] = rolling_weight_policy(fn)

        from crlpa.evaluation.metrics import summarise

        for name, policy in policies.items():
            res = run_policy(make_env(cfg, prefix), policy, periods_per_year=ppy)
            fold_slice = res.returns.iloc[va_end:te_end].reset_index(drop=True)
            oos[name].append(fold_slice)
            m = summarise(fold_slice, periods_per_year=ppy)
            m["fold"] = len(bounds) - 1
            folds[name].append(m)

        start += wf.step

    oos_returns = {v: pd.concat(s, ignore_index=True) for v, s in oos.items() if s}
    fold_metrics = {v: pd.DataFrame(rows) for v, rows in folds.items() if rows}
    return WalkForwardResult(oos_returns=oos_returns, fold_metrics=fold_metrics, fold_bounds=bounds)
