from __future__ import annotations

import copy

import pandas as pd

from crlpa.evaluation.backtest import run_policy
from crlpa.experiment import make_env
from crlpa.training.differentiable import DiffConfig, diff_policy, train_differentiable
from crlpa.utils.config import Config


def stress_split(returns: pd.DataFrame, train_end: int = 560, val_end: int = 620):
    """Train pre-crisis / validate / test on a stress-inclusive tail window."""
    return (
        returns.iloc[:train_end].reset_index(drop=True),
        returns.iloc[train_end:val_end].reset_index(drop=True),
        returns.iloc[val_end:].reset_index(drop=True),
    )


def override_cfg(cfg: Config, **overrides) -> Config:
    """Deep-copy a config and apply ``section.key=value`` overrides."""
    new = Config(copy.deepcopy(dict(cfg)))
    for dotted, value in overrides.items():
        section, key = dotted.split(".")
        new.setdefault(section, {})[key] = value
    return new


def train_eval_variant(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cfg: Config,
    constrained: bool,
    limit: float,
    seeds: tuple[int, ...] = (7, 13, 23),
    updates: int = 2000,
) -> dict[str, float]:
    """Train the return-greedy/CVaR-constrained allocator across seeds; mean test metrics."""
    lookback = int(cfg.get_path("environment.lookback", 26))
    cost_bps = float(cfg.get_path("environment.transaction_cost_bps", 5.0))
    alpha = float(cfg.get_path("risk.cvar_alpha", 0.95))
    cvar_window = int(cfg.get_path("risk.cvar_window", 52))
    ppy = int(cfg.get_path("backtest.periods_per_year", 52))

    per_seed = []
    for seed in seeds:
        actor, _ = train_differentiable(
            train, cost_bps, alpha, limit, cvar_window, lookback,
            config=DiffConfig(n_updates=updates, horizon=min(104, len(train) - 1),
                              objective="return", risk_aversion=0.0,
                              constrained=constrained, lagrange_lr=5.0, seed=seed),
            val_returns=val,
        )
        res = run_policy(make_env(cfg, test), diff_policy(actor, lookback), periods_per_year=ppy)
        per_seed.append(res.metrics)
    return pd.DataFrame(per_seed).mean().to_dict()
