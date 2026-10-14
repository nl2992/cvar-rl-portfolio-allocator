from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from crlpa.data.synthetic import make_synthetic_panel
from crlpa.envs.allocation import AllocationEnv
from crlpa.envs.constraints import PortfolioConstraints
from crlpa.models.cvar_actor_critic import ActorCriticConfig
from crlpa.training.train_allocator import TrainConfig
from crlpa.utils.config import Config


@dataclass
class DataSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    full: pd.DataFrame
    regimes: pd.Series | None


def load_returns(cfg: Config) -> tuple[pd.DataFrame, pd.Series | None]:
    data = cfg.get_path("data", {})
    source = data.get("source", "synthetic")
    if source == "synthetic":
        returns, regimes = make_synthetic_panel(
            n_steps=int(data.get("n_steps", 520)),
            assets=tuple(cfg.get_path("universe.assets")),
            seed=int(cfg.get("seed", 42)),
        )
        return returns, regimes
    if source == "parquet":
        path = Path(data["parquet_path"])
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Build it first with: "
                "python scripts/build_dataset.py --config <experiment config>"
            )
        returns = pd.read_parquet(path)
        return returns, None
    if source == "yahoo":
        from crlpa.data.build_dataset import ETF_UNIVERSE, build_etf_dataset

        tickers = list(data.get("tickers", list(ETF_UNIVERSE.keys())))
        returns = build_etf_dataset(
            tickers,
            start=str(data.get("start", "2008-01-01")),
            end=str(data.get("end", "2024-12-31")),
            rule=str(data.get("weekly_rule", "W-FRI")),
        )
        return returns, None
    raise ValueError(f"unknown data source: {source}")


def split_returns(cfg: Config, returns: pd.DataFrame, regimes: pd.Series | None) -> DataSplits:
    data = cfg.get_path("data", {})
    n = len(returns)
    n_train = int(n * float(data.get("train_frac", 0.6)))
    n_val = int(n * float(data.get("val_frac", 0.2)))
    train = returns.iloc[:n_train].reset_index(drop=True)
    val = returns.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test = returns.iloc[n_train + n_val :].reset_index(drop=True)
    return DataSplits(train=train, val=val, test=test, full=returns, regimes=regimes)


def make_constraints(cfg: Config) -> PortfolioConstraints:
    c = cfg.get_path("constraints", {})
    return PortfolioConstraints(
        long_only=bool(c.get("long_only", True)),
        max_weight=float(c.get("max_weight", 1.0)),
        cash_floor=float(c.get("cash_floor", 0.0)),
        cash_index=c.get("cash_index", None),
        turnover_cap=c.get("turnover_cap", None),
        gross_cap=float(c.get("gross_cap", 1.0)),
    )


def make_env(cfg: Config, returns: pd.DataFrame) -> AllocationEnv:
    env_cfg = cfg.get_path("environment", {})
    risk = cfg.get_path("risk", {})
    return AllocationEnv(
        returns,
        transaction_cost_bps=float(env_cfg.get("transaction_cost_bps", 2.0)),
        constraints=make_constraints(cfg),
        cvar_alpha=float(risk.get("cvar_alpha", 0.95)),
        cvar_limit=float(risk.get("cvar_limit", 0.03)),
        cvar_window=int(risk.get("cvar_window", 52)),
        lookback=int(env_cfg.get("lookback", 26)),
    )


def make_agent_config(cfg: Config, constrained: bool = True) -> ActorCriticConfig:
    m = cfg.get_path("model", {})
    return ActorCriticConfig(
        hidden=tuple(m.get("hidden", (64, 64))),
        actor_lr=float(m.get("actor_lr", 3e-4)),
        critic_lr=float(m.get("critic_lr", 1e-3)),
        gamma=float(m.get("gamma", 0.95)),
        entropy_coef=float(m.get("entropy_coef", 0.01)),
        value_coef=float(m.get("value_coef", 0.5)),
        log_std_init=float(m.get("log_std_init", -0.5)),
        constrained=constrained,
        cost_mode=str(m.get("cost_mode", "breach")),
        lagrange_lr=float(m.get("lagrange_lr", 1.0)),
        cvar_budget=float(m.get("cvar_budget", 0.05)),
    )


def make_train_config(cfg: Config, seed: int) -> TrainConfig:
    t = cfg.get_path("training", {})
    return TrainConfig(
        n_episodes=int(t.get("n_episodes", 150)),
        turnover_penalty=float(t.get("turnover_penalty", 0.0)),
        entropy_decay=float(t.get("entropy_decay", 0.1)),
        eval_every=int(t.get("eval_every", 10)),
        seed=seed,
    )
