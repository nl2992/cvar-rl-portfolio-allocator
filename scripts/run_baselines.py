"""Run deterministic baselines over the test split and write a metrics table.

Usage:
    python scripts/run_baselines.py --config configs/experiment_etf.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crlpa.evaluation.backtest import rolling_weight_policy, run_policy, static_policy
from crlpa.experiment import load_returns, make_env, split_returns
from crlpa.policies import baselines as b
from crlpa.utils.config import load_config


def build_policies(cfg, n_assets: int, max_weight: float):
    return {
        "equal_weight": static_policy(b.equal_weight(n_assets)),
        "inverse_vol": rolling_weight_policy(lambda h: b.inverse_volatility(h)),
        "min_variance": rolling_weight_policy(lambda h: b.min_variance(h, max_weight=max_weight)),
        "mean_variance": rolling_weight_policy(
            lambda h: b.mean_variance(h, max_weight=max_weight)
        ),
        "risk_parity": rolling_weight_policy(lambda h: b.risk_parity(h, max_weight=max_weight)),
        "cvar_optimizer": rolling_weight_policy(
            lambda h: b.cvar_optimizer(h, alpha=cfg.get_path("risk.cvar_alpha", 0.95), max_weight=max_weight)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_etf.yaml")
    parser.add_argument("--out", default="results/tables/baseline_metrics_etf.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    returns, regimes = load_returns(cfg)
    splits = split_returns(cfg, returns, regimes)
    ppy = int(cfg.get_path("backtest.periods_per_year", 52))
    max_weight = float(cfg.get_path("constraints.max_weight", 1.0))

    policies = build_policies(cfg, returns.shape[1], max_weight)
    rows = {}
    for name, policy in policies.items():
        env = make_env(cfg, splits.test)
        result = run_policy(env, policy, periods_per_year=ppy)
        rows[name] = result.metrics
        print(f"{name:16s} sharpe={result.metrics['sharpe']:+.3f} "
              f"cvar95={result.metrics['cvar_95']:.4f} mdd={result.metrics['max_drawdown']:.3f} "
              f"breach={result.metrics['cvar_breach_rate']:.3f}")

    table = pd.DataFrame(rows).T
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
