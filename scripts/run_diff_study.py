"""Differentiable-allocator stress study (train pre-crisis, test crisis).

Trains the differentiable allocator (return-greedy unconstrained vs. CVaR-
constrained) across seeds on a pre-2018 window and evaluates on a stress window
that contains the 2020 COVID crash and the 2022 selloff, alongside adaptive
baselines. Demonstrates the CVaR constraint's tail-risk reduction versus
unconstrained RL where breaches actually occur.

Usage:
    python scripts/run_diff_study.py --config configs/experiment_etf.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from crlpa.evaluation.backtest import rolling_weight_policy, run_policy, static_policy
from crlpa.evaluation.bootstrap import paired_bootstrap
from crlpa.experiment import load_returns, make_env
from crlpa.policies import baselines as b
from crlpa.training.differentiable import DiffConfig, diff_policy, train_differentiable
from crlpa.utils.config import load_config

METRIC_COLS = ["sharpe", "ann_return", "max_drawdown", "cvar_95", "cvar_99",
               "cvar_breach_rate", "avg_turnover"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_etf.yaml")
    parser.add_argument("--out", default="results/tables_diff")
    parser.add_argument("--seeds", type=int, nargs="*", default=[7, 13, 23, 42, 2025])
    parser.add_argument("--updates", type=int, default=2500)
    parser.add_argument("--train_end", type=int, default=560)
    parser.add_argument("--val_end", type=int, default=620)
    parser.add_argument("--constrained_limit", type=float, default=0.012)
    args = parser.parse_args()

    cfg = load_config(args.config)
    R, _ = load_returns(cfg)
    lookback = int(cfg.get_path("environment.lookback", 26))
    cost_bps = float(cfg.get_path("environment.transaction_cost_bps", 5.0))
    alpha = float(cfg.get_path("risk.cvar_alpha", 0.95))
    cvar_window = int(cfg.get_path("risk.cvar_window", 52))
    max_weight = float(cfg.get_path("constraints.max_weight", 0.4))

    train = R.iloc[: args.train_end].reset_index(drop=True)
    val = R.iloc[args.train_end : args.val_end].reset_index(drop=True)
    test = R.iloc[args.val_end :].reset_index(drop=True)
    print(f"train {len(train)}  val {len(val)}  test(stress) {len(test)} weeks")

    rows: dict[str, dict] = {}
    test_returns: dict[str, pd.Series] = {}

    # adaptive baselines on the stress window
    baselines = {
        "equal_weight": static_policy(b.equal_weight(R.shape[1])),
        "inverse_vol": rolling_weight_policy(lambda h: b.inverse_volatility(h)),
        "min_variance": rolling_weight_policy(lambda h: b.min_variance(h, max_weight=max_weight)),
        "cvar_optimizer": rolling_weight_policy(lambda h: b.cvar_optimizer(h, max_weight=max_weight)),
    }
    for name, policy in baselines.items():
        res = run_policy(make_env(cfg, test), policy)
        rows[name] = res.metrics
        test_returns[name] = res.returns

    # differentiable allocators across seeds
    variants = {
        "rl_unconstrained": dict(constrained=False, lim=0.03),
        "rl_cvar_constrained": dict(constrained=True, lim=args.constrained_limit),
    }
    for name, spec in variants.items():
        per_seed, series0 = [], None
        for seed in args.seeds:
            actor, _ = train_differentiable(
                train, cost_bps, alpha, spec["lim"], cvar_window, lookback,
                config=DiffConfig(n_updates=args.updates, horizon=104, objective="return",
                                  risk_aversion=0.0, constrained=spec["constrained"],
                                  lagrange_lr=5.0, seed=seed),
                val_returns=val,
            )
            res = run_policy(make_env(cfg, test), diff_policy(actor, lookback))
            per_seed.append(res.metrics)
            if series0 is None:
                series0 = res.returns
        rows[name] = pd.DataFrame(per_seed).mean().to_dict()
        test_returns[name] = series0
        print(f"{name}: " + "  ".join(f"{k}={rows[name][k]:.4f}" for k in METRIC_COLS))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).T
    table.to_csv(out / "stress_metrics.csv")
    print("\n" + table[METRIC_COLS].round(4).to_string())

    # constraint effect: paired bootstrap of CVaR-99 (constrained - unconstrained)
    if {"rl_cvar_constrained", "rl_unconstrained"} <= set(test_returns):

        def cvar99(x: np.ndarray) -> float:
            arr = np.sort(np.asarray(x))
            return float(-np.mean(arr[: max(1, int(0.01 * len(arr)))]))

        bs = paired_bootstrap(test_returns["rl_cvar_constrained"], test_returns["rl_unconstrained"],
                              statistic=cvar99, block_size=4, n_resamples=2000)
        comp = pd.DataFrame([{
            "comparison": "cvar99_constrained_minus_unconstrained",
            "diff": bs.point_estimate, "ci_low": bs.ci_low, "ci_high": bs.ci_high,
            "p_value": bs.p_value,
        }])
        comp.to_csv(out / "stress_constraint_test.csv", index=False)
        print("\n" + comp.round(4).to_string(index=False))

    print(f"\nwrote tables to {out}")


if __name__ == "__main__":
    main()
