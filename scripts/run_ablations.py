"""Ablations for the CVaR-constrained differentiable allocator (stress split).

Varies the anchor, CVaR confidence level, risk budget, and objective, training
the constrained allocator across seeds and reporting test-window risk/return.

Usage:
    python scripts/run_ablations.py --config configs/experiment_etf.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crlpa.evaluation.backtest import run_policy
from crlpa.evaluation.stress import stress_split
from crlpa.experiment import make_env
from crlpa.training.differentiable import DiffConfig, diff_policy, train_differentiable
from crlpa.utils.config import load_config


def _run(returns, cfg, seeds, updates, *, anchor, alpha, limit, objective):
    tr, va, te = stress_split(returns)
    lookback = int(cfg.get_path("environment.lookback", 26))
    cost_bps = float(cfg.get_path("environment.transaction_cost_bps", 5.0))
    cvar_window = int(cfg.get_path("risk.cvar_window", 52))
    per_seed = []
    for seed in seeds:
        actor, _ = train_differentiable(
            tr, cost_bps, alpha, limit, cvar_window, lookback,
            config=DiffConfig(n_updates=updates, horizon=104, objective=objective,
                              anchor=anchor, constrained=True, lagrange_lr=5.0, seed=seed),
            val_returns=va,
        )
        # eval CVaR uses the trained alpha for consistency
        env = make_env(cfg, te)
        env.cvar_alpha = alpha
        per_seed.append(run_policy(env, diff_policy(actor, lookback, anchor)).metrics)
    return pd.DataFrame(per_seed).mean().to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_etf.yaml")
    parser.add_argument("--out", default="results/tables_ablations")
    parser.add_argument("--seeds", type=int, nargs="*", default=[7, 13, 23])
    parser.add_argument("--updates", type=int, default=2000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    from crlpa.experiment import load_returns

    returns, _ = load_returns(cfg)
    base_limit = float(cfg.get_path("risk.cvar_limit", 0.03)) * 0.4
    seeds = tuple(args.seeds)
    common = dict(anchor="inverse_vol", alpha=0.95, limit=base_limit, objective="return")

    ablations = {
        "base": {},
        "no_anchor": {"anchor": None},
        "alpha_0.90": {"alpha": 0.90},
        "alpha_0.99": {"alpha": 0.99},
        "budget_tight": {"limit": base_limit * 0.6},
        "budget_loose": {"limit": base_limit * 1.6},
        "objective_sharpe": {"objective": "sharpe"},
    }

    rows = []
    for name, override in ablations.items():
        params = {**common, **override}
        m = _run(returns, cfg, seeds, args.updates, **params)
        rows.append({"ablation": name, **{k: params[k] for k in ("anchor", "alpha", "limit", "objective")},
                     "sharpe": m["sharpe"], "cvar_95": m["cvar_95"], "cvar_99": m["cvar_99"],
                     "max_drawdown": m["max_drawdown"], "avg_turnover": m["avg_turnover"]})
        r = rows[-1]
        print(f"{name:18s} sharpe={r['sharpe']:.3f} cvar95={r['cvar_95']:.4f} "
              f"cvar99={r['cvar_99']:.4f} maxDD={r['max_drawdown']:.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "ablation_metrics.csv", index=False)
    print(f"\nwrote {out / 'ablation_metrics.csv'}")


if __name__ == "__main__":
    main()
