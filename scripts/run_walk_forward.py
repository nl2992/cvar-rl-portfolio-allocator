"""Rolling walk-forward of the differentiable allocator across regimes.

Refits the allocator on a rolling window, concatenates the out-of-sample folds
into one OOS path per strategy, and reports overall metrics, per-regime metrics,
and a fold-level paired test of the CVaR constraint's effect.

Usage:
    python scripts/run_walk_forward.py --config configs/experiment_etf.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from crlpa.evaluation.metrics import summarise
from crlpa.evaluation.regimes import label_regimes, metrics_by_regime
from crlpa.evaluation.walk_forward import WalkForwardConfig, walk_forward
from crlpa.experiment import load_returns
from crlpa.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_etf.yaml")
    parser.add_argument("--out", default="results/tables_wf")
    parser.add_argument("--seeds", type=int, nargs="*", default=[7, 13, 23])
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--train_window", type=int, default=312)
    parser.add_argument("--test_window", type=int, default=52)
    args = parser.parse_args()

    cfg = load_config(args.config)
    returns, _ = load_returns(cfg)
    ppy = int(cfg.get_path("backtest.periods_per_year", 52))
    wf = WalkForwardConfig(
        train_window=args.train_window, test_window=args.test_window, step=args.test_window,
        seeds=tuple(args.seeds), n_updates=args.updates,
        constrained_limit=float(cfg.get_path("risk.cvar_limit", 0.03)) * 0.4,
        max_weight=float(cfg.get_path("constraints.max_weight", 0.4)),
    )
    print(f"walk-forward: train {wf.train_window}  val {wf.val_window}  test {wf.test_window}  "
          f"step {wf.step}  seeds {wf.seeds}")
    result = walk_forward(returns, cfg, wf)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # overall OOS metrics across all concatenated folds
    overall = {name: summarise(s, periods_per_year=ppy) for name, s in result.oos_returns.items()}
    for name, s in result.oos_returns.items():
        overall[name]["n_weeks"] = len(s)
    table = pd.DataFrame(overall).T
    table.to_csv(out / "walkforward_oos_metrics.csv")
    cols = ["sharpe", "ann_return", "max_drawdown", "cvar_95", "cvar_99", "n_weeks"]
    print("\n=== Concatenated OOS metrics ===")
    print(table[[c for c in cols if c in table.columns]].round(4).to_string())

    # fold-level paired test: does the constraint cut CVaR-99 fold by fold?
    fc = result.fold_metrics.get("rl_cvar_constrained")
    fu = result.fold_metrics.get("rl_unconstrained")
    if fc is not None and fu is not None:
        merged = fc.merge(fu, on="fold", suffixes=("_con", "_unc"))
        diffs = merged["cvar_99_con"] - merged["cvar_99_unc"]
        tstat, pval = stats.wilcoxon(diffs) if len(diffs) >= 6 else (np.nan, np.nan)
        fold_test = pd.DataFrame([{
            "n_folds": len(diffs),
            "mean_cvar99_con": merged["cvar_99_con"].mean(),
            "mean_cvar99_unc": merged["cvar_99_unc"].mean(),
            "mean_diff": diffs.mean(),
            "folds_improved": int((diffs < 0).sum()),
            "wilcoxon_p": pval,
        }])
        fold_test.to_csv(out / "walkforward_fold_test.csv", index=False)
        print("\n=== Fold-level CVaR-99 paired test (constrained - unconstrained) ===")
        print(fold_test.round(4).to_string(index=False))

    # per-regime metrics on the OOS path
    regimes_full = label_regimes(returns)
    for name in ["rl_cvar_constrained", "rl_unconstrained", *wf.include_baselines]:
        if name not in result.oos_returns:
            continue
        oos_len = len(result.oos_returns[name])
        regimes_oos = regimes_full.iloc[len(returns) - oos_len :].reset_index(drop=True)
        by_regime = metrics_by_regime(result.oos_returns[name], regimes_oos, summarise)
        by_regime.to_csv(out / f"walkforward_regime_{name}.csv")
        print(f"\n=== {name}: per-regime ===")
        print(by_regime[[c for c in ["sharpe", "cvar_95", "cvar_99", "max_drawdown", "n_weeks"] if c in by_regime.columns]].round(4).to_string())

    print(f"\nwrote tables to {out}")


if __name__ == "__main__":
    main()
