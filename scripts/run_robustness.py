"""Robustness suite for the CVaR constraint's tail-risk reduction.

Re-runs the stress study (return-greedy unconstrained vs. CVaR-constrained
differentiable allocator) under perturbations — transaction costs, CVaR limit,
and a reduced universe — and checks that the constraint still cuts CVaR-99 and
max drawdown versus unconstrained RL.

Usage:
    python scripts/run_robustness.py --config configs/experiment_etf.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crlpa.evaluation.stress import override_cfg, stress_split, train_eval_variant
from crlpa.experiment import load_returns
from crlpa.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_etf.yaml")
    parser.add_argument("--out", default="results/tables_robustness")
    parser.add_argument("--seeds", type=int, nargs="*", default=[7, 13, 23])
    parser.add_argument("--updates", type=int, default=2000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_returns, _ = load_returns(cfg)
    base_cost = float(cfg.get_path("environment.transaction_cost_bps", 5.0))
    tight = float(cfg.get_path("risk.cvar_limit", 0.03)) * 0.4
    seeds = tuple(args.seeds)

    # (label, cfg-overrides, returns, constrained_limit)
    scenarios = [
        ("baseline", {}, base_returns, tight),
        ("cost_2x", {"environment.transaction_cost_bps": base_cost * 2}, base_returns, tight),
        ("cost_3x", {"environment.transaction_cost_bps": base_cost * 3}, base_returns, tight),
        ("limit_tight", {}, base_returns, tight * 0.7),
        ("limit_loose", {}, base_returns, tight * 1.5),
        ("drop_gold", {"constraints.cash_index": None}, base_returns.drop(columns=["GLD"], errors="ignore"), tight),
        ("drop_commodity", {"constraints.cash_index": None}, base_returns.drop(columns=["DBC"], errors="ignore"), tight),
    ]

    rows = []
    for label, overrides, returns, limit in scenarios:
        scfg = override_cfg(cfg, **overrides) if overrides else cfg
        train, val, test = stress_split(returns)
        unc = train_eval_variant(train, val, test, scfg, False, 0.03, seeds, args.updates)
        con = train_eval_variant(train, val, test, scfg, True, limit, seeds, args.updates)
        rows.append({
            "scenario": label,
            "unc_sharpe": unc["sharpe"], "con_sharpe": con["sharpe"],
            "unc_cvar99": unc["cvar_99"], "con_cvar99": con["cvar_99"],
            "cvar99_reduction": unc["cvar_99"] - con["cvar_99"],
            "unc_maxdd": unc["max_drawdown"], "con_maxdd": con["max_drawdown"],
            "maxdd_reduction": unc["max_drawdown"] - con["max_drawdown"],
            "constraint_helps": bool(con["cvar_99"] < unc["cvar_99"]),
        })
        r = rows[-1]
        print(f"{label:16s} CVaR99 {r['unc_cvar99']:.4f}->{r['con_cvar99']:.4f} "
              f"(cut {r['cvar99_reduction']:+.4f})  maxDD {r['unc_maxdd']:.3f}->{r['con_maxdd']:.3f}  "
              f"helps={r['constraint_helps']}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "robustness_metrics.csv", index=False)
    n_help = int(table["constraint_helps"].sum())
    print(f"\nconstraint reduced CVaR-99 in {n_help}/{len(table)} scenarios")
    print(f"wrote {out / 'robustness_metrics.csv'}")


if __name__ == "__main__":
    main()
