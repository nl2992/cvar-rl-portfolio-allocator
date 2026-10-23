"""Feature study: does adding macro + factor state help the allocator?

Compares the CVaR-constrained differentiable allocator trained on market state
only vs. market state augmented with lagged macro covariates (term spread, credit
spread, VIX) and rolling market betas, on the stress test window.

Usage:
    python scripts/run_macro_study.py --config configs/experiment_etf.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crlpa.data.build_dataset import ETF_UNIVERSE, build_etf_panel
from crlpa.data.load_macro import load_macro
from crlpa.envs.allocation import AllocationEnv
from crlpa.envs.constraints import PortfolioConstraints
from crlpa.evaluation.backtest import run_policy
from crlpa.features.build import build_exog
from crlpa.training.differentiable import DiffConfig, diff_policy, train_differentiable
from crlpa.utils.config import load_config


def _env(returns, cfg, exog):
    c = cfg.get_path("constraints", {})
    return AllocationEnv(
        returns,
        transaction_cost_bps=float(cfg.get_path("environment.transaction_cost_bps", 5.0)),
        constraints=PortfolioConstraints(
            long_only=True, max_weight=float(c.get("max_weight", 0.4)),
            turnover_cap=c.get("turnover_cap", 0.5),
        ),
        cvar_alpha=float(cfg.get_path("risk.cvar_alpha", 0.95)),
        cvar_limit=float(cfg.get_path("risk.cvar_limit", 0.03)),
        cvar_window=int(cfg.get_path("risk.cvar_window", 52)),
        lookback=int(cfg.get_path("environment.lookback", 26)),
        exog=exog,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_etf.yaml")
    parser.add_argument("--out", default="results/tables_macro")
    parser.add_argument("--seeds", type=int, nargs="*", default=[7, 13, 23])
    parser.add_argument("--updates", type=int, default=2000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tickers = list(cfg.get_path("data.tickers", list(ETF_UNIVERSE.keys())))
    start, end = str(cfg.get_path("data.start", "2008-01-01")), str(cfg.get_path("data.end", "2024-12-31"))

    panel = build_etf_panel(tickers, start, end, str(cfg.get_path("data.weekly_rule", "W-FRI")))
    returns = panel.reset_index(drop=True)
    macro, macro_src = None, "none"
    try:
        macro, macro_src = load_macro(start="2007-01-01", end=end), "fred"
    except Exception as exc:  # FRED throttling/outage -> Yahoo proxy
        print(f"WARNING: FRED macro fetch failed ({exc}); trying Yahoo proxies")
        try:
            from crlpa.data.load_macro import load_macro_yahoo

            macro, macro_src = load_macro_yahoo(start="2008-01-01", end=end), "yahoo"
        except Exception as exc2:
            print(f"WARNING: Yahoo macro failed ({exc2}); factor betas only")
    exog_full = build_exog(panel, macro, market_col=0,
                           beta_window=int(cfg.get_path("risk.cvar_window", 52)))
    augmented = f"with_macro({macro_src})+factors" if macro is not None else "with_factor_betas"
    print(f"panel {panel.shape}, exog {exog_full.shape} ({augmented})")

    tr_end, va_end = 560, 620
    lookback = int(cfg.get_path("environment.lookback", 26))
    limit = float(cfg.get_path("risk.cvar_limit", 0.03)) * 0.4
    test = returns.iloc[va_end:].reset_index(drop=True)

    rows = []
    for use_exog in (False, True):
        ex = exog_full if use_exog else None
        per_seed = []
        for seed in args.seeds:
            actor, _ = train_differentiable(
                returns.iloc[:tr_end].reset_index(drop=True),
                cost_bps=float(cfg.get_path("environment.transaction_cost_bps", 5.0)),
                cvar_alpha=float(cfg.get_path("risk.cvar_alpha", 0.95)),
                cvar_limit=limit, cvar_window=int(cfg.get_path("risk.cvar_window", 52)),
                lookback=lookback,
                config=DiffConfig(n_updates=args.updates, horizon=104, objective="return",
                                  constrained=True, lagrange_lr=5.0, seed=seed),
                val_returns=returns.iloc[tr_end:va_end].reset_index(drop=True),
                exog=None if ex is None else ex[:tr_end],
                val_exog=None if ex is None else ex[tr_end:va_end],
            )
            env = _env(test, cfg, None if ex is None else ex[va_end:])
            per_seed.append(run_policy(env, diff_policy(actor, lookback)).metrics)
        label = augmented if use_exog else "market_only"
        rows.append({"variant": label, **pd.DataFrame(per_seed).mean().to_dict()})
        r = rows[-1]
        print(f"{label:20s} sharpe={r['sharpe']:.3f} cvar95={r['cvar_95']:.4f} "
              f"cvar99={r['cvar_99']:.4f} maxDD={r['max_drawdown']:.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).set_index("variant").to_csv(out / "macro_study.csv")
    print(f"\nwrote {out / 'macro_study.csv'}")


if __name__ == "__main__":
    main()
