"""Camera-ready: recompute the stress-window CVaR-99 bootstrap intervals.

``scripts/run_diff_study.py`` reports a paired block bootstrap (block 4, 2000
resamples, seed 42) of CVaR-99(constrained) - CVaR-99(unconstrained) on the
seed-7 test-return series, with a *percentile* interval. For an extreme-tail
statistic (CVaR-99 averages the worst ~2 of 267 weeks) resamples often omit the
crash weeks, so the percentile interval sits away from the full-sample estimate.
This script retrains seed 7 exactly as ``run_diff_study.py`` does, checks the
percentile interval reproduces the reported one, and adds the basic and BCa
intervals computed from the same resamples.

Usage:
    python scripts/cr_bootstrap_intervals.py --universe etf
    python scripts/cr_bootstrap_intervals.py --universe sector
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from crlpa.evaluation.backtest import run_policy
from crlpa.evaluation.bootstrap import paired_bootstrap
from crlpa.experiment import load_returns, make_env
from crlpa.training.differentiable import DiffConfig, diff_policy, train_differentiable
from crlpa.utils.config import load_config

CONFIGS = {"etf": "configs/experiment_etf.yaml", "sector": "configs/experiment_sector.yaml"}
# Reported percentile intervals (paper Sections 7.2 and 7.5), for the reproduction check.
REPORTED = {"etf": (-0.032980381413746526, -0.03474649072854247, -0.007645280644879828),
            "sector": (-0.044, -0.050, -0.014)}


def cvar99(x: np.ndarray) -> float:
    """Identical to the statistic in run_diff_study.py."""
    arr = np.sort(np.asarray(x))
    return float(-np.mean(arr[: max(1, int(0.01 * len(arr)))]))


def main(universe: str, seed: int = 7, updates: int = 2500, train_end: int = 560,
         val_end: int = 620, constrained_limit: float = 0.012) -> None:
    cfg = load_config(CONFIGS[universe])
    R, _ = load_returns(cfg)
    lookback = int(cfg.get_path("environment.lookback", 26))
    cost_bps = float(cfg.get_path("environment.transaction_cost_bps", 5.0))
    alpha = float(cfg.get_path("risk.cvar_alpha", 0.95))
    cvar_window = int(cfg.get_path("risk.cvar_window", 52))
    train = R.iloc[:train_end].reset_index(drop=True)
    val = R.iloc[train_end:val_end].reset_index(drop=True)
    test = R.iloc[val_end:].reset_index(drop=True)

    series = {}
    for name, constrained, lim in (("rl_unconstrained", False, 0.03),
                                   ("rl_cvar_constrained", True, constrained_limit)):
        actor, _ = train_differentiable(
            train, cost_bps, alpha, lim, cvar_window, lookback,
            config=DiffConfig(n_updates=updates, horizon=104, objective="return",
                              risk_aversion=0.0, constrained=constrained,
                              lagrange_lr=5.0, seed=seed),
            val_returns=val,
        )
        res = run_policy(make_env(cfg, test), diff_policy(actor, lookback))
        series[name] = res.returns
        print(f"{universe} {name} seed={seed}: cvar99={cvar99(res.returns):.4f} "
              f"sharpe={res.metrics['sharpe']:.4f}", flush=True)

    rows = []
    for method in ("percentile", "basic", "bca"):
        bs = paired_bootstrap(series["rl_cvar_constrained"], series["rl_unconstrained"],
                              statistic=cvar99, block_size=4, n_resamples=2000, method=method)
        rows.append({"universe": universe, "method": method, "diff": bs.point_estimate,
                     "ci_low": bs.ci_low, "ci_high": bs.ci_high, "p_value": bs.p_value,
                     "n_weeks": len(series["rl_cvar_constrained"])})
    df = pd.DataFrame(rows)
    rep = REPORTED[universe]
    pct = df[df.method == "percentile"].iloc[0]
    print(f"reported (percentile): diff={rep[0]:.4f} ci=[{rep[1]:.4f}, {rep[2]:.4f}]")
    print(f"recomputed (percentile): diff={pct['diff']:.4f} ci=[{pct.ci_low:.4f}, {pct.ci_high:.4f}]")
    print(df.round(4).to_string(index=False))
    out = Path("results/tables_camera_ready")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"bootstrap_intervals_{universe}.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--universe", choices=list(CONFIGS), required=True)
    main(p.parse_args().universe)
