"""Weekly stress-window returns of the unconstrained and constrained allocators, per seed.

Re-runs the two differentiable arms of scripts/run_diff_study.py with identical settings
(train weeks 0-560, validation 560-620, test 620 onward, 2500 updates, seeds 7/13/23/42/2025)
and keeps every seed's weekly test returns, so the paper can plot the drawdown paths behind
tab:stress. Also prints each arm's mean max drawdown as a check against tab:stress.

Output: results/tables_camera_ready/stress_paths.csv (date, arm, seed, ret)

Usage:
    PYTHONPATH=src python scripts/cr_stress_paths.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from crlpa.evaluation.backtest import run_policy
from crlpa.evaluation.metrics import max_drawdown
from crlpa.experiment import load_returns, make_env
from crlpa.training.differentiable import DiffConfig, diff_policy, train_differentiable
from crlpa.utils.config import load_config

SEEDS = [7, 13, 23, 42, 2025]
TRAIN_END, VAL_END, UPDATES = 560, 620, 2500
ARMS = {"unconstrained": dict(constrained=False, lim=0.03),
        "constrained": dict(constrained=True, lim=0.012)}
OUT = Path("results/tables_camera_ready/stress_paths.csv")


def main() -> None:
    cfg = load_config("configs/experiment_etf.yaml")
    R, _ = load_returns(cfg)
    dates = R.index[VAL_END:]
    lookback = int(cfg.get_path("environment.lookback", 26))
    cost_bps = float(cfg.get_path("environment.transaction_cost_bps", 5.0))
    alpha = float(cfg.get_path("risk.cvar_alpha", 0.95))
    cvar_window = int(cfg.get_path("risk.cvar_window", 52))
    train = R.iloc[:TRAIN_END].reset_index(drop=True)
    val = R.iloc[TRAIN_END:VAL_END].reset_index(drop=True)
    test = R.iloc[VAL_END:].reset_index(drop=True)

    frames = []
    for arm, spec in ARMS.items():
        mdd = []
        for seed in SEEDS:
            actor, _ = train_differentiable(
                train, cost_bps, alpha, spec["lim"], cvar_window, lookback,
                config=DiffConfig(n_updates=UPDATES, horizon=104, objective="return",
                                  risk_aversion=0.0, constrained=spec["constrained"],
                                  lagrange_lr=5.0, seed=seed),
                val_returns=val,
            )
            res = run_policy(make_env(cfg, test), diff_policy(actor, lookback))
            ret = res.returns.to_numpy()
            mdd.append(max_drawdown(ret))
            frames.append(pd.DataFrame({"date": dates[: len(ret)], "arm": arm, "seed": seed, "ret": ret}))
            print(f"{arm} seed={seed}: max_dd={mdd[-1]:.4f}", flush=True)
        print(f"{arm}: mean max_dd={sum(mdd) / len(mdd):.4f}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
