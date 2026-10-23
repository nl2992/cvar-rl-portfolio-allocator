"""Record the Lagrange multiplier over training for the mis-scaled and scaled dual.

Re-trains the two coupling arms of tab:coupling (stress split, 1500 updates, five seeds)
with exactly the settings of cr_selection_log.py and saves the per-update multiplier, so
the paper can show the trace that Finding 1 says is the only visible symptom.

Output: results/tables_camera_ready/lambda_trace.csv (arm, seed, update, lagrange)

Usage:
    PYTHONPATH=src python scripts/cr_lambda_trace.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from crlpa.evaluation.stress import stress_split
from crlpa.experiment import load_returns
from crlpa.training.differentiable import DiffConfig, train_differentiable
from crlpa.utils.config import load_config

SEEDS = [7, 13, 23, 42, 2025]
N_UPDATES = 1500
ARMS = {"mis-scaled dual": 0.001, "scaled dual": 5.0}  # dual learning rate, as in tab:coupling
OUT = Path("results/tables_camera_ready/lambda_trace.csv")


def main() -> None:
    cfg = load_config("configs/experiment_etf.yaml")
    returns, _ = load_returns(cfg)
    alpha = float(cfg.get_path("risk.cvar_alpha", 0.95))
    limit = float(cfg.get_path("risk.cvar_limit", 0.03)) * 0.4
    cvar_window = int(cfg.get_path("risk.cvar_window", 52))
    lookback = int(cfg.get_path("environment.lookback", 26))
    cost_bps = float(cfg.get_path("environment.transaction_cost_bps", 5.0))
    tr, va, _ = stress_split(returns)

    frames = []
    for arm, lr in ARMS.items():
        for seed in SEEDS:
            _, hist = train_differentiable(
                tr, cost_bps, alpha, limit, cvar_window, lookback,
                config=DiffConfig(n_updates=N_UPDATES, horizon=104, objective="return",
                                  anchor="inverse_vol", constrained=True,
                                  lagrange_lr=lr, seed=seed),
                val_returns=va,
            )
            df = pd.DataFrame({"arm": arm, "seed": seed,
                               "update": range(1, len(hist) + 1), "lagrange": hist["lagrange"].to_numpy()})
            frames.append(df)
            print(f"{arm} seed={seed}: final lambda={df['lagrange'].iloc[-1]:.4f}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.groupby(["arm", "update"]).lagrange.mean().groupby("arm").last().round(4).to_string())


if __name__ == "__main__":
    main()
