"""TASK 3, Rule 3 (corrected scoring) -- per-fold adaptive budget walk-forward.

Same pre-specified rule as before: at each fold of a 10-fold rolling
walk-forward, re-estimate the budget d_fold from that fold's OWN validation
window as the realised CVaR_0.95 of the deterministic inverse-vol anchor
(validation data only, never the test slice). Train the scaled-dual policy
per fold at d_fold and score OOS breach on that fold's test slice.

BUGFIX vs. the first run: the first version scored the 52-week test slice with
a 52-week rolling CVaR window drawn from within the slice, leaving zero
scoreable weeks (n_test_weeks_scored=0, breach trivially 0.0 -- an artifact,
not a result). Here the rolling CVaR at each test week t uses the preceding 52
weeks of the CONTINUOUS realised path (warmup drawn from before the test slice,
exactly as the 267-week stress window is scored: 267 - 52 = 215 scored weeks).
No look-ahead is introduced: the window at t uses only returns strictly before t.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from crlpa.evaluation.backtest import rolling_weight_policy, run_policy
from crlpa.evaluation.metrics import cvar as compute_cvar
from crlpa.experiment import load_returns, make_env
from crlpa.policies import baselines as b
from crlpa.training.differentiable import DiffConfig, diff_policy, train_differentiable
from crlpa.utils.config import load_config

N_UPDATES = 1500
FOLD_SEEDS = [7, 13, 23]


def main():
    cfg = load_config("configs/experiment_etf.yaml")
    returns, _ = load_returns(cfg)
    lookback = int(cfg.get_path("environment.lookback", 26))
    cost_bps = float(cfg.get_path("environment.transaction_cost_bps", 5.0))
    alpha = float(cfg.get_path("risk.cvar_alpha", 0.95))
    cvar_window = int(cfg.get_path("risk.cvar_window", 52))
    ppy = int(cfg.get_path("backtest.periods_per_year", 52))

    anchor_policy = rolling_weight_policy(lambda h: b.inverse_volatility(h))
    train_window, val_window, test_window, step = 312, 52, 52, 52

    rows = []
    start, fold_idx, n = 0, 0, len(returns)
    while True:
        tr_end = start + train_window
        va_end = tr_end + val_window
        te_end = va_end + test_window
        if te_end > n:
            break
        f_tr = returns.iloc[start:tr_end].reset_index(drop=True)
        f_va = returns.iloc[tr_end:va_end].reset_index(drop=True)
        f_prefix = returns.iloc[:te_end].reset_index(drop=True)

        # budget from THIS fold's validation window only
        anchor_val = run_policy(make_env(cfg, f_va), anchor_policy, periods_per_year=ppy)
        d_fold = compute_cvar(anchor_val.returns.to_numpy(), alpha)

        for seed in FOLD_SEEDS:
            diff_cfg = DiffConfig(
                n_updates=N_UPDATES, horizon=min(104, len(f_tr) - 1), objective="return",
                anchor="inverse_vol", constrained=True, lagrange_lr=5.0, seed=seed,
            )
            actor, history = train_differentiable(
                f_tr, cost_bps, alpha, d_fold, cvar_window, lookback,
                config=diff_cfg, val_returns=f_va,
            )
            env = make_env(cfg, f_prefix)
            env.cvar_alpha = alpha
            res = run_policy(env, diff_policy(actor, lookback, "inverse_vol"), periods_per_year=ppy)
            path = res.returns.to_numpy()  # continuous realised path over the whole prefix

            # score every test week using the preceding 52 weeks of the continuous path
            breaches, excesses = [], []
            for t in range(va_end, min(te_end, len(path))):
                w_ret = path[max(0, t - cvar_window):t]
                if len(w_ret) >= 5:
                    cv = compute_cvar(w_ret, alpha)
                    breaches.append(int(cv > d_fold + 1e-6))
                    excesses.append(cv - d_fold)
            rows.append({
                "fold": fold_idx, "seed": seed, "d_fold": d_fold,
                "breach_rate": float(np.mean(breaches)) if breaches else np.nan,
                "mean_excess_when_breaching": (
                    float(np.mean([e for e in excesses if e > 1e-6]))
                    if any(e > 1e-6 for e in excesses) else 0.0
                ),
                "n_test_weeks_scored": len(breaches),
                "final_lam": float(history["lagrange"].iloc[-1]),
            })
        fm = np.mean([r["breach_rate"] for r in rows if r["fold"] == fold_idx])
        print(f"  fold {fold_idx}: d_fold={d_fold:.4f}  mean_breach={fm:.3f}  "
              f"scored={rows[-1]['n_test_weeks_scored']}", flush=True)
        start += step
        fold_idx += 1

    df = pd.DataFrame(rows)
    out = Path("results/tables_reanalysis")
    df.to_csv(out / "task3_rule3_adaptive_walkforward.csv", index=False)

    summary = {
        "n_folds": fold_idx,
        "seeds_per_fold": len(FOLD_SEEDS),
        "breach_rate_mean": float(df["breach_rate"].mean()),
        "breach_rate_std": float(df["breach_rate"].std()),
        "mean_excess_when_breaching": float(df["mean_excess_when_breaching"].mean()),
        "d_fold_mean": float(df["d_fold"].mean()),
        "d_fold_min": float(df["d_fold"].min()),
        "d_fold_max": float(df["d_fold"].max()),
        "final_lam_mean": float(df["final_lam"].mean()),
        "total_test_weeks_scored": int(df["n_test_weeks_scored"].sum()),
        "per_fold_breach": df.groupby("fold")["breach_rate"].mean().round(4).to_dict(),
    }
    (out / "task3_rule3_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Rule 3 (corrected scoring) ===")
    print(df.round(4).to_string(index=False))
    print("\n" + json.dumps(summary, indent=2))
    print("wrote results/tables_reanalysis/task3_rule3_*")


if __name__ == "__main__":
    main()
