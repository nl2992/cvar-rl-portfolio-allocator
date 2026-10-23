"""TASK 3 -- one principled retry of the budget-feasibility fix.

Pre-specified rules (see task3_prereg.md, written before this ran):
  Rule 1: d = validation-realised CVaR_0.95 of the scaled-dual policy itself
          (trained at the tight d=1.2% budget), then retrain at that d.
  Rule 2: d = validation-realised CVaR_0.95 of the deterministic inverse-vol
          anchor (no training needed for the estimate), then train at that d.
  Rule 3: expanding/per-fold adaptive budget across a 10-fold walk-forward,
          d_fold = validation-realised CVaR_0.95 of the anchor on that fold's
          own val window; train+evaluate per fold, average OOS breach.

Selection uses validation data only; the stress test window is touched only
at final evaluation, once per rule (never for tuning).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from crlpa.evaluation.backtest import rolling_weight_policy, run_policy
from crlpa.evaluation.metrics import cvar as compute_cvar
from crlpa.evaluation.stress import stress_split
from crlpa.experiment import load_returns, make_env
from crlpa.policies import baselines as b
from crlpa.training.differentiable import DiffConfig, diff_policy, train_differentiable
from crlpa.utils.config import load_config

SEEDS = [7, 13, 23, 42, 2025]
N_UPDATES = 1500
ALPHA = 0.95
CVAR_WINDOW = 52
D_TIGHT = 0.012


def rolling_breach_rate(returns: np.ndarray, limit: float, alpha: float, window: int) -> float:
    violations, total = 0, 0
    for t in range(window, len(returns)):
        w_ret = returns[max(0, t - window): t]
        if len(w_ret) >= 5:
            cvar_val = compute_cvar(w_ret, alpha)
            violations += int(cvar_val > limit + 1e-6)
            total += 1
    return violations / max(1, total)


def train_and_eval(tr, va, te, cfg, d, seeds, updates, lookback, cost_bps, alpha, cvar_window):
    rows = []
    for seed in seeds:
        diff_cfg = DiffConfig(
            n_updates=updates, horizon=104, objective="return", anchor="inverse_vol",
            constrained=True, lagrange_lr=5.0, seed=seed,
        )
        actor, history = train_differentiable(
            tr, cost_bps, alpha, d, cvar_window, lookback, config=diff_cfg, val_returns=va,
        )
        env = make_env(cfg, te)
        env.cvar_alpha = alpha
        res = run_policy(env, diff_policy(actor, lookback, "inverse_vol"))
        breach = rolling_breach_rate(res.returns.to_numpy(), d, alpha, cvar_window)
        final_lam = float(history["lagrange"].iloc[-1]) if "lagrange" in history.columns else 0.0
        rows.append({"seed": seed, "d": d, "sharpe": res.metrics["sharpe"],
                      "cvar_99": res.metrics.get("cvar_99", np.nan),
                      "breach_rate": breach, "final_lam": final_lam})
    return pd.DataFrame(rows)


def main():
    cfg = load_config("configs/experiment_etf.yaml")
    returns, _ = load_returns(cfg)
    lookback = int(cfg.get_path("environment.lookback", 26))
    cost_bps = float(cfg.get_path("environment.transaction_cost_bps", 5.0))
    alpha = float(cfg.get_path("risk.cvar_alpha", 0.95))
    cvar_window = int(cfg.get_path("risk.cvar_window", 52))

    tr, va, te = stress_split(returns)
    out = Path("results/tables_reanalysis")
    out.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ---- Rule 1: self-consistent validation quantile ----
    print("=== Rule 1: training reference scaled-dual policy at d=1.2% ===")
    ref_diff_cfg = DiffConfig(n_updates=N_UPDATES, horizon=104, objective="return",
                               anchor="inverse_vol", constrained=True, lagrange_lr=5.0, seed=7)
    ref_actor, _ = train_differentiable(tr, cost_bps, alpha, D_TIGHT, cvar_window, lookback,
                                         config=ref_diff_cfg, val_returns=va)
    env_val = make_env(cfg, va)
    env_val.cvar_alpha = alpha
    res_val = run_policy(env_val, diff_policy(ref_actor, lookback, "inverse_vol"))
    d1 = compute_cvar(res_val.returns.to_numpy(), alpha)
    print(f"Rule 1: d_1 (validation-realised CVaR_0.95 of reference constrained policy) = {d1:.4f}")

    df1 = train_and_eval(tr, va, te, cfg, d1, SEEDS, N_UPDATES, lookback, cost_bps, alpha, cvar_window)
    df1.to_csv(out / "task3_rule1_selfconsistent.csv", index=False)
    all_results["rule1_selfconsistent_val_quantile"] = {
        "d": d1, "breach_rate_mean": df1["breach_rate"].mean(), "breach_rate_std": df1["breach_rate"].std(),
        "sharpe_mean": df1["sharpe"].mean(), "final_lam_mean": df1["final_lam"].mean(),
    }
    print(df1.round(4).to_string(index=False))

    # ---- Rule 2: anchor-achievable budget ----
    print("\n=== Rule 2: anchor-achievable budget (no training needed for estimate) ===")
    anchor_policy = rolling_weight_policy(lambda h: b.inverse_volatility(h))
    res_anchor_val = run_policy(make_env(cfg, va), anchor_policy)
    d2 = compute_cvar(res_anchor_val.returns.to_numpy(), alpha)
    print(f"Rule 2: d_2 (validation-realised CVaR_0.95 of inverse-vol anchor) = {d2:.4f}")

    df2 = train_and_eval(tr, va, te, cfg, d2, SEEDS, N_UPDATES, lookback, cost_bps, alpha, cvar_window)
    df2.to_csv(out / "task3_rule2_anchor.csv", index=False)
    all_results["rule2_anchor_achievable"] = {
        "d": d2, "breach_rate_mean": df2["breach_rate"].mean(), "breach_rate_std": df2["breach_rate"].std(),
        "sharpe_mean": df2["sharpe"].mean(), "final_lam_mean": df2["final_lam"].mean(),
    }
    print(df2.round(4).to_string(index=False))

    # ---- Rule 3: expanding / per-fold adaptive budget ----
    print("\n=== Rule 3: per-fold adaptive budget, 10-fold walk-forward ===")
    ppy = int(cfg.get_path("backtest.periods_per_year", 52))
    train_window, val_window, test_window, step = 312, 52, 52, 52
    fold_seeds = [7, 13, 23]
    fold_rows = []
    start = 0
    n = len(returns)
    fold_idx = 0
    while True:
        tr_end = start + train_window
        va_end = tr_end + val_window
        te_end = va_end + test_window
        if te_end > n:
            break
        f_tr = returns.iloc[start:tr_end].reset_index(drop=True)
        f_va = returns.iloc[tr_end:va_end].reset_index(drop=True)
        f_prefix = returns.iloc[:te_end].reset_index(drop=True)

        anchor_res_fold = run_policy(make_env(cfg, f_va), anchor_policy, periods_per_year=ppy)
        d_fold = compute_cvar(anchor_res_fold.returns.to_numpy(), alpha)

        for seed in fold_seeds:
            diff_cfg = DiffConfig(n_updates=N_UPDATES, horizon=min(104, len(f_tr) - 1),
                                   objective="return", anchor="inverse_vol", constrained=True,
                                   lagrange_lr=5.0, seed=seed)
            actor, _ = train_differentiable(f_tr, cost_bps, alpha, d_fold, cvar_window, lookback,
                                             config=diff_cfg, val_returns=f_va)
            env = make_env(cfg, f_prefix)
            env.cvar_alpha = alpha
            res = run_policy(env, diff_policy(actor, lookback, "inverse_vol"), periods_per_year=ppy)
            fold_test_returns = res.returns.iloc[va_end:te_end].to_numpy()
            breach = rolling_breach_rate(fold_test_returns, d_fold, alpha, cvar_window)
            fold_rows.append({"fold": fold_idx, "seed": seed, "d_fold": d_fold,
                               "breach_rate": breach,
                               "n_test_weeks_scored": max(0, len(fold_test_returns) - cvar_window)})
        print(f"  fold {fold_idx}: d_fold={d_fold:.4f}  "
              f"mean_breach={np.mean([r['breach_rate'] for r in fold_rows if r['fold'] == fold_idx]):.3f}")
        start += step
        fold_idx += 1

    df3 = pd.DataFrame(fold_rows)
    df3.to_csv(out / "task3_rule3_adaptive_walkforward.csv", index=False)
    scored = df3[df3["n_test_weeks_scored"] > 0]
    all_results["rule3_adaptive_per_fold"] = {
        "n_folds": fold_idx,
        "breach_rate_mean": scored["breach_rate"].mean() if len(scored) else float("nan"),
        "breach_rate_std": scored["breach_rate"].std() if len(scored) else float("nan"),
        "d_fold_mean": df3["d_fold"].mean(),
        "d_fold_min": df3["d_fold"].min(),
        "d_fold_max": df3["d_fold"].max(),
    }
    print(df3.round(4).to_string(index=False))

    print("\n=== TASK 3 SUMMARY (all rules, all reported) ===")
    print(json.dumps(all_results, indent=2, default=float))
    (out / "task3_summary.json").write_text(json.dumps(all_results, indent=2, default=float))
    print("\nBaseline for comparison (existing tab:coupling rows): "
          "tight d=1.2% breach=0.751, loose d=1.92% breach=0.780")
    print(f"wrote tables to {out}")


if __name__ == "__main__":
    main()
