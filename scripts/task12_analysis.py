"""TASK 1 + TASK 2 re-analysis.

TASK 1: breach rate / magnitude against the stated CVaR budget d=1.2% (the
budget used throughout Section 5 / tab:coupling) for the constrained learner
AND every classical comparator, on the same stress-window OOS path used
throughout that section (train pre-2018, test through 2020+2022 crisis,
stress_split(returns, train_end=560, val_end=620)). Baselines need no
training (closed-form / rolling optimisers); the constrained learner is
retrained with the exact recipe already used for tab:coupling's "scaled
dual" (fixed) row (lagrange_lr=5.0, d=1.2%, 5 seeds, 1500 updates) so the
weights match a result already reported in the paper (Sharpe 0.911, CVaR99
0.028, breach 0.751 mean over 5 seeds).

TASK 2: weight-space distance between the constrained learner and rolling
min-variance on the same window (L1/L2 per rebalance, correlation, turnover,
top-asset-agreement).
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
D_BUDGET = 0.012  # base_limit = cfg.risk.cvar_limit(0.03) * 0.4, matches tab:coupling


def rolling_breach_rate(returns: np.ndarray, limit: float, alpha: float, window: int):
    """Fraction of test weeks where rolling CVaR(alpha) > limit, + magnitude stats."""
    excess = []
    cvars = []
    for t in range(window, len(returns)):
        w_ret = returns[max(0, t - window): t]
        if len(w_ret) >= 5:
            cvar_val = compute_cvar(w_ret, alpha)
            cvars.append(cvar_val)
            excess.append(cvar_val - limit)
    excess = np.asarray(excess)
    cvars = np.asarray(cvars)
    breach_mask = excess > 1e-6
    return {
        "breach_rate": float(breach_mask.mean()) if len(excess) else float("nan"),
        "mean_breach_magnitude": float(excess[breach_mask].mean()) if breach_mask.any() else 0.0,
        "worst_breach_magnitude": float(excess.max()) if len(excess) else float("nan"),
        "worst_rolling_cvar99_like": float(cvars.max()) if len(cvars) else float("nan"),
        "n_weeks_scored": int(len(excess)),
    }


def main():
    cfg = load_config("configs/experiment_etf.yaml")
    returns, _ = load_returns(cfg)
    max_weight = float(cfg.get_path("constraints.max_weight", 0.4))
    lookback = int(cfg.get_path("environment.lookback", 26))
    cost_bps = float(cfg.get_path("environment.transaction_cost_bps", 5.0))

    tr, va, te = stress_split(returns)
    print(f"train {len(tr)}  val {len(va)}  test(stress) {len(te)} weeks")

    # ---------------- classical comparators: no training required ----------------
    baseline_fns = {
        "equal_weight": lambda h: b.equal_weight(h.shape[1]),
        "inverse_vol": lambda h: b.inverse_volatility(h),
        "min_variance": lambda h: b.min_variance(h, max_weight=max_weight),
        "risk_parity": lambda h: b.risk_parity(h, max_weight=max_weight),
        "cvar_optimizer": lambda h: b.cvar_optimizer(h, max_weight=max_weight),
    }

    breach_rows = []
    baseline_weights = {}
    baseline_returns = {}
    for name, fn in baseline_fns.items():
        policy = rolling_weight_policy(fn)
        res = run_policy(make_env(cfg, te), policy)
        baseline_weights[name] = res.weights
        baseline_returns[name] = res.returns
        stats = rolling_breach_rate(res.returns.to_numpy(), D_BUDGET, ALPHA, CVAR_WINDOW)
        stats["strategy"] = name
        stats["sharpe"] = res.metrics["sharpe"]
        breach_rows.append(stats)
        print(f"{name:16s} breach_rate={stats['breach_rate']:.3f}  "
              f"worst_excess={stats['worst_breach_magnitude']:.4f}  sharpe={stats['sharpe']:.3f}")

    # ---------------- constrained learner: retrain (matches tab:coupling "fixed") ----------------
    alpha = float(cfg.get_path("risk.cvar_alpha", 0.95))
    cvar_window = int(cfg.get_path("risk.cvar_window", 52))
    base_limit = float(cfg.get_path("risk.cvar_limit", 0.03)) * 0.4
    assert abs(base_limit - D_BUDGET) < 1e-9, (base_limit, D_BUDGET)

    con_seed_returns = {}
    con_seed_weights = {}
    con_breach_rows = []
    for seed in SEEDS:
        diff_cfg = DiffConfig(
            n_updates=N_UPDATES, horizon=104, objective="return", anchor="inverse_vol",
            constrained=True, lagrange_lr=5.0, seed=seed,
        )
        actor, history = train_differentiable(
            tr, cost_bps, alpha, base_limit, cvar_window, lookback,
            config=diff_cfg, val_returns=va,
        )
        env = make_env(cfg, te)
        env.cvar_alpha = alpha
        res = run_policy(env, diff_policy(actor, lookback, "inverse_vol"))
        con_seed_returns[seed] = res.returns
        con_seed_weights[seed] = res.weights
        stats = rolling_breach_rate(res.returns.to_numpy(), D_BUDGET, ALPHA, CVAR_WINDOW)
        stats["seed"] = seed
        stats["sharpe"] = res.metrics["sharpe"]
        con_breach_rows.append(stats)
        print(f"  rl_cvar_constrained seed={seed}: breach_rate={stats['breach_rate']:.3f}  "
              f"sharpe={stats['sharpe']:.3f}")

    con_df = pd.DataFrame(con_breach_rows)
    con_summary = {
        "strategy": "rl_cvar_constrained",
        "breach_rate": con_df["breach_rate"].mean(),
        "mean_breach_magnitude": con_df["mean_breach_magnitude"].mean(),
        "worst_breach_magnitude": con_df["worst_breach_magnitude"].max(),
        "worst_rolling_cvar99_like": con_df["worst_rolling_cvar99_like"].max(),
        "n_weeks_scored": int(con_df["n_weeks_scored"].iloc[0]),
        "sharpe": con_df["sharpe"].mean(),
    }
    breach_rows.append(con_summary)

    out_dir = Path("results/tables_reanalysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    breach_table = pd.DataFrame(breach_rows).set_index("strategy")
    breach_table.to_csv(out_dir / "breach_vs_budget_table1.csv")
    con_df.to_csv(out_dir / "constrained_breach_per_seed.csv", index=False)
    print("\n=== TASK 1: breach vs budget d=1.2% (stress window) ===")
    print(breach_table.round(4).to_string())

    # ---------------- TASK 2: weight-space distance to min-variance ----------------
    mv_weights = baseline_weights["min_variance"]  # DataFrame [T x n_assets]
    dist_rows = []
    for seed, w_con in con_seed_weights.items():
        n = min(len(w_con), len(mv_weights))
        wc = w_con.iloc[:n].to_numpy()
        wm = mv_weights.iloc[:n].to_numpy()
        l1 = np.abs(wc - wm).sum(axis=1)
        l2 = np.sqrt(((wc - wm) ** 2).sum(axis=1))
        corrs = [np.corrcoef(wc[t], wm[t])[0, 1] for t in range(n)]
        top_con = wc.argmax(axis=1)
        top_mv = wm.argmax(axis=1)
        turnover_con = np.abs(np.diff(wc, axis=0)).sum(axis=1)
        turnover_mv = np.abs(np.diff(wm, axis=0)).sum(axis=1)
        dist_rows.append({
            "seed": seed,
            "mean_L1": float(np.mean(l1)),
            "mean_L2": float(np.mean(l2)),
            "mean_corr": float(np.nanmean(corrs)),
            "frac_top_asset_differs": float(np.mean(top_con != top_mv)),
            "mean_turnover_constrained": float(np.mean(turnover_con)),
            "mean_turnover_minvar": float(np.mean(turnover_mv)),
        })
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(out_dir / "weight_distance_to_minvar_table2.csv", index=False)
    print("\n=== TASK 2: weight-space distance to rolling min-variance ===")
    print(dist_df.round(4).to_string(index=False))
    print("\nSummary (mean over 5 seeds):")
    print(dist_df.drop(columns="seed").mean().round(4).to_string())

    summary = {
        "d_budget": D_BUDGET,
        "test_window_weeks": len(te),
        "weight_distance_summary_mean": dist_df.drop(columns="seed").mean().to_dict(),
        "weight_distance_summary_std": dist_df.drop(columns="seed").std().to_dict(),
    }
    (out_dir / "task2_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote tables to {out_dir}")


if __name__ == "__main__":
    main()
