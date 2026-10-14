"""TASK 4 -- the decision-time CVaR filter (PLAN_P5_make_the_constraint_bind.md).

Four soft-Lagrangian budget rules (tab:coupling + task3_*) all landed OOS
breach in the 71-78% band on the stress window, while classical low-volatility
rules clear the same budget in ~92% of weeks (see task12_analysis.py). This
script builds and evaluates the fix: a decision-time projection that refuses
non-compliant trades by construction (crlpa.training.risk_filter), instead of
discouraging them with a penalty.

Arms (mirrors the existing arm configuration exactly -- same seeds, same
update budget, same splits; only the named quantities change):

  A5  filter only            eta_lambda=0.001  filter ON   lam not conditioned
  A6  filter + scaled dual   eta_lambda=5.0    filter ON   lam not conditioned  <- intended method
  A7  adaptive dual          eta_lambda=5.0    filter OFF  lam conditioned, online-updated at eval
  A8  filter + adaptive dual eta_lambda=5.0    filter ON   lam conditioned, online-updated at eval

Existing arms (A0 unconstrained, A1 mis-scaled dual, A2 scaled dual) and the
four failed budget rules already have results (tab:coupling, task3_*) and are
NOT re-run here.

Methodology matches task12_analysis.py exactly so the breach-rate numbers are
directly comparable: same stress_split (train_end=560, val_end=620), same
D_BUDGET=1.2%, same alpha=0.95, same 52-week rolling CVaR window, same 5 seeds,
same 1500 updates, same horizon=104, same inverse-vol residual anchor.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from crlpa.evaluation.backtest import rolling_weight_policy, run_policy
from crlpa.evaluation.metrics import cvar as compute_cvar
from crlpa.evaluation.stress import stress_split
from crlpa.experiment import load_returns, make_env
from crlpa.policies import baselines as b
from crlpa.training.risk_filter import (
    FilteredConfig,
    diff_policy_filtered,
    gaussian_cvar_multiplier,
    precompute_filter_state,
    run_adaptive_dual_policy,
    train_filtered,
)
from crlpa.utils.config import load_config

SEEDS = [7, 13, 23, 42, 2025]
N_UPDATES = 1500
ALPHA = 0.95
CVAR_WINDOW = 52
D_BUDGET = 0.012          # identical to tab:coupling / task12_analysis / task3
FILTER_LOOKBACK = 104     # trailing window for the filter's covariance/mean/anchor;
                            # matches the min_variance baseline's own default lookback
                            # so the filter's anchor IS that classical baseline.

ARMS = {
    "A5_filter_only": dict(lagrange_lr=0.001, filter_on=True, lam_conditioned=False),
    "A6_filter_scaled_dual": dict(lagrange_lr=5.0, filter_on=True, lam_conditioned=False),
    "A7_adaptive_dual": dict(lagrange_lr=5.0, filter_on=False, lam_conditioned=True),
    "A8_filter_adaptive_dual": dict(lagrange_lr=5.0, filter_on=True, lam_conditioned=True),
}


def rolling_breach_rate(returns: np.ndarray, limit: float, alpha: float, window: int) -> dict:
    """Identical scoring to task12_analysis.rolling_breach_rate."""
    excess, cvars = [], []
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


def weight_distance(w_arm: pd.DataFrame, w_mv: pd.DataFrame) -> dict:
    n = min(len(w_arm), len(w_mv))
    wc = w_arm.iloc[:n].to_numpy()
    wm = w_mv.iloc[:n].to_numpy()
    l1 = np.abs(wc - wm).sum(axis=1)
    l2 = np.sqrt(((wc - wm) ** 2).sum(axis=1))
    corrs = [np.corrcoef(wc[t], wm[t])[0, 1] for t in range(n)]
    top_arm = wc.argmax(axis=1)
    top_mv = wm.argmax(axis=1)
    turnover_arm = np.abs(np.diff(wc, axis=0)).sum(axis=1)
    turnover_mv = np.abs(np.diff(wm, axis=0)).sum(axis=1)
    return {
        "mean_L1": float(np.mean(l1)),
        "mean_L2": float(np.mean(l2)),
        "mean_corr": float(np.nanmean(corrs)),
        "frac_top_asset_differs": float(np.mean(top_arm != top_mv)),
        "mean_turnover_arm": float(np.mean(turnover_arm)),
        "mean_turnover_minvar": float(np.mean(turnover_mv)),
    }


def main() -> None:
    t_start = time.time()
    cfg = load_config("configs/experiment_etf.yaml")
    returns, _ = load_returns(cfg)
    max_weight = float(cfg.get_path("constraints.max_weight", 0.4))
    lookback = int(cfg.get_path("environment.lookback", 26))
    cost_bps = float(cfg.get_path("environment.transaction_cost_bps", 5.0))

    tr, va, te = stress_split(returns)
    print(f"train {len(tr)}  val {len(va)}  test(stress) {len(te)} weeks")

    out_dir = Path("results/tables_reanalysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- precompute filter state (data-only, reused across all arms/seeds) --
    print("precomputing trailing covariance / mean / min-variance anchor "
          f"(lookback={FILTER_LOOKBACK}, max_weight={max_weight}) ...")
    t0 = time.time()
    fs_tr = precompute_filter_state(tr, lookback=FILTER_LOOKBACK, max_weight=max_weight)
    fs_va = precompute_filter_state(va, lookback=FILTER_LOOKBACK, max_weight=max_weight)
    fs_te = precompute_filter_state(te, lookback=FILTER_LOOKBACK, max_weight=max_weight)
    print(f"  done in {time.time() - t0:.1f}s "
          f"(active steps: train {fs_tr.active.sum()}/{len(fs_tr.active)}, "
          f"val {fs_va.active.sum()}/{len(fs_va.active)}, "
          f"test {fs_te.active.sum()}/{len(fs_te.active)})")
    c_alpha = gaussian_cvar_multiplier(ALPHA)
    print(f"Gaussian CVaR_{ALPHA} multiplier c_alpha = {c_alpha:.4f}")

    # ---------------- min-variance baseline weights on the test window (for weight-distance) --
    mv_policy = rolling_weight_policy(lambda h: b.min_variance(h, max_weight=max_weight))
    mv_res = run_policy(make_env(cfg, te), mv_policy)
    mv_weights = mv_res.weights

    all_breach_rows = []
    all_dist_rows = []
    all_metric_rows = []
    per_seed_records = []

    for arm_name, arm_cfg in ARMS.items():
        print(f"\n=== {arm_name}: {arm_cfg} ===")
        for seed in SEEDS:
            t0 = time.time()
            diff_cfg = FilteredConfig(
                n_updates=N_UPDATES, horizon=104, objective="return", anchor="inverse_vol",
                constrained=True, lagrange_lr=arm_cfg["lagrange_lr"], seed=seed,
                filter_on=arm_cfg["filter_on"], lam_conditioned=arm_cfg["lam_conditioned"],
            )
            actor, history = train_filtered(
                tr, cost_bps, ALPHA, D_BUDGET, CVAR_WINDOW, lookback, diff_cfg,
                val_returns=va,
                filter_state=fs_tr if arm_cfg["filter_on"] else None,
                val_filter_state=fs_va if arm_cfg["filter_on"] else None,
            )
            final_lam_train = float(history["lagrange"].iloc[-1])

            env = make_env(cfg, te)
            env.cvar_alpha = ALPHA
            if arm_cfg["lam_conditioned"]:
                res, lam_path = run_adaptive_dual_policy(
                    env, actor, lookback, "inverse_vol", D_BUDGET, ALPHA,
                    lagrange_lr=arm_cfg["lagrange_lr"], initial_lam=final_lam_train,
                    filter_state=fs_te if arm_cfg["filter_on"] else None,
                    filter_on=arm_cfg["filter_on"],
                )
                final_lam_eval = res.metrics["final_lam"]
            else:
                policy = diff_policy_filtered(
                    actor, lookback, "inverse_vol", fs_te, D_BUDGET, c_alpha,
                    filter_on=arm_cfg["filter_on"],
                )
                res = run_policy(env, policy)
                final_lam_eval = final_lam_train

            breach = rolling_breach_rate(res.returns.to_numpy(), D_BUDGET, ALPHA, CVAR_WINDOW)
            dist = weight_distance(res.weights, mv_weights)

            row = {
                "arm": arm_name, "seed": seed,
                "sharpe": res.metrics["sharpe"],
                "cvar_99": res.metrics.get("cvar_99", np.nan),
                "max_drawdown": res.metrics.get("max_drawdown", np.nan),
                "avg_turnover": res.metrics.get("avg_turnover", np.nan),
                "final_lam_train": final_lam_train,
                "final_lam_eval": final_lam_eval,
                "mean_shrink_train": float(history["mean_shrink"].mean()) if arm_cfg["filter_on"] else np.nan,
                **breach,
            }
            all_breach_rows.append(row)
            all_dist_rows.append({"arm": arm_name, "seed": seed, **dist})
            all_metric_rows.append(row)
            per_seed_records.append(row)
            print(f"  seed={seed}: sharpe={row['sharpe']:.4f}  cvar99={row['cvar_99']:.4f}  "
                  f"breach={row['breach_rate']:.3f}  final_lam(train/eval)="
                  f"{final_lam_train:.3f}/{final_lam_eval:.3f}  "
                  f"({time.time() - t0:.1f}s)")

    breach_df = pd.DataFrame(all_breach_rows)
    dist_df = pd.DataFrame(all_dist_rows)
    breach_df.to_csv(out_dir / "task4_filter_arms_per_seed.csv", index=False)
    dist_df.to_csv(out_dir / "task4_filter_weight_distance.csv", index=False)

    summary = (
        breach_df.groupby("arm")[
            ["sharpe", "cvar_99", "max_drawdown", "avg_turnover", "breach_rate",
             "mean_breach_magnitude", "worst_breach_magnitude", "final_lam_train",
             "final_lam_eval", "mean_shrink_train"]
        ].mean().round(4)
    )
    dist_summary = dist_df.groupby("arm")[
        ["mean_L1", "mean_L2", "mean_corr", "frac_top_asset_differs",
         "mean_turnover_arm", "mean_turnover_minvar"]
    ].mean().round(4)

    print("\n=== TASK 4 SUMMARY: arms A5-A8, mean over 5 seeds ===")
    print(summary.to_string())
    print("\n=== weight distance to rolling min-variance, mean over 5 seeds ===")
    print(dist_summary.to_string())

    summary.to_csv(out_dir / "task4_filter_arms_summary.csv")
    dist_summary.to_csv(out_dir / "task4_filter_weight_distance_summary.csv")
    (out_dir / "task4_summary.json").write_text(json.dumps({
        "d_budget": D_BUDGET, "alpha": ALPHA, "cvar_window": CVAR_WINDOW,
        "filter_lookback": FILTER_LOOKBACK, "c_alpha": c_alpha,
        "arms": summary.to_dict(orient="index"),
        "weight_distance": dist_summary.to_dict(orient="index"),
        "elapsed_seconds": time.time() - t_start,
    }, indent=2, default=float))
    print(f"\nwrote tables to {out_dir}  (total {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
