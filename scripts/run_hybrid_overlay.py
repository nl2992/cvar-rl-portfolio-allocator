"""Option B: Regime-switching hybrid overlay analysis.

Constructs a strategy that uses the CVaR-constrained allocator during
high-volatility regimes and min-variance during calm and selloff regimes.
Shows the hybrid beats pure min-variance without requiring retraining.

Uses:
  - data/processed/aligned_portfolio_panel_etf.parquet  (weekly returns)
  - results/tables_stats/oos_etf7.csv                   (constrained RL OOS path)

Outputs:
  results/tables/hybrid_overlay_results.json
  results/tables/hybrid_overlay_table.csv

Usage:
    python scripts/run_hybrid_overlay.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

PARQUET = _ROOT / "data/processed/aligned_portfolio_panel_etf.parquet"
OOS_CON_PATH = _ROOT / "results/tables_stats/oos_etf7.csv"
OUT_DIR = _ROOT / "results/tables"
PPY = 52
COST_BPS = 5.0
# Walk-forward config matching run_stats_study.py
TRAIN_WINDOW = 312
VAL_WINDOW = 52
TEST_WINDOW = 26
N_OOS_FOLDS = 20  # 20 folds * 26 weeks = 520 OOS weeks

# Regime labelling parameters (must match regimes.py)
VOL_WINDOW = 13
SELLOFF_THRESHOLD = -0.05
VOL_QUANTILE = 0.75
MAX_WEIGHT = 0.40
N_BOOT = 50_000
RNG_SEED = 2025


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def label_regimes(market: pd.Series) -> pd.Series:
    trailing_ret = (1 + market).rolling(VOL_WINDOW).apply(np.prod, raw=True) - 1
    trailing_vol = market.rolling(VOL_WINDOW).std()
    vol_cut = trailing_vol.quantile(VOL_QUANTILE)
    labels = np.full(len(market), "calm", dtype=object)
    labels[(trailing_vol >= vol_cut).to_numpy()] = "high_vol"
    labels[(trailing_ret <= SELLOFF_THRESHOLD).to_numpy()] = "selloff"
    return pd.Series(labels, index=market.index, name="regime")


def summarise(returns: np.ndarray) -> dict:
    ann_ret = float(np.prod(1 + returns) ** (PPY / len(returns)) - 1)
    ann_vol = float(returns.std(ddof=1) * np.sqrt(PPY))
    sharpe = ann_ret / ann_vol if ann_vol > 1e-9 else 0.0
    wealth = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(wealth)
    mdd = float(-(wealth / peak - 1).min())
    tail = returns[returns <= np.quantile(returns, 0.01)]
    cvar99 = float(-tail.mean()) if len(tail) > 0 else 0.0
    return dict(ann_return=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                max_drawdown=mdd, cvar_99=cvar99, n_weeks=len(returns))


def min_variance_weights(cov: np.ndarray, max_w: float = MAX_WEIGHT) -> np.ndarray:
    n = cov.shape[0]
    w0 = np.ones(n) / n
    bounds = [(0.0, max_w)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    res = minimize(lambda w: w @ cov @ w, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-10, "maxiter": 500})
    if res.success:
        return np.clip(res.x, 0, max_w)
    return w0


def rolling_minvar_returns(returns_df: pd.DataFrame,
                            lookback: int = 52,
                            cost_bps: float = COST_BPS) -> np.ndarray:
    """Compute rolling min-var portfolio returns (no lookahead)."""
    ret_arr = returns_df.to_numpy()
    n, m = ret_arr.shape
    weights = np.zeros((n, m))
    w_prev = np.ones(m) / m
    for t in range(n):
        if t < lookback:
            weights[t] = w_prev
            continue
        hist = ret_arr[t - lookback:t]
        cov = np.cov(hist.T)
        w = min_variance_weights(cov)
        weights[t] = w
        w_prev = w
    # portfolio return: dot(w, next_period_ret) minus costs
    port_ret = (weights[:-1] * ret_arr[1:]).sum(axis=1)
    turnover = np.abs(np.diff(weights, axis=0)).sum(axis=1)
    costs = turnover * cost_bps * 1e-4
    return port_ret - costs


def bootstrap_sharpe_ci(returns: np.ndarray, n_boot: int = N_BOOT,
                          seed: int = RNG_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.choice(returns, size=len(returns), replace=True)
        ann_r = np.prod(1 + s) ** (PPY / len(s)) - 1
        ann_v = s.std(ddof=1) * np.sqrt(PPY)
        sharpes[i] = ann_r / ann_v if ann_v > 1e-9 else 0.0
    return float(np.percentile(sharpes, 2.5)), float(np.percentile(sharpes, 97.5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 64)
    print("HYBRID OVERLAY ANALYSIS  (Option B)")
    print("=" * 64)

    # 1. Load data
    print("\n[1] Loading ETF parquet...")
    returns_df = pd.read_parquet(PARQUET)
    dates = pd.date_range(start="2008-01-04", periods=len(returns_df), freq="W-FRI")
    returns_df.index = dates
    print(f"    {len(returns_df)} weeks  {returns_df.index[0].date()} → {returns_df.index[-1].date()}")

    # 2. Reconstruct OOS slice (train=312 + val=52 → OOS starts at week 364)
    oos_start_idx = TRAIN_WINDOW + VAL_WINDOW  # 364
    oos_end_idx = oos_start_idx + N_OOS_FOLDS * TEST_WINDOW  # 364 + 520 = 884
    oos_dates = dates[oos_start_idx:oos_end_idx]
    oos_returns_full = returns_df.iloc[oos_start_idx:oos_end_idx]
    print(f"    OOS period: {oos_dates[0].date()} → {oos_dates[-1].date()} ({len(oos_dates)} weeks)")

    # 3. Constrained RL OOS returns (already computed and saved)
    print("\n[2] Loading constrained RL OOS returns...")
    con_rl = pd.read_csv(OOS_CON_PATH)["ret"].to_numpy()
    if len(con_rl) != len(oos_dates):
        raise RuntimeError(f"Constrained RL OOS length mismatch: {len(con_rl)} vs {len(oos_dates)}")
    print(f"    Loaded {len(con_rl)} weekly returns")
    con_summary = summarise(con_rl)
    print(f"    Constrained RL: Sharpe={con_summary['sharpe']:.3f}  "
          f"Return={con_summary['ann_return']:.1%}  CVaR99={con_summary['cvar_99']:.3f}")

    # 4. Compute min-var rolling returns on OOS + sufficient warmup
    print("\n[3] Computing rolling min-variance returns (no retraining)...")
    warmup = 52  # 1 year of warmup before OOS
    warmup_start = oos_start_idx - warmup
    mv_full = rolling_minvar_returns(
        returns_df.iloc[warmup_start:oos_end_idx], lookback=52)
    # mv_full covers (warmup_start+1 : oos_end_idx), trim to OOS period
    mv_oos = mv_full[warmup - 1:]  # first warmup-1 rows are pre-OOS
    assert len(mv_oos) == len(oos_dates), f"{len(mv_oos)} vs {len(oos_dates)}"
    mv_summary = summarise(mv_oos)
    print(f"    Min-var: Sharpe={mv_summary['sharpe']:.3f}  "
          f"Return={mv_summary['ann_return']:.1%}  CVaR99={mv_summary['cvar_99']:.3f}")

    # 5. Label regimes on OOS using SPY (col 0)
    print("\n[4] Labelling market regimes on OOS window...")
    spy_oos = returns_df.iloc[oos_start_idx:oos_end_idx]["SPY"]
    regimes_oos = label_regimes(spy_oos).to_numpy()
    regime_counts = {r: int((regimes_oos == r).sum()) for r in ["calm", "high_vol", "selloff"]}
    print(f"    Regime counts: {regime_counts}")

    # 6. Construct hybrid: constrained RL in high_vol, min-var otherwise
    print("\n[5] Constructing regime-switching hybrid...")
    high_vol_mask = regimes_oos == "high_vol"
    hybrid = mv_oos.copy()
    hybrid[high_vol_mask] = con_rl[high_vol_mask]
    hybrid_summary = summarise(hybrid)
    print(f"    Hybrid: Sharpe={hybrid_summary['sharpe']:.3f}  "
          f"Return={hybrid_summary['ann_return']:.1%}  CVaR99={hybrid_summary['cvar_99']:.3f}")
    print(f"    High-vol weeks using constrained RL: {high_vol_mask.sum()} / {len(hybrid)}")

    # 7. Bootstrap confidence intervals for each strategy
    print("\n[6] Bootstrapping Sharpe CIs (50k resamples)...")
    ci_hybrid = bootstrap_sharpe_ci(hybrid)
    ci_mv = bootstrap_sharpe_ci(mv_oos)
    ci_con = bootstrap_sharpe_ci(con_rl)
    print(f"    Hybrid:         {ci_hybrid[0]:.3f} – {ci_hybrid[1]:.3f}")
    print(f"    Min-var:        {ci_mv[0]:.3f} – {ci_mv[1]:.3f}")
    print(f"    Constrained RL: {ci_con[0]:.3f} – {ci_con[1]:.3f}")

    # 8. Regime-conditional performance for hybrid subcomponents
    regime_stats = {}
    for label in ["calm", "high_vol", "selloff"]:
        mask = regimes_oos == label
        if mask.sum() < 10:
            continue
        regime_stats[label] = {
            "n_weeks": int(mask.sum()),
            "hybrid": summarise(hybrid[mask]),
            "min_var": summarise(mv_oos[mask]),
            "constrained_rl": summarise(con_rl[mask]),
        }
    print("\n[7] Regime-conditional Sharpe comparison:")
    for r, s in regime_stats.items():
        print(f"    {r:10s} | hybrid={s['hybrid']['sharpe']:+.3f}  "
              f"min_var={s['min_var']['sharpe']:+.3f}  "
              f"con_rl={s['constrained_rl']['sharpe']:+.3f}  "
              f"(n={s['n_weeks']})")

    # 9. Summary table
    table = pd.DataFrame({
        "strategy": ["min_var (pure)", "con_rl (pure)", "hybrid (B)"],
        "sharpe": [mv_summary["sharpe"], con_summary["sharpe"], hybrid_summary["sharpe"]],
        "sharpe_ci_lo": [ci_mv[0], ci_con[0], ci_hybrid[0]],
        "sharpe_ci_hi": [ci_mv[1], ci_con[1], ci_hybrid[1]],
        "ann_return": [mv_summary["ann_return"], con_summary["ann_return"], hybrid_summary["ann_return"]],
        "ann_vol": [mv_summary["ann_vol"], con_summary["ann_vol"], hybrid_summary["ann_vol"]],
        "max_drawdown": [mv_summary["max_drawdown"], con_summary["max_drawdown"], hybrid_summary["max_drawdown"]],
        "cvar_99": [mv_summary["cvar_99"], con_summary["cvar_99"], hybrid_summary["cvar_99"]],
        "high_vol_weeks_used": [0, regime_counts.get("high_vol", 0), high_vol_mask.sum()],
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "hybrid_overlay_table.csv", index=False)

    results = {
        "hybrid_sharpe": hybrid_summary["sharpe"],
        "min_var_sharpe": mv_summary["sharpe"],
        "con_rl_sharpe": con_summary["sharpe"],
        "hybrid_sharpe_ci": list(ci_hybrid),
        "min_var_sharpe_ci": list(ci_mv),
        "con_rl_sharpe_ci": list(ci_con),
        "hybrid_ann_return": hybrid_summary["ann_return"],
        "min_var_ann_return": mv_summary["ann_return"],
        "hybrid_cvar_99": hybrid_summary["cvar_99"],
        "min_var_cvar_99": mv_summary["cvar_99"],
        "hybrid_max_drawdown": hybrid_summary["max_drawdown"],
        "min_var_max_drawdown": mv_summary["max_drawdown"],
        "regime_counts": regime_counts,
        "high_vol_fraction": float(high_vol_mask.sum()) / len(hybrid),
        "sharpe_gain_vs_minvar": hybrid_summary["sharpe"] - mv_summary["sharpe"],
        "regime_stats": {r: {k: v for k, v in s.items() if k != "n_weeks"}
                         for r, s in regime_stats.items()},
    }
    (OUT_DIR / "hybrid_overlay_results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'='*64}")
    print("HYBRID OVERLAY SUMMARY")
    print(f"{'='*64}")
    print(f"  Min-var (pure):   Sharpe {mv_summary['sharpe']:.3f}  "
          f"[{ci_mv[0]:.3f}, {ci_mv[1]:.3f}]")
    print(f"  Constrained RL:   Sharpe {con_summary['sharpe']:.3f}  "
          f"[{ci_con[0]:.3f}, {ci_con[1]:.3f}]")
    print(f"  Hybrid (Option B): Sharpe {hybrid_summary['sharpe']:.3f}  "
          f"[{ci_hybrid[0]:.3f}, {ci_hybrid[1]:.3f}]")
    print(f"\n  Sharpe gain vs pure min-var: {results['sharpe_gain_vs_minvar']:+.3f}")
    print(f"  High-vol weeks: {high_vol_mask.sum()} of {len(hybrid)} "
          f"({results['high_vol_fraction']:.1%})")
    print(f"\nSaved: {OUT_DIR}/hybrid_overlay_results.json")
    print(f"Saved: {OUT_DIR}/hybrid_overlay_table.csv")


if __name__ == "__main__":
    main()
