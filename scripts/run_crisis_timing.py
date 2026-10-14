"""Option C: Crisis de-risking timing — constrained RL vs min-var.

Measures how early each strategy entered drawdown relative to the SPY
market trough during two major crises:
  - COVID crash (trough: 2020-03-23)
  - 2022 rates selloff (trough: 2022-10-13)

A strategy "de-risks" when its rolling 4-week drawdown first crosses
-3% (entering a meaningful loss zone). We measure the lead or lag
between each strategy's de-risking date and the SPY trough.

Uses:
  - data/processed/aligned_portfolio_panel_etf.parquet (weekly returns)
  - results/tables_stats/oos_etf7.csv                  (constrained RL OOS)

Outputs:
  results/tables/crisis_timing_results.json
  results/tables/crisis_timing_table.csv

Usage:
    python scripts/run_crisis_timing.py
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
TRAIN_WINDOW = 312
VAL_WINDOW = 52
N_OOS_FOLDS = 20
TEST_WINDOW = 26

# De-risking threshold: rolling 4-week drawdown crosses -3%
DRAWDOWN_THRESHOLD = -0.03
DRAWDOWN_WINDOW = 4  # weeks

# Crisis trough dates (Fridays closest to actual troughs)
CRISES = {
    "covid_2020": "2020-03-20",   # SPY hit intraday low 2020-03-23; prev Friday
    "rates_2022": "2022-10-14",   # SPY/TLT trough October 2022
}

MAX_WEIGHT = 0.40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def min_variance_weights(cov: np.ndarray, max_w: float = MAX_WEIGHT) -> np.ndarray:
    n = cov.shape[0]
    w0 = np.ones(n) / n
    bounds = [(0.0, max_w)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    res = minimize(lambda w: w @ cov @ w, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-10, "maxiter": 500})
    return np.clip(res.x, 0, max_w) if res.success else w0


def rolling_minvar_returns(returns_df: pd.DataFrame,
                            lookback: int = 52,
                            cost_bps: float = COST_BPS) -> pd.Series:
    """Rolling min-var returns with date index (no lookahead)."""
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
    port_ret = (weights[:-1] * ret_arr[1:]).sum(axis=1)
    turnover = np.abs(np.diff(weights, axis=0)).sum(axis=1)
    costs = turnover * cost_bps * 1e-4
    return pd.Series(port_ret - costs, index=returns_df.index[1:])


def rolling_drawdown(returns: pd.Series, window: int = DRAWDOWN_WINDOW) -> pd.Series:
    """Rolling window-period return (not peak-to-trough, but recent cum. return)."""
    cum = (1 + returns).cumprod()
    roll_start = cum.shift(window)
    return (cum / roll_start - 1).fillna(0)


def find_derisk_date(returns: pd.Series, threshold: float = DRAWDOWN_THRESHOLD,
                      before_date: str | None = None) -> pd.Timestamp | None:
    """Find the first date rolling drawdown crosses threshold (before a given date)."""
    dd = rolling_drawdown(returns)
    if before_date is not None:
        dd = dd[dd.index <= pd.Timestamp(before_date)]
    crossed = dd[dd <= threshold]
    return crossed.index[0] if len(crossed) > 0 else None


def weeks_to_trough(derisk_date: pd.Timestamp | None,
                     trough_date: str) -> int | None:
    """Positive = de-risked BEFORE trough (good). Negative = de-risked AFTER (bad)."""
    if derisk_date is None:
        return None
    trough = pd.Timestamp(trough_date)
    delta_days = (trough - derisk_date).days
    return round(delta_days / 7)


def max_drawdown_before(returns: pd.Series, trough_date: str) -> float:
    """Maximum drawdown from wealth peak up to the trough date."""
    r = returns[returns.index <= pd.Timestamp(trough_date)]
    if len(r) == 0:
        return 0.0
    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    dd = (wealth / peak - 1)
    return float(-dd.min())


def summarise_crisis_window(returns: pd.Series, start: str, end: str) -> dict:
    """Summarise performance in a crisis window."""
    window = returns[
        (returns.index >= pd.Timestamp(start)) &
        (returns.index <= pd.Timestamp(end))
    ]
    if len(window) < 2:
        return {}
    cum_ret = float((1 + window).prod() - 1)
    vol = float(window.std(ddof=1) * np.sqrt(PPY))
    tail = window.to_numpy()[window.to_numpy() <= np.quantile(window.to_numpy(), 0.01)]
    cvar99 = float(-tail.mean()) if len(tail) > 0 else 0.0
    wealth = (1 + window).cumprod()
    peak = wealth.cummax()
    mdd = float(-(wealth / peak - 1).min())
    return dict(cum_return=cum_ret, ann_vol=vol, cvar_99=cvar99,
                max_drawdown=mdd, n_weeks=len(window))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 64)
    print("CRISIS DE-RISKING TIMING  (Option C)")
    print("=" * 64)

    # 1. Load data
    print("\n[1] Loading ETF parquet...")
    returns_df = pd.read_parquet(PARQUET)
    dates = pd.date_range(start="2008-01-04", periods=len(returns_df), freq="W-FRI")
    returns_df.index = dates
    print(f"    {len(returns_df)} weeks  {dates[0].date()} → {dates[-1].date()}")

    # 2. OOS slice
    oos_start = TRAIN_WINDOW + VAL_WINDOW  # 364
    oos_end = oos_start + N_OOS_FOLDS * TEST_WINDOW  # 884
    oos_dates = dates[oos_start:oos_end]
    oos_returns_full = returns_df.iloc[oos_start:oos_end]
    print(f"    OOS: {oos_dates[0].date()} → {oos_dates[-1].date()}")

    # 3. SPY (market benchmark) — OOS period
    spy_oos = oos_returns_full["SPY"].copy()
    spy_oos.index = oos_dates

    # 4. Constrained RL OOS returns with dates
    print("\n[2] Loading constrained RL OOS returns...")
    con_rl_arr = pd.read_csv(OOS_CON_PATH)["ret"].to_numpy()
    if len(con_rl_arr) != len(oos_dates):
        raise RuntimeError(f"Length mismatch: {len(con_rl_arr)} vs {len(oos_dates)}")
    con_rl = pd.Series(con_rl_arr, index=oos_dates, name="con_rl")
    print(f"    Loaded {len(con_rl)} weeks")

    # 5. Min-var rolling returns (with warmup pre-OOS)
    print("\n[3] Computing rolling min-var returns...")
    warmup = 52
    mv_full = rolling_minvar_returns(
        returns_df.iloc[oos_start - warmup:oos_end], lookback=52)
    # mv_full index starts at oos_start - warmup + 1; trim to OOS period
    mv_oos = mv_full[mv_full.index.isin(oos_dates)]
    # re-index to match exactly
    mv_oos = mv_oos.reindex(oos_dates).fillna(0)
    mv_oos.name = "min_var"
    print(f"    Min-var: {len(mv_oos)} weeks  "
          f"ann_ret={float(np.prod(1+mv_oos.values)**(PPY/len(mv_oos))-1):.1%}")

    # 6. SPY cumulative wealth for trough identification
    print("\n[4] Identifying crisis windows and troughs...")
    strategies = {"constrained_rl": con_rl, "min_var": mv_oos, "spy": spy_oos}

    results = {}
    table_rows = []

    for crisis_name, trough_date_str in CRISES.items():
        trough_date = pd.Timestamp(trough_date_str)
        print(f"\n  Crisis: {crisis_name}  (trough: {trough_date.date()})")

        # SPY drawdown at trough
        spy_dd_at_trough = summarise_crisis_window(
            spy_oos, str(trough_date - pd.Timedelta(weeks=26)), trough_date_str
        )
        print(f"    SPY 26-week drawdown to trough: {spy_dd_at_trough.get('max_drawdown', 0):.1%}")

        crisis_results = {"trough_date": trough_date_str, "strategies": {}}

        for strat_name, strat_ret in strategies.items():
            # De-risking: first week rolling drawdown crosses threshold (before trough)
            derisk = find_derisk_date(strat_ret, DRAWDOWN_THRESHOLD, trough_date_str)
            lead_weeks = weeks_to_trough(derisk, trough_date_str)

            # Performance in a 20-week crisis window centred before trough
            window_start = trough_date - pd.Timedelta(weeks=13)
            window_end = trough_date + pd.Timedelta(weeks=6)
            perf = summarise_crisis_window(strat_ret,
                                            str(window_start.date()),
                                            str(window_end.date()))

            crisis_results["strategies"][strat_name] = {
                "derisk_date": str(derisk.date()) if derisk else "never",
                "lead_weeks_before_trough": lead_weeks,
                "crisis_window_perf": perf,
            }

            lead_str = (f"+{lead_weeks}w before trough" if lead_weeks and lead_weeks > 0
                        else f"{lead_weeks}w after trough" if lead_weeks else "never crossed")
            print(f"    {strat_name:18s} de-risked: {lead_str}  |  "
                  f"drawdown={perf.get('max_drawdown', 0):.1%}  "
                  f"cum_ret={perf.get('cum_return', 0):+.1%}")

            table_rows.append({
                "crisis": crisis_name,
                "trough_date": trough_date_str,
                "strategy": strat_name,
                "derisk_date": str(derisk.date()) if derisk else "never",
                "lead_weeks_before_trough": lead_weeks,
                "crisis_drawdown": perf.get("max_drawdown", 0),
                "crisis_cum_return": perf.get("cum_return", 0),
                "crisis_cvar_99": perf.get("cvar_99", 0),
            })

        # Key comparison: constrained RL vs min-var timing
        con_lead = crisis_results["strategies"]["constrained_rl"]["lead_weeks_before_trough"]
        mv_lead = crisis_results["strategies"]["min_var"]["lead_weeks_before_trough"]
        if con_lead is not None and mv_lead is not None:
            timing_advantage = con_lead - mv_lead
            crisis_results["timing_advantage_constrained_vs_minvar"] = timing_advantage
            print(f"    >> Constrained RL de-risked {abs(timing_advantage)}w "
                  f"{'earlier' if timing_advantage > 0 else 'later'} than min-var")
        results[crisis_name] = crisis_results

    # 7. Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "crisis_timing_results.json").write_text(json.dumps(results, indent=2))
    table = pd.DataFrame(table_rows)
    table.to_csv(OUT_DIR / "crisis_timing_table.csv", index=False)

    print(f"\n{'='*64}")
    print("CRISIS TIMING SUMMARY")
    print(f"{'='*64}")
    for crisis_name, cr in results.items():
        adv = cr.get("timing_advantage_constrained_vs_minvar")
        if adv is not None:
            direction = "earlier" if adv > 0 else "later"
            print(f"  {crisis_name}: constrained RL de-risked {abs(adv)}w {direction} than min-var")
    print(f"\nSaved: {OUT_DIR}/crisis_timing_results.json")
    print(f"Saved: {OUT_DIR}/crisis_timing_table.csv")


if __name__ == "__main__":
    main()
