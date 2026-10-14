"""Threshold-sensitivity sweep for the regime-switching hybrid (referee defence).

The hybrid overlay (run_hybrid_overlay.py) switches to the CVaR-constrained RL
allocator in the high-volatility regime and uses min-variance otherwise. Its
headline Sharpe advantage over pure min-variance depends on the regime-labelling
thresholds (vol_quantile, selloff_threshold). A reviewer will ask whether the
advantage is an artefact of a cherry-picked threshold.

This script reuses the *exact* overlay pipeline (same data, same min-var rolling,
same constrained-RL OOS path) and recomputes the hybrid-vs-min-variance Sharpe
gain across a grid of vol_quantile and selloff_threshold values. No retraining.

Outputs:
  results/tables/hybrid_threshold_sweep.csv
  results/tables/hybrid_threshold_sweep.json

Usage:
    python scripts/run_hybrid_threshold_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import run_hybrid_overlay as ovl  # noqa: E402

VOL_QUANTILES = [0.65, 0.70, 0.75, 0.80, 0.85]
SELLOFF_THRESHOLDS = [-0.04, -0.05, -0.06]


def label_regimes_param(market: pd.Series, vol_quantile: float,
                        selloff_threshold: float) -> np.ndarray:
    trailing_ret = (1 + market).rolling(ovl.VOL_WINDOW).apply(np.prod, raw=True) - 1
    trailing_vol = market.rolling(ovl.VOL_WINDOW).std()
    vol_cut = trailing_vol.quantile(vol_quantile)
    labels = np.full(len(market), "calm", dtype=object)
    labels[(trailing_vol >= vol_cut).to_numpy()] = "high_vol"
    labels[(trailing_ret <= selloff_threshold).to_numpy()] = "selloff"
    return labels


def main() -> None:
    returns_df = pd.read_parquet(ovl.PARQUET)
    dates = pd.date_range(start="2008-01-04", periods=len(returns_df), freq="W-FRI")
    returns_df.index = dates

    oos_start = ovl.TRAIN_WINDOW + ovl.VAL_WINDOW
    oos_end = oos_start + ovl.N_OOS_FOLDS * ovl.TEST_WINDOW
    oos_dates = dates[oos_start:oos_end]

    con_rl = pd.read_csv(ovl.OOS_CON_PATH)["ret"].to_numpy()
    assert len(con_rl) == len(oos_dates)

    warmup = 52
    mv_full = ovl.rolling_minvar_returns(
        returns_df.iloc[oos_start - warmup:oos_end], lookback=52)
    mv_oos = mv_full[warmup - 1:]
    assert len(mv_oos) == len(oos_dates)

    spy_oos = returns_df.iloc[oos_start:oos_end]["SPY"]
    mv_sharpe = ovl.summarise(mv_oos)["sharpe"]

    rows = []
    for q in VOL_QUANTILES:
        for s in SELLOFF_THRESHOLDS:
            regimes = label_regimes_param(spy_oos, q, s)
            hv = regimes == "high_vol"
            hybrid = mv_oos.copy()
            hybrid[hv] = con_rl[hv]
            hs = ovl.summarise(hybrid)
            rows.append({
                "vol_quantile": q,
                "selloff_threshold": s,
                "high_vol_weeks": int(hv.sum()),
                "hybrid_sharpe": round(hs["sharpe"], 4),
                "min_var_sharpe": round(mv_sharpe, 4),
                "sharpe_gain": round(hs["sharpe"] - mv_sharpe, 4),
                "hybrid_beats_minvar": bool(hs["sharpe"] > mv_sharpe),
                "hybrid_cvar99": round(hs["cvar_99"], 4),
            })

    df = pd.DataFrame(rows)
    out = _ROOT / "results/tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "hybrid_threshold_sweep.csv", index=False)

    gains = df["sharpe_gain"].to_numpy()
    summary = {
        "n_configs": len(df),
        "min_var_sharpe": round(float(mv_sharpe), 4),
        "n_configs_hybrid_wins": int(df["hybrid_beats_minvar"].sum()),
        "sharpe_gain_min": round(float(gains.min()), 4),
        "sharpe_gain_max": round(float(gains.max()), 4),
        "sharpe_gain_mean": round(float(gains.mean()), 4),
        "default_config_gain": round(
            float(df[(df.vol_quantile == 0.75) & (df.selloff_threshold == -0.05)]
                  ["sharpe_gain"].iloc[0]), 4),
    }
    (out / "hybrid_threshold_sweep.json").write_text(json.dumps(summary, indent=2))

    print(df.to_string(index=False))
    print("\nSUMMARY:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
