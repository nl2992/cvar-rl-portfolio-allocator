"""Compile regime-conditional performance comparison from existing walkforward regime files.

Reads results/tables_wf/walkforward_regime_*.csv (calm / high_vol / selloff)
and produces a unified comparison table: regime × model × metric.

Outputs:
  results/tables/regime_comparison.csv
  results/tables/regime_comparison.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
WF_DIR = ROOT / "results/tables_wf"
OUT = ROOT / "results/tables"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "rl_cvar_constrained": "RL Constrained",
    "rl_unconstrained":    "RL Unconstrained",
    "min_variance":        "Min-Variance",
    "inverse_vol":         "Inverse-Vol",
}

REGIME_ORDER = ["calm", "high_vol", "selloff"]
METRICS = ["sharpe", "cvar_99", "max_drawdown", "n_weeks"]


def load_regime_file(model_key: str) -> pd.DataFrame:
    path = WF_DIR / f"walkforward_regime_{model_key}.csv"
    df = pd.read_csv(path, index_col=0)
    df.index.name = "regime"
    df = df.reset_index()
    df["model_key"] = model_key
    df["model_label"] = MODELS[model_key]
    return df


def main():
    frames = [load_regime_file(k) for k in MODELS]
    all_df = pd.concat(frames, ignore_index=True)

    # Pivot to wide format: regime → (model, metric)
    rows = []
    for regime in REGIME_ORDER:
        row = {"regime": regime}
        for mk, label in MODELS.items():
            sub = all_df[(all_df["regime"] == regime) & (all_df["model_key"] == mk)]
            if sub.empty:
                continue
            for m in METRICS:
                if m in sub.columns:
                    row[f"{mk}_{m}"] = round(float(sub[m].iloc[0]), 4)
        rows.append(row)

    wide = pd.DataFrame(rows)
    wide.to_csv(OUT / "regime_comparison.csv", index=False)

    # Print readable summary
    print("=" * 80)
    print("REGIME-CONDITIONAL PERFORMANCE COMPARISON")
    print("=" * 80)
    for regime in REGIME_ORDER:
        r = all_df[all_df["regime"] == regime]
        n = r[r["model_key"] == "min_variance"]["n_weeks"].values
        n_str = f"(n={int(n[0])} weeks)" if len(n) else ""
        print(f"\n{regime.upper()} {n_str}")
        print(f"  {'Model':25s}  {'Sharpe':>8}  {'CVaR-99':>8}  {'MaxDD':>8}")
        print(f"  {'-'*25}  {'--------':>8}  {'--------':>8}  {'--------':>8}")
        for mk, label in MODELS.items():
            sub = r[r["model_key"] == mk]
            if sub.empty:
                continue
            sh = sub["sharpe"].values[0] if "sharpe" in sub else float("nan")
            cv = sub["cvar_99"].values[0] if "cvar_99" in sub else float("nan")
            dd = sub["max_drawdown"].values[0] if "max_drawdown" in sub else float("nan")
            print(f"  {label:25s}  {sh:>8.3f}  {cv:>8.4f}  {dd:>8.4f}")

    # Key finding: does constrained RL narrow the gap to min-var in stress?
    print("\n=== KEY COMPARISON: RL Constrained vs Min-Variance ===")
    for regime in REGIME_ORDER:
        r = all_df[all_df["regime"] == regime]
        rl  = r[r["model_key"] == "rl_cvar_constrained"]
        mv  = r[r["model_key"] == "min_variance"]
        if rl.empty or mv.empty:
            continue
        sh_gap = float(rl["sharpe"].values[0]) - float(mv["sharpe"].values[0])
        cv_gap = float(rl["cvar_99"].values[0]) - float(mv["cvar_99"].values[0])
        print(f"  {regime:10s}  Sharpe gap (RL - MV): {sh_gap:+.3f}  CVaR-99 gap: {cv_gap:+.4f}")

    # JSON summary
    result = {
        "regimes": REGIME_ORDER,
        "models": list(MODELS.keys()),
        "data": all_df.to_dict(orient="records"),
        "wide": wide.to_dict(orient="records"),
    }
    (OUT / "regime_comparison.json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {OUT}/regime_comparison.csv")
    print(f"Saved: {OUT}/regime_comparison.json")


if __name__ == "__main__":
    main()
