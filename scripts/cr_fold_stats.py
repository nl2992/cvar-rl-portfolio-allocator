"""Camera-ready: dependence-aware fold statistics for the two-universe walk-forward.

Reads the committed fold-level CVaR-99 results (``results/tables_stats/folds_*.csv``,
from ``scripts/run_stats_study.py``: 312-week train, 52-week validation, 26-week
test folds, step 26) and reports, without retraining:

* the improved / tied / worse split per universe (a tie means the CVaR budget
  never bound, so the constrained and unconstrained learners are identical);
* the sign test with ties excluded (the standard convention);
* a calendar-clustered analysis: fold k covers the same weeks in both universes
  (verified row-by-row on shared tickers), so the two universes' differences are
  averaged per fold, giving 20 calendar clusters, tested with Wilcoxon, an exact
  sign-flip permutation, and a moving-block bootstrap over consecutive clusters;
* the committed pooled statistics, recomputed, for comparison.

Usage:
    python scripts/cr_fold_stats.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TRAIN, VAL, TEST, STEP = 312, 52, 26, 26
OUT = Path("results/tables_camera_ready")
TIE_TOL = 1e-12


def fold_dates(n_folds: int) -> pd.DataFrame:
    """Calendar test window of each fold, from the dated 31-asset panel (same clock)."""
    idx = pd.read_parquet("data/processed/aligned_portfolio_panel_large.parquet").index
    rows = []
    for k in range(n_folds):
        s = k * STEP + TRAIN + VAL
        rows.append({"fold": k, "test_start": idx[s].date(), "test_end": idx[s + TEST - 1].date()})
    return pd.DataFrame(rows)


def check_alignment() -> float:
    """Max abs difference of shared tickers between the macro/sector panels and the dated panel."""
    large = pd.read_parquet("data/processed/aligned_portfolio_panel_large.parquet")
    worst = 0.0
    for path in ("data/processed/aligned_portfolio_panel_etf.parquet",
                 "data/processed/aligned_portfolio_panel_sector.parquet"):
        panel = pd.read_parquet(path)
        for col in set(panel.columns) & set(large.columns):
            worst = max(worst, float(np.nanmax(np.abs(panel[col].to_numpy() - large[col].to_numpy()))))
    return worst


def split_counts(d: np.ndarray) -> dict:
    return {"improved": int((d < -TIE_TOL).sum()), "tied": int((np.abs(d) <= TIE_TOL).sum()),
            "worse": int((d > TIE_TOL).sum())}


def sign_test_excl_ties(d: np.ndarray) -> float:
    nz = d[np.abs(d) > TIE_TOL]
    return float(stats.binomtest(int((nz < 0).sum()), len(nz)).pvalue) if len(nz) else float("nan")


def exact_sign_flip_p(x: np.ndarray) -> float:
    """Exact two-sided sign-flip permutation p-value for the mean (2^n enumeration)."""
    x = np.asarray(x, dtype=float)
    obs = abs(x.mean())
    ax = np.abs(x)
    count = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(x)):
        count += abs(np.dot(signs, ax) / len(x)) >= obs - 1e-15
        total += 1
    return count / total


def block_bootstrap_ci(x: np.ndarray, block: int, n_boot: int = 20000, seed: int = 0) -> tuple[float, float]:
    """Circular moving-block bootstrap 95% CI for the mean of consecutive clusters."""
    rng = np.random.default_rng(seed)
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        means[i] = x[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    e = pd.read_csv("results/tables_stats/folds_etf7.csv").sort_values("fold")
    s = pd.read_csv("results/tables_stats/folds_sector10.csv").sort_values("fold")
    de = (e.cvar99_con - e.cvar99_unc).to_numpy()
    ds = (s.cvar99_con - s.cvar99_unc).to_numpy()
    pooled = np.concatenate([de, ds])
    clusters = (de + ds) / 2

    folds = fold_dates(len(de))
    folds["diff_etf7"], folds["diff_sector10"], folds["diff_cluster_mean"] = de, ds, clusters

    res = {
        "alignment_max_abs_diff_shared_tickers": check_alignment(),
        "per_universe": {},
        "pooled_as_printed": {
            "n": len(pooled), **split_counts(pooled),
            "wilcoxon_p": float(stats.wilcoxon(pooled).pvalue),
            "sign_test_counting_ties_as_failures_p": float(
                stats.binomtest(int((pooled < -TIE_TOL).sum()), len(pooled)).pvalue),
            "sign_test_excl_ties_p": sign_test_excl_ties(pooled),
        },
        "calendar_clusters": {
            "n_clusters": len(clusters), **split_counts(clusters),
            "mean_diff": float(clusters.mean()),
            "wilcoxon_p": float(stats.wilcoxon(clusters).pvalue),
            "sign_test_excl_ties_p": sign_test_excl_ties(clusters),
            "exact_sign_flip_p": exact_sign_flip_p(clusters),
            "block_bootstrap_ci95": {f"block_{b}": block_bootstrap_ci(clusters, b) for b in (2, 3, 4)},
        },
    }
    for name, d in (("etf7", de), ("sector10", ds)):
        res["per_universe"][name] = {
            "n": len(d), **split_counts(d), "mean_diff": float(d.mean()),
            "wilcoxon_p": float(stats.wilcoxon(d).pvalue),
            "sign_test_excl_ties_p": sign_test_excl_ties(d),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUT / "fold_stats_per_fold.csv", index=False)
    (OUT / "fold_stats.json").write_text(json.dumps(res, indent=2, default=float))
    print(folds.round(5).to_string(index=False))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
