# Camera-ready analyses (ICAIF '26, paper 873)

These analyses answer the reviewers' points. Every file here was regenerated on 4 Oct 2026 in
a fresh environment built from `requirements.lock.txt` (Python 3.12.2, torch 2.12.0,
numpy 1.26.4, scipy 1.17.1, pandas 2.3.3; 83/83 packages matched; 81 tests pass). Under that
environment every re-run arm reproduces the committed per-seed results exactly (max abs
difference 0.0), A6 included.

| File | Script | Content |
|---|---|---|
| `selection_log.csv`, `selection_log_summary.csv` | `scripts/cr_selection_log.py` | Checkpoint selection per arm and seed (tab:coupling). Mis-scaled dual: selected update 250 to 500 (mean 370), 8.4/30 checkpoints meet the validation budget. Scaled dual: 600 to 1300 (mean 1010), 27.2/30. A5's filter fired in 0 of 1285 test weeks; A6's in 582 of 1285. |
| `fold_stats.json`, `fold_stats_per_fold.csv` | `scripts/cr_fold_stats.py` | Two-universe walk-forward. 26 folds improved, 13 tied (budget never binds), 1 worse. Sign test excluding ties: pooled p = 4.2e-7, macro 9/9, sector 17/18. Calendar-clustered (20 clusters): Wilcoxon p = 3.3e-4, exact sign-flip p = 3.8e-5, block-bootstrap 95% CIs exclude 0. Pooled Wilcoxon 8.8e-6 reproduces. |
| `bootstrap_intervals_{etf,sector}.csv` | `scripts/cr_bootstrap_intervals.py` | Seed-7 CVaR99 difference. The percentile interval reproduces the submitted one. BCa: macro −0.033 [−0.035, −0.022], sector −0.044 [−0.050, −0.032]. |
| `shrinkage_large31_stress.csv` | `scripts/cr_shrinkage_large31.py` | 31-asset stress split Sharpe: min-variance 0.23, inverse vol 0.38, Ledoit-Wolf min-variance 0.25, min-CVaR LP 0.26, equal weight 0.68; learner 0.85. |
| (not committed) | FRED series `SP500`, `VIXCLS` | Intro facts: S&P 500 −14.98% over 13 to 20 Mar 2020 (five sessions); VIX record close 82.69 on 16 Mar. The repository ships no vendor data; fetch the series from FRED to check. |

Newer releases (numpy 2.5.3, torch 2.14.1) reproduce every arm except A6 seed 42, whose Sharpe
moves from 0.991 to 0.824 because the eta = 5 dual amplifies ~1e-7 numerical differences. Use
the lock file.
