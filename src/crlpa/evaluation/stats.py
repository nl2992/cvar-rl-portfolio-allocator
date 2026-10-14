from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

_EULER = 0.5772156649015329


def probabilistic_sharpe_ratio(
    returns: np.ndarray, sr_benchmark: float = 0.0
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado).

    Probability that the true (per-period) Sharpe exceeds ``sr_benchmark``, given
    the sample length, skew, and kurtosis of ``returns``. All Sharpes here are
    per-period (not annualised), matching the PSR formula.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    sd = r.std(ddof=1)
    if n < 3 or sd == 0:
        return float("nan")
    sr = r.mean() / sd
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)  # non-excess kurtosis
    denom = np.sqrt(1 - skew * sr + ((kurt - 1) / 4) * sr**2)
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def deflated_sharpe_ratio(returns: np.ndarray, trial_sharpes: np.ndarray) -> float:
    """Deflated Sharpe Ratio: PSR against a benchmark inflated by the number of trials.

    ``trial_sharpes`` are the per-period Sharpes of every configuration tried; their
    dispersion and count set the expected maximum Sharpe under the null, which
    becomes the PSR benchmark — correcting for selection across many trials.
    """
    trials = np.asarray(trial_sharpes, dtype=float)
    n_trials = max(2, trials.size)
    sr_std = trials.std(ddof=1) if trials.size > 1 else 0.0
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr_benchmark = sr_std * ((1 - _EULER) * z1 + _EULER * z2)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr_benchmark)


@dataclass
class PairedFoldTest:
    n: int
    mean_diff: float
    improved: int
    wilcoxon_p: float
    sign_p: float
    ci_low: float
    ci_high: float


def paired_fold_test(
    treatment: np.ndarray, control: np.ndarray, seed: int = 42, n_boot: int = 5000
) -> PairedFoldTest:
    """Paired test that ``treatment < control`` fold-by-fold (e.g. CVaR per fold).

    Reports the mean difference, count improved, Wilcoxon signed-rank and exact
    sign-test p-values, and a bootstrap CI of the mean difference.
    """
    a = np.asarray(treatment, dtype=float)
    b = np.asarray(control, dtype=float)
    diffs = a - b
    n = diffs.size
    improved = int((diffs < 0).sum())

    if n >= 6 and np.any(diffs != 0):
        wilcoxon_p = float(stats.wilcoxon(diffs).pvalue)
    else:
        wilcoxon_p = float("nan")
    # two-sided exact sign test
    sign_p = float(stats.binomtest(improved, n, 0.5).pvalue) if n > 0 else float("nan")

    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(diffs, n, replace=True).mean() for _ in range(n_boot)])
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    return PairedFoldTest(n, float(diffs.mean()), improved, wilcoxon_p, sign_p,
                          float(ci_low), float(ci_high))


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini–Hochberg FDR control; returns a reject mask aligned to ``pvalues``."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        kmax = np.max(np.where(passed))
        reject[order[: kmax + 1]] = True
    return reject.tolist()
