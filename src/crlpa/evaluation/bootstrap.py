from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from crlpa.evaluation.metrics import sharpe


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    p_value: float

    def significant(self, level: float = 0.05) -> bool:
        return self.p_value < level


def paired_bootstrap(
    returns_a: pd.Series | np.ndarray,
    returns_b: pd.Series | np.ndarray,
    statistic: Callable[[np.ndarray], float] = sharpe,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    block_size: int = 1,
    seed: int = 42,
    method: str = "percentile",
) -> BootstrapResult:
    """Paired (optionally block) bootstrap of ``statistic(a) - statistic(b)``.

    Resampling indices are shared between the two series so the comparison stays
    paired. ``block_size > 1`` uses a circular block bootstrap to preserve serial
    dependence. The two-sided p-value tests the null that the difference is zero.

    ``method`` selects the confidence interval. ``"percentile"`` (default) takes
    quantiles of the resampled differences. ``"basic"`` reflects them about the
    full-sample estimate (``2*theta - q``), which corrects the percentile
    interval's shift when resamples are systematically biased relative to the
    full sample -- the case for extreme-tail statistics such as CVaR-99, whose
    resamples often omit the few worst weeks. ``"bca"`` is bias-corrected and
    accelerated, with the acceleration from a delete-one-block jackknife.
    """
    if method not in {"percentile", "basic", "bca"}:
        raise ValueError(f"unknown bootstrap interval method: {method!r}")
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap requires equal-length series")
    n = a.size
    rng = np.random.default_rng(seed)
    observed = statistic(a) - statistic(b)

    diffs = np.empty(n_resamples)
    n_blocks = int(np.ceil(n / block_size))
    for i in range(n_resamples):
        if block_size <= 1:
            idx = rng.integers(0, n, size=n)
        else:
            starts = rng.integers(0, n, size=n_blocks)
            idx = np.concatenate([(s + np.arange(block_size)) % n for s in starts])[:n]
        diffs[i] = statistic(a[idx]) - statistic(b[idx])

    tail = (1 - confidence) / 2
    if method == "percentile":
        ci_low, ci_high = np.quantile(diffs, [tail, 1 - tail])
    elif method == "basic":
        q_low, q_high = np.quantile(diffs, [tail, 1 - tail])
        ci_low, ci_high = 2 * observed - q_high, 2 * observed - q_low
    else:  # bca
        ci_low, ci_high = _bca_interval(a, b, statistic, diffs, observed, tail, max(1, block_size))
    centred = diffs - diffs.mean()
    p_value = float((np.abs(centred) >= abs(observed)).mean())
    return BootstrapResult(float(observed), float(ci_low), float(ci_high), p_value)


def _bca_interval(
    a: np.ndarray,
    b: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    diffs: np.ndarray,
    observed: float,
    tail: float,
    block_size: int,
) -> tuple[float, float]:
    """BCa interval for ``statistic(a) - statistic(b)`` (Efron 1987).

    Bias correction ``z0`` from the share of resampled differences below the
    full-sample estimate; acceleration from a delete-one-block jackknife so the
    estimate respects the same dependence structure as the block resampling.
    """
    from scipy.stats import norm

    share_below = np.clip(np.mean(diffs < observed), 1e-6, 1 - 1e-6)
    z0 = norm.ppf(share_below)
    n = a.size
    starts = np.arange(0, n, block_size)
    jack = np.empty(starts.size)
    for j, s in enumerate(starts):
        keep = np.r_[0:s, min(n, s + block_size):n]
        jack[j] = statistic(a[keep]) - statistic(b[keep])
    dev = jack.mean() - jack
    denom = 6.0 * (np.sum(dev**2) ** 1.5)
    accel = float(np.sum(dev**3) / denom) if denom > 0 else 0.0

    def adjusted(q: float) -> float:
        z = norm.ppf(q)
        return float(norm.cdf(z0 + (z0 + z) / (1 - accel * (z0 + z))))

    lo_q, hi_q = adjusted(tail), adjusted(1 - tail)
    ci_low, ci_high = np.quantile(diffs, [lo_q, hi_q])
    return float(ci_low), float(ci_high)
