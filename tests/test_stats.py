from __future__ import annotations

import numpy as np

from crlpa.evaluation.stats import (
    benjamini_hochberg,
    deflated_sharpe_ratio,
    paired_fold_test,
    probabilistic_sharpe_ratio,
)


def test_psr_high_for_strong_track_record():
    rng = np.random.default_rng(0)
    r = rng.normal(0.01, 0.02, 400)  # strongly positive Sharpe, long sample
    assert probabilistic_sharpe_ratio(r) > 0.95


def test_psr_near_half_for_zero_mean():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.02, 400)
    assert 0.2 < probabilistic_sharpe_ratio(r) < 0.8


def test_deflated_le_probabilistic():
    rng = np.random.default_rng(1)
    r = rng.normal(0.006, 0.02, 300)
    trials = rng.normal(0.1, 0.15, 20)  # many trials with dispersion -> deflation
    psr = probabilistic_sharpe_ratio(r)
    dsr = deflated_sharpe_ratio(r, trials)
    assert dsr <= psr + 1e-9


def test_paired_fold_test_detects_improvement():
    rng = np.random.default_rng(2)
    control = rng.uniform(0.04, 0.07, 20)
    treatment = control - rng.uniform(0.005, 0.02, 20)  # always lower (better tail)
    ft = paired_fold_test(treatment, control)
    assert ft.improved == 20
    assert ft.mean_diff < 0
    assert ft.ci_high < 0  # CI entirely below zero
    assert ft.sign_p < 0.05


def test_benjamini_hochberg_rejects_small_pvalues():
    reject = benjamini_hochberg([0.001, 0.04, 0.8], alpha=0.05)
    assert reject[0] is True
    assert reject[2] is False
