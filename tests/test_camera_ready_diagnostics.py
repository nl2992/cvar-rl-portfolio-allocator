"""Tests for the camera-ready diagnostics: bootstrap interval methods and the
checkpoint-selection metadata recorded by the differentiable trainer."""

from __future__ import annotations

import numpy as np
import pytest

from crlpa.evaluation.bootstrap import paired_bootstrap

torch = pytest.importorskip("torch")

from crlpa.data.synthetic import make_synthetic_returns  # noqa: E402
from crlpa.training.differentiable import DiffConfig, train_differentiable  # noqa: E402


def _pair(seed: int = 0, n: int = 260):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.002, 0.02, n)
    b = a + rng.normal(-0.001, 0.005, n)
    return a, b


def test_default_bootstrap_is_unchanged_percentile():
    a, b = _pair()
    default = paired_bootstrap(a, b, statistic=np.mean, block_size=4, n_resamples=500)
    pct = paired_bootstrap(a, b, statistic=np.mean, block_size=4, n_resamples=500, method="percentile")
    assert (default.ci_low, default.ci_high) == (pct.ci_low, pct.ci_high)


def test_basic_interval_reflects_percentile_about_estimate():
    a, b = _pair(1)
    pct = paired_bootstrap(a, b, statistic=np.mean, n_resamples=500)
    basic = paired_bootstrap(a, b, statistic=np.mean, n_resamples=500, method="basic")
    est = pct.point_estimate
    assert basic.ci_low == pytest.approx(2 * est - pct.ci_high)
    assert basic.ci_high == pytest.approx(2 * est - pct.ci_low)


def test_bca_close_to_percentile_for_smooth_statistic():
    a, b = _pair(2, n=800)
    pct = paired_bootstrap(a, b, statistic=np.mean, n_resamples=2000)
    bca = paired_bootstrap(a, b, statistic=np.mean, n_resamples=2000, method="bca")
    width = pct.ci_high - pct.ci_low
    assert abs(bca.ci_low - pct.ci_low) < 0.25 * width
    assert abs(bca.ci_high - pct.ci_high) < 0.25 * width


def test_unknown_bootstrap_method_raises():
    a, b = _pair()
    with pytest.raises(ValueError):
        paired_bootstrap(a, b, statistic=np.mean, method="studentised")


@pytest.mark.parametrize("constrained,rule", [(True, {"feasible", "lowest_cvar_fallback"}), (False, {"best_val"})])
def test_selection_metadata_recorded(constrained, rule):
    returns = make_synthetic_returns(n_steps=160, seed=7)
    train, val = returns.iloc[:120], returns.iloc[120:]
    cfg = DiffConfig(n_updates=60, horizon=52, constrained=constrained, seed=7, eval_every=10)
    _, history = train_differentiable(
        train, cost_bps=5.0, cvar_alpha=0.95, cvar_limit=0.03, cvar_window=52, lookback=26,
        config=cfg, val_returns=val,
    )
    sel = history.attrs["selection"]
    assert sel["rule"] in rule
    assert sel["n_evals"] == 6
    assert 0 <= sel["n_feasible"] <= sel["n_evals"]
    assert sel["selected_update"] % cfg.eval_every == 0


def test_selection_metadata_does_not_change_training():
    returns = make_synthetic_returns(n_steps=160, seed=11)
    train, val = returns.iloc[:120], returns.iloc[120:]
    kwargs = dict(cost_bps=5.0, cvar_alpha=0.95, cvar_limit=0.03, cvar_window=52, lookback=26,
                  config=DiffConfig(n_updates=40, horizon=52, constrained=True, seed=11, eval_every=10),
                  val_returns=val)
    a1, h1 = train_differentiable(train, **kwargs)
    a2, h2 = train_differentiable(train, **kwargs)
    for p1, p2 in zip(a1.parameters(), a2.parameters()):
        assert torch.equal(p1, p2)
    assert h1.attrs["selection"] == h2.attrs["selection"]
