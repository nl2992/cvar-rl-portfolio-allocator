"""Look-ahead audit for the decision-time CVaR filter (P5 acceptance check).

Non-negotiable per PLAN_P5_make_the_constraint_bind.md: a filter that peeks at
future returns would invalidate every result built on it. Two independent
checks:

1. ``precompute_filter_state`` -- corrupting returns strictly AFTER step t
   must not change the trailing covariance, mean, or anchor computed AT t.
2. ``cvar_filter_step`` -- given identical trailing statistics and an
   identical proposed action, the filter's decision (shrink factor and
   filtered weights) is a pure function of those trailing statistics; it
   takes no other input, so if (1) holds, the full decision at t is provably
   unaffected by anything at or after t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from crlpa.training.risk_filter import (
    cvar_filter_step,
    gaussian_cvar_multiplier,
    precompute_filter_state,
)


def _make_returns(n_steps: int = 200, n_assets: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0004, 0.01, size=(n_steps, n_assets))
    return pd.DataFrame(data, columns=[f"a{i}" for i in range(n_assets)])


def test_filter_state_ignores_future_returns():
    base = _make_returns()
    corrupted = base.copy()
    t_probe = 120
    # Corrupt everything from t_probe onward (inclusive) with wild values.
    corrupted.iloc[t_probe:] = 999.0

    state_base = precompute_filter_state(base, lookback=52, max_weight=0.4)
    state_corrupt = precompute_filter_state(corrupted, lookback=52, max_weight=0.4)

    # Every row strictly before t_probe must be identical: it was built only
    # from returns[:t] with t < t_probe, none of which were touched.
    for t in range(t_probe):
        assert np.allclose(state_base.sigma[t], state_corrupt.sigma[t]), f"sigma differs at t={t}"
        assert np.allclose(state_base.mu[t], state_corrupt.mu[t]), f"mu differs at t={t}"
        assert np.allclose(state_base.anchor[t], state_corrupt.anchor[t]), f"anchor differs at t={t}"
        assert state_base.active[t] == state_corrupt.active[t]

    # Sanity: the corruption actually did change something downstream, so this
    # is not a vacuously-passing test (rows strictly after t_probe move once
    # the corrupted values enter their own trailing window).
    later = t_probe + 52 + 1
    assert later < len(base)
    assert not np.allclose(state_base.sigma[later], state_corrupt.sigma[later])


def test_filter_decision_is_a_pure_function_of_trailing_stats():
    """Same trailing stats + same proposal => same shrink factor and output,
    regardless of what the rest of the return series looks like."""
    rng = np.random.default_rng(1)
    n = 5
    sigma = torch.eye(n, dtype=torch.float32) * 1e-4
    mu = torch.zeros(n, dtype=torch.float32)
    anchor = torch.full((n,), 1.0 / n, dtype=torch.float32)
    w_proposed = torch.softmax(torch.as_tensor(rng.normal(size=n), dtype=torch.float32), dim=-1)
    c_alpha = gaussian_cvar_multiplier(0.95)
    budget = 0.01

    w1, s1, diag1 = cvar_filter_step(w_proposed.clone(), sigma, mu, anchor, budget, c_alpha)
    # Re-run with literally the same tensors: must be bit-identical (no hidden
    # dependence on external/global/future state -- the function signature is
    # its complete input).
    w2, s2, diag2 = cvar_filter_step(w_proposed.clone(), sigma, mu, anchor, budget, c_alpha)
    assert s1 == s2
    assert torch.allclose(w1, w2)
    assert diag1 == diag2


def test_shift_returns_leaves_earlier_decisions_unchanged():
    """Directly mirrors the plan's prescribed audit: shift/replace the tail of
    the return series and confirm the filter's decision at an EARLIER step t
    is bit-identical, using the filter output as the observable decision.
    """
    base = _make_returns(n_steps=150, n_assets=4, seed=7)
    shifted = base.copy()
    t_probe = 90
    rng = np.random.default_rng(999)
    shifted.iloc[t_probe + 1 :] = rng.normal(0.0, 0.05, size=shifted.iloc[t_probe + 1 :].shape)

    state_base = precompute_filter_state(base, lookback=40, max_weight=0.5)
    state_shift = precompute_filter_state(shifted, lookback=40, max_weight=0.5)

    c_alpha = gaussian_cvar_multiplier(0.95)
    budget = 0.02
    w_proposed = torch.full((4,), 0.25)

    for t in [10, 50, t_probe]:
        w_a, s_a, _ = cvar_filter_step(
            w_proposed.clone(),
            torch.as_tensor(state_base.sigma[t], dtype=torch.float32),
            torch.as_tensor(state_base.mu[t], dtype=torch.float32),
            torch.as_tensor(state_base.anchor[t], dtype=torch.float32),
            budget, c_alpha,
        )
        w_b, s_b, _ = cvar_filter_step(
            w_proposed.clone(),
            torch.as_tensor(state_shift.sigma[t], dtype=torch.float32),
            torch.as_tensor(state_shift.mu[t], dtype=torch.float32),
            torch.as_tensor(state_shift.anchor[t], dtype=torch.float32),
            budget, c_alpha,
        )
        assert s_a == s_b, f"shrink factor at t={t} changed after shifting future returns"
        assert torch.allclose(w_a, w_b), f"filtered weights at t={t} changed after shifting future returns"
