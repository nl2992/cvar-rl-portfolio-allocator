from __future__ import annotations

import numpy as np
import pytest

from crlpa.envs.constraints import (
    PortfolioConstraints,
    apply_turnover_cap,
    cap_max_weight,
    project_weights,
    violations,
)


def test_projection_satisfies_all_constraints():
    c = PortfolioConstraints(max_weight=0.4, turnover_cap=0.3, cash_index=4, cash_floor=0.1)
    prev = np.full(5, 0.2)
    w = project_weights(np.array([5.0, 0.0, 0.0, 0.0, 0.0]), prev, c)
    assert np.isclose(w.sum(), 1.0)
    assert w.max() <= 0.4 + 1e-9
    assert w[4] >= 0.1 - 1e-9
    assert np.abs(w - prev).sum() <= 0.3 + 1e-9
    assert (w >= -1e-12).all()


def test_cap_max_weight_redistributes_excess():
    w = cap_max_weight(np.array([0.9, 0.05, 0.05]), max_weight=0.5)
    assert np.isclose(w.sum(), 1.0)
    assert w.max() <= 0.5 + 1e-9


def test_turnover_cap_preserves_budget():
    prev = np.array([0.5, 0.3, 0.2])
    proposed = np.array([0.1, 0.1, 0.8])
    capped = apply_turnover_cap(proposed, prev, turnover_cap=0.2)
    assert np.isclose(capped.sum(), 1.0)
    assert np.abs(capped - prev).sum() <= 0.2 + 1e-9


def test_infeasible_max_weight_raises():
    c = PortfolioConstraints(max_weight=0.1)
    with pytest.raises(ValueError):
        c.feasible_max_weight(n_assets=5)


def test_violations_flag_breaches():
    c = PortfolioConstraints(max_weight=0.5, turnover_cap=0.1)
    prev = np.full(3, 1 / 3)
    bad = np.array([0.8, 0.1, 0.1])
    v = violations(bad, prev, c)
    assert v["max_weight"] > 0
    assert v["turnover"] > 0
