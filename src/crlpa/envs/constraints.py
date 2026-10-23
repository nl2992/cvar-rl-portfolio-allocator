from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_TOL = 1e-9


@dataclass(frozen=True)
class PortfolioConstraints:
    """Admissible-set definition for long-only allocation weights.

    Weights are interpreted as fractions of a fully-invested book that sums to
    one. ``turnover`` is measured as the one-way L1 distance ``sum(|w - w_prev|)``,
    consistent with the transaction-cost model in :class:`AllocationEnv`.
    """

    long_only: bool = True
    max_weight: float = 1.0
    cash_floor: float = 0.0
    cash_index: int | None = None
    turnover_cap: float | None = None
    gross_cap: float = 1.0

    def feasible_max_weight(self, n_assets: int) -> None:
        """Raise if the max-weight cap cannot admit a fully-invested book."""
        if self.max_weight * n_assets < 1.0 - _TOL:
            raise ValueError(
                f"max_weight={self.max_weight} with {n_assets} assets cannot sum to 1"
            )


def normalise_long_only(weights: np.ndarray) -> np.ndarray:
    """Clip negatives and renormalise to sum to one."""
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = w.sum()
    if total <= _TOL:
        return np.full(w.shape, 1.0 / w.size)
    return w / total


def cap_max_weight(weights: np.ndarray, max_weight: float) -> np.ndarray:
    """Water-fill so no weight exceeds ``max_weight`` while the book sums to one."""
    w = np.array(weights, dtype=float)
    if max_weight >= 1.0 - _TOL:
        return w
    for _ in range(w.size + 1):
        over = w > max_weight + _TOL
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        room = w[under].sum()
        if room <= _TOL:
            w[:] = 1.0 / w.size
            break
        w[under] += excess * w[under] / room
    return w


def apply_cash_floor(weights: np.ndarray, cash_index: int, cash_floor: float) -> np.ndarray:
    """Ensure the designated cash asset holds at least ``cash_floor``."""
    w = np.array(weights, dtype=float)
    if cash_floor <= _TOL or w[cash_index] >= cash_floor - _TOL:
        return w
    others = np.arange(w.size) != cash_index
    other_total = w[others].sum()
    if other_total <= _TOL:
        w[others] = (1.0 - cash_floor) / max(others.sum(), 1)
    else:
        w[others] *= (1.0 - cash_floor) / other_total
    w[cash_index] = cash_floor
    return w


def apply_turnover_cap(
    weights: np.ndarray, prev_weights: np.ndarray, turnover_cap: float
) -> np.ndarray:
    """Scale the trade vector toward ``prev_weights`` so turnover <= cap.

    The trade vector ``w - w_prev`` sums to zero, so scaling it preserves the
    unit budget and (being a convex move from one feasible point toward another)
    preserves long-only and max-weight feasibility.
    """
    w = np.asarray(weights, dtype=float)
    prev = np.asarray(prev_weights, dtype=float)
    turnover = float(np.abs(w - prev).sum())
    if turnover_cap is None or turnover <= turnover_cap + _TOL or turnover <= _TOL:
        return w
    scale = turnover_cap / turnover
    return prev + scale * (w - prev)


def project_weights(
    proposed: np.ndarray,
    prev_weights: np.ndarray,
    constraints: PortfolioConstraints,
) -> np.ndarray:
    """Project proposed weights onto the admissible set.

    Order matters: normalise -> cash floor -> max-weight cap -> turnover cap.
    The turnover step comes last because it only moves *toward* the (already
    feasible) previous weights, preserving the earlier constraints.
    """
    w = np.asarray(proposed, dtype=float)
    if constraints.long_only:
        w = normalise_long_only(w)
    if constraints.cash_index is not None and constraints.cash_floor > 0:
        w = apply_cash_floor(w, constraints.cash_index, constraints.cash_floor)
    w = cap_max_weight(w, constraints.max_weight)
    if constraints.turnover_cap is not None:
        w = apply_turnover_cap(w, prev_weights, constraints.turnover_cap)
    return w


def violations(
    weights: np.ndarray,
    prev_weights: np.ndarray,
    constraints: PortfolioConstraints,
) -> dict[str, float]:
    """Report constraint slack/violation magnitudes for diagnostics.

    Positive values indicate a breach amount; zero means the constraint holds.
    """
    w = np.asarray(weights, dtype=float)
    prev = np.asarray(prev_weights, dtype=float)
    out: dict[str, float] = {}
    out["long_only"] = float(max(0.0, -w.min())) if constraints.long_only else 0.0
    out["max_weight"] = float(max(0.0, w.max() - constraints.max_weight))
    out["gross"] = float(max(0.0, np.abs(w).sum() - constraints.gross_cap))
    if constraints.cash_index is not None and constraints.cash_floor > 0:
        out["cash_floor"] = float(max(0.0, constraints.cash_floor - w[constraints.cash_index]))
    if constraints.turnover_cap is not None:
        turnover = float(np.abs(w - prev).sum())
        out["turnover"] = float(max(0.0, turnover - constraints.turnover_cap))
    return out
