from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LagrangeMultiplier:
    """Dual variable for the CVaR constraint, updated by projected ascent.

    The constraint is ``E[discounted CVaR cost] <= budget``. The multiplier rises
    when the observed cost exceeds the budget and decays toward zero when the
    constraint is satisfied, staying non-negative and capped for stability.
    """

    value: float = 0.0
    lr: float = 0.05
    budget: float = 0.0
    max_value: float = 100.0

    def update(self, observed_cost: float) -> float:
        gradient = observed_cost - self.budget
        self.value = float(min(self.max_value, max(0.0, self.value + self.lr * gradient)))
        return self.value
