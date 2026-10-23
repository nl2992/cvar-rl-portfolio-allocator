from __future__ import annotations

import torch
from torch import nn

from crlpa.models.actor import mlp


class SafetyCritic(nn.Module):
    """Estimates the expected discounted CVaR-constraint cost-to-go.

    The constraint cost per step is the non-negative CVaR breach amount
    ``max(0, cvar_estimate - cvar_limit)``, so the output is passed through a
    softplus to keep the predicted cost-to-go non-negative.
    """

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        self.net = mlp(obs_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.net(obs).squeeze(-1))
