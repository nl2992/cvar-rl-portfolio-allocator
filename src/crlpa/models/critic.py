from __future__ import annotations

import torch
from torch import nn

from crlpa.models.actor import mlp


class ValueCritic(nn.Module):
    """Estimates the expected discounted reward-to-go V(s)."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        self.net = mlp(obs_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class QCritic(nn.Module):
    """Estimates the state-action value Q(s, a).

    The action is the pre-softmax latent (matching the Gaussian-simplex actor, where
    weights = softmax(latent)), so the SAC critic and actor share the same action
    parameterisation as the PPO/A2C arms and stay directly comparable.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        self.net = mlp(obs_dim + action_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)
