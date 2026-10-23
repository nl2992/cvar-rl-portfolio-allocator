from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.Tanh()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class GaussianSimplexActor(nn.Module):
    """Stochastic policy over the long-only simplex.

    The network outputs a mean vector of pre-softmax logits. A diagonal Gaussian
    with learnable log-std is placed over these logits; a sampled latent ``z`` is
    mapped to portfolio weights by a softmax. Policy gradients use the score
    function of the latent Gaussian, so the softmax is treated as a deterministic
    part of the environment transition.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: tuple[int, ...] = (64, 64),
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.net = mlp(obs_dim, hidden, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = self.log_std.exp().clamp(1e-3, 2.0)
        return Normal(mean, std)

    @staticmethod
    def to_weights(latent: torch.Tensor) -> torch.Tensor:
        return torch.softmax(latent, dim=-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """Return (weights, log_prob, entropy) for a single observation."""
        dist = self.distribution(obs)
        latent = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(latent).sum(-1)
        entropy = dist.entropy().sum(-1)
        return self.to_weights(latent), log_prob, entropy
