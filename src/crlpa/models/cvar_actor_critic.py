from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from crlpa.models.actor import GaussianSimplexActor
from crlpa.models.critic import ValueCritic
from crlpa.models.safety_critic import SafetyCritic
from crlpa.training.lagrangian import LagrangeMultiplier


@dataclass
class ActorCriticConfig:
    hidden: tuple[int, ...] = (64, 64)
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    log_std_init: float = -0.5
    constrained: bool = True  # False reproduces an unconstrained A2C allocator
    cost_mode: str = "breach"  # "breach" (0/1 indicator) | "magnitude" (raw CVaR excess)
    lagrange_lr: float = 1.0
    cvar_budget: float = 0.05  # tolerated breach rate when cost_mode="breach"
    max_grad_norm: float = 1.0


@dataclass
class StepCache:
    """Per-step tensors retained for the episodic policy-gradient update."""

    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    safety_value: torch.Tensor


class CVaRActorCritic:
    """Constrained actor-critic with a reward critic, safety critic, and dual.

    Setting ``config.constrained = False`` freezes the Lagrange multiplier at zero
    and ignores the safety critic in the policy objective, recovering a standard
    unconstrained actor-critic allocator for the Phase 7 baseline.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: ActorCriticConfig | None = None):
        self.config = config or ActorCriticConfig()
        self.actor = GaussianSimplexActor(
            obs_dim, action_dim, self.config.hidden, self.config.log_std_init
        )
        self.critic = ValueCritic(obs_dim, self.config.hidden)
        self.safety_critic = SafetyCritic(obs_dim, self.config.hidden)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.safety_critic.parameters()),
            lr=self.config.critic_lr,
        )
        self.lagrange = LagrangeMultiplier(
            lr=self.config.lagrange_lr, budget=self.config.cvar_budget
        )

    # --- acting -------------------------------------------------------------

    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, StepCache]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        weights, log_prob, entropy = self.actor.act(obs_t, deterministic=deterministic)
        cache = StepCache(
            log_prob=log_prob,
            entropy=entropy,
            value=self.critic(obs_t),
            safety_value=self.safety_critic(obs_t),
        )
        return weights.detach().numpy(), cache

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic weights for evaluation/backtest."""
        with torch.no_grad():
            weights, _, _ = self.actor.act(
                torch.as_tensor(obs, dtype=torch.float32), deterministic=True
            )
        return weights.numpy()

    # --- learning -----------------------------------------------------------

    def _discounted(self, values: list[float], gamma: float) -> torch.Tensor:
        out = np.zeros(len(values), dtype=np.float32)
        running = 0.0
        for i in reversed(range(len(values))):
            running = values[i] + gamma * running
            out[i] = running
        return torch.as_tensor(out)

    def update(
        self,
        cache: list[StepCache],
        rewards: list[float],
        costs: list[float],
        entropy_coef: float | None = None,
    ) -> dict[str, float]:
        gamma = self.config.gamma
        entropy_coef = self.config.entropy_coef if entropy_coef is None else entropy_coef

        reward_to_go = self._discounted(rewards, gamma)
        cost_to_go = self._discounted(costs, gamma)
        log_probs = torch.stack([c.log_prob for c in cache])
        entropies = torch.stack([c.entropy for c in cache])
        values = torch.stack([c.value for c in cache])
        safety_values = torch.stack([c.safety_value for c in cache])

        advantage = reward_to_go - values.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        lam = self.lagrange.value if self.config.constrained else 0.0
        if self.config.constrained:
            cost_advantage = cost_to_go - safety_values.detach()
            # Standardise the cost advantage to the same unit scale as the reward
            # advantage. Without this the raw CVaR cost (~1e-3) is dwarfed by the
            # standardised reward advantage (O(1)), so the constraint never binds
            # for any reasonable Lagrange multiplier.
            cost_advantage = (cost_advantage - cost_advantage.mean()) / (
                cost_advantage.std() + 1e-8
            )
            penalised = advantage - lam * cost_advantage
        else:
            penalised = advantage

        policy_loss = -(log_probs * penalised.detach()).mean()
        entropy_loss = -entropy_coef * entropies.mean()
        actor_loss = policy_loss + entropy_loss

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_opt.step()

        value_loss = nn.functional.mse_loss(values, reward_to_go)
        safety_loss = nn.functional.mse_loss(safety_values, cost_to_go)
        critic_loss = self.config.value_coef * (value_loss + safety_loss)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) + list(self.safety_critic.parameters()),
            self.config.max_grad_norm,
        )
        self.critic_opt.step()

        mean_cost = float(np.mean(costs)) if costs else 0.0
        if self.config.constrained:
            self.lagrange.update(mean_cost)

        return {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "safety_loss": float(safety_loss.detach()),
            "entropy": float(entropies.mean().detach()),
            "lagrange": float(lam),
            "mean_cost": mean_cost,
        }

    # --- persistence --------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "safety_critic": self.safety_critic.state_dict(),
            "lagrange": self.lagrange.value,
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.safety_critic.load_state_dict(state["safety_critic"])
        self.lagrange.value = state.get("lagrange", 0.0)
