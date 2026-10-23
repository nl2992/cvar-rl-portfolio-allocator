from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from crlpa.envs.allocation import AllocationEnv
from crlpa.models.actor import GaussianSimplexActor
from crlpa.models.critic import ValueCritic
from crlpa.training.lagrangian import LagrangeMultiplier
from crlpa.utils.seeds import set_global_seed


@dataclass
class PPOConfig:
    n_iterations: int = 150
    rollout_len: int = 104  # steps collected per iteration from a random start
    epochs: int = 10
    minibatch: int = 64
    clip: float = 0.2
    gamma: float = 0.95
    gae_lambda: float = 0.95
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.01
    hidden: tuple[int, ...] = (64, 64)
    constrained: bool = False
    cost_mode: str = "breach"  # breach indicator drives the dual (well scaled)
    lagrange_lr: float = 1.0
    cvar_budget: float = 0.05
    seed: int = 42


class PPOAllocator:
    """Clipped-surrogate PPO over the long-only simplex with start-point sampling.

    A proper on-policy baseline for the model-free arm: GAE advantages, a clipped
    surrogate with multiple epochs/minibatches, entropy bonus, and an optional CVaR
    Lagrangian (breach-rate budget). The actor reuses the Gaussian-softmax policy so
    it is directly comparable to the A2C and differentiable allocators.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig | None = None):
        self.config = config or PPOConfig()
        self.actor = GaussianSimplexActor(obs_dim, action_dim, self.config.hidden)
        self.critic = ValueCritic(obs_dim, self.config.hidden)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.lagrange = LagrangeMultiplier(lr=self.config.lagrange_lr, budget=self.config.cvar_budget)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            w, _, _ = self.actor.act(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)
        return w.numpy()

    @staticmethod
    def _gae(rewards, values, gamma, lam):
        adv = np.zeros(len(rewards), dtype=np.float32)
        last = 0.0
        for t in reversed(range(len(rewards))):
            next_v = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + gamma * next_v - values[t]
            last = delta + gamma * lam * last
            adv[t] = last
        return adv

    def _collect(self, env: AllocationEnv, rng: np.random.Generator):
        """Roll a random-start segment, returning transitions and the cost signal."""
        cfg = self.config
        start = int(rng.integers(0, max(1, len(env.returns) - cfg.rollout_len)))
        env.reset()
        env.state.step = start  # random episode start point for trajectory diversity
        obs_list, act_logp, rewards, values, costs = [], [], [], [], []
        for _ in range(cfg.rollout_len):
            obs = env.observation()
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                dist = self.actor.distribution(obs_t)
                latent = dist.rsample()
                logp = dist.log_prob(latent).sum(-1)
                w = self.actor.to_weights(latent)
                value = self.critic(obs_t)
            _, reward, done, info = env.step(w.numpy())
            obs_list.append(obs)
            act_logp.append((latent.numpy(), float(logp)))
            rewards.append(reward)
            values.append(float(value))
            costs.append(info["cvar_breach"] if cfg.cost_mode == "breach" else info["cvar_cost"])
            if done:
                break
        return obs_list, act_logp, rewards, values, costs

    def update(self, env: AllocationEnv, rng: np.random.Generator) -> dict[str, float]:
        cfg = self.config
        obs_list, act_logp, rewards, values, costs = self._collect(env, rng)
        n = len(rewards)
        rewards = np.asarray(rewards, dtype=np.float32)
        if cfg.constrained:
            rewards = rewards - self.lagrange.value * np.asarray(costs, dtype=np.float32)

        adv = self._gae(rewards, values, cfg.gamma, cfg.gae_lambda)
        returns = adv + np.asarray(values, dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.as_tensor(np.asarray(obs_list), dtype=torch.float32)
        latents = torch.as_tensor(np.asarray([a for a, _ in act_logp]), dtype=torch.float32)
        old_logp = torch.as_tensor(np.asarray([lp for _, lp in act_logp]), dtype=torch.float32)
        adv_t = torch.as_tensor(adv)
        ret_t = torch.as_tensor(returns)

        idx = np.arange(n)
        for _ in range(cfg.epochs):
            rng.shuffle(idx)
            for s in range(0, n, cfg.minibatch):
                b = idx[s : s + cfg.minibatch]
                dist = self.actor.distribution(obs_t[b])
                logp = dist.log_prob(latents[b]).sum(-1)
                ratio = torch.exp(logp - old_logp[b])
                clipped = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)
                policy_loss = -torch.min(ratio * adv_t[b], clipped * adv_t[b]).mean()
                entropy = dist.entropy().sum(-1).mean()
                actor_loss = policy_loss - cfg.entropy_coef * entropy

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_opt.step()

                value_loss = nn.functional.mse_loss(self.critic(obs_t[b]), ret_t[b])
                self.critic_opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_opt.step()

        breach_rate = float(np.mean(np.asarray(costs) > 0))
        if cfg.constrained:
            self.lagrange.update(float(np.mean(costs)))
        return {"breach_rate": breach_rate, "lagrange": self.lagrange.value,
                "mean_reward": float(rewards.mean())}


def train_ppo(
    train_env: AllocationEnv, agent: PPOAllocator, config: PPOConfig | None = None
) -> tuple[PPOAllocator, pd.DataFrame]:
    config = config or agent.config
    set_global_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    history = []
    for it in range(config.n_iterations):
        stats = agent.update(train_env, rng)
        history.append({"iteration": it, **stats})
    return agent, pd.DataFrame(history)
