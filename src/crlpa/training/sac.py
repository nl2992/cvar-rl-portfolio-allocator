from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from crlpa.envs.allocation import AllocationEnv
from crlpa.models.actor import GaussianSimplexActor
from crlpa.models.critic import QCritic
from crlpa.training.lagrangian import LagrangeMultiplier
from crlpa.utils.seeds import set_global_seed


@dataclass
class SACConfig:
    n_iterations: int = 150
    rollout_len: int = 104       # env steps collected per iteration (random start)
    updates_per_step: int = 1    # gradient updates per collected step
    batch_size: int = 128
    buffer_size: int = 50_000
    warmup_steps: int = 256      # random-action steps before learning starts
    gamma: float = 0.95
    tau: float = 0.01            # target soft-update rate
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    alpha_lr: float = 3e-4
    init_alpha: float = 0.1      # entropy temperature (initial / fixed)
    autotune_alpha: bool = True  # learn alpha toward a target entropy
    hidden: tuple[int, ...] = (64, 64)
    constrained: bool = False
    cost_mode: str = "breach"    # breach indicator drives the dual (well scaled)
    lagrange_lr: float = 1.0
    cvar_budget: float = 0.05
    seed: int = 42


class _ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: deque = deque(maxlen=capacity)

    def add(self, obs, act, rew, next_obs, done):
        self.buf.append((obs, act, rew, next_obs, done))

    def sample(self, n: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self.buf), size=n)
        items = [self.buf[i] for i in idx]
        obs, act, rew, next_obs, done = zip(*items)
        return (
            torch.as_tensor(np.asarray(obs), dtype=torch.float32),
            torch.as_tensor(np.asarray(act), dtype=torch.float32),
            torch.as_tensor(np.asarray(rew), dtype=torch.float32),
            torch.as_tensor(np.asarray(next_obs), dtype=torch.float32),
            torch.as_tensor(np.asarray(done), dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)


class SACAllocator:
    """Soft Actor-Critic over the long-only simplex (off-policy model-free arm).

    Twin Q-critics with target networks, a stochastic Gaussian-simplex actor, and an
    (optionally auto-tuned) entropy temperature. The action is the pre-softmax latent
    — the env applies the softmax to obtain weights — so the policy parameterisation
    matches the PPO and A2C arms and the comparison is like-for-like. An optional CVaR
    Lagrangian shapes the reward by a breach-rate budget, as in the PPO arm.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: SACConfig | None = None):
        self.config = config or SACConfig()
        self.action_dim = action_dim
        self.actor = GaussianSimplexActor(obs_dim, action_dim, self.config.hidden)
        self.q1 = QCritic(obs_dim, action_dim, self.config.hidden)
        self.q2 = QCritic(obs_dim, action_dim, self.config.hidden)
        self.q1_t = QCritic(obs_dim, action_dim, self.config.hidden)
        self.q2_t = QCritic(obs_dim, action_dim, self.config.hidden)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.config.critic_lr
        )
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.tensor(float(np.log(self.config.init_alpha)), requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
        self.lagrange = LagrangeMultiplier(lr=self.config.lagrange_lr, budget=self.config.cvar_budget)

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp()) if self.config.autotune_alpha else self.config.init_alpha

    def predict(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            w, _, _ = self.actor.act(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)
        return w.numpy()

    def _sample_action(self, obs_t: torch.Tensor):
        """Sample latent action and its log-prob (sum over assets)."""
        dist = self.actor.distribution(obs_t)
        latent = dist.rsample()
        logp = dist.log_prob(latent).sum(-1)
        return latent, logp

    def _soft_update(self) -> None:
        tau = self.config.tau
        for net, net_t in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
            for p, pt in zip(net.parameters(), net_t.parameters()):
                pt.data.mul_(1 - tau).add_(tau * p.data)

    def _update(self, buffer: _ReplayBuffer, rng: np.random.Generator) -> None:
        cfg = self.config
        obs, act, rew, next_obs, done = buffer.sample(cfg.batch_size, rng)

        with torch.no_grad():
            next_latent, next_logp = self._sample_action(next_obs)
            q1_t = self.q1_t(next_obs, next_latent)
            q2_t = self.q2_t(next_obs, next_latent)
            min_q_t = torch.min(q1_t, q2_t) - self.alpha * next_logp
            target = rew + cfg.gamma * (1 - done) * min_q_t

        q1 = self.q1(obs, act)
        q2 = self.q2(obs, act)
        q_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        latent, logp = self._sample_action(obs)
        q1_pi = self.q1(obs, latent)
        q2_pi = self.q2(obs, latent)
        actor_loss = (self.alpha * logp - torch.min(q1_pi, q2_pi)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        if cfg.autotune_alpha:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        self._soft_update()

    def _collect(self, env: AllocationEnv, buffer: _ReplayBuffer, rng: np.random.Generator,
                 learning: bool) -> list[float]:
        """Roll a random-start segment, pushing transitions; return per-step costs."""
        cfg = self.config
        start = int(rng.integers(0, max(1, len(env.returns) - cfg.rollout_len)))
        env.reset()
        env.state.step = start
        costs: list[float] = []
        for _ in range(cfg.rollout_len):
            obs = env.observation()
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                if learning:
                    latent, _ = self._sample_action(obs_t)
                else:  # warmup: sample latent from the prior (unconditioned exploration)
                    latent = torch.as_tensor(rng.normal(size=self.action_dim), dtype=torch.float32)
                w = self.actor.to_weights(latent)
            _, reward, done, info = env.step(w.numpy())
            cost = info["cvar_breach"] if cfg.cost_mode == "breach" else info["cvar_cost"]
            shaped = reward - (self.lagrange.value * cost if cfg.constrained else 0.0)
            next_obs = env.observation()
            buffer.add(obs, latent.numpy(), float(shaped), next_obs, float(done))
            costs.append(cost)
            if done:
                break
        return costs


def train_sac(
    train_env: AllocationEnv, agent: SACAllocator, config: SACConfig | None = None
) -> tuple[SACAllocator, pd.DataFrame]:
    config = config or agent.config
    set_global_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    buffer = _ReplayBuffer(config.buffer_size)
    history = []
    total_steps = 0
    for it in range(config.n_iterations):
        learning = total_steps >= config.warmup_steps
        costs = agent._collect(train_env, buffer, rng, learning)
        total_steps += len(costs)
        if len(buffer) >= config.batch_size and learning:
            for _ in range(len(costs) * config.updates_per_step):
                agent._update(buffer, rng)
        if config.constrained:
            agent.lagrange.update(float(np.mean(costs)))
        history.append({
            "iteration": it,
            "breach_rate": float(np.mean(np.asarray(costs) > 0)),
            "alpha": agent.alpha,
            "lagrange": agent.lagrange.value,
        })
    return agent, pd.DataFrame(history)
