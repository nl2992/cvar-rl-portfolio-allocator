from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from crlpa.evaluation.metrics import cvar as hist_cvar
from crlpa.models.actor import mlp
from crlpa.utils.seeds import set_global_seed


class DiffAllocator(nn.Module):
    """Deterministic state -> long-only weights policy (softmax over an MLP).

    Trained by differentiating a risk-adjusted objective directly through the
    portfolio return, rather than by score-function policy gradients. The
    observation layout matches :meth:`AllocationEnv.observation`
    (``[momentum(n), vol(n), weights(n), drawdown, cvar]``) so the same network
    plugs straight into the backtest harness at evaluation time.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        self.net = mlp(obs_dim, hidden, action_dim)

    def forward(self, obs: torch.Tensor, log_anchor: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.net(obs)
        if log_anchor is not None:
            # Residual policy: start from the (adaptive) anchor and learn a tilt.
            # softmax(log(anchor)) == anchor, so a zero net output reproduces it.
            logits = logits + log_anchor
        return torch.softmax(logits, dim=-1)

    def predict(self, obs: np.ndarray, log_anchor: np.ndarray | None = None) -> np.ndarray:
        with torch.no_grad():
            la = None if log_anchor is None else torch.as_tensor(log_anchor, dtype=torch.float32)
            return self.forward(torch.as_tensor(obs, dtype=torch.float32), la).numpy()


@dataclass
class DiffConfig:
    n_updates: int = 1500
    horizon: int = 104
    lr: float = 1e-3
    objective: str = "sharpe"  # "sharpe" (risk-adjusted) | "return" (return-greedy)
    risk_aversion: float = 0.0  # >0 adds a mean-variance term on top of the objective
    turnover_penalty: float = 0.0  # extra penalty on mean one-way turnover (beyond costs)
    weight_decay: float = 1e-4  # L2 regularisation to curb backtest overfitting
    constrained: bool = True
    lagrange_lr: float = 1.0
    eval_every: int = 50  # validation-best early stopping cadence
    anchor: str | None = "inverse_vol"  # adaptive base allocation; None = from scratch
    seed: int = 42
    hidden: tuple[int, ...] = (64, 64)


def _anchor_log_weights(returns: np.ndarray, lookback: int, mode: str | None) -> np.ndarray | None:
    """Per-step log of an adaptive anchor allocation (no look-ahead). None disables it."""
    if mode is None:
        return None
    t_total, n = returns.shape
    log_anchor = np.zeros((t_total, n), dtype=np.float32)  # log(uniform) up to a constant
    for t in range(t_total):
        hist = returns[max(0, t - lookback) : t]
        if hist.shape[0] >= 2 and mode == "inverse_vol":
            vol = hist.std(axis=0)
            vol = np.where(vol <= 1e-8, np.nanmean(vol) + 1e-8, vol)
            w = (1.0 / vol)
            w = w / w.sum()
        else:
            w = np.full(n, 1.0 / n)
        log_anchor[t] = np.log(np.clip(w, 1e-6, None))
    return log_anchor


@torch.no_grad()
def _rollout_sharpe(
    actor: DiffAllocator,
    R: torch.Tensor,
    feats: torch.Tensor,
    cost: float,
    cvar_window: int,
    cvar_alpha: float,
    log_anchor: torch.Tensor | None = None,
    exog: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    """Deterministic full-pass (Sharpe, CVaR, mean return) of net returns for selection."""
    n = R.shape[1]
    w_prev = torch.full((n,), 1.0 / n)
    drawdown, cvar_feat = 0.0, 0.0
    wealth, peak = 1.0, 1.0
    recent: deque = deque(maxlen=cvar_window)
    rets: list[float] = []
    for t in range(R.shape[0]):
        state_tail = torch.tensor([*w_prev.tolist(), drawdown, cvar_feat])
        obs_parts = [feats[t], state_tail] if exog is None else [feats[t], state_tail, exog[t]]
        la = None if log_anchor is None else log_anchor[t]
        w = actor(torch.cat(obs_parts), la)
        ret = float((w * R[t]).sum() - cost * (w - w_prev).abs().sum())
        rets.append(ret)
        wealth *= 1 + ret
        peak = max(peak, wealth)
        drawdown = 1.0 - wealth / peak
        recent.append(ret)
        cvar_feat = hist_cvar(np.asarray(recent), cvar_alpha) if len(recent) >= 5 else 0.0
        w_prev = w
    arr = np.asarray(rets)
    return float(arr.mean() / (arr.std() + 1e-8)), hist_cvar(arr, cvar_alpha), float(arr.mean())


def _market_features(returns: np.ndarray, lookback: int) -> np.ndarray:
    """Precompute [momentum(n), vol(n)] per step using only returns[:t] (no look-ahead)."""
    t_total, n = returns.shape
    feats = np.zeros((t_total, 2 * n), dtype=np.float32)
    for t in range(t_total):
        hist = returns[max(0, t - lookback) : t]
        if hist.shape[0] >= 2:
            feats[t, :n] = hist.mean(axis=0)
            feats[t, n:] = hist.std(axis=0)
    return feats


def train_differentiable(
    returns: pd.DataFrame,
    cost_bps: float,
    cvar_alpha: float,
    cvar_limit: float,
    cvar_window: int,
    lookback: int,
    config: DiffConfig | None = None,
    val_returns: pd.DataFrame | None = None,
    exog: np.ndarray | None = None,
    val_exog: np.ndarray | None = None,
) -> tuple[DiffAllocator, pd.DataFrame]:
    """Train a differentiable allocator on randomised windows of the return series.

    Objective per window: maximise the realised Sharpe ratio (optionally minus a
    mean-variance term), with a differentiable CVaR penalty (mean of the worst
    ``(1-alpha)`` tail of net returns) scaled by a Lagrange multiplier that tracks
    the CVaR-limit violation via dual ascent. Optional ``exog`` adds per-step
    exogenous state (e.g. macro/factor features) appended to the observation.
    """
    config = config or DiffConfig()
    set_global_seed(config.seed)
    R = torch.as_tensor(returns.to_numpy().copy(), dtype=torch.float32)
    t_total, n = R.shape
    cost = cost_bps / 10_000.0
    feats = torch.as_tensor(_market_features(returns.to_numpy(), lookback))
    anchor_np = _anchor_log_weights(returns.to_numpy(), lookback, config.anchor)
    log_anchor = None if anchor_np is None else torch.as_tensor(anchor_np)
    exog_t = None if exog is None else torch.as_tensor(np.asarray(exog), dtype=torch.float32)
    exog_dim = 0 if exog_t is None else exog_t.shape[1]
    obs_dim = 3 * n + 2 + exog_dim
    horizon = min(config.horizon, t_total - 1)
    k_tail = max(1, int(round((1 - cvar_alpha) * horizon)))

    actor = DiffAllocator(obs_dim, n, config.hidden)
    opt = torch.optim.Adam(actor.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    lam = 0.0
    history: list[dict[str, float]] = []

    val_R = val_feats = None
    best_val = -np.inf
    best_feasible_val = -np.inf  # best val Sharpe among checkpoints meeting the CVaR limit
    best_cvar = np.inf  # fallback: lowest val CVaR if none are feasible
    best_state = {k: v.clone() for k, v in actor.state_dict().items()}
    best_feasible_state = None
    # Selection bookkeeping (diagnostics only; does not affect training or selection).
    best_it = best_feasible_it = best_cvar_it = None
    n_evals = n_feasible = 0
    val_anchor = val_exog_t = None
    if val_returns is not None:
        val_R = torch.as_tensor(val_returns.to_numpy().copy(), dtype=torch.float32)
        val_feats = torch.as_tensor(_market_features(val_returns.to_numpy(), lookback))
        va = _anchor_log_weights(val_returns.to_numpy(), lookback, config.anchor)
        val_anchor = None if va is None else torch.as_tensor(va)
        val_exog_t = None if val_exog is None else torch.as_tensor(np.asarray(val_exog), dtype=torch.float32)

    for it in range(config.n_updates):
        start = int(rng.integers(0, max(1, t_total - horizon)))
        w_prev = torch.full((n,), 1.0 / n)
        wealth, peak, drawdown, cvar_feat = 1.0, 1.0, 0.0, 0.0
        recent: deque = deque(maxlen=cvar_window)
        rets: list[torch.Tensor] = []
        turnovers: list[torch.Tensor] = []

        for t in range(start, start + horizon):
            state_tail = torch.tensor([*w_prev.detach().tolist(), drawdown, cvar_feat])
            obs_parts = [feats[t], state_tail] if exog_t is None else [feats[t], state_tail, exog_t[t]]
            obs = torch.cat(obs_parts)
            la = None if log_anchor is None else log_anchor[t]
            w = actor(obs, la)
            turnover = (w - w_prev.detach()).abs().sum()
            ret = (w * R[t]).sum() - cost * turnover
            rets.append(ret)
            turnovers.append(turnover)

            r_val = float(ret.item())
            wealth *= 1 + r_val
            peak = max(peak, wealth)
            drawdown = 1.0 - wealth / peak
            recent.append(r_val)
            cvar_feat = hist_cvar(np.asarray(recent), cvar_alpha) if len(recent) >= 5 else 0.0
            w_prev = w

        rets_t = torch.stack(rets)
        sharpe = rets_t.mean() / (rets_t.std() + 1e-8)
        tail = torch.topk(-rets_t, k_tail).values
        cvar_t = tail.mean()

        if config.objective == "return":
            loss = -torch.log1p(rets_t.clamp(min=-0.999)).mean()
        else:
            loss = -sharpe
        if config.risk_aversion > 0:
            loss = loss + config.risk_aversion * rets_t.var()
        if config.turnover_penalty > 0:
            loss = loss + config.turnover_penalty * torch.stack(turnovers).mean()
        if config.constrained:
            loss = loss + lam * torch.relu(cvar_t - cvar_limit)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt.step()

        if config.constrained:
            lam = float(min(100.0, max(0.0, lam + config.lagrange_lr * (cvar_t.item() - cvar_limit))))

        record = {
            "update": it,
            "sharpe": float(sharpe.item()),
            "cvar": float(cvar_t.item()),
            "lagrange": lam,
            "loss": float(loss.item()),
        }
        if val_R is not None and it % config.eval_every == 0:
            val_sharpe, val_cvar, val_ret = _rollout_sharpe(
                actor, val_R, val_feats, cost, cvar_window, cvar_alpha, val_anchor, val_exog_t
            )
            # Select by the same criterion the objective optimises.
            val_metric = val_ret if config.objective == "return" else val_sharpe
            record["val_sharpe"] = val_sharpe
            record["val_cvar"] = val_cvar
            snapshot = {k: v.clone() for k, v in actor.state_dict().items()}
            n_evals += 1
            if val_metric > best_val:
                best_val, best_state, best_it = val_metric, snapshot, it
            # Constrained selection: best objective metric among checkpoints meeting the limit.
            if val_cvar <= cvar_limit + 1e-6:
                n_feasible += 1
                if val_metric > best_feasible_val:
                    best_feasible_val, best_feasible_state, best_feasible_it = val_metric, snapshot, it
            if val_cvar < best_cvar:
                best_cvar, best_cvar_state, best_cvar_it = val_cvar, snapshot, it
        history.append(record)

    selection: dict = {"rule": "final", "selected_update": config.n_updates - 1,
                       "n_evals": n_evals, "n_feasible": n_feasible}
    if val_R is not None:
        if config.constrained:
            # Prefer the highest-Sharpe checkpoint that satisfies the CVaR limit on
            # validation; if none do, fall back to the lowest-CVaR checkpoint.
            actor.load_state_dict(best_feasible_state or best_cvar_state)
            if best_feasible_state is not None:
                selection.update(rule="feasible", selected_update=best_feasible_it)
            else:
                selection.update(rule="lowest_cvar_fallback", selected_update=best_cvar_it)
        else:
            actor.load_state_dict(best_state)
            selection.update(rule="best_val", selected_update=best_it)
    history_df = pd.DataFrame(history)
    history_df.attrs["selection"] = selection
    return actor, history_df


def _step_log_anchor(env, lookback: int, anchor: str | None) -> np.ndarray | None:
    """Per-step log anchor from history before the current step (no look-ahead)."""
    if anchor is None:
        return None
    t = env.state.step
    hist = env.returns.iloc[max(0, t - lookback) : t].to_numpy()
    n = env.n_assets
    if hist.shape[0] >= 2 and anchor == "inverse_vol":
        vol = hist.std(axis=0)
        vol = np.where(vol <= 1e-8, np.nanmean(vol) + 1e-8, vol)
        w = 1.0 / vol
        w = w / w.sum()
    else:
        w = np.full(n, 1.0 / n)
    return np.log(np.clip(w, 1e-6, None)).astype(np.float32)


def diff_policy(actor: DiffAllocator, lookback: int, anchor: str | None = "inverse_vol"):
    """Backtest policy wrapper that recomputes the adaptive anchor each step."""

    def policy(env) -> np.ndarray:
        return actor.predict(env.observation(), _step_log_anchor(env, lookback, anchor))

    return policy


def ensemble_diff_policy(
    actors: list[DiffAllocator], lookback: int, anchor: str | None = "inverse_vol"
):
    """Average the weights of several seed-trained actors (a robust inference policy)."""

    def policy(env) -> np.ndarray:
        log_anchor = _step_log_anchor(env, lookback, anchor)
        obs = env.observation()
        weights = np.mean([a.predict(obs, log_anchor) for a in actors], axis=0)
        return weights / weights.sum()

    return policy

