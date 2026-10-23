"""Decision-time CVaR risk filter (post-action projection).

Motivation (see PLAN_P5_make_the_constraint_bind.md): a soft Lagrangian penalty
discourages CVaR breaches but cannot enforce the budget out of sample -- the
multiplier is fitted on in-sample tail behaviour and is simply too small once
the realised tail is fatter than anything seen in training. This module
inserts a *hard* projection between the actor's proposed weights and the
environment:

  1. Estimate the forward CVaR of the proposed portfolio from trailing
     information only (a rolling covariance/mean and a Gaussian tail
     multiplier -- no future returns, ever).
  2. If the estimate exceeds the budget, shrink the proposal along the
     straight line toward a trailing long-only minimum-variance anchor,
     solving numerically for the *largest* mixing weight ``s`` (the smallest
     shrink away from the actor's own proposal) that clears the budget.
  3. Trade the projected weights. The existing turnover cap / max-weight /
     long-only projection in :class:`crlpa.envs.allocation.AllocationEnv`
     composes AFTER this step, unchanged.

The shrink factor ``s`` is solved numerically (bisection on a closed-form
Gaussian CVaR-of-mixture expression) from *detached* trailing statistics, then
applied to the actor's output as an affine combination
``w_filtered = anchor + s * (w_proposed - anchor)``. Gradients flow back to
the actor through this affine map (a straight-through style projection layer:
the "how much to shrink" decision is computed from data, not backprop'd
through), so a policy trained with the filter active learns against the
filtered dynamics rather than being surprised by the filter at test time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from torch import nn

from crlpa.evaluation.backtest import BacktestResult
from crlpa.evaluation.metrics import cvar as hist_cvar
from crlpa.evaluation.metrics import summarise
from crlpa.policies.baselines import min_variance
from crlpa.training.differentiable import (
    DiffAllocator,
    _anchor_log_weights,
    _market_features,
    _step_log_anchor,
)
from crlpa.utils.seeds import set_global_seed


def gaussian_cvar_multiplier(alpha: float) -> float:
    """Gaussian-tail CVaR multiplier: for a loss ~ N(-mu, sigma^2),
    CVaR_alpha(loss) = c(alpha) * sigma - mu, with c(alpha) = phi(z_alpha)/(1-alpha).

    This is the standard closed-form parametric CVaR used as a fast, trailing
    (no-future-data) risk estimate; it is what makes the filter's forward-CVaR
    evaluation cheap enough to call at every decision step.
    """
    z = norm.ppf(alpha)
    return float(norm.pdf(z) / (1.0 - alpha))


# --------------------------------------------------------------------------
# Trailing statistics: covariance, mean, and a long-only min-variance anchor,
# every one of them computed from returns[:t] only.
# --------------------------------------------------------------------------


@dataclass
class FilterState:
    sigma: np.ndarray   # (T, n, n) trailing covariance, from returns[:t]
    mu: np.ndarray       # (T, n)    trailing mean return, from returns[:t]
    anchor: np.ndarray   # (T, n)    trailing long-only min-variance anchor
    active: np.ndarray   # (T,) bool -- False during warmup (insufficient history)


def precompute_filter_state(
    returns: pd.DataFrame,
    lookback: int = 104,
    max_weight: float = 0.4,
    min_history: int = 10,
) -> FilterState:
    """Precompute per-step trailing covariance / mean / min-variance anchor.

    Row ``t`` uses only ``returns.iloc[max(0, t-lookback):t]`` -- strictly
    before ``t``, matching the no-look-ahead convention used throughout the
    repo (``_market_features``, ``_anchor_log_weights``,
    ``rolling_weight_policy``). Rows with fewer than ``min_history``
    observations are marked inactive; the filter passes the proposal through
    unchanged during that warmup, exactly as the other rolling estimators do.
    """
    data = returns.to_numpy(dtype=float)
    t_total, n = data.shape
    sigma = np.zeros((t_total, n, n))
    mu = np.zeros((t_total, n))
    anchor = np.full((t_total, n), 1.0 / n)
    active = np.zeros(t_total, dtype=bool)
    for t in range(t_total):
        hist = data[max(0, t - lookback) : t]
        if hist.shape[0] < min_history:
            continue
        sigma[t] = np.cov(hist, rowvar=False)
        mu[t] = hist.mean(axis=0)
        hist_df = returns.iloc[max(0, t - lookback) : t]
        anchor[t] = min_variance(hist_df, lookback=lookback, max_weight=max_weight)
        active[t] = True
    return FilterState(sigma=sigma, mu=mu, anchor=anchor, active=active)


# --------------------------------------------------------------------------
# The projection itself.
# --------------------------------------------------------------------------


def cvar_filter_step(
    w_proposed: torch.Tensor,
    sigma_t: torch.Tensor,
    mu_t: torch.Tensor,
    anchor_t: torch.Tensor,
    budget: float,
    c_alpha: float,
    n_bisect: int = 40,
) -> tuple[torch.Tensor, float, dict]:
    """Project ``w_proposed`` toward ``anchor_t`` so trailing CVaR clears ``budget``.

    Parametrises the line ``w(s) = anchor_t + s * (w_proposed - anchor_t)``,
    ``s in [0, 1]``. Along this line the (Gaussian, trailing) CVaR estimate is

        CVaR(s) = c_alpha * sqrt(A + 2*B*s + C*s^2) - (M0 + M1*s)

    with ``A, B, C`` the quadratic form of the trailing covariance evaluated
    at the anchor / cross / delta terms and ``M0, M1`` the trailing mean
    evaluated the same way. Because the anchor is a covariance minimiser,
    ``A + 2Bs + Cs^2`` is non-decreasing in ``s`` on ``[0, 1]`` (a convex
    quadratic minimised at or near ``s=0``), so CVaR(s) is (numerically)
    increasing in ``s`` and bisection finds the unique crossing. We solve for
    the LARGEST feasible ``s`` -- the smallest shrink away from the actor's
    own proposal that still clears the budget -- from *detached* trailing
    statistics (no gradient through the numeric solve itself), then apply it
    to the actor's live output as an affine mix so gradients still flow to
    the actor through ``(w_proposed - anchor_t)``.
    """
    with torch.no_grad():
        w_det = w_proposed.detach()
        d_vec = w_det - anchor_t
        A = float(anchor_t @ sigma_t @ anchor_t)
        B = float(d_vec @ sigma_t @ anchor_t)
        C = float(d_vec @ sigma_t @ d_vec)
        M0 = float(mu_t @ anchor_t)
        M1 = float(mu_t @ d_vec)

        def cvar_at(s: float) -> float:
            var = max(A + 2 * B * s + C * s * s, 1e-14)
            return c_alpha * var**0.5 - (M0 + M1 * s)

        cvar_proposed = cvar_at(1.0)
        cvar_anchor = cvar_at(0.0)
        anchor_breach = cvar_anchor > budget

        if cvar_proposed <= budget:
            s = 1.0
        elif anchor_breach:
            # Even the anchor is estimated to breach on this trailing window:
            # the estimator, not the mechanism, is the bottleneck here. Fall
            # back to the anchor itself (the best available point) and flag it.
            s = 0.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                if cvar_at(mid) <= budget:
                    lo = mid
                else:
                    hi = mid
            s = lo

    w_filtered = anchor_t + s * (w_proposed - anchor_t)
    diag = {
        "shrink_s": s,
        "cvar_proposed": cvar_proposed,
        "cvar_anchor": cvar_anchor,
        "anchor_breach": anchor_breach,
        "filter_active": cvar_proposed > budget,
    }
    return w_filtered, s, diag


# --------------------------------------------------------------------------
# Training with the filter (optionally) active, and with the actor's
# observation (optionally) conditioned on the current Lagrange multiplier so
# the SAME frozen network can be made more conservative at evaluation time by
# an online-updated multiplier (the "adaptive dual" arms A7/A8).
# --------------------------------------------------------------------------


@dataclass
class FilteredConfig:
    n_updates: int = 1500
    horizon: int = 104
    lr: float = 1e-3
    objective: str = "return"
    risk_aversion: float = 0.0
    turnover_penalty: float = 0.0
    weight_decay: float = 1e-4
    constrained: bool = True
    lagrange_lr: float = 1.0
    eval_every: int = 50
    anchor: str | None = "inverse_vol"
    seed: int = 42
    hidden: tuple[int, ...] = (64, 64)
    filter_on: bool = False
    lam_conditioned: bool = False


def _obs_dim(n: int, lam_conditioned: bool) -> int:
    return 3 * n + 2 + (1 if lam_conditioned else 0)


@torch.no_grad()
def _rollout_metrics_filtered(
    actor: DiffAllocator,
    R: torch.Tensor,
    feats: torch.Tensor,
    cost: float,
    cvar_window: int,
    cvar_alpha: float,
    log_anchor: torch.Tensor | None,
    lam: float,
    config: FilteredConfig,
    filter_state: FilterState | None,
    cvar_limit: float,
    c_alpha: float,
) -> tuple[float, float, float]:
    n = R.shape[1]
    w_prev = torch.full((n,), 1.0 / n)
    drawdown, cvar_feat = 0.0, 0.0
    wealth, peak = 1.0, 1.0
    recent: deque = deque(maxlen=cvar_window)
    rets: list[float] = []
    sigma_all = mu_all = anchor_all = None
    active_all = None
    if filter_state is not None:
        sigma_all = torch.as_tensor(filter_state.sigma, dtype=torch.float32)
        mu_all = torch.as_tensor(filter_state.mu, dtype=torch.float32)
        anchor_all = torch.as_tensor(filter_state.anchor, dtype=torch.float32)
        active_all = filter_state.active
    for t in range(R.shape[0]):
        tail_vals = [*w_prev.tolist(), drawdown, cvar_feat]
        if config.lam_conditioned:
            tail_vals.append(lam)
        state_tail = torch.tensor(tail_vals, dtype=torch.float32)
        la = None if log_anchor is None else log_anchor[t]
        w = actor(torch.cat([feats[t], state_tail]), la)
        if config.filter_on and active_all is not None and t < len(active_all) and active_all[t]:
            w, _s, _d = cvar_filter_step(w, sigma_all[t], mu_all[t], anchor_all[t], cvar_limit, c_alpha)
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


def train_filtered(
    returns: pd.DataFrame,
    cost_bps: float,
    cvar_alpha: float,
    cvar_limit: float,
    cvar_window: int,
    lookback: int,
    config: FilteredConfig,
    val_returns: pd.DataFrame | None = None,
    filter_state: FilterState | None = None,
    val_filter_state: FilterState | None = None,
) -> tuple[DiffAllocator, pd.DataFrame]:
    """Same training recipe as ``train_differentiable`` (identical loss, optimiser,
    horizon sampling, dual ascent, validation-feasible checkpoint selection),
    with two optional additions gated by ``config.filter_on`` /
    ``config.lam_conditioned`` so that with both False this reduces, step for
    step, to the unfiltered training loop used for the existing arms.
    """
    set_global_seed(config.seed)
    c_alpha = gaussian_cvar_multiplier(cvar_alpha)
    R = torch.as_tensor(returns.to_numpy().copy(), dtype=torch.float32)
    t_total, n = R.shape
    cost = cost_bps / 10_000.0
    feats = torch.as_tensor(_market_features(returns.to_numpy(), lookback))
    anchor_np = _anchor_log_weights(returns.to_numpy(), lookback, config.anchor)
    log_anchor = None if anchor_np is None else torch.as_tensor(anchor_np)
    obs_dim = _obs_dim(n, config.lam_conditioned)
    horizon = min(config.horizon, t_total - 1)
    k_tail = max(1, int(round((1 - cvar_alpha) * horizon)))

    sigma_all = mu_all = anchor_all = None
    active_all = None
    if filter_state is not None:
        sigma_all = torch.as_tensor(filter_state.sigma, dtype=torch.float32)
        mu_all = torch.as_tensor(filter_state.mu, dtype=torch.float32)
        anchor_all = torch.as_tensor(filter_state.anchor, dtype=torch.float32)
        active_all = filter_state.active

    actor = DiffAllocator(obs_dim, n, config.hidden)
    opt = torch.optim.Adam(actor.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    lam = 0.0
    history: list[dict] = []

    val_R = val_feats = None
    best_val = -np.inf
    best_feasible_val = -np.inf
    best_cvar = np.inf
    best_state = {k: v.clone() for k, v in actor.state_dict().items()}
    best_feasible_state = None
    best_cvar_state = best_state
    # Selection bookkeeping (diagnostics only; does not affect training or selection).
    best_it = best_feasible_it = best_cvar_it = None
    n_evals = n_feasible = 0
    val_anchor = None
    if val_returns is not None:
        val_R = torch.as_tensor(val_returns.to_numpy().copy(), dtype=torch.float32)
        val_feats = torch.as_tensor(_market_features(val_returns.to_numpy(), lookback))
        va = _anchor_log_weights(val_returns.to_numpy(), lookback, config.anchor)
        val_anchor = None if va is None else torch.as_tensor(va)

    for it in range(config.n_updates):
        start = int(rng.integers(0, max(1, t_total - horizon)))
        w_prev = torch.full((n,), 1.0 / n)
        wealth, peak, drawdown, cvar_feat = 1.0, 1.0, 0.0, 0.0
        recent: deque = deque(maxlen=cvar_window)
        rets: list[torch.Tensor] = []
        turnovers: list[torch.Tensor] = []
        shrinks: list[float] = []

        for t in range(start, start + horizon):
            tail_vals = [*w_prev.detach().tolist(), drawdown, cvar_feat]
            if config.lam_conditioned:
                tail_vals.append(lam)
            state_tail = torch.tensor(tail_vals, dtype=torch.float32)
            obs = torch.cat([feats[t], state_tail])
            la = None if log_anchor is None else log_anchor[t]
            w_raw = actor(obs, la)

            if config.filter_on and active_all is not None and active_all[t]:
                w, s, _diag = cvar_filter_step(
                    w_raw, sigma_all[t], mu_all[t], anchor_all[t], cvar_limit, c_alpha
                )
                shrinks.append(s)
            else:
                w = w_raw

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
            "mean_shrink": float(np.mean(shrinks)) if shrinks else 1.0,
        }
        if val_R is not None and it % config.eval_every == 0:
            val_sharpe, val_cvar, val_ret = _rollout_metrics_filtered(
                actor, val_R, val_feats, cost, cvar_window, cvar_alpha, val_anchor,
                lam, config, val_filter_state, cvar_limit, c_alpha,
            )
            val_metric = val_ret if config.objective == "return" else val_sharpe
            record["val_sharpe"] = val_sharpe
            record["val_cvar"] = val_cvar
            snapshot = {k: v.clone() for k, v in actor.state_dict().items()}
            n_evals += 1
            if val_metric > best_val:
                best_val, best_state, best_it = val_metric, snapshot, it
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


# --------------------------------------------------------------------------
# Evaluation-time policies.
# --------------------------------------------------------------------------


def diff_policy_filtered(
    actor: DiffAllocator,
    lookback: int,
    anchor: str | None,
    filter_state: FilterState | None,
    budget: float,
    c_alpha: float,
    filter_on: bool = True,
):
    """Backtest policy for the non-adaptive filter arms (A5, A6): the actor's
    proposal is passed through the CVaR filter each step, using the
    precomputed trailing statistics for the returns series backing ``env``.
    """

    def policy(env) -> np.ndarray:
        t = env.state.step
        obs = env.observation()
        la = _step_log_anchor(env, lookback, anchor)
        la_t = None if la is None else torch.as_tensor(la, dtype=torch.float32)
        # Build the torch forward pass by hand (rather than actor.predict, which
        # returns numpy) so we can feed the raw proposal straight into cvar_filter_step.
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits = actor.net(obs_t)
            if la_t is not None:
                logits = logits + la_t
            w_raw = torch.softmax(logits, dim=-1)
        if (
            filter_on
            and filter_state is not None
            and t < len(filter_state.active)
            and filter_state.active[t]
        ):
            sigma_t = torch.as_tensor(filter_state.sigma[t], dtype=torch.float32)
            mu_t = torch.as_tensor(filter_state.mu[t], dtype=torch.float32)
            anchor_t = torch.as_tensor(filter_state.anchor[t], dtype=torch.float32)
            w_f, _s, _d = cvar_filter_step(w_raw, sigma_t, mu_t, anchor_t, budget, c_alpha)
            policy.n_filter_steps += 1
            policy.n_fired += int(bool(_d["filter_active"]))
            return w_f.numpy()
        return w_raw.numpy()

    # Diagnostics: steps where the filter was evaluated, and steps where the
    # proposal exceeded the budget and was shrunk toward the anchor.
    policy.n_filter_steps = 0
    policy.n_fired = 0
    return policy


def run_adaptive_dual_policy(
    env,
    actor: DiffAllocator,
    lookback: int,
    anchor: str | None,
    cvar_limit: float,
    cvar_alpha: float,
    lagrange_lr: float,
    initial_lam: float = 0.0,
    filter_state: FilterState | None = None,
    filter_on: bool = False,
    periods_per_year: int = 52,
) -> tuple[BacktestResult, list[float]]:
    """Roll the (frozen, lam-conditioned) actor through ``env`` with the dual
    multiplier kept updating at evaluation time, using only realised PAST
    losses.

    No look-ahead: ``info["cvar_estimate"]`` returned by ``env.step`` at
    decision t is the rolling CVaR of the *trailing* return history INCLUDING
    the return just realised at t, i.e. it is available only after t's trade
    has already happened. It is used here purely to set ``lam`` for the
    NEXT decision (t+1); it never feeds back into the action taken at t.
    """
    c_alpha = gaussian_cvar_multiplier(cvar_alpha)
    env.reset()
    lam = initial_lam
    rewards: list[float] = []
    weight_rows: list[np.ndarray] = []
    info_rows: list[dict] = []
    lam_path: list[float] = []
    done = False
    while not done:
        t = env.state.step
        obs = env.observation()
        obs_full = np.concatenate([obs, [lam]]).astype(np.float32)
        la = _step_log_anchor(env, lookback, anchor)
        w_raw = torch.as_tensor(actor.predict(obs_full, la), dtype=torch.float32)
        if (
            filter_on
            and filter_state is not None
            and t < len(filter_state.active)
            and filter_state.active[t]
        ):
            sigma_t = torch.as_tensor(filter_state.sigma[t], dtype=torch.float32)
            mu_t = torch.as_tensor(filter_state.mu[t], dtype=torch.float32)
            anchor_t = torch.as_tensor(filter_state.anchor[t], dtype=torch.float32)
            w_f, _s, _d = cvar_filter_step(w_raw, sigma_t, mu_t, anchor_t, cvar_limit, c_alpha)
            action = w_f.numpy()
        else:
            action = w_raw.numpy()

        state, reward, done, info = env.step(action)
        rewards.append(reward)
        weight_rows.append(state.weights.copy())
        info_rows.append(info)
        lam_path.append(lam)

        realised_cvar = float(info["cvar_estimate"])  # trailing, includes the step just taken
        lam = float(min(100.0, max(0.0, lam + lagrange_lr * (realised_cvar - cvar_limit))))

    returns = pd.Series(rewards, name="portfolio_return")
    weights = pd.DataFrame(weight_rows, columns=list(env.returns.columns))
    info_df = pd.DataFrame(info_rows)
    metrics = summarise(returns, periods_per_year=periods_per_year)
    metrics.update({
        "avg_turnover": float(info_df["turnover"].mean()),
        "total_costs": float(info_df["costs"].sum()),
        "cvar_breach_rate": float(info_df["cvar_breach"].mean()),
        "constraint_violations": float((info_df["constraint_violation"] > 1e-8).sum()),
        "final_wealth": float(info_df["wealth"].iloc[-1]),
        "final_lam": lam,
        "mean_lam": float(np.mean(lam_path)) if lam_path else initial_lam,
    })
    return BacktestResult(returns=returns, weights=weights, info=info_df, metrics=metrics), lam_path
