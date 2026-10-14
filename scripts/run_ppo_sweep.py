"""Proper hyperparameter sweep of the model-free PPO allocator.

The paper reports the model-free arm (A2C/PPO) as weak. This script settles that
honestly: it sweeps PPO over learning rate, entropy bonus, discount, rollout
horizon, and training length, selecting the configuration by *validation* Sharpe
(no test leakage), then retrains the winner across seeds and evaluates it on the
held-out stress window alongside the differentiable allocators and the strong
deterministic baselines. The output is a like-for-like, same-window comparison so
the paper can state what a tuned model-free allocator actually achieves here.

Usage:
    python scripts/run_ppo_sweep.py --config configs/experiment_etf.yaml
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from crlpa.evaluation.backtest import rolling_weight_policy, run_policy, static_policy
from crlpa.experiment import load_returns, make_env
from crlpa.policies import baselines as b
from crlpa.training.differentiable import DiffConfig, diff_policy, train_differentiable
from crlpa.training.ppo import PPOAllocator, PPOConfig, train_ppo
from crlpa.utils.config import load_config

METRIC_COLS = ["sharpe", "ann_return", "max_drawdown", "cvar_95", "cvar_99",
               "cvar_breach_rate", "avg_turnover"]


def ppo_policy(agent: PPOAllocator):
    return lambda env: agent.predict(env.observation())


def train_eval_ppo(cfg, train, eval_slice, pcfg: PPOConfig) -> tuple[PPOAllocator, dict]:
    """Train a PPO agent on ``train`` and score it on ``eval_slice``."""
    train_env = make_env(cfg, train)
    agent = PPOAllocator(train_env.obs_dim, train_env.action_dim, pcfg)
    train_ppo(train_env, agent)
    res = run_policy(make_env(cfg, eval_slice), ppo_policy(agent))
    return agent, res.metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_etf.yaml")
    parser.add_argument("--out", default="results/tables_ppo")
    parser.add_argument("--select_seed", type=int, default=7)
    parser.add_argument("--final_seeds", type=int, nargs="*", default=[7, 13, 23])
    parser.add_argument("--diff_seeds", type=int, nargs="*", default=[7, 13])
    parser.add_argument("--diff_updates", type=int, default=2500)
    parser.add_argument("--train_end", type=int, default=560)
    parser.add_argument("--val_end", type=int, default=620)
    parser.add_argument("--constrained_limit", type=float, default=0.012)
    args = parser.parse_args()

    cfg = load_config(args.config)
    R, _ = load_returns(cfg)
    lookback = int(cfg.get_path("environment.lookback", 26))
    cost_bps = float(cfg.get_path("environment.transaction_cost_bps", 5.0))
    alpha = float(cfg.get_path("risk.cvar_alpha", 0.95))
    cvar_window = int(cfg.get_path("risk.cvar_window", 52))
    max_weight = float(cfg.get_path("constraints.max_weight", 0.4))

    train = R.iloc[: args.train_end].reset_index(drop=True)
    val = R.iloc[args.train_end : args.val_end].reset_index(drop=True)
    test = R.iloc[args.val_end :].reset_index(drop=True)
    print(f"train {len(train)}  val {len(val)}  test(stress) {len(test)} weeks")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1) sweep grid, selected on validation Sharpe (no test leakage) --------
    grid = {
        "actor_lr": [1e-4, 3e-4, 1e-3],
        "entropy_coef": [0.0, 0.01, 0.03],
        "gamma": [0.95, 0.99],
        "n_iterations": [150, 300],
    }
    keys = list(grid)
    sweep_rows = []
    print(f"\n=== PPO sweep: {np.prod([len(grid[k]) for k in keys])} configs "
          f"(select seed {args.select_seed}) ===")
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        pcfg = PPOConfig(rollout_len=104, epochs=10, clip=0.2, constrained=False,
                         seed=args.select_seed, **params)
        _, m = train_eval_ppo(cfg, train, val, pcfg)
        row = {**params, "val_sharpe": m["sharpe"], "val_cvar99": m["cvar_99"],
               "val_maxdd": m["max_drawdown"]}
        sweep_rows.append(row)
        print(f"  lr={params['actor_lr']:.0e} ent={params['entropy_coef']:.2f} "
              f"g={params['gamma']:.2f} it={params['n_iterations']:>3d} -> "
              f"val_sharpe={m['sharpe']:+.3f} val_cvar99={m['cvar_99']:.4f}")
    sweep = pd.DataFrame(sweep_rows).sort_values("val_sharpe", ascending=False)
    sweep.to_csv(out / "ppo_sweep_grid.csv", index=False)
    best = sweep.iloc[0]
    best_params = {k: best[k] for k in keys}
    best_params["n_iterations"] = int(best_params["n_iterations"])
    print(f"\nbest by val Sharpe: {best_params}  (val_sharpe={best['val_sharpe']:+.3f})")

    # ---- 2) retrain winner across seeds, evaluate on the stress window ---------
    rows: dict[str, dict] = {}

    def avg_test_metrics(make_agent_seed) -> dict:
        per_seed = []
        for seed in args.final_seeds:
            agent = make_agent_seed(seed)
            res = run_policy(make_env(cfg, test), ppo_policy(agent))
            per_seed.append(res.metrics)
        return pd.DataFrame(per_seed).mean(numeric_only=True).to_dict()

    def make_ppo(seed, constrained):
        train_env = make_env(cfg, train)
        pcfg = PPOConfig(rollout_len=104, epochs=10, clip=0.2, constrained=constrained,
                         cvar_budget=float(cfg.get_path("model.cvar_budget", 0.05)),
                         lagrange_lr=float(cfg.get_path("model.lagrange_lr", 1.0)),
                         seed=seed, **best_params)
        agent = PPOAllocator(train_env.obs_dim, train_env.action_dim, pcfg)
        train_ppo(train_env, agent)
        return agent

    rows["ppo_best_unconstrained"] = avg_test_metrics(lambda s: make_ppo(s, False))
    rows["ppo_best_cvar_constrained"] = avg_test_metrics(lambda s: make_ppo(s, True))

    # ---- 3) reference arms on the same stress window ---------------------------
    rows["equal_weight"] = run_policy(make_env(cfg, test),
                                      static_policy(b.equal_weight(R.shape[1]))).metrics
    rows["inverse_vol"] = run_policy(make_env(cfg, test),
                                     rolling_weight_policy(lambda h: b.inverse_volatility(h))).metrics
    rows["min_variance"] = run_policy(make_env(cfg, test),
                                      rolling_weight_policy(lambda h: b.min_variance(h, max_weight=max_weight))).metrics

    for name, constrained, lim in [("rl_unconstrained", False, 0.03),
                                   ("rl_cvar_constrained", True, args.constrained_limit)]:
        per_seed = []
        for seed in args.diff_seeds:
            actor, _ = train_differentiable(
                train, cost_bps, alpha, lim, cvar_window, lookback,
                config=DiffConfig(n_updates=args.diff_updates, horizon=104, objective="return",
                                  risk_aversion=0.0, constrained=constrained, lagrange_lr=5.0, seed=seed),
                val_returns=val)
            per_seed.append(run_policy(make_env(cfg, test), diff_policy(actor, lookback)).metrics)
        rows[name] = pd.DataFrame(per_seed).mean(numeric_only=True).to_dict()

    table = pd.DataFrame(rows).T
    table.to_csv(out / "ppo_stress_comparison.csv")
    print("\n=== Stress-window comparison (best-tuned PPO vs. references) ===")
    print(table[METRIC_COLS].round(4).to_string())
    print(f"\nwrote tables to {out}")


if __name__ == "__main__":
    main()
