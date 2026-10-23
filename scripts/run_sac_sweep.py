"""Hyperparameter sweep of the model-free SAC allocator.

Companion to run_ppo_sweep.py for the off-policy model-free arm. Sweeps SAC over
learning rate, discount, and entropy-temperature handling, selects by validation
Sharpe (no test leakage), retrains the winner across seeds, and evaluates on the
held-out stress window alongside the strong deterministic baselines. Combined with
the PPO sweep output, this lets the paper report what tuned model-free allocators
(both on- and off-policy) actually achieve on this tail-risk problem.

Usage:
    python scripts/run_sac_sweep.py --config configs/experiment_etf.yaml
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from crlpa.evaluation.backtest import rolling_weight_policy, run_policy
from crlpa.experiment import load_returns, make_env
from crlpa.policies import baselines as b
from crlpa.training.sac import SACAllocator, SACConfig, train_sac
from crlpa.utils.config import load_config

METRIC_COLS = ["sharpe", "ann_return", "max_drawdown", "cvar_95", "cvar_99",
               "cvar_breach_rate", "avg_turnover"]


def sac_policy(agent: SACAllocator):
    return lambda env: agent.predict(env.observation())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_etf.yaml")
    parser.add_argument("--out", default="results/tables_sac")
    parser.add_argument("--select_seed", type=int, default=7)
    parser.add_argument("--final_seeds", type=int, nargs="*", default=[7, 13, 23])
    parser.add_argument("--n_iterations", type=int, default=120)
    parser.add_argument("--train_end", type=int, default=560)
    parser.add_argument("--val_end", type=int, default=620)
    args = parser.parse_args()

    cfg = load_config(args.config)
    R, _ = load_returns(cfg)
    max_weight = float(cfg.get_path("constraints.max_weight", 0.4))

    train = R.iloc[: args.train_end].reset_index(drop=True)
    val = R.iloc[args.train_end : args.val_end].reset_index(drop=True)
    test = R.iloc[args.val_end :].reset_index(drop=True)
    print(f"train {len(train)}  val {len(val)}  test(stress) {len(test)} weeks")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def make_sac(seed, constrained, params):
        env = make_env(cfg, train)
        scfg = SACConfig(n_iterations=args.n_iterations, rollout_len=104, batch_size=128,
                         warmup_steps=208, constrained=constrained,
                         cvar_budget=float(cfg.get_path("model.cvar_budget", 0.05)),
                         lagrange_lr=float(cfg.get_path("model.lagrange_lr", 1.0)),
                         seed=seed, **params)
        agent = SACAllocator(env.obs_dim, env.action_dim, scfg)
        train_sac(env, agent)
        return agent

    # ---- 1) sweep, selected on validation Sharpe ------------------------------
    grid = {
        "actor_lr": [3e-4, 1e-3],
        "gamma": [0.95, 0.99],
        "autotune_alpha": [True, False],
    }
    keys = list(grid)
    sweep_rows = []
    print(f"\n=== SAC sweep: {int(np.prod([len(grid[k]) for k in keys]))} configs "
          f"(select seed {args.select_seed}) ===")
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        if not params["autotune_alpha"]:
            params = {**params, "init_alpha": 0.1}
        agent = make_sac(args.select_seed, False, params)
        m = run_policy(make_env(cfg, val), sac_policy(agent)).metrics
        row = {**params, "val_sharpe": m["sharpe"], "val_cvar99": m["cvar_99"]}
        sweep_rows.append(row)
        print(f"  lr={params['actor_lr']:.0e} g={params['gamma']:.2f} "
              f"autoalpha={params['autotune_alpha']!s:5s} -> "
              f"val_sharpe={m['sharpe']:+.3f} val_cvar99={m['cvar_99']:.4f}")
    sweep = pd.DataFrame(sweep_rows).sort_values("val_sharpe", ascending=False)
    sweep.to_csv(out / "sac_sweep_grid.csv", index=False)
    best = sweep.iloc[0].to_dict()
    best_params = {k: best[k] for k in keys}
    if not best_params["autotune_alpha"]:
        best_params["init_alpha"] = 0.1
    print(f"\nbest by val Sharpe: {best_params}  (val_sharpe={best['val_sharpe']:+.3f})")

    # ---- 2) retrain winner across seeds, evaluate on the stress window ---------
    def avg_test_metrics(constrained) -> dict:
        per_seed = []
        for seed in args.final_seeds:
            agent = make_sac(seed, constrained, best_params)
            per_seed.append(run_policy(make_env(cfg, test), sac_policy(agent)).metrics)
        return pd.DataFrame(per_seed).mean(numeric_only=True).to_dict()

    rows: dict[str, dict] = {}
    rows["sac_best_unconstrained"] = avg_test_metrics(False)
    rows["sac_best_cvar_constrained"] = avg_test_metrics(True)

    # cheap deterministic references on the same window for a standalone table
    rows["inverse_vol"] = run_policy(make_env(cfg, test),
                                     rolling_weight_policy(lambda h: b.inverse_volatility(h))).metrics
    rows["min_variance"] = run_policy(make_env(cfg, test),
                                      rolling_weight_policy(lambda h: b.min_variance(h, max_weight=max_weight))).metrics

    table = pd.DataFrame(rows).T
    table.to_csv(out / "sac_stress_comparison.csv")
    print("\n=== Stress-window comparison (best-tuned SAC vs. references) ===")
    print(table[METRIC_COLS].round(4).to_string())
    print(f"\nwrote tables to {out}")


if __name__ == "__main__":
    main()
