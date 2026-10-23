"""Evaluate trained allocators on the test split against deterministic baselines.

Loads every checkpoint in the training manifest, backtests it on the held-out
test split, averages metrics across seeds per variant, and runs a paired
bootstrap of the constrained allocator's Sharpe against the unconstrained RL
baseline and equal weight.

Usage:
    python scripts/evaluate_allocator.py --config configs/experiment_etf.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from crlpa.evaluation.backtest import rolling_weight_policy, run_policy, static_policy
from crlpa.evaluation.bootstrap import paired_bootstrap
from crlpa.experiment import load_returns, make_agent_config, make_env, split_returns
from crlpa.models.cvar_actor_critic import CVaRActorCritic
from crlpa.policies import baselines as b
from crlpa.utils.config import load_config


def agent_policy(agent: CVaRActorCritic):
    return lambda env: agent.predict(env.observation())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_etf.yaml")
    parser.add_argument("--checkpoints", default="results/checkpoints_etf")
    parser.add_argument("--out", default="results/tables_etf")
    args = parser.parse_args()

    cfg = load_config(args.config)
    returns, regimes = load_returns(cfg)
    splits = split_returns(cfg, returns, regimes)
    ppy = int(cfg.get_path("backtest.periods_per_year", 52))
    max_weight = float(cfg.get_path("constraints.max_weight", 1.0))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- deterministic baselines on the test split -------------------------
    baseline_policies = {
        "equal_weight": static_policy(b.equal_weight(returns.shape[1])),
        "min_variance": rolling_weight_policy(lambda h: b.min_variance(h, max_weight=max_weight)),
        "cvar_optimizer": rolling_weight_policy(
            lambda h: b.cvar_optimizer(h, max_weight=max_weight)
        ),
    }
    rows: dict[str, dict] = {}
    return_series: dict[str, pd.Series] = {}
    for name, policy in baseline_policies.items():
        res = run_policy(make_env(cfg, splits.test), policy, periods_per_year=ppy)
        rows[name] = res.metrics
        return_series[name] = res.returns

    # --- trained allocators (averaged across seeds) ------------------------
    manifest_path = Path(args.checkpoints) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    per_variant: dict[str, list[dict]] = {}
    per_variant_returns: dict[str, list[pd.Series]] = {}
    for entry in manifest:
        variant = entry["variant"]
        env = make_env(cfg, splits.test)
        agent = CVaRActorCritic(
            env.obs_dim, env.action_dim, make_agent_config(cfg, constrained=variant == "constrained")
        )
        agent.load_state_dict(torch.load(entry["checkpoint"], weights_only=True))
        res = run_policy(env, agent_policy(agent), periods_per_year=ppy)
        per_variant.setdefault(variant, []).append(res.metrics)
        per_variant_returns.setdefault(variant, []).append(res.returns)

    for variant, metric_list in per_variant.items():
        df = pd.DataFrame(metric_list)
        rows[f"rl_{variant}"] = df.mean().to_dict()
        # representative return series = first seed, for bootstrap comparisons
        return_series[f"rl_{variant}"] = per_variant_returns[variant][0]

    table = pd.DataFrame(rows).T
    table.to_csv(out / "allocator_metrics.csv")
    print(table[["sharpe", "ann_return", "max_drawdown", "cvar_95", "cvar_breach_rate", "avg_turnover"]].round(4).to_string())

    # --- statistical comparisons -------------------------------------------
    n_resamples = int(cfg.get_path("backtest.bootstrap_resamples", 2000))
    block = int(cfg.get_path("backtest.block_size", 4))
    comparisons = []
    if "rl_constrained" in return_series:
        for ref in ["rl_unconstrained", "equal_weight"]:
            if ref in return_series:
                bs = paired_bootstrap(
                    return_series["rl_constrained"], return_series[ref],
                    n_resamples=n_resamples, block_size=block,
                )
                comparisons.append({
                    "comparison": f"constrained_minus_{ref}",
                    "sharpe_diff": bs.point_estimate,
                    "ci_low": bs.ci_low, "ci_high": bs.ci_high, "p_value": bs.p_value,
                })
    if comparisons:
        comp = pd.DataFrame(comparisons)
        comp.to_csv(out / "statistical_tests.csv", index=False)
        print("\n" + comp.round(4).to_string(index=False))

    print(f"\nwrote tables to {out}")


if __name__ == "__main__":
    main()
