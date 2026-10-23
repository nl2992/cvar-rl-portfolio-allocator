"""Camera-ready: checkpoint-selection and filter diagnostics for tab:coupling.

Re-runs the committed coupling arms (``scripts/coupling_fix_ablation.py``) and the
decision-time filter arms A5/A6 (``scripts/task4_risk_filter.py``) with identical
configs and seeds, and records per seed:

* which validation checkpoint the trainer kept and under which rule,
* how many evaluated checkpoints met the validation CVaR budget,
* for A5/A6, how often the filter fired on the stress test window,
* mean L1 distance of the test-window weights from the inverse-vol anchor.

Each row's metrics are compared against the committed per-seed CSVs, so the
diagnostics describe exactly the runs reported in the paper.

Usage:
    python scripts/cr_selection_log.py --arms buggy fixed
    python scripts/cr_selection_log.py --summarise
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from crlpa.evaluation.backtest import run_policy
from crlpa.evaluation.metrics import cvar as compute_cvar
from crlpa.evaluation.stress import stress_split
from crlpa.experiment import load_returns, make_env
from crlpa.training.differentiable import (
    DiffConfig,
    _step_log_anchor,
    diff_policy,
    train_differentiable,
)
from crlpa.training.risk_filter import (
    FilteredConfig,
    diff_policy_filtered,
    gaussian_cvar_multiplier,
    precompute_filter_state,
    train_filtered,
)
from crlpa.utils.config import load_config

SEEDS = [7, 13, 23, 42, 2025]
N_UPDATES = 1500
FILTER_LOOKBACK = 104  # as in task4_risk_filter.py
OUT = Path("results/tables_camera_ready")

# name -> (kind, constrained, lagrange_lr, limit multiplier of base_limit, filter_on)
ARMS = {
    "buggy": ("coupling", True, 0.001, 1.0, False),
    "fixed": ("coupling", True, 5.0, 1.0, False),
    "unconstrained": ("coupling", False, 0.0, 1.0, False),
    "feasible": ("coupling", True, 5.0, 1.6, False),
    "A5_filter_only": ("filter", True, 0.001, 1.0, True),
    "A6_filter_scaled_dual": ("filter", True, 5.0, 1.0, True),
}


def rolling_breach_rate(returns: np.ndarray, limit: float, alpha: float, window: int) -> float:
    """Identical scoring to coupling_fix_ablation.py / task4_risk_filter.py."""
    hits, total = 0, 0
    for t in range(window, len(returns)):
        w_ret = returns[max(0, t - window): t]
        if len(w_ret) >= 5:
            hits += int(compute_cvar(w_ret, alpha) > limit + 1e-6)
            total += 1
    return hits / max(1, total)


def committed_row(arm: str, seed: int) -> dict:
    if ARMS[arm][0] == "coupling":
        df = pd.read_csv("results/tables_ablations/coupling_fix_ablation.csv")
        r = df[(df.variant == arm) & (df.seed == seed)].iloc[0]
    else:
        df = pd.read_csv("results/tables_reanalysis/task4_filter_arms_per_seed.csv")
        r = df[(df.arm == arm) & (df.seed == seed)].iloc[0]
    return {"sharpe": float(r.sharpe), "cvar_99": float(r.cvar_99), "breach_rate": float(r.breach_rate)}


def run_arms(arms: list[str]) -> None:
    cfg = load_config("configs/experiment_etf.yaml")
    returns, _ = load_returns(cfg)
    alpha = float(cfg.get_path("risk.cvar_alpha", 0.95))
    base_limit = float(cfg.get_path("risk.cvar_limit", 0.03)) * 0.4
    cvar_window = int(cfg.get_path("risk.cvar_window", 52))
    lookback = int(cfg.get_path("environment.lookback", 26))
    cost_bps = float(cfg.get_path("environment.transaction_cost_bps", 5.0))
    max_weight = float(cfg.get_path("constraints.max_weight", 0.4))
    tr, va, te = stress_split(returns)

    def anchor_policy(env):
        la = _step_log_anchor(env, lookback, "inverse_vol")
        w = np.exp(la)
        return w / w.sum()

    env = make_env(cfg, te)
    env.cvar_alpha = alpha
    w_anchor = run_policy(env, anchor_policy).weights.to_numpy()

    fs = None
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        kind, constrained, lr, mult, filter_on = ARMS[arm]
        limit = base_limit * mult
        rows = []
        for seed in SEEDS:
            t0 = time.time()
            env = make_env(cfg, te)
            env.cvar_alpha = alpha
            fired = steps = np.nan
            if kind == "coupling":
                actor, hist = train_differentiable(
                    tr, cost_bps, alpha, limit, cvar_window, lookback,
                    config=DiffConfig(n_updates=N_UPDATES, horizon=104, objective="return",
                                      anchor="inverse_vol", constrained=constrained,
                                      lagrange_lr=lr, seed=seed),
                    val_returns=va,
                )
                res = run_policy(env, diff_policy(actor, lookback, "inverse_vol"))
            else:
                if fs is None:
                    fs = {name: precompute_filter_state(x, lookback=FILTER_LOOKBACK, max_weight=max_weight)
                          for name, x in (("tr", tr), ("va", va), ("te", te))}
                actor, hist = train_filtered(
                    tr, cost_bps, alpha, limit, cvar_window, lookback,
                    FilteredConfig(n_updates=N_UPDATES, horizon=104, objective="return",
                                   anchor="inverse_vol", constrained=True, lagrange_lr=lr,
                                   seed=seed, filter_on=filter_on, lam_conditioned=False),
                    val_returns=va, filter_state=fs["tr"], val_filter_state=fs["va"],
                )
                policy = diff_policy_filtered(actor, lookback, "inverse_vol", fs["te"], limit,
                                              gaussian_cvar_multiplier(alpha), filter_on=filter_on)
                res = run_policy(env, policy)
                fired, steps = policy.n_fired, policy.n_filter_steps

            sel = hist.attrs["selection"]
            w = res.weights.to_numpy()
            n = min(len(w), len(w_anchor))
            row = {
                "arm": arm, "seed": seed,
                "sharpe": res.metrics["sharpe"],
                "cvar_99": res.metrics.get("cvar_99", np.nan),
                "breach_rate": rolling_breach_rate(res.returns.to_numpy(), limit, alpha, cvar_window),
                "final_lam": float(hist["lagrange"].iloc[-1]) if "lagrange" in hist else 0.0,
                "selection_rule": sel["rule"],
                "selected_update": sel["selected_update"],
                "n_feasible": sel["n_feasible"],
                "n_evals": sel["n_evals"],
                "filter_fired": fired,
                "filter_steps": steps,
                "mean_l1_to_anchor": float(np.abs(w[:n] - w_anchor[:n]).sum(axis=1).mean()),
            }
            ref = committed_row(arm, seed)
            row["max_abs_diff_vs_committed"] = max(abs(row[k] - ref[k]) for k in ref)
            rows.append(row)
            print(f"{arm} seed={seed}: sharpe={row['sharpe']:.4f} cvar99={row['cvar_99']:.4f} "
                  f"breach={row['breach_rate']:.3f} selected={row['selected_update']} "
                  f"({row['selection_rule']}, {row['n_feasible']}/{row['n_evals']} feasible) "
                  f"fired={fired}/{steps} L1={row['mean_l1_to_anchor']:.3f} "
                  f"diff_vs_committed={row['max_abs_diff_vs_committed']:.2e} ({time.time() - t0:.0f}s)",
                  flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"selection_log_{arm}.csv", index=False)


def summarise() -> None:
    frames = [pd.read_csv(p) for p in sorted(OUT.glob("selection_log_*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / "selection_log.csv", index=False)
    summary = df.groupby("arm").agg(
        sharpe=("sharpe", "mean"), cvar_99=("cvar_99", "mean"), breach_rate=("breach_rate", "mean"),
        final_lam=("final_lam", "mean"), selected_update_mean=("selected_update", "mean"),
        selected_update_min=("selected_update", "min"), selected_update_max=("selected_update", "max"),
        n_feasible_mean=("n_feasible", "mean"), n_evals=("n_evals", "first"),
        filter_fired_total=("filter_fired", "sum"), filter_steps_total=("filter_steps", "sum"),
        mean_l1_to_anchor=("mean_l1_to_anchor", "mean"),
        max_abs_diff_vs_committed=("max_abs_diff_vs_committed", "max"),
    )
    summary.to_csv(OUT / "selection_log_summary.csv")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    parser.add_argument("--summarise", action="store_true")
    args = parser.parse_args()
    if args.summarise:
        summarise()
    else:
        run_arms(args.arms)
