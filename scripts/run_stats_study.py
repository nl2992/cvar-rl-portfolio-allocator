"""Strengthen the statistics: multi-universe walk-forward + Deflated Sharpe.

Runs the rolling walk-forward on two distinct universes (macro ETFs and sector
ETFs) with shorter, non-overlapping folds for more independent observations, then:
  * pools fold-level CVaR-99 differences (constrained - unconstrained) across both
    universes and runs a paired test (Wilcoxon + sign test + bootstrap CI);
  * reports per-universe Probabilistic and Deflated Sharpe for the constrained
    allocator's concatenated OOS path (the latter correcting for trial selection).

Usage:
    python scripts/run_stats_study.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from crlpa.evaluation.stats import (
    benjamini_hochberg,
    deflated_sharpe_ratio,
    paired_fold_test,
    probabilistic_sharpe_ratio,
)
from crlpa.evaluation.walk_forward import WalkForwardConfig, walk_forward
from crlpa.experiment import load_returns
from crlpa.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*",
                        default=["configs/experiment_etf.yaml", "configs/experiment_sector.yaml"])
    parser.add_argument("--seeds", type=int, nargs="*", default=[7, 13])
    parser.add_argument("--updates", type=int, default=900)
    parser.add_argument("--train_window", type=int, default=312)
    parser.add_argument("--test_window", type=int, default=26)  # shorter -> more folds
    parser.add_argument("--out", default="results/tables_stats")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pooled_con: list[float] = []
    pooled_unc: list[float] = []
    per_universe = []
    trial_sharpes_all: list[float] = []
    fold_rows: list[dict] = []

    for cfg_path in args.configs:
        cfg = load_config(cfg_path)
        uid = cfg.get_path("universe.universe_id", Path(cfg_path).stem)
        returns, _ = load_returns(cfg)
        wf = WalkForwardConfig(
            train_window=args.train_window, val_window=52,
            test_window=args.test_window, step=args.test_window,
            seeds=tuple(args.seeds), n_updates=args.updates,
            constrained_limit=float(cfg.get_path("risk.cvar_limit", 0.03)) * 0.4,
            max_weight=float(cfg.get_path("constraints.max_weight", 0.4)),
            include_baselines=("min_variance",),
        )
        print(f"\n=== {uid}: walk-forward (test_window={wf.test_window}) ===")
        res = walk_forward(returns, cfg, wf)

        fc = res.fold_metrics["rl_cvar_constrained"].sort_values("fold")
        fu = res.fold_metrics["rl_unconstrained"].sort_values("fold")
        con99 = fc["cvar_99"].to_numpy()
        unc99 = fu["cvar_99"].to_numpy()
        pooled_con.extend(con99.tolist())
        pooled_unc.extend(unc99.tolist())
        for f, c, u in zip(fc["fold"].to_numpy(), con99, unc99):
            fold_rows.append({"universe": uid, "fold": int(f),
                              "cvar99_con": float(c), "cvar99_unc": float(u),
                              "diff": float(c - u)})

        # per-fold constrained Sharpe values act as the "trials" for DSR deflation
        trial_sharpes_all.extend((fc["sharpe"] / np.sqrt(52)).tolist())

        ft = paired_fold_test(con99, unc99)
        con_oos = res.oos_returns["rl_cvar_constrained"].to_numpy()
        fold_trial_sr = (fc["sharpe"] / np.sqrt(52)).to_numpy()  # per-period SRs as trials
        psr = probabilistic_sharpe_ratio(con_oos)
        dsr = deflated_sharpe_ratio(con_oos, fold_trial_sr)
        per_universe.append({
            "universe": uid, "folds": ft.n,
            "mean_cvar99_con": con99.mean(), "mean_cvar99_unc": unc99.mean(),
            "cvar99_mean_diff": ft.mean_diff, "folds_improved": ft.improved,
            "wilcoxon_p": ft.wilcoxon_p, "sign_p": ft.sign_p,
            "oos_sharpe_con": con_oos.mean() / (con_oos.std() + 1e-9) * np.sqrt(52),
            "psr_con": psr, "deflated_sharpe_con": dsr,
        })
        print(f"{uid}: folds={ft.n} mean CVaR99 {unc99.mean():.4f}->{con99.mean():.4f} "
              f"improved {ft.improved}/{ft.n} wilcoxon_p={ft.wilcoxon_p:.4f} "
              f"PSR={psr:.3f} DSR={dsr:.3f}")

    # pooled paired test across universes
    pooled = paired_fold_test(np.array(pooled_con), np.array(pooled_unc))
    # multiple-testing correction across per-universe Wilcoxon tests
    pvals = [r["wilcoxon_p"] for r in per_universe]
    reject = benjamini_hochberg(pvals)
    for r, rej in zip(per_universe, reject):
        r["bh_reject_0.05"] = rej

    # deflated Sharpe for the pooled constrained OOS, deflating by all fold trials
    dsr_trials = np.array(trial_sharpes_all)

    summary = pd.DataFrame(per_universe)
    summary.to_csv(out / "per_universe_stats.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "fold_level_cvar99.csv", index=False)
    pooled_row = pd.DataFrame([{
        "n_folds_pooled": pooled.n,
        "mean_diff": pooled.mean_diff,
        "folds_improved": pooled.improved,
        "wilcoxon_p": pooled.wilcoxon_p,
        "sign_p": pooled.sign_p,
        "ci_low": pooled.ci_low, "ci_high": pooled.ci_high,
        "deflated_sharpe_benchmark_trials": dsr_trials.size,
    }])
    pooled_row.to_csv(out / "pooled_fold_test.csv", index=False)

    print("\n=== Per-universe ===")
    print(summary.round(4).to_string(index=False))
    print("\n=== Pooled fold-level paired test (CVaR-99, constrained - unconstrained) ===")
    print(pooled_row.round(4).to_string(index=False))
    print(f"\nDeflated-Sharpe trial pool size: {dsr_trials.size}")
    print(f"wrote tables to {out}")


if __name__ == "__main__":
    main()
