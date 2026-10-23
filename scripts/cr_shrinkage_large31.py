"""Camera-ready: classical optimisers, including Ledoit-Wolf shrinkage, on the
31-asset single stress split (paper tab:protocol, "stress split" column).

Answers whether the learner's single-split advantage over minimum variance is an
artefact of an unshrunk sample covariance on 31 assets. Deterministic (no
training): every baseline is re-estimated weekly exactly as in
``scripts/run_diff_study.py``. The unshrunk minimum-variance and inverse-vol rows
are checked against the reported values (0.23 and 0.38).

Usage:
    python scripts/cr_shrinkage_large31.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crlpa.evaluation.backtest import rolling_weight_policy, run_policy, static_policy
from crlpa.evaluation.stress import stress_split
from crlpa.experiment import load_returns, make_env
from crlpa.policies import baselines as b
from crlpa.utils.config import load_config

REPORTED_SHARPE = {"min_variance": 0.23, "inverse_vol": 0.38, "learner_constrained": 0.85}


def main() -> None:
    cfg = load_config("configs/experiment_large.yaml")
    returns, _ = load_returns(cfg)
    max_weight = float(cfg.get_path("constraints.max_weight", 0.4))
    _, _, test = stress_split(returns)
    dates = returns.index[620:]
    print(f"31-asset stress test window: {dates[0].date()} to {dates[-1].date()} ({len(test)} weeks), "
          f"max_weight={max_weight}")

    policies = {
        "equal_weight": static_policy(b.equal_weight(returns.shape[1])),
        "inverse_vol": rolling_weight_policy(lambda h: b.inverse_volatility(h)),
        "min_variance": rolling_weight_policy(lambda h: b.min_variance(h, max_weight=max_weight)),
        "min_variance_ledoit_wolf": rolling_weight_policy(
            lambda h: b.min_variance_shrunk(h, max_weight=max_weight)),
        "cvar_optimizer": rolling_weight_policy(lambda h: b.cvar_optimizer(h, max_weight=max_weight)),
    }
    rows = []
    for name, policy in policies.items():
        m = run_policy(make_env(cfg, test.reset_index(drop=True)), policy).metrics
        rows.append({"strategy": name, "sharpe": m["sharpe"], "cvar_99": m.get("cvar_99"),
                     "max_drawdown": m.get("max_drawdown"), "sortino": m.get("sortino"),
                     "reported_sharpe": REPORTED_SHARPE.get(name)})
        print(f"{name}: sharpe={m['sharpe']:.3f} cvar99={m.get('cvar_99'):.4f} "
              f"maxdd={m.get('max_drawdown'):.3f}", flush=True)
    df = pd.DataFrame(rows)
    out = Path("results/tables_camera_ready")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "shrinkage_large31_stress.csv", index=False)
    print(df.round(4).to_string(index=False))
    print(f"learner (constrained, reported): sharpe={REPORTED_SHARPE['learner_constrained']}")


if __name__ == "__main__":
    main()
