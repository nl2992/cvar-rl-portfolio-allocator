"""Train the CVaR-constrained allocator (and an unconstrained RL baseline).

Trains both variants across the configured seeds, saves the validation-best
checkpoint for each, and writes per-episode training curves.

Usage:
    python scripts/train_allocator.py --config configs/experiment_etf.yaml
    python scripts/train_allocator.py --episodes 40 --seeds 7 13   # quick run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from crlpa.experiment import (
    load_returns,
    make_agent_config,
    make_env,
    make_train_config,
    split_returns,
)
from crlpa.models.cvar_actor_critic import CVaRActorCritic
from crlpa.training.train_allocator import train
from crlpa.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_etf.yaml")
    parser.add_argument("--episodes", type=int, default=None, help="override training.n_episodes")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="override training.seeds")
    parser.add_argument("--variants", nargs="*", default=["constrained", "unconstrained"])
    parser.add_argument("--out", default="results/checkpoints_etf")
    args = parser.parse_args()

    cfg = load_config(args.config)
    returns, regimes = load_returns(cfg)
    splits = split_returns(cfg, returns, regimes)
    seeds = args.seeds if args.seeds is not None else cfg.get_path("training.seeds", [42])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    curves_dir = Path("results/training_curves")
    curves_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for variant in args.variants:
        constrained = variant == "constrained"
        for seed in seeds:
            train_cfg = make_train_config(cfg, seed=seed)
            if args.episodes is not None:
                train_cfg.n_episodes = args.episodes

            train_env = make_env(cfg, splits.train)
            val_env = make_env(cfg, splits.val)
            agent = CVaRActorCritic(
                train_env.obs_dim,
                train_env.action_dim,
                make_agent_config(cfg, constrained=constrained),
            )
            agent, history, best_state = train(train_env, agent, train_cfg, val_env=val_env)

            tag = f"{variant}_seed{seed}"
            ckpt = out / f"{tag}.pt"
            torch.save(best_state, ckpt)
            history.to_csv(curves_dir / f"{tag}.csv", index=False)
            final = history.iloc[-1]
            print(f"{tag:24s} ep_return={final['ep_return']:+.3f} "
                  f"breach_rate={final['breach_rate']:.3f} lagrange={final['lagrange']:.3f}")
            index.append({"variant": variant, "seed": seed, "checkpoint": str(ckpt)})

    manifest = out / "manifest.json"
    manifest.write_text(json.dumps(index, indent=2))
    print(f"\nwrote {len(index)} checkpoints; manifest at {manifest}")
    pd.DataFrame(index).to_csv(out / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
