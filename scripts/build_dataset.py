"""Materialise the modelling dataset to parquet.

For a paper configuration this fetches the configured ETF prices from the
Yahoo chart API and writes a frozen weekly-return panel to
``data.parquet_path``. Training then reads that panel (``data.source: parquet``)
for offline, reproducible runs.

Usage:
    python scripts/build_dataset.py --config configs/experiment_etf.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from crlpa.experiment import load_returns
from crlpa.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_etf.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--source", default="yahoo", help="data source to build from (default: yahoo)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["source"] = args.source
    returns, _ = load_returns(cfg)

    out = Path(args.out or cfg.get_path("data.parquet_path"))
    out.parent.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(out)
    print(f"wrote {out} with shape {returns.shape}")


if __name__ == "__main__":
    main()
