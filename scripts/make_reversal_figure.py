"""Write the recorded 31-asset protocol comparison used by the paper.

The raw 31-asset walk-forward folds were not retained. This script keeps the
published stress-split and walk-forward Sharpe values in one reproducible CSV.

Usage:
    python scripts/make_reversal_figure.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Authoritative values in paper Table~\\ref{tab:protocol}.
PROTOCOL_SHARPE = pd.DataFrame(
    [
        {"strategy": "minimum variance", "stress": 0.23, "walkforward": 1.40},
        {"strategy": "inverse vol", "stress": 0.38, "walkforward": 0.78},
        {"strategy": "diff. constrained (ours)", "stress": 0.85, "walkforward": 0.74},
        {"strategy": "diff. unconstrained", "stress": 0.70, "walkforward": 0.56},
    ]
)


def main() -> None:
    data_dir = Path("results/tables_protocol")
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / "protocol_sharpe.csv"
    PROTOCOL_SHARPE.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
