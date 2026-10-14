"""Paper figure: drawdown paths of the unconstrained and constrained allocators, stress window.

Plots results/tables_camera_ready/stress_paths.csv (scripts/cr_stress_paths.py): the
drawdown of each seed (thin) and of the five-seed mean return path (thick). Test week 16
is the week of 16-20 March 2020 (SPY -14.6%, the worst week in the window), which fixes
the calendar: week k ends on 20 March 2020 + 7(k-16) days.

Usage:
    PYTHONPATH=src python scripts/make_stress_paths_figure.py
"""
from __future__ import annotations

import _paper_style as ps
import numpy as np
import pandas as pd
from _paper_style import plt

PATHS = pd.read_csv("results/tables_camera_ready/stress_paths.csv")
COVID_WEEK, COVID_DATE = 16, pd.Timestamp("2020-03-20")
STYLE = {"unconstrained": (ps.RED, ps.RED_2, "unconstrained RL"),
         "constrained": (ps.BLUE, ps.BLUE_2, "CVaR-constrained")}


def drawdown(ret: np.ndarray) -> np.ndarray:
    wealth = np.cumprod(1 + ret)
    return wealth / np.maximum.accumulate(wealth) - 1


def week_of(date: str) -> float:
    return COVID_WEEK + (pd.Timestamp(date) - COVID_DATE).days / 7


def main() -> None:
    ps.apply()
    fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH_IN, 1.45))
    means, troughs = {}, {}
    for arm, (colour, light, label) in STYLE.items():
        d = PATHS[PATHS.arm == arm]
        for _, g in d.groupby("seed"):
            ax.plot(np.arange(len(g)), 100 * drawdown(g.ret.to_numpy()), color=light, lw=ps.LW_SEED,
                    alpha=0.6, zorder=1)
        mean_ret = d.groupby(d.groupby("seed").cumcount()).ret.mean().to_numpy()
        dd = 100 * drawdown(mean_ret)
        ax.plot(np.arange(len(dd)), dd, color=colour, lw=ps.LW_MAIN, label=label, zorder=3,
                solid_capstyle="round")
        k = int(np.argmin(dd[:40]))  # the March 2020 trough
        means[arm], troughs[arm] = dd, (k, dd[k])
    x = np.arange(len(means["constrained"]))
    ax.fill_between(x, means["unconstrained"], means["constrained"],
                    where=means["unconstrained"] < means["constrained"], color=ps.RED_2, alpha=0.22, lw=0,
                    zorder=0.5, interpolate=True)
    ax.axvline(COVID_WEEK, color=ps.GREY, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.text(COVID_WEEK + 3, -21.5, "March 2020", fontsize=ps.FONT - 1, color=ps.DARK, ha="left", va="bottom")
    for arm, (k, v) in troughs.items():
        ax.plot(k, v, "o", ms=3.4, color=STYLE[arm][0], mec="white", mew=0.6, zorder=4)
    years = ["2020-01-03", "2021-01-01", "2022-01-07", "2023-01-06", "2024-01-05"]
    ax.set_xticks([week_of(y) for y in years], [y[:4] for y in years])
    ax.set_xlim(0, PATHS.groupby(["arm", "seed"]).size().max() - 1)
    ax.set_ylim(-25, 1)
    ax.set_yticks([0, -10, -20])
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.98), handlelength=1.5, ncols=2, columnspacing=1.5,
              borderaxespad=0.0)
    fig.tight_layout(pad=0.2)
    issues = ps.check_layout(fig)
    print("layout issues:", issues or "none")
    ps.save(fig, "figure_stress_paths")


if __name__ == "__main__":
    main()
