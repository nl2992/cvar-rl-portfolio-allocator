"""Paper figure: the Lagrange multiplier over training, mis-scaled vs scaled dual.

Plots results/tables_camera_ready/lambda_trace.csv (scripts/cr_lambda_trace.py): one thin
line per seed and the five-seed mean, for the two coupling arms of tab:coupling. The flat
trace is the symptom that Finding 1 says is the only visible sign of the coupling failure.

Usage:
    PYTHONPATH=src python scripts/make_lambda_trace_figure.py
"""
from __future__ import annotations

import _paper_style as ps
import pandas as pd
from _paper_style import plt

TRACE = pd.read_csv("results/tables_camera_ready/lambda_trace.csv")
STYLE = {"scaled dual": (ps.BLUE, ps.BLUE_2, r"scaled dual ($\eta_\lambda=5$)"),
         "mis-scaled dual": (ps.RED, ps.RED_2, r"mis-scaled dual ($\eta_\lambda=0.001$)")}


def main() -> None:
    ps.apply()
    fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH_IN, 1.35))
    for arm, (colour, light, label) in STYLE.items():
        d = TRACE[TRACE.arm == arm]
        for _, g in d.groupby("seed"):
            ax.plot(g["update"], g["lagrange"], color=light, lw=ps.LW_SEED, alpha=0.45, zorder=1)
        mean = d.groupby("update").lagrange.mean()
        if arm == "scaled dual":
            ax.fill_between(mean.index, 0, mean.values, color=ps.BLUE, alpha=0.12, lw=0, zorder=0)
        ax.plot(mean.index, mean.values, color=colour, lw=ps.LW_MAIN, zorder=3)
    top = TRACE.lagrange.max()
    ax.set_xlim(0, 1500)
    ax.set_ylim(-0.01, top * 1.05)
    ax.text(30, top * 0.93, STYLE["scaled dual"][2], color=ps.BLUE, fontsize=ps.FONT - 0.5, va="top",
            bbox=ps.LABEL_BOX)
    ax.text(15, top * 0.06, STYLE["mis-scaled dual"][2].replace(" dual (", "\n("), color=ps.RED,
            fontsize=ps.FONT - 0.5, ha="left", va="bottom", linespacing=1.0, bbox=ps.LABEL_BOX)
    ax.set_xlabel("Training update")
    ax.set_ylabel(r"Multiplier $\lambda$")
    fig.tight_layout(pad=0.2)
    issues = ps.check_layout(fig)
    print("layout issues:", issues or "none")
    ps.save(fig, "figure_lambda_trace")


if __name__ == "__main__":
    main()
