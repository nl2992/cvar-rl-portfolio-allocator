"""Compliance frontier: breach rate vs Sharpe on the stress window, one point per
arm plus the classical comparators, with the 5% target drawn (paper Figure
fig:frontier). Single-column print figure in the shared paper style
(scripts/_paper_style.py, the poster's palette): blue for the decision-time filter
arms, coral for the soft-Lagrangian arms, grey for classical rules, green for the
region within the tolerated breach rate; identity is also carried by marker shape
and direct labels.

Data: results/tables_reanalysis/task4_filter_arms_summary.csv (A5-A8),
results/tables_reanalysis/breach_vs_budget_table1.csv (classical comparators and
the scaled-dual RL), and the coupling-ablation values in paper Table tab:coupling
(unconstrained, mis-scaled dual). A5 and the mis-scaled dual coincide because
A5's filter never fires on the test window, so they share one point.

Usage: python scripts/make_filter_frontier_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# (label, breach_rate, sharpe, group)
POINTS = [
    ("unconstrained", 1.000, 0.626, "penalty"),
    ("A5 = mis-scaled dual", 0.079, 0.987, "filter"),
    ("scaled dual", 0.751, 0.911, "penalty"),
    ("scaled dual, loose budget", 0.780, 0.543, "penalty"),
    ("A7", 0.819, 0.826, "adaptive"),
    ("equal weight", 0.628, 0.878, "classical"),
    ("risk parity", 0.753, 0.677, "classical"),
    ("inverse vol", 0.079, 0.780, "classical"),
    ("min-CVaR LP", 0.079, 0.817, "classical"),
    ("min variance", 0.079, 0.901, "classical"),
    ("A6 filter + scaled dual", 0.074, 0.851, "filter"),
    ("A8 filter + adaptive dual", 0.146, 0.800, "filter"),
]
STYLE = {  # group: (marker, face, edge, size)
    "classical": ("s", ps.GREY, "white", 17),
    "penalty": ("o", ps.RED_2, "white", 20),
    "adaptive": ("^", "white", ps.RED, 20),
    "filter": ("D", ps.BLUE, "white", 18),
}
# Label anchors (data coordinates) chosen so that no label overlaps another
# label, a marker, or a leader line (verified by _paper_style.check_layout).
LABELS = {
    "A5 = mis-scaled dual": (0.20, 1.005, "left"),
    "min variance": (0.20, 0.945, "left"),
    "A6 filter + scaled dual": (0.20, 0.885, "left"),
    "min-CVaR LP": (0.20, 0.825, "left"),
    "A8 filter + adaptive dual": (0.20, 0.765, "left"),
    "inverse vol": (0.20, 0.705, "left"),
    "scaled dual": (0.80, 0.965, "left"),
    "equal weight": (0.62, 0.935, "center"),
    "A7": (0.86, 0.826, "left"),
    "risk parity": (0.70, 0.677, "right"),
    "unconstrained": (1.04, 0.69, "right"),
    "scaled dual, loose budget": (0.73, 0.520, "right"),
}
DISPLAY: dict[str, str] = {}
TARGET = 0.05  # tolerated weekly breach rate (cfg model.cvar_budget)


def build():
    ps.apply()
    fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH_IN, 1.8))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.2, top=0.828)
    ax.axvspan(-0.03, TARGET, color=ps.GREEN_1, lw=0, zorder=0)
    ax.axvline(TARGET, color=ps.DARK, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    ax.text(TARGET - 0.012, 0.485, "5% target", rotation=90, ha="right", va="bottom",
            fontsize=ps.FONT - 0.5, color=ps.GREEN_DARK)
    # A5 and the mis-scaled dual share a point: ring the diamond with an open circle.
    ax.scatter([0.079], [0.987], s=62, marker="o", facecolors="none", edgecolors=ps.RED_2,
               linewidths=0.7, zorder=3)
    for group, (marker, face, edge, size) in STYLE.items():
        pts = [p for p in POINTS if p[3] == group]
        ax.scatter([p[1] for p in pts], [p[2] for p in pts], s=size, marker=marker,
                   facecolors=face, edgecolors=edge, linewidths=0.6, zorder=4)
    for label, x, y, _ in POINTS:
        tx, ty, ha = LABELS[label]
        ax.annotate(DISPLAY.get(label, label), (x, y), xytext=(tx, ty), ha=ha, va="center", fontsize=ps.FONT - 0.5,
                    color=ps.INK, bbox=ps.LABEL_BOX, arrowprops=dict(arrowstyle="-", color=ps.GREY, linewidth=0.4,
                                                  shrinkA=1.5, shrinkB=3.5))
    ax.set_xlim(-0.03, 1.06)
    ax.set_ylim(0.48, 1.04)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xlabel("Breach rate (share of weeks above budget)")
    ax.set_ylabel("Sharpe ratio (stress window)")
    handles = [plt.Line2D([0], [0], marker=STYLE[g][0], linestyle="none", markersize=3.8,
                          markerfacecolor=STYLE[g][1], markeredgecolor=STYLE[g][2], markeredgewidth=0.6)
               for g in ("classical", "penalty", "adaptive", "filter")]
    fig.legend(handles, ["classical optimiser", "soft Lagrangian", "adaptive dual (A7)",
                         "decision-time filter"],
               loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=2, handletextpad=0.3,
               columnspacing=1.2, labelspacing=0.3, borderaxespad=0.1)
    return fig


def main() -> None:
    fig = build()
    issues = ps.check_layout(fig)
    print("layout issues:", issues or "none")
    ps.save(fig, "figure_filter_frontier")


if __name__ == "__main__":
    main()
