"""CVaR hero figure (paper Figure fig:hero), in the shared print style.

Two risk--return panels (x = CVaR99 in %, lower is safer; y = Sharpe ratio):
  (a) Stress window: the constrained allocator sits up and to the left of the
      unconstrained learner (higher Sharpe, lower tail), next to min-variance.
  (b) Ablation: tightening the tail budget improves CVaR99 and Sharpe together;
      min-variance on the same stress window is shown for reference.

Coordinates are the values reported in the paper's tables (stress window,
tuned model-free comparison, ablation, walk-forward), so the figure matches the
text by construction. Style: scripts/_paper_style.py (the poster's palette: blue
for the proposed method, red for unconstrained RL, grey for minimum variance;
identity is also carried by marker shape and direct labels).

    python scripts/make_hero_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Panel (a): stress window, (CVaR99 %, Sharpe) from tab:stress and tab:modelfree.
STRESS = [  # label, cvar99_pct, sharpe, marker, face, edge, label anchor (x, y, ha)
    ("CVaR-constrained RL", 3.27, 0.88, "D", ps.BLUE, "white", (3.20, 0.926, "center")),
    ("unconstrained RL", 6.47, 0.63, "o", ps.RED, "white", (6.30, 0.598, "right")),
    ("PPO (best)", 7.00, 0.67, "o", "white", ps.RED, (6.92, 0.715, "center")),
    ("min variance", 4.40, 0.90, "s", ps.GREY, "white", (4.58, 0.90, "left")),
]
# Panel (b): constraint-tightening ablation (tab:ablation) and min-variance on
# the same stress window (tab:modelfree), so both panels compare like with like.
ABLATION = [  # label, cvar99_pct, sharpe, label anchor
    ("loose budget", 4.93, 0.515, (4.80, 0.462, "right")),
    ("base", 3.01, 0.865, (3.17, 0.96, "left")),
    ("tight budget", 1.35, 1.272, (1.55, 1.345, "left")),
]
MV_REF = ("min variance", 4.40, 0.90, (4.40, 1.02, "center"))


def build():
    ps.apply()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(ps.TEXT_WIDTH_IN, 1.52))
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.25, top=0.85, wspace=0.22)

    # (a) stress window
    ax_a.annotate("", xy=(3.36, 0.866), xytext=(6.38, 0.637),
                  arrowprops=dict(arrowstyle="-|>", color=ps.GREY, linewidth=0.6, mutation_scale=6,
                                  connectionstyle="arc3,rad=-0.12", shrinkA=3, shrinkB=3), zorder=2)
    for label, x, y, marker, face, edge, (tx, ty, ha) in STRESS:
        ax_a.scatter([x], [y], s=30, marker=marker, facecolors=face, edgecolors=edge,
                     linewidths=0.7, zorder=4)
        colour = ps.BLUE if face == ps.BLUE else ps.INK
        ax_a.annotate(label, (x, y), xytext=(tx, ty), ha=ha, va="center", fontsize=ps.FONT - 0.2,
                      color=colour, bbox=ps.LABEL_BOX)
    ax_a.set_xlim(2.4, 7.5)
    ax_a.set_ylim(0.57, 0.95)
    ax_a.set_xlabel("CVaR$_{99}$ (%), lower is safer")
    ax_a.set_ylabel("Sharpe ratio")
    ax_a.set_title("(a) Stress window, 2020 and 2022 drawdowns", loc="left", fontsize=ps.FONT + 0.5, pad=4)

    # (b) ablation
    xs = [c for _, c, _, _ in ABLATION]
    ys = [s for _, _, s, _ in ABLATION]
    ax_b.plot(xs, ys, color=ps.BLUE, linewidth=ps.LW_MAIN, solid_capstyle="round", marker="D", markersize=4.6,
              markerfacecolor=ps.BLUE, markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    for label, x, y, (tx, ty, ha) in ABLATION:
        ax_b.annotate(label, (x, y), xytext=(tx, ty), ha=ha, va="center", fontsize=ps.FONT - 0.2, color=ps.BLUE,
                      bbox=ps.LABEL_BOX)
    label, x, y, (tx, ty, ha) = MV_REF
    ax_b.scatter([x], [y], s=30, marker="s", facecolors=ps.GREY, edgecolors="white",
                 linewidths=0.7, zorder=4)
    ax_b.annotate(label, (x, y), xytext=(tx, ty), ha=ha, va="center", fontsize=ps.FONT - 0.2, color=ps.INK,
                  bbox=ps.LABEL_BOX)
    ax_b.set_xlim(0.7, 5.4)
    ax_b.set_ylim(0.36, 1.42)
    ax_b.set_xlabel("CVaR$_{99}$ (%), lower is safer")
    ax_b.set_ylabel("Sharpe ratio")
    ax_b.set_title("(b) Tightening the tail budget (stress window)", loc="left", fontsize=ps.FONT + 0.5, pad=4)
    return fig


def main() -> None:
    fig = build()
    issues = ps.check_layout(fig)
    print("layout issues:", issues or "none")
    ps.save(fig, "figure_cvar_hero")


if __name__ == "__main__":
    main()
