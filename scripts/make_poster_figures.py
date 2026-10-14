"""Poster figures, in the poster style (scripts/_poster_style.py, after figures4papers).

The paper's figures (paper/figures/) are left untouched; the poster gets its own
versions drawn at printed size with the same data:

figure_stress_paths   Drawdown of both allocators over the stress window, five seeds.
                      Data: results/tables_camera_ready/stress_paths.csv.
figure_cvar_tail      Weekly return distribution of the constrained and unconstrained
                      allocators on the stress window, with -VaR_0.95 and -CVaR_0.95
                      marked in the left tail. Data: results/tables_camera_ready/stress_paths.csv.
figure_lambda_trace   The Lagrange multiplier over training, mis-scaled vs scaled dual.
                      Data: results/tables_camera_ready/lambda_trace.csv.
figure_protocol       Sharpe of four methods under a single stress split and under a
                      rolling walk-forward, 31 assets. Data: results/tables_protocol/protocol_sharpe.csv.
figure_coupling       Sharpe, CVaR_0.99 and breach rate of the ablation arms on the stress
                      window, five-seed means, as bars in the figures4papers style. Values:
                      paper tab:coupling (results/tables_camera_ready/selection_log_summary.csv)
                      and the budget-reach paragraph for minimum variance.
figure_regime         Out-of-sample CVaR_0.99 by market regime (calm, high volatility, selloff)
                      for unconstrained RL, constrained RL and minimum variance, 10-fold
                      walk-forward 2010 to 2024, macro ETF universe (paper tab:regime).
                      Data: results/tables_wf/walkforward_regime_*.csv.
figure_fold_scatter   Each of the 40 walk-forward folds as one point, CVaR_0.99
                      without against with the constraint; below the diagonal means
                      lower tail risk. Data: results/tables_stats/folds_{etf7,sector10}.csv.

Usage:
    PYTHONPATH=src python scripts/make_poster_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _poster_style as ps  # noqa: E402
from _paper_style import check_layout  # noqa: E402
from _poster_style import plt  # noqa: E402

from crlpa.evaluation.metrics import cvar, value_at_risk  # noqa: E402

ALPHA = 0.95
POSTER_W = 11.5  # inches, a 0.78-linewidth figure on the A0 poster


def report(fig, name: str) -> None:
    issues = check_layout(fig, marker_radius_pt=6)
    print(f"{name} layout issues:", issues or "none")


def tail_stats(loss: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    """VaR and CVaR of a loss series, with the evaluation code's estimators."""
    return value_at_risk(-loss, alpha), cvar(-loss, alpha)


# ---------------------------------------------------------------------------
PATHS = pd.read_csv("results/tables_camera_ready/stress_paths.csv")
COVID_WEEK, COVID_DATE = 16, pd.Timestamp("2020-03-20")
ARMS = {"unconstrained": (ps.RED, ps.RED_2, "Unconstrained RL"),
        "constrained": (ps.BLUE, ps.BLUE_2, "CVaR-constrained (ours)")}


def drawdown(ret: np.ndarray) -> np.ndarray:
    wealth = np.cumprod(1 + ret)
    return wealth / np.maximum.accumulate(wealth) - 1


def week_of(date: str) -> float:
    return COVID_WEEK + (pd.Timestamp(date) - COVID_DATE).days / 7


def stress_paths() -> None:
    """Drawdown of each seed (thin) and of the five-seed mean return path (thick)."""
    ps.apply()
    fig, ax = plt.subplots(figsize=(POSTER_W, 4.6))
    troughs, means = {}, {}
    for arm, (colour, light, label) in ARMS.items():
        d = PATHS[PATHS.arm == arm]
        for _, g in d.groupby("seed"):
            ax.plot(np.arange(len(g)), 100 * drawdown(g.ret.to_numpy()), color=light, lw=ps.LW_SEED,
                    alpha=0.6, zorder=1)
        mean_ret = d.groupby(d.groupby("seed").cumcount()).ret.mean().to_numpy()
        dd = 100 * drawdown(mean_ret)
        ax.plot(np.arange(len(dd)), dd, color=colour, lw=ps.LW_MAIN, label=label, zorder=3,
                solid_capstyle="round")
        covid = slice(0, 40)
        k = int(np.argmin(dd[covid]))
        troughs[arm] = (k, dd[k])
        means[arm] = dd
    x = np.arange(len(means["constrained"]))
    ax.fill_between(x, means["unconstrained"], means["constrained"],
                    where=means["unconstrained"] < means["constrained"], color=ps.RED_2, alpha=0.22, lw=0, zorder=0.5,
                    interpolate=True)
    ax.axvline(COVID_WEEK, color=ps.GREY, lw=2.5, ls="--", zorder=0)
    ax.text(COVID_WEEK + 3, -21.5, "March 2020", fontsize=ps.FONT - 4, color=ps.DARK, ha="left", va="bottom")
    for arm, (k, v) in troughs.items():
        colour = ARMS[arm][0]
        ax.plot(k, v, "o", ms=11, color=colour, mec="white", mew=2, zorder=4)
    years = ["2020-01-03", "2021-01-01", "2022-01-07", "2023-01-06", "2024-01-05"]
    ax.set_xticks([week_of(y) for y in years], [y[:4] for y in years])
    ax.set_xlim(0, PATHS.groupby(["arm", "seed"]).size().max() - 1)
    ax.set_ylim(-25, 1)
    ax.set_yticks([0, -5, -10, -15, -20, -25])
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), handlelength=1.8, ncols=2, columnspacing=2.5)
    fig.tight_layout(pad=0.4)
    report(fig, "stress_paths")
    ps.save(fig, "figure_stress_paths")


def cvar_tail() -> None:
    """Pooled five-seed weekly-return histograms with VaR and CVaR of the plotted sample.

    Returns are on the x-axis, so the loss tail is the left tail. VaR and CVaR are
    losses (positive numbers); their lines sit at -VaR and -CVaR. The lines are
    computed on the pooled sample, so the shaded region holds the worst 5% of the
    plotted constrained weeks and each CVaR line is the mean of its own tail.
    """
    ps.apply()
    fig, ax = plt.subplots(figsize=(POSTER_W, 4.9))
    rets = {arm: 100 * PATHS[PATHS.arm == arm].ret.to_numpy() for arm in ("unconstrained", "constrained")}
    lo = np.floor(min(r.min() for r in rets.values()))
    hi = np.ceil(max(r.max() for r in rets.values()))
    bins = np.arange(lo, hi + 1e-9, 1 / 3)
    assert all(((r >= bins[0]) & (r <= bins[-1])).all() for r in rets.values()), "weeks outside bins"
    ax.hist(rets["unconstrained"], bins=bins, density=True, histtype="stepfilled", facecolor=ps.RED_2,
            alpha=0.3, edgecolor="none", zorder=1)
    ax.hist(rets["unconstrained"], bins=bins, density=True, histtype="step", edgecolor=ps.RED, linewidth=1.8,
            zorder=1)
    ax.hist(rets["constrained"], bins=bins, density=True, histtype="stepfilled", facecolor=ps.BLUE,
            alpha=0.28, edgecolor="none", zorder=2)
    ax.hist(rets["constrained"], bins=bins, density=True, histtype="step", edgecolor=ps.BLUE, linewidth=2.5,
            zorder=3)
    var_c, cvar_c = tail_stats(-rets["constrained"])
    var_u, cvar_u = tail_stats(-rets["unconstrained"])
    share = (rets["constrained"] <= -var_c).mean()
    top = ax.get_ylim()[1]
    ax.axvspan(bins[0], -var_c, color=ps.BLUE, alpha=0.07, lw=0, zorder=0)
    ax.vlines(-var_c, 0, 1.16 * top, color=ps.BLUE, lw=2.5, ls=(0, (4, 3)))
    ax.vlines(-cvar_c, 0, 1.16 * top, color=ps.BLUE, lw=ps.LW_MAIN)
    ax.vlines(-cvar_u, 0, 0.98 * top, color=ps.RED, lw=ps.LW_MAIN)
    fs = ps.FONT - 3
    ax.text(-var_c + 0.15, 1.2 * top, r"$-$VaR$_{0.95}$", color=ps.BLUE, fontsize=fs, ha="left", va="bottom")
    ax.text(-cvar_c - 0.12, 1.2 * top, rf"$-$CVaR$_{{0.95}}$ = $-${cvar_c:.2f}%", color=ps.BLUE, fontsize=fs,
            ha="right", va="bottom", fontweight="bold")
    ax.text(-cvar_u - 0.2, 0.92 * top, f"$-$CVaR$_{{0.95}}$ = $-${cvar_u:.2f}%\n(unconstrained)", color=ps.RED,
            fontsize=fs, ha="right", va="top", linespacing=1.15)
    ax.text(bins[0] + 0.2, 0.40 * top, "shaded: worst 5%\nof constrained weeks", color=ps.BLUE, fontsize=fs,
            ha="left", va="top", linespacing=1.15)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=(*plt.matplotlib.colors.to_rgb(ps.BLUE), 0.28),
                             edgecolor=ps.BLUE, lw=2.5, label="CVaR-constrained (ours)"),
               plt.Rectangle((0, 0), 1, 1, facecolor=(*plt.matplotlib.colors.to_rgb(ps.RED_2), 0.3),
                             edgecolor=ps.RED, lw=1.8,
                             label="Unconstrained RL")]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.98), handlelength=1.2)
    ax.set_ylim(0, 1.45 * top)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xticks(np.arange(-8, 9, 2))
    ax.set_xlabel("Weekly return (%), stress window, five seeds pooled")
    ax.set_ylabel("Density")
    ax.set_yticks([])
    fig.tight_layout(pad=0.4)
    report(fig, "cvar_tail")
    print(f"cvar_tail pooled: constrained VaR {var_c:.3f} CVaR {cvar_c:.3f} (share in tail {share:.4f}); "
          f"unconstrained VaR {var_u:.3f} CVaR {cvar_u:.3f}")
    ps.save(fig, "figure_cvar_tail")


# ---------------------------------------------------------------------------
TRACE = pd.read_csv("results/tables_camera_ready/lambda_trace.csv")


def lambda_trace() -> None:
    """One thin line per seed and the five-seed mean, for the two dual step sizes."""
    ps.apply()
    fig, ax = plt.subplots(figsize=(POSTER_W, 3.7))
    style = {"scaled dual": (ps.BLUE, ps.BLUE_2, r"scaled dual step, $\eta_\lambda = 5$"),
             "mis-scaled dual": (ps.RED, ps.RED_2, r"mis-scaled dual step, $\eta_\lambda = 0.001$")}
    handles = []
    for arm, (colour, light, label) in style.items():
        d = TRACE[TRACE.arm == arm]
        for _, g in d.groupby("seed"):
            ax.plot(g["update"], g["lagrange"], color=light, lw=ps.LW_SEED, alpha=0.45, zorder=1)
        mean = d.groupby("update").lagrange.mean()
        if arm == "scaled dual":
            ax.fill_between(mean.index, 0, mean.values, color=ps.BLUE, alpha=0.12, lw=0, zorder=0)
        ax.plot(mean.index, mean.values, color=colour, lw=ps.LW_MAIN, zorder=3)
        handles.append(Line2D([0], [0], color=colour, lw=ps.LW_MAIN, label=label))
    handles.append(Line2D([0], [0], color=ps.BLUE_2, lw=ps.LW_SEED + 0.5, label="individual seeds"))
    top = TRACE.lagrange.max()
    ax.set_xlim(0, 1500)
    ax.set_ylim(-0.015, top * 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    fig.legend(handles=handles, loc="upper center", ncols=3, handlelength=1.4, columnspacing=1.2,
               bbox_to_anchor=(0.5, 1.0), fontsize=ps.FONT - 5)
    ax.set_xlabel("Training update")
    ax.set_ylabel(r"Multiplier $\lambda$")
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.86))
    report(fig, "lambda_trace")
    ps.save(fig, "figure_lambda_trace")


# ---------------------------------------------------------------------------
def protocol() -> None:
    """Slope chart: each line is one strategy, its Sharpe under two evaluation protocols."""
    sharpe = pd.read_csv("results/tables_protocol/protocol_sharpe.csv").set_index("strategy")
    display = {"minimum variance": ("Minimum variance", ps.DARK, 3.5),
               "inverse vol": ("Inverse vol", ps.GREY, 3.0),
               "diff. constrained (ours)": ("Constrained (ours)", ps.BLUE, 5.0),
               "diff. unconstrained": ("Unconstrained RL", ps.RED_2, 3.0)}
    nudge = {("walkforward", "inverse vol"): 0.045, ("walkforward", "diff. constrained (ours)"): -0.045}
    ps.apply()
    fig, ax = plt.subplots(figsize=(POSTER_W, 4.0))
    for name, row in sharpe.iterrows():
        label, colour, lw = display[name]
        ours = "ours" in name
        ax.plot([0, 1], [row.stress, row.walkforward], color=colour, lw=lw, zorder=4 if ours else 2,
                solid_capstyle="round")
        ax.plot([0, 1], [row.stress, row.walkforward], "o", ms=15 if ours else 12, color=colour, mec="white",
                mew=2.5, zorder=5 if ours else 3)
        text_colour = ps.RED if colour == ps.RED_2 else colour
        weight = "bold" if ours else "normal"
        for x, col, ha, dx in ((0, "stress", "right", -0.07), (1, "walkforward", "left", 0.07)):
            y = row[col] + nudge.get((col, name), 0.0)
            text = f"{label}  {row[col]:.2f}" if x == 0 else f"{row[col]:.2f}  {label}"
            ax.text(x + dx, y, text, ha=ha, va="center", fontsize=ps.FONT - 3, color=text_colour,
                    fontweight=weight)
    ax.set_xlim(-1.3, 2.05)
    ax.set_ylim(0.1, 1.55)
    ax.set_yticks([0.5, 1.0, 1.5])
    ax.set_xticks([0, 1], ["Single stress split", "Rolling walk-forward"])
    ax.set_ylabel("Sharpe ratio")
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_bounds(0.5, 1.5)
    ax.tick_params(axis="x", length=0, pad=12)
    fig.tight_layout(pad=0.4)
    report(fig, "protocol")
    ps.save(fig, "figure_protocol")


# ---------------------------------------------------------------------------
# Stress window, five-seed means. A1, A2, A3 and A6 are in
# results/tables_camera_ready/selection_log_summary.csv (unconstrained, buggy, fixed,
# A6_filter_scaled_dual); A7 is in paper tab:coupling; minimum variance is the classical
# rule re-scored against the same rolling budget (paper Section 7, budget reach).
COUPLING = [  # label, Sharpe, CVaR_0.99 (%), breach (%), face colour, alpha
    ("No constraint", 0.626, 6.5, 100.0, ps.RED_2, 0.55),
    (r"Penalty, $\eta_\lambda = 0.001$", 0.987, 2.5, 7.9, ps.BLUE, 0.15),
    (r"Penalty, $\eta_\lambda = 5$", 0.911, 2.8, 75.1, ps.BLUE, 0.3),
    (r"Penalty, $\lambda$ raised in testing", 0.826, 3.1, 81.9, ps.BLUE, 0.45),
    (r"Penalty, $\eta_\lambda = 5$, with filter (ours)", 0.851, 1.6, 7.4, ps.BLUE, 0.9),
    ("Minimum variance", 0.901, 4.4, 7.9, ps.GREY, 0.4),
]
TOLERATED = 5.0


def coupling() -> None:
    """Three metric panels of grouped bars, one bar per arm, values printed above each bar."""
    ps.apply()
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.2))
    panels = [(r"Sharpe ratio $\uparrow$", 1, "{:.2f}", (0, 1.25), [0, 0.5, 1.0]),
              (r"CVaR$_{0.99}$ (%) $\downarrow$", 2, "{:.1f}", (0, 8.0), [0, 2, 4, 6]),
              (r"Breach rate (%) $\downarrow$", 3, "{:.1f}", (0, 118), [0, 25, 50, 75, 100])]
    x = np.arange(len(COUPLING))
    for ax, (title, col, fmt, ylim, ticks) in zip(axes, panels):
        for i, row in enumerate(COUPLING):
            face = plt.matplotlib.colors.to_rgba(row[4], row[5])
            ours = "ours" in row[0]
            ax.bar(i, row[col], width=0.78, color=face, edgecolor=ps.darken(row[4]), lw=2.6 if ours else 1.8, zorder=2)
            ax.text(i, row[col] + 0.025 * ylim[1], fmt.format(row[col]), ha="center", va="bottom",
                    fontsize=ps.FONT - 6, fontweight="bold" if ours else "normal",
                    color=ps.BLUE if ours else ps.INK)
        if col == 3:
            ax.axhline(TOLERATED, color=ps.DARK, lw=2.2, ls="--", zorder=3)
        ax.set_title(title, fontsize=ps.FONT - 2, pad=14)
        ax.set_xticks([])
        ax.set_xlim(-0.6, len(COUPLING) - 0.4)
        ax.set_ylim(*ylim)
        ax.set_yticks(ticks)
        ax.tick_params(axis="y", labelsize=ps.FONT - 5)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=plt.matplotlib.colors.to_rgba(r[4], r[5]),
                             edgecolor=ps.darken(r[4]), lw=1.8, label=r[0]) for r in COUPLING]
    handles.append(Line2D([0], [0], color=ps.DARK, lw=2.2, ls="--", label="5% tolerated breach rate"))
    fig.legend(handles=handles, loc="lower center", ncols=4, bbox_to_anchor=(0.5, 0.0), fontsize=ps.FONT - 5,
               handlelength=1.3, columnspacing=2.0, labelspacing=0.5)
    fig.tight_layout(pad=0.3, w_pad=2.5, rect=(0, 0.25, 1, 1))
    report(fig, "coupling")
    ps.save(fig, "figure_coupling")


# ---------------------------------------------------------------------------
REGIMES = [("calm", "Calm"), ("high_vol", "High volatility"), ("selloff", "Selloff")]
REGIME_ARMS = [("rl_unconstrained", "Unconstrained RL", ps.RED_2, 0.55),
               ("rl_cvar_constrained", "CVaR-constrained RL (ours)", ps.BLUE, 0.9),
               ("min_variance", "Minimum variance", ps.GREY, 0.4)]


def regime() -> None:
    """Grouped bars of CVaR_0.99: one group per regime, one bar per strategy.

    Sized to sit beside the fold scatter on the poster. Maximum drawdown by regime is in the
    same tables (paper tab:regime) and is not drawn here.
    """
    tables = {key: pd.read_csv(f"results/tables_wf/walkforward_regime_{key}.csv", index_col=0)
              for key, *_ in REGIME_ARMS}
    n_weeks = {r: int(tables["rl_cvar_constrained"].loc[r, "n_weeks"]) for r, _ in REGIMES}
    ps.apply()
    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    width, top = 0.26, 7.8
    for j, (key, label, colour, alpha) in enumerate(REGIME_ARMS):
        ours = "ours" in label
        for i, (r, _) in enumerate(REGIMES):
            v = 100 * tables[key].loc[r, "cvar_99"]
            ax.bar(i + (j - 1) * width, v, width=width * 0.92, color=plt.matplotlib.colors.to_rgba(colour, alpha),
                   edgecolor=ps.darken(colour), lw=2.6 if ours else 1.8, zorder=2,
                   label=label if i == 0 else None)
            ax.text(i + (j - 1) * width, v + 0.02 * top, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=ps.FONT - 6, color=ps.BLUE if ours else ps.INK, fontweight="bold" if ours else "normal")
    ax.set_xticks(range(len(REGIMES)), [f"{name}\n{n_weeks[r]} weeks" for r, name in REGIMES], fontsize=ps.FONT - 4)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, top)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_ylabel(r"CVaR$_{0.99}$ (%)")
    ax.legend(loc="upper left", fontsize=ps.FONT - 5, handlelength=1.3, labelspacing=0.4, borderaxespad=0.1)
    fig.tight_layout(pad=0.3)
    report(fig, "regime")
    for r, _ in REGIMES:
        u, c = tables["rl_unconstrained"].loc[r, "cvar_99"], tables["rl_cvar_constrained"].loc[r, "cvar_99"]
        print(f"regime {r}: CVaR99 {u:.4f} -> {c:.4f} ({100 * (c / u - 1):+.1f}%)")
    ps.save(fig, "figure_regime")


# ---------------------------------------------------------------------------
TIE_TOL = 1e-12  # as in scripts/cr_fold_stats.py: a tie means the budget never bound
COVID_FOLD = 10   # fold k tests panel rows 364 + 26k .. 389 + 26k; k = 10 is the first half of 2020


def fold_scatter() -> None:
    """Each walk-forward fold as one point: unconstrained against constrained CVaR_0.99.

    Points below the diagonal are folds in which the constraint lowered tail risk;
    points on it are ties (the budget never bound and both learners coincide).
    Log axes, because fold CVaR ranges from 0.3% to 11% and the COVID fold dominates.
    """
    ps.apply()
    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    lo, hi = 0.25, 14.0
    grid = np.geomspace(lo, hi, 50)
    ax.fill_between(grid, lo, grid, color=ps.GREEN_1, lw=0, zorder=0)
    ax.plot(grid, grid, color=ps.GREY, lw=2.5, ls="--", zorder=1)
    styles = {"etf7": ("o", ps.BLUE, "white", "Macro ETF (7)"),
              "sector10": ("s", "white", ps.BLUE, "Sector ETF (10)")}
    for key, (marker, face, edge, name) in styles.items():
        d = pd.read_csv(f"results/tables_stats/folds_{key}.csv").sort_values("fold")
        unc, con = 100 * d.cvar99_unc.to_numpy(), 100 * d.cvar99_con.to_numpy()
        delta = con - unc
        lower, tied = delta < -100 * TIE_TOL, np.abs(delta) <= 100 * TIE_TOL
        ax.scatter(unc, con, s=170, marker=marker, facecolor=face, edgecolor=edge, linewidth=2.2, zorder=3,
                   label=f"{name}: lower in {lower.sum()} of {(~tied).sum()}")
        print(f"fold_scatter {key}: lower {lower.sum()}, tied {tied.sum()}, higher {(~lower & ~tied).sum()}, "
              f"mean CVaR99 {unc.mean():.2f}% -> {con.mean():.2f}%")
        k = int(np.where(d.fold.to_numpy() == COVID_FOLD)[0][0])
        offset = (1.45, 0.42) if key == "etf7" else (0.95, 0.5)
        ax.annotate("2020 H1", (unc[k], con[k]), xytext=(unc[k] * offset[0], con[k] * offset[1]),
                    fontsize=ps.FONT - 4, ha="center", va="bottom", color=ps.DARK, bbox=ps.LABEL_BOX,
                    arrowprops=dict(arrowstyle="-", color=ps.GREY, lw=1.5, shrinkA=2, shrinkB=7))
    ax.text(4.6, 0.33, "Below the diagonal:\nconstraint lowered\ntail risk", fontsize=ps.FONT - 3,
            color=ps.GREEN_DARK, ha="center", va="bottom", linespacing=1.1, fontweight="bold")
    ax.text(0.3, 1.25, "Above: constraint\nraised it", fontsize=ps.FONT - 4, color=ps.GREY, ha="left",
            va="bottom", linespacing=1.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [0.5, 1, 2, 5, 10]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(plt.FixedLocator(ticks))
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))
        axis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(axis="both")
    ax.set_xlabel(r"CVaR$_{0.99}$ without the constraint")
    ax.set_ylabel(r"CVaR$_{0.99}$ with the constraint")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), handletextpad=0.2, labelspacing=0.4,
              fontsize=ps.FONT - 4, title="Untied folds", title_fontsize=ps.FONT - 4, alignment="left")
    fig.tight_layout(pad=0.4)
    report(fig, "fold_scatter")
    ps.save(fig, "figure_fold_scatter")


if __name__ == "__main__":
    stress_paths()
    cvar_tail()
    lambda_trace()
    protocol()
    coupling()
    regime()
    fold_scatter()
