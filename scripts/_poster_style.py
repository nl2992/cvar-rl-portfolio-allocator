"""Poster figure style: figures4papers layout conventions with a vivid, soft rendering.

Layout follows figures4papers (github.com/ChenLiu-1996/figures4papers): Helvetica, open
top/right spines, frameless legends, large type. Rendering follows the look of the vivid
figures skill (github.com/yjz211/vivid-figures-skill), re-implemented here rather than
copied, since that repository's licence forbids derivative works: translucent fills with a
same-hue outline, white-edged markers, thin grey axes, a faint dashed y-grid, individual
paths drawn lightly and the representative one strongly. The palette is its Blue Sky
colours mapped to meaning: blue for the proposed method, red and coral for unconstrained
RL and failing settings, green for regions within budget, grey for classical baselines.

Figures are drawn at the size they are printed on the A0 poster (a 0.78-linewidth figure
is about 11.5 in wide), so font sizes here are the printed point sizes. Body text on the
poster is 21 pt; figure labels sit just above it.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

BLUE = "#377EB8"         # proposed method
BLUE_2 = "#86B3C7"       # its individual seeds
RED = "#B23648"          # unconstrained RL / failing configuration
RED_2 = "#DC7369"
RED_1 = "#F3D0CC"        # light coral fill
GREEN_1 = "#DCE8D5"      # region within budget / where the constraint lowered risk
GREEN_3 = "#8AAA7B"
GREEN_DARK = "#4F7342"   # text on the green region
NEUTRAL = "#BCC3C6"      # light grey fill
GREY = "#8E999D"         # classical baselines
DARK = "#4D4D4D"
INK = "#333333"
AXIS = "#6B6B6B"
GRID = "#E3E3E3"
LABEL_BOX = dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none")

FONT = 24
LW_AXES = 1.6
LW_MAIN = 4.0
LW_SEED = 1.2


def darken(colour: str, factor: float = 0.72) -> tuple[float, float, float]:
    """The same hue, darker: the outline drawn around a translucent fill."""
    return tuple(factor * c for c in to_rgb(colour))


def apply() -> None:
    plt.rcParams.update({
        "font.family": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
        "font.size": FONT,
        "axes.labelsize": FONT + 2,
        "xtick.labelsize": FONT - 2,
        "ytick.labelsize": FONT - 2,
        "legend.fontsize": FONT - 2,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "axes.linewidth": LW_AXES,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelpad": 10,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linestyle": (0, (5, 4)),
        "grid.linewidth": 1.2,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save(fig, stem: str, out: str = "poster/figures") -> None:
    Path(out).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(out) / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(Path(out) / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", Path(out) / f"{stem}.pdf", "and .png")
