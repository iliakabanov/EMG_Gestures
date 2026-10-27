"""Shared matplotlib style for report figures: one palette, recessive axes, consistent sizes."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

# Categorical slots in fixed order (validated palette; never cycle or reorder).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Reference / baseline series (e.g. rest) in muted ink rather than a categorical hue.
REFERENCE = "#898781"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def setup() -> None:
    """Apply the project style to matplotlib."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.size": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "lines.linewidth": 1.4, "legend.frameon": False, "legend.labelcolor": INK_2,
    })


def save(fig: plt.Figure, name: str) -> Path:
    """Save a figure to reports/figures/<name>.png and return the path."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    return path
