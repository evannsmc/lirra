"""Matplotlib defaults for publication figures (TrueType fonts, never Type 3)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
}


def setup() -> None:
    plt.rcParams.update(RC)


def save(fig, path_without_suffix: str | Path) -> list[Path]:
    path = Path(path_without_suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for suffix in (".pdf", ".png"):
        target = path.with_suffix(suffix)
        fig.savefig(target)
        out.append(target)
    plt.close(fig)
    return out
