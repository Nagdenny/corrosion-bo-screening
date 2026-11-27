"""Shared figure style for the corrosion-BO manuscript.

Every figure script imports this module so that all figures use the same
fonts, line weights, colours and journal sizes (Elsevier: single column
90 mm, 1.5 column 140 mm, double column 190 mm).
"""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4                       # millimetres -> inches
WIDTH_SINGLE = 90 * MM
WIDTH_ONEHALF = 140 * MM
WIDTH_DOUBLE = 180 * MM             # inside the 190 mm double-column limit

# ------------------------------------------------------------------
# One palette for the whole manuscript. Each colour keeps the same
# meaning in every figure in which it appears.
# ------------------------------------------------------------------
COLORS = {
    # model development (Figures 3 and 6)
    "baseline": "#A6A6A6",          # baseline / reference
    "optimized": "#0F4D92",         # optimized model (hero)
    "optimized_light": "#8FB3DC",   # five-seed mean of the optimized model
    # descriptor type (Figure 5)
    "environment": "#42949E",
    "composition": "#9A4D8E",
    # corrosion-behaviour classes (Figures 4 and 9): ordered from dark red
    # to dark blue, with a light yellow-green for Good so that Good and
    # Resistant differ in both hue and lightness (colour-blind safe)
    "Severe": "#B2182B",
    "Poor": "#E69F00",
    "Good": "#A6D854",
    "Resistant": "#0072B2",
    # selection strategies (Figures 7, 8 and 10): Okabe-Ito hues that stay
    # distinct as lines and as light shaded bands
    "EI": "#E69F00",       # orange
    "PI": "#0072B2",       # blue
    "UCB": "#B2182B",      # reddish purple
    "Greedy": "#A6D854",   # bluish green
    "Random": "#808080",   # grey
    # neutrals
    "text": "#272727",
    "grid": "#E6E6E6",
}

CLASS_ORDER = ["Severe", "Poor", "Good", "Resistant"]
SURROGATE_STYLE = {"RF": "-", "GP": "--"}       # Figure 7
SURROGATE_MARKER = {"RF": "o", "GP": "s"}
BAND_ALPHA = 0.18          # shaded +/- std bands (Figures 7 and 8)


def _sans_font():
    """First installed font from the preferred list (Arial on Windows)."""
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]:
        if name in installed:
            return name
    return "DejaVu Sans"


def apply_style():
    sans = _sans_font()
    mpl.rcParams.update({
        # sub/superscripts written as $_3$ or $^{-}$ use the same font
        # (Arial has no Unicode subscript digits such as U+2083)
        "mathtext.fontset": "custom",
        "mathtext.rm": sans,
        "mathtext.it": f"{sans}:italic",
        "mathtext.bf": f"{sans}:bold",
        "mathtext.default": "regular",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans",
                            "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",       # editable text in SVG
        "pdf.fonttype": 42,           # editable TrueType text in PDF
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.6,       # thicker data lines
        "lines.markersize": 4,
        "errorbar.capsize": 2.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.edgecolor": COLORS["text"],
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "legend.frameon": False,
        "savefig.dpi": 600,
    })


def panel_label(ax, letter, x=-0.14, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left", color=COLORS["text"])


# Default output folder: <repository root>/results/figures_publication
_DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "figures_publication")


def save(fig, stem, out_dir=_DEFAULT_OUT, multipanel=False):
    """Export SVG, PDF, 600-dpi TIFF and a 300-dpi PNG for the Word file.

    For multi-panel figures, the panel-alignment gate of the
    nature-figure skill is run first when it is available.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, stem)
    if multipanel:
        try:
            from audit_panel_alignment import (
                require_matplotlib_panel_alignment)
            fig.canvas.draw()
            require_matplotlib_panel_alignment(
                fig, json_out=f"{path}.alignment.json",
                tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
        except ImportError:
            print("  (alignment audit not available; skipped)",
                  file=sys.stderr)
    fig.savefig(f"{path}.svg", bbox_inches="tight")
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    fig.savefig(f"{path}.tiff", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(f"{path}.png", dpi=300, bbox_inches="tight")
    print(f"Saved {path}.[svg|pdf|tiff|png]")
