"""Shared figure styling so every plot in :mod:`reproduce` looks identical.

The paper uses TeX Gyre Termes (a Times clone matching the body font) and a
fixed colour-blind-safe palette keyed by model.  The research scripts
re-registered the font and re-declared the palette in ~15 places; this module is
the single definition.

Font handling is best-effort and portable: :func:`register_fonts` searches the
TeX Live install (where the OTFs ship) and a few common locations, and silently
falls back to the default serif if none are found, so plots still render on a
machine without TeX Live.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

__all__ = [
    "MODEL_COLORS",
    "MODEL_LABELS",
    "SEMANTIC_COLORS",
    "register_fonts",
    "apply_paper_style",
]

# Per-model line/marker colours (colour-blind-safe Dark2 subset).
MODEL_COLORS = {
    "Qwen3-8B": "#1b9e77",
    "Olmo-3-7B-Think": "#d95f02",
    "Phi-4-reasoning": "#7570b3",
    "GPT-OSS-20B": "#e7298a",
}

# Display names (paper capitalization).
MODEL_LABELS = {
    "Qwen3-8B": "Qwen3-8B",
    "Olmo-3-7B-Think": "OLMo-3-7B-Think",
    "Phi-4-reasoning": "Phi-4-Reasoning",
    "GPT-OSS-20B": "GPT-OSS-20B",
}

# Harmful / benign semantic colours (refusal-valley fills).
SEMANTIC_COLORS = {
    "harmful": "#C0392B", "harmful_fill": "#F5B7B1",
    "benign": "#2471A3", "benign_fill": "#AED6F1",
}

# Where the TeX Gyre Termes OTFs typically live.
_FONT_SEARCH_DIRS = [
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre",
    "/usr/share/texmf/fonts/opentype/public/tex-gyre",
    "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/tex-gyre",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/.local/share/fonts"),
]

_FONTS_REGISTERED = False


def register_fonts() -> bool:
    """Register TeX Gyre Termes OTFs with matplotlib if available.

    Returns ``True`` if at least one face was registered, else ``False`` (in
    which case plots fall back to the default serif font).
    """
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return True
    from matplotlib import font_manager

    registered = False
    for pattern in _FONT_SEARCH_DIRS:
        for d in glob.glob(pattern):
            for otf in glob.glob(os.path.join(d, "texgyretermes-*.otf")):
                try:
                    font_manager.fontManager.addfont(otf)
                    registered = True
                except Exception:
                    pass
    _FONTS_REGISTERED = registered
    return registered


def apply_paper_style(scale: float = 1.0, *, base_size: float = 15.0) -> None:
    """Apply the paper's matplotlib rcParams (serif font, tick/label sizes).

    ``scale`` multiplies every font size; ``base_size`` is the unscaled body size.
    Safe to call repeatedly; registers fonts on first use.
    """
    import matplotlib as mpl

    register_fonts()
    s = scale * base_size
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": s,
        "axes.titlesize": s * 1.4,
        "axes.labelsize": s * 1.15,
        "xtick.labelsize": s * 0.9,
        "ytick.labelsize": s * 0.9,
        "legend.fontsize": s * 0.8,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.45,
        "grid.color": "#9a9a9a",
        "axes.axisbelow": True,
    })


def model_color(name: str) -> str:
    """Colour for a (possibly defense-cell) model, falling back to its base."""
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    for base, color in MODEL_COLORS.items():
        if name.endswith(base):
            return color
    return "#444444"
