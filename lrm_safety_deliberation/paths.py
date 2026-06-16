"""Central path configuration for the ``lrm_safety_deliberation`` release.

Every script in :mod:`reproduce` and every heavy driver in ``pipeline/`` resolves
its inputs and outputs through this module rather than hard-coding absolute
paths.  This makes the release portable: clone it anywhere, optionally point a
few environment variables at where your data / models / intermediate artifacts
live, and everything resolves correctly.

Layout
------
``PROJECT_ROOT``    the ``camera_ready/`` directory (repo root of the release).
``DATA_ROOT``       benchmark *input* manifests (``data/<bench>/<bench>.jsonl``).
``MODELS_ROOT``     model weights (HuggingFace snapshots / local checkpoints).
``ARTIFACTS_ROOT``  intermediate artifacts (generations, classifications,
                    representations, cut rollouts, analysis caches).  These are
                    large and regenerable; the release ships *code* to make them
                    and a manifest to *download* them from the HuggingFace Hub.
``FIGURES_ROOT``    where ``reproduce/`` writes the final figures and tables.

Overrides
---------
Each root honours an environment variable so you can keep heavy data outside the
checkout (or point the library at the original research tree for verification)::

    LRM_SAFETY_DATA        -> DATA_ROOT
    LRM_SAFETY_MODELS      -> MODELS_ROOT
    LRM_SAFETY_ARTIFACTS   -> ARTIFACTS_ROOT
    LRM_SAFETY_FIGURES     -> FIGURES_ROOT

Intermediate artifacts keep the *same relative subtree* they had in the original
research repository (e.g. ``eval_results/neurips_final/<model>/...``), so the
downloadable HuggingFace bundle and the original tree are layout-compatible: set
``LRM_SAFETY_ARTIFACTS`` to either one and the reproduce scripts just work.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "DATA_ROOT",
    "MODELS_ROOT",
    "ARTIFACTS_ROOT",
    "FIGURES_ROOT",
    "artifact",
    "data_manifest",
    "model_path",
    "eval_results",
    "neurips_final",
    "defenses_eval",
    "figure_out",
]

# camera_ready/lrm_safety_deliberation/paths.py -> parents[1] == camera_ready/
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _root(env_var: str, default: Path) -> Path:
    override = os.environ.get(env_var)
    return Path(override).expanduser().resolve() if override else default


DATA_ROOT = _root("LRM_SAFETY_DATA", PROJECT_ROOT / "data")
MODELS_ROOT = _root("LRM_SAFETY_MODELS", PROJECT_ROOT / "models")
ARTIFACTS_ROOT = _root("LRM_SAFETY_ARTIFACTS", PROJECT_ROOT / "artifacts")
FIGURES_ROOT = _root("LRM_SAFETY_FIGURES", PROJECT_ROOT / "figures")


# --------------------------------------------------------------------------- #
# Generic resolvers
# --------------------------------------------------------------------------- #
def artifact(*parts: str | Path) -> Path:
    """Resolve a path under ``ARTIFACTS_ROOT`` (the intermediate-artifact tree)."""
    return ARTIFACTS_ROOT.joinpath(*[str(p) for p in parts])


def data_manifest(benchmark: str) -> Path:
    """Path to a benchmark input manifest: ``data/<benchmark>/<benchmark>.jsonl``."""
    return DATA_ROOT / benchmark / f"{benchmark}.jsonl"


def model_path(name_or_relpath: str) -> Path:
    """Resolve a model weight directory.

    Accepts either a registry-style relative path (``models/Qwen3-8B`` or
    ``models/final/STAR1-Qwen3-8B``) or a bare model name.  A leading ``models/``
    is stripped so the result is anchored at :data:`MODELS_ROOT`.
    """
    rel = str(name_or_relpath)
    if rel.startswith("models/"):
        rel = rel[len("models/"):]
    return MODELS_ROOT / rel


# --------------------------------------------------------------------------- #
# Named artifact subtrees (mirror the original research layout)
# --------------------------------------------------------------------------- #
def eval_results(*parts: str | Path) -> Path:
    """``<ARTIFACTS_ROOT>/eval_results/...``"""
    return artifact("eval_results", *parts)


def neurips_final(model: str, *parts: str | Path) -> Path:
    """Per-model canonical eval tree: ``eval_results/neurips_final/<model>/...``"""
    return eval_results("neurips_final", model, *parts)


def defenses_eval(cell: str, *parts: str | Path) -> Path:
    """Per-defense-cell eval tree: ``eval_results/defenses_eval/<cell>/...``"""
    return eval_results("defenses_eval", cell, *parts)


def figure_out(name: str) -> Path:
    """Output path for a final figure/table under :data:`FIGURES_ROOT`."""
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    return FIGURES_ROOT / name
