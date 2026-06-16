"""lrm_safety_deliberation — reproduction library for *the refusal valley* paper.

A small, well-factored toolkit behind the paper's experiments: a model registry,
the benchmark/split loaders, the four-guardrail safety evaluator, hidden-state
and probe utilities, the causal-cut / oscillation machinery, shared statistics,
and a single plotting style.  The thin per-figure/per-table drivers live in
``reproduce/``; the heavy GPU/training drivers live in ``pipeline/``.

Typical use::

    from lrm_safety_deliberation import models, data, guardrails, plotting
    spec = models.get("Qwen3-8B")
    records = data.build_split("asr_full")
    plotting.apply_paper_style()
"""

from __future__ import annotations

from . import behavioral, data, guardrails, io, models, paths, plotting, stats

__all__ = ["paths", "io", "models", "data", "guardrails", "behavioral", "stats", "plotting"]

__version__ = "0.1.0"
