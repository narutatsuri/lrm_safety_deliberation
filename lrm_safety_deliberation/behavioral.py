"""Behavioral ASR / ORR readouts from the full-eval summaries.

ASR = comply rate on the harmful pool (lower is safer); ORR = refusal rate on
the benign pool (lower is more useful).  Both come from the per-cell
``__fullpipe__/<asr|orr>/<cell>/summary.json`` written by the evaluation
pipeline.  Shared by the defense table and the ASR-ORR Pareto figure.
"""

from __future__ import annotations

from typing import Optional

from .io import read_json
from .paths import defenses_eval, neurips_final

__all__ = ["cell_summary_path", "load_cell_summary", "cell_asr_orr"]


def cell_summary_path(method: str, base: str, split: str):
    """Path to a cell's summary for one split.

    ``method="Base"`` reads the base-model tree (``eval_results/neurips_final``);
    any other method reads the defense cell ``<method>-<base>``.
    ``split`` is ``"asr"`` or ``"orr"``.
    """
    if method == "Base":
        return neurips_final(base, "__fullpipe__", split, base, "summary.json")
    cell = f"{method}-{base}"
    return defenses_eval(cell, "__fullpipe__", split, cell, "summary.json")


def load_cell_summary(method: str, base: str, split: str) -> Optional[dict]:
    p = cell_summary_path(method, base, split)
    return read_json(p) if p.exists() else None


def _read(summary: Optional[dict], vote: str, kind: str) -> Optional[float]:
    if summary is None:
        return None
    if vote == "soft":
        if kind == "asr":
            return summary.get("soft_asr")
        return summary.get("soft_orr", summary.get("soft_refusal_fraction"))
    if kind == "asr":
        return summary.get("majority_asr")
    return summary.get("majority_orr", summary.get("majority_refusal_rate"))


def cell_asr_orr(method: str, base: str, vote: str = "majority") -> tuple[Optional[float], Optional[float]]:
    """Return ``(asr, orr)`` as fractions in [0, 1] for one (method, base) cell.

    ``vote`` is ``"majority"`` (>=3/4, ties -> comply) or ``"soft"`` (mean
    refusal_votes/4).  Missing cells return ``(None, None)``.
    """
    asr = _read(load_cell_summary(method, base, "asr"), vote, "asr")
    orr = _read(load_cell_summary(method, base, "orr"), vote, "orr")
    return asr, orr
