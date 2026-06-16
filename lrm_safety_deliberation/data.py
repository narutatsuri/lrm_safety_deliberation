"""Benchmark loading and the ASR / ORR split registry.

All splits are sourced from portable input manifests under
``data/<bench>/<bench>.jsonl`` — no network access, no subsampling.

The paper uses two evaluation suites:

* **Main suite** (paper Table ``tab:eval_datasets``; used throughout the body —
  the probe, oscillation, defense table, Pareto, slopegraph):
    - ``asr``  — WildJailbreak (2,000 adversarial-harmful) + FORTRESS (500) = 2,500.
    - ``orr``  — OR-Bench-Hard (1,319) + FalseReject (1,187) + CoCoNot (379) = 2,885.
* **Appendix battery** (paper Table ``tab:base_asr_orr``; the famous-benchmark
  sweep at 16K):
    - ``asr_full``  — 8 benchmarks (advbench, harmbench, strongreject, sorrybench,
      jailbreakbench, wildjailbreak, fortress, hexphi) = 4,573.
    - ``orr_full``  — 4 benchmarks (or_bench, falsereject, coconot, xstest_safe) = 3,135.

The main suite is a subset of the appendix battery over the shared manifests
(``or_bench`` is OR-Bench-Hard). The over-refusal *supplementary* sources
(PHTest, ORFuzzSet) split ``orr_supplementary`` is fetched from the HuggingFace
Hub on demand (not redistributed in ``data/``).
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .io import read_jsonl
from .paths import DATA_ROOT

__all__ = [
    "MAIN_ASR",
    "MAIN_ORR",
    "ASR_FULL",
    "ORR_FULL",
    "SPLIT_SPECS",
    "load_manifest",
    "build_split",
    "split_benchmarks",
]

# Main-suite benchmarks (paper Table tab:eval_datasets), in paper order.
MAIN_ASR = ["wildjailbreak", "fortress"]                 # 2,000 + 500 = 2,500
MAIN_ORR = ["or_bench", "falsereject", "coconot"]        # 1,319 + 1,187 + 379 = 2,885

# Appendix-battery benchmarks (paper Table tab:base_asr_orr), in paper order.
ASR_FULL = [
    "advbench", "harmbench", "strongreject", "sorrybench",
    "jailbreakbench", "wildjailbreak", "fortress", "hexphi",
]
ORR_FULL = ["or_bench", "falsereject", "coconot", "xstest_safe"]


# --------------------------------------------------------------------------- #
# Manifest loader (portable, the single source of truth for the full battery)
# --------------------------------------------------------------------------- #
def load_manifest(benchmark: str) -> list[dict[str, Any]]:
    """Load a normalized benchmark from ``data/<benchmark>/<benchmark>.jsonl``.

    Each row has a uniform schema ``{sample_id, prompt, category, **extras}``.
    """
    path = DATA_ROOT / benchmark / f"{benchmark}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark manifest not found: {path}\n"
            f"Expected layout: data/{benchmark}/{benchmark}.jsonl"
        )
    items: list[dict[str, Any]] = []
    for r in read_jsonl(path):
        prompt = str(r.get("prompt", "")).strip()
        if not prompt:
            continue
        extras = {
            k: v for k, v in r.items()
            if k not in {"id", "prompt", "benchmark", "split_type", "category"}
        }
        items.append({
            "sample_id": r.get("id", len(items)),
            "prompt": prompt,
            "category": str(r.get("category", benchmark)),
            **extras,
        })
    return items


def _subsample(items: list[dict], max_samples: int | None, seed: int) -> list[dict]:
    if not max_samples or max_samples <= 0 or max_samples >= len(items):
        return items
    rng = random.Random(seed)
    idxs = list(range(len(items)))
    rng.shuffle(idxs)
    return [items[i] for i in sorted(idxs[:max_samples])]


# --------------------------------------------------------------------------- #
# Optional HuggingFace-hosted over-refusal supplementary sources
# --------------------------------------------------------------------------- #
def _hf_load(*args, **kwargs):
    import datasets  # imported lazily; only needed for the supplementary splits
    return datasets.load_dataset(*args, **kwargs)


def load_phtest_harmless(max_samples=None, seed=42, cache_dir=None, **_):
    ds = _hf_load("furonghuang-lab/PHTest", split="train", cache_dir=cache_dir)
    items = []
    for i, row in enumerate(ds):
        if str(row.get("Harmfulness", "")).strip().lower() != "harmless":
            continue
        prompt = str(row.get("Request", "")).strip()
        if prompt:
            items.append({"sample_id": i, "prompt": prompt, "category": "phtest_harmless"})
    return _subsample(items, max_samples, seed)


def load_orfuzzset(max_samples=None, seed=42, csv_path=None, **_):
    import pandas as pd
    source = csv_path or (DATA_ROOT / "orfuzzset" / "ORFuzz.csv")
    df = pd.read_csv(source)
    col = next((c for c in ("input", "prompt", "Request", "request") if c in df.columns), None)
    if col is None:
        raise ValueError(f"ORFuzzSet csv missing a prompt column: {list(df.columns)}")
    items = [
        {"sample_id": int(i), "prompt": str(row[col]).strip(), "category": "orfuzzset"}
        for i, row in df.iterrows() if str(row[col]).strip()
    ]
    return _subsample(items, max_samples, seed)


def _manifest_loader(benchmark: str) -> Callable[..., list[dict]]:
    def _load(max_samples=None, seed=42, **_):
        return _subsample(load_manifest(benchmark), max_samples, seed)
    return _load


# --------------------------------------------------------------------------- #
# Split registry
# --------------------------------------------------------------------------- #
SPLIT_SPECS: dict[str, list[tuple[str, Callable[..., list[dict]]]]] = {
    # Main body suite (tab:eval_datasets).
    "asr": [(b, _manifest_loader(b)) for b in MAIN_ASR],
    "orr": [(b, _manifest_loader(b)) for b in MAIN_ORR],
    # Appendix battery (tab:base_asr_orr).
    "asr_full": [(b, _manifest_loader(b)) for b in ASR_FULL],
    "orr_full": [(b, _manifest_loader(b)) for b in ORR_FULL],
    # Over-refusal supplementary (HuggingFace, not redistributed).
    "orr_supplementary": [
        ("phtest_harmless", load_phtest_harmless),
        ("orfuzzset", load_orfuzzset),
    ],
}


def split_benchmarks(split: str) -> list[str]:
    """Benchmark names that make up a split, in paper order."""
    if split not in SPLIT_SPECS:
        raise KeyError(f"Unknown split {split!r}. Known: {sorted(SPLIT_SPECS)}")
    return [name for name, _ in SPLIT_SPECS[split]]


def build_split(
    split: str,
    *,
    max_samples: int | None = None,
    seed: int = 42,
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Materialize a split into flat prompt records with a contiguous ``global_id``.

    Each record: ``{split, benchmark, benchmark_sample_id, prompt, category,
    [source_meta], global_id}`` — the schema the generation pipeline consumes.
    """
    if split not in SPLIT_SPECS:
        raise KeyError(f"Unknown split {split!r}. Known: {sorted(SPLIT_SPECS)}")

    records: list[dict[str, Any]] = []
    for benchmark, loader in SPLIT_SPECS[split]:
        for idx, item in enumerate(loader(max_samples=max_samples, seed=seed, cache_dir=cache_dir)):
            source_meta = {
                k: v for k, v in item.items()
                if k not in {"sample_id", "prompt", "category"}
            }
            record = {
                "split": split,
                "benchmark": benchmark,
                "benchmark_sample_id": item.get("sample_id", idx),
                "prompt": item["prompt"],
                "category": item.get("category", benchmark),
            }
            if source_meta:
                record["source_meta"] = source_meta
            records.append(record)

    for global_id, record in enumerate(records):
        record["global_id"] = global_id
    return records


def split_counts(split: str, **kwargs) -> dict[str, int]:
    """Per-benchmark prompt counts for a split (handy for sanity checks)."""
    return dict(Counter(r["benchmark"] for r in build_split(split, **kwargs)))
