"""Unified benchmark registries for the full eval pipeline."""

from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _data_dir() -> Path:
    """Resolve the benchmark-manifest directory (the shipped camera_ready/data).

    Honors LRM_SAFETY_DATA, then the installed lrm_safety_deliberation library, then a local
    fallback — so the moved pipeline finds the manifests wherever the release lives.
    """
    env = os.environ.get("LRM_SAFETY_DATA")
    if env:
        return Path(env)
    try:
        from lrm_safety_deliberation.paths import DATA_ROOT
        return DATA_ROOT
    except Exception:
        return _PROJECT_ROOT / "data"


def _load_hf_dataset(*args, **kwargs):
    """Load the HuggingFace datasets package without clashing with our module."""
    import importlib
    import sys

    hf_mod = sys.modules.get("datasets")
    if hf_mod is not None and hasattr(hf_mod, "load_dataset"):
        return hf_mod.load_dataset(*args, **kwargs)

    hf_mod = importlib.import_module("datasets")
    sys.modules["_hf_datasets_pkg"] = hf_mod
    return hf_mod.load_dataset(*args, **kwargs)


def _subsample(items: list[dict], max_samples: int | None, seed: int) -> list[dict]:
    if max_samples is None or max_samples <= 0 or max_samples >= len(items):
        return items
    rng = random.Random(seed)
    idxs = list(range(len(items)))
    rng.shuffle(idxs)
    keep = sorted(idxs[:max_samples])
    return [items[i] for i in keep]


def load_wj2k_items(
    max_samples: int | None = None,
    seed: int = 42,
    **_,
) -> list[dict]:
    """Load the full WildJailbreak eval split of adversarial_harmful prompts."""
    eval_tsv = _PROJECT_ROOT / "evaluation" / "wildjailbreak" / "eval" / "eval.tsv"
    items = []
    with open(eval_tsv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            if row.get("data_type") != "adversarial_harmful":
                continue
            items.append(
                {
                    "sample_id": len(items),
                    "prompt": row["adversarial"],
                    "category": row.get("data_type", "adversarial_harmful"),
                    "label": int(row.get("label", "1")),
                    "data_type": row.get("data_type", "adversarial_harmful"),
                    "source_row": row_idx,
                }
            )
    return _subsample(items, max_samples, seed)


def load_fortress_items(
    max_samples: int | None = None,
    seed: int = 42,
    **_,
) -> list[dict]:
    """Load the full FORTRESS public split."""
    par_path = _PROJECT_ROOT / "evaluation" / "fortress" / "fortress_public" / "data" / "train-00000-of-00001.parquet"
    df = pd.read_parquet(par_path)
    items = []
    for row_idx, row in df.iterrows():
        rubric = row["rubric"]
        if hasattr(rubric, "tolist"):
            rubric = rubric.tolist()
        items.append(
            {
                "sample_id": int(row.get("ID", row_idx)),
                "prompt": str(row["adversarial_prompt"]),
                "category": str(row.get("risk_domain", "fortress")),
                "risk_domain": str(row.get("risk_domain", "fortress")),
                "risk_subdomain": str(row.get("risk_subdomain", "")),
                "rubric": list(rubric),
                "source_row": int(row_idx),
            }
        )
    return _subsample(items, max_samples, seed)


def load_or_bench_items(
    max_samples: int | None = None,
    seed: int = 42,
    **_,
) -> list[dict]:
    from evaluation.or_bench.evaluator import load_or_bench_hard

    raw = load_or_bench_hard()
    items = [
        {
            "sample_id": i,
            "prompt": x["prompt"],
            "category": x.get("category", "or_bench"),
        }
        for i, x in enumerate(raw)
    ]
    return _subsample(items, max_samples, seed)


def load_false_reject_test_items(
    max_samples: int | None = None,
    seed: int = 42,
    cache_dir: str | None = None,
    **_,
) -> list[dict]:
    load_dataset = _load_hf_dataset
    ds = load_dataset("AmazonScience/FalseReject", split="test", cache_dir=cache_dir)
    items = []
    for i, row in enumerate(ds):
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        cat = row.get("category_text")
        if cat is None:
            cat = str(row.get("category", "falsereject"))
        items.append(
            {
                "sample_id": int(row.get("id", i)) if str(row.get("id", "")).isdigit() else i,
                "prompt": prompt,
                "category": str(cat),
            }
        )
    return _subsample(items, max_samples, seed)


def load_coconot_items(
    max_samples: int | None = None,
    seed: int = 42,
    cache_dir: str | None = None,
    **_,
) -> list[dict]:
    load_dataset = _load_hf_dataset
    ds = load_dataset("allenai/coconot", "contrast", split="test", cache_dir=cache_dir)
    items = []
    for i, row in enumerate(ds):
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        items.append(
            {
                "sample_id": i,
                "prompt": prompt,
                "category": str(row.get("subcategory", row.get("category", "coconot"))),
            }
        )
    return _subsample(items, max_samples, seed)


def load_phtest_items(
    max_samples: int | None = None,
    seed: int = 42,
    include_controversial: bool = False,
    phtest_split: str | None = None,
    split: str = "train",
    cache_dir: str | None = None,
    **_,
) -> list[dict]:
    load_dataset = _load_hf_dataset
    actual_split = phtest_split or split
    ds = load_dataset("furonghuang-lab/PHTest", split=actual_split, cache_dir=cache_dir)
    allowed = {"harmless", "controversial"} if include_controversial else {"harmless"}
    items = []
    for i, row in enumerate(ds):
        harm = str(row.get("Harmfulness", "")).strip().lower()
        if harm not in allowed:
            continue
        prompt = str(row.get("Request", "")).strip()
        if not prompt:
            continue
        items.append(
            {
                "sample_id": int(row.get("ID", i)) if str(row.get("ID", "")).isdigit() else i,
                "prompt": prompt,
                "category": f"phtest_{harm}",
            }
        )
    return _subsample(items, max_samples, seed)


def load_orfuzzset_items(
    max_samples: int | None = None,
    seed: int = 42,
    csv_path: str | None = None,
    **_,
) -> list[dict]:
    source = csv_path or (_PROJECT_ROOT / "evaluation" / "overrefusal_suite" / "ORFuzz.csv")
    df = pd.read_csv(source)
    prompt_col = None
    for cand in ("input", "prompt", "Request", "request"):
        if cand in df.columns:
            prompt_col = cand
            break
    if prompt_col is None:
        raise ValueError(f"ORFuzzSet csv missing prompt column. Columns: {list(df.columns)}")
    items = []
    for i, row in df.iterrows():
        prompt = str(row[prompt_col]).strip()
        if not prompt:
            continue
        items.append(
            {
                "sample_id": int(i),
                "prompt": prompt,
                "category": "orfuzzset",
            }
        )
    return _subsample(items, max_samples, seed)


def load_data_manifest_items(
    max_samples: int | None = None,
    seed: int = 42,
    benchmark: str | None = None,
    **_,
) -> list[dict]:
    """Load a normalized benchmark from the canonical `data/<bench>/<bench>.jsonl`
    layout produced by `data/prepare_all_benchmarks.py` (see data/benchmarks_manifest.json).

    Every row has a uniform schema {id, prompt, benchmark, split_type, ...extras}.
    This is the single source of truth for the full ASR/ORR benchmark suite; the
    asr_full / orr_full splits below source exclusively from it (no HF download at
    runtime, no subsampling)."""
    if not benchmark:
        raise ValueError("load_data_manifest_items requires benchmark=<name>")
    path = _data_dir() / benchmark / f"{benchmark}.jsonl"
    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            prompt = str(r.get("prompt", "")).strip()
            if not prompt:
                continue
            extras = {
                k: v
                for k, v in r.items()
                if k not in {"id", "prompt", "benchmark", "split_type", "category"}
            }
            items.append(
                {
                    "sample_id": r.get("id", len(items)),
                    "prompt": prompt,
                    "category": str(r.get("category", benchmark)),
                    **extras,
                }
            )
    return _subsample(items, max_samples, seed)


def load_math500_items(max_samples: int | None = None, seed: int = 42, cache_dir: str | None = None, **_):
    """MATH-500 (500 items). Task prompt = raw problem text."""
    load_dataset = _load_hf_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=cache_dir)
    items = []
    for i, row in enumerate(ds):
        items.append({
            "sample_id": i,
            "prompt": str(row["problem"]).strip(),
            "category": str(row.get("subject", "math")).lower().replace(" ", "_"),
            "level": str(row.get("level", "")),
            "gold_answer": str(row.get("answer", "")),
        })
    return _subsample(items, max_samples, seed)


def load_gpqa_diamond_items(max_samples: int | None = None, seed: int = 42, **_):
    """GPQA-Diamond (198 items). Task prompt already includes multiple-choice a/b/c/d."""
    from datasets import Dataset
    path = "/scratch/gpfs/ARORA/nr3764/latent_reasoning/general_inference_eval/benchmarks/gpqa/gpqa.ds"
    ds = Dataset.load_from_disk(path)
    items = []
    for i, row in enumerate(ds):
        items.append({
            "sample_id": int(row.get("id", i)),
            "prompt": str(row["problem"]).strip(),
            "category": "gpqa_diamond",
            "gold_answer": str(row.get("answer", "")),
        })
    return _subsample(items, max_samples, seed)


SPLIT_SPECS: dict[str, list[tuple[str, Callable[..., list[dict]], dict[str, Any]]]] = {
    # Main body suite (paper tab:eval_datasets), sourced from the shipped data/
    # manifests so the from-scratch path needs no extra raw benchmark files.
    "asr": [
        ("wildjailbreak", load_data_manifest_items, {"benchmark": "wildjailbreak"}),
        ("fortress", load_data_manifest_items, {"benchmark": "fortress"}),
    ],
    "orr": [
        ("or_bench", load_data_manifest_items, {"benchmark": "or_bench"}),
        ("falsereject", load_data_manifest_items, {"benchmark": "falsereject"}),
        ("coconot", load_data_manifest_items, {"benchmark": "coconot"}),
    ],
    "orr_supplementary": [
        ("phtest_harmless", load_phtest_items, {}),
        ("orfuzzset", load_orfuzzset_items, {}),
    ],
    "math_gpqa": [
        ("math500", load_math500_items, {}),
        ("gpqa_diamond", load_gpqa_diamond_items, {}),
    ],
    # Full ASR/ORR suite sourced from the canonical data/ manifest (no HF at runtime,
    # no subsampling). asr_full = 7 benchmarks (4,273), orr_full = 3 benchmarks (2,885).
    "asr_full": [
        ("advbench", load_data_manifest_items, {"benchmark": "advbench"}),
        ("harmbench", load_data_manifest_items, {"benchmark": "harmbench"}),
        ("strongreject", load_data_manifest_items, {"benchmark": "strongreject"}),
        ("sorrybench", load_data_manifest_items, {"benchmark": "sorrybench"}),
        ("jailbreakbench", load_data_manifest_items, {"benchmark": "jailbreakbench"}),
        ("wildjailbreak", load_data_manifest_items, {"benchmark": "wildjailbreak"}),
        ("fortress", load_data_manifest_items, {"benchmark": "fortress"}),
        ("hexphi", load_data_manifest_items, {"benchmark": "hexphi"}),
    ],
    "orr_full": [
        ("or_bench", load_data_manifest_items, {"benchmark": "or_bench"}),
        ("falsereject", load_data_manifest_items, {"benchmark": "falsereject"}),
        ("coconot", load_data_manifest_items, {"benchmark": "coconot"}),
        ("xstest_safe", load_data_manifest_items, {"benchmark": "xstest_safe"}),
    ],
}


def build_split_records(
    split: str,
    *,
    max_samples: int | None = None,
    seed: int = 42,
    cache_dir: str | None = None,
    sample_limit: int | None = None,
) -> list[dict]:
    if split not in SPLIT_SPECS:
        raise KeyError(f"Unknown split: {split}")

    records: list[dict] = []
    for benchmark, loader, extra_kwargs in SPLIT_SPECS[split]:
        kwargs = dict(extra_kwargs)
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        kwargs["max_samples"] = max_samples
        kwargs["seed"] = seed
        items = loader(**kwargs)
        for idx, item in enumerate(items):
            source_meta = {
                k: v
                for k, v in item.items()
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

    if sample_limit is not None and sample_limit > 0:
        records = records[:sample_limit]
        for global_id, record in enumerate(records):
            record["global_id"] = global_id

    return records


def write_split_records(
    split: str,
    output_dir: str | Path,
    *,
    max_samples: int | None = None,
    seed: int = 42,
    cache_dir: str | None = None,
    sample_limit: int | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path, list[dict]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split}.jsonl"
    meta_path = output_dir / f"{split}.meta.json"

    if out_path.exists() and not overwrite:
        return out_path, meta_path, read_jsonl(out_path)

    records = build_split_records(
        split,
        max_samples=max_samples,
        seed=seed,
        cache_dir=cache_dir,
        sample_limit=sample_limit,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    meta = {
        "split": split,
        "total": len(records),
        "by_benchmark": dict(Counter(r["benchmark"] for r in records)),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return out_path, meta_path, records


def read_jsonl(path: str | Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items
