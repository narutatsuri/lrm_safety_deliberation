"""Shared helpers for NeurIPS final reanalysis campaign.

Provenance: copied verbatim (logic unchanged) from the research repo
scripts/neurips_final/common.py. The MODELS registry + 4-guardrail majority
vote kept here are superseded by the shared `lrm_safety_deliberation` library
(`lrm_safety_deliberation.models` for the registry/decoding, `lrm_safety_deliberation.guardrails` for the
classifiers + vote); this copy is retained only so the pipeline drivers run
standalone. Set LRM_SAFETY_ARTIFACTS to relocate the artifact tree.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
OUT_ROOT = ROOT / "eval_results" / "neurips_final"

# Suite → output tree + per-split choice. "safety" is the original campaign.
SUITES: dict[str, dict] = {
    "safety": {
        "out_root": ROOT / "eval_results" / "neurips_final",
        "splits": ["asr", "orr"],
        "intents": [("harmful", "asr"), ("benign", "orr")],
    },
    "pretrained": {
        "out_root": ROOT / "eval_results" / "neurips_final_pretrained",
        "splits": ["asr", "orr"],
        "intents": [("harmful", "asr"), ("benign", "orr")],
    },
    "negctrl": {
        "out_root": ROOT / "eval_results" / "neurips_final_negctrl",
        "splits": ["math_gpqa"],
        # split-level intents: "math" (MATH-500) + "gpqa" (GPQA-Diamond)
        "intents": [("math", "math_gpqa"), ("gpqa", "math_gpqa")],
    },
    "supplementary": {
        "out_root": ROOT / "eval_results" / "neurips_final_supplementary",
        "splits": ["orr_supplementary"],
        # PHTest-harmless + ORFuzzSet ship in one split file; post-split by benchmark.
        "intents": [("phtest", "orr_supplementary"), ("orfuzz", "orr_supplementary")],
    },
    "defense": {
        "out_root": ROOT / "eval_results" / "defenses_eval",
        "splits": ["asr", "orr"],
        "intents": [("harmful", "asr"), ("benign", "orr")],
    },
}

MODELS: dict[str, dict[str, Any]] = {
    "Qwen3-8B": {
        "model_path": "models/Qwen3-8B",
        "size_class": "8b",
        "batch_size_hf": 8,
        "max_model_len": 24576,  # universal protocol 2026-05-16
        "max_new_tokens": 16384,  # universal protocol 2026-05-16
        "gpu_memory_utilization": 0.90,
        "is_channel": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    },
    "Qwen3-32B": {
        "model_path": "models/Qwen3-32B",
        "size_class": "32b",
        "batch_size_hf": 1,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.88,
        "is_channel": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    },
    "Olmo-3-7B-Think": {
        "model_path": "models/Olmo-3-7B-Think",
        "size_class": "8b",
        "batch_size_hf": 8,
        "max_model_len": 16384,
        "gpu_memory_utilization": 0.90,
        "is_channel": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95},
    },
    "Phi-4-reasoning": {
        "model_path": "models/Phi-4-reasoning",
        "size_class": "8b",
        "batch_size_hf": 8,
        "max_model_len": 24576,  # universal protocol 2026-05-16: 16K think + 8K prompt/response/headroom
        "max_new_tokens": 16384,  # universal protocol 2026-05-16: max thinking budget; force-close past this
        "gpu_memory_utilization": 0.85,
        "is_channel": False,
        "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
    },
    "GPT-OSS-20B": {
        "model_path": "models/GPT-OSS-20B",
        "size_class": "20b",
        "batch_size_hf": 2,
        "max_model_len": 16384,
        "gpu_memory_utilization": 0.88,
        "is_channel": True,
        "stream_with_cache": True,
        "stream_chunk_size": 128,
        "sampling": {"temperature": 1.0, "top_p": 1.0},
    },
    "DeepSeek-R1-0528-Qwen3-8B": {
        "model_path": "models/DeepSeek-R1-0528-Qwen3-8B",
        "size_class": "8b",
        "batch_size_hf": 8,
        "max_model_len": 16384,
        "gpu_memory_utilization": 0.90,
        "is_channel": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95},
    },
    "DeepSeek-R1-Distill-Llama-8B": {
        "model_path": "models/DeepSeek-R1-Distill-Llama-8B",
        "size_class": "8b",
        "batch_size_hf": 8,
        "max_model_len": 16384,
        "gpu_memory_utilization": 0.90,
        "is_channel": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95},
    },
    "Phi-4-reasoning-plus": {
        "model_path": "models/Phi-4-reasoning-plus",
        "size_class": "8b",
        "batch_size_hf": 8,
        "max_model_len": 16384,
        "gpu_memory_utilization": 0.75,
        "is_channel": False,
        "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
    },
    "Qwen3.5-9B": {
        "model_path": "models/Qwen3.5-9B",
        "size_class": "8b",
        "batch_size_hf": 8,
        "max_model_len": 16384,
        "gpu_memory_utilization": 0.85,
        "is_channel": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    },
    "TARS-7B": {
        "model_path": "models/TARS-7B",
        "size_class": "8b",
        "batch_size_hf": 8,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.90,
        "is_channel": False,
        "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
    },
    # -------- Pretrained / base models --------
    "Qwen3-8B-Base": {
        "model_path": "models/Qwen3-8B-Base",
        "size_class": "8b",
        "batch_size_hf": 8,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.90,
        "is_channel": False,
        "is_pretrained": True,
        "sampling": {"temperature": 0.6, "top_p": 0.95},
    },
    "OLMo-3-1025-7B": {
        "model_path": "models/OLMo-3-1025-7B",
        "size_class": "8b",
        "batch_size_hf": 8,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.90,
        "is_channel": False,
        "is_pretrained": True,
        "sampling": {"temperature": 0.6, "top_p": 0.95},
    },
    "phi-4": {
        "model_path": "models/phi-4",
        "size_class": "14b",
        "batch_size_hf": 4,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.90,
        "is_channel": False,
        "is_pretrained": True,
        "sampling": {"temperature": 0.6, "top_p": 0.95},
    },
    "pythia-6.9b": {
        "model_path": "models/pythia-6.9b",
        "size_class": "7b",
        "batch_size_hf": 8,
        "max_model_len": 2048,  # Pythia's training ctx
        "max_new_tokens": 512,  # leaves ~1472 for prompt in 2048 window
        "gpu_memory_utilization": 0.90,
        "is_channel": False,
        "is_pretrained": True,
        "sampling": {"temperature": 0.6, "top_p": 0.95},
    },
    "Mistral-7B-v0.1": {
        "model_path": "models/Mistral-7B-v0.1",
        "size_class": "7b",
        "batch_size_hf": 8,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.90,
        "is_channel": False,
        "is_pretrained": True,
        "sampling": {"temperature": 0.6, "top_p": 0.95},
    },
    "Llama-3.1-8B": {
        "model_path": "models/Llama-3.1-8B",
        "size_class": "8b",
        "batch_size_hf": 8,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.90,
        "is_channel": False,
        "is_pretrained": True,
        "sampling": {"temperature": 0.6, "top_p": 0.95},
    },
    "safelm-1.7b": {
        "model_path": "models/safelm-1.7b",
        "size_class": "2b",
        "batch_size_hf": 16,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.85,
        "is_channel": False,
        "is_pretrained": True,
        "sampling": {"temperature": 0.6, "top_p": 0.95},
    },
    # -------- Defense baselines (defense suite) --------
    # Each entry = a finished defense × base-model checkpoint, materialized
    # as a standalone HF model on disk (LoRA-merged where applicable). The
    # `is_channel`, sampling, and batch settings are inherited from the base
    # model's entry above. See .agent/defenses/EVAL_PLAN_2026-05-01.md.
    "STAR1-Qwen3-8B": {
        "model_path": "models/final/STAR1-Qwen3-8B",
        "size_class": "8b", "batch_size_hf": 8, "max_model_len": 16384,
        "gpu_memory_utilization": 0.90, "is_channel": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    },
    "STAR1-Olmo-3-7B-Think": {
        "model_path": "models/final/STAR1-Olmo-3-7B-Think",
        "size_class": "8b", "batch_size_hf": 8, "max_model_len": 16384,
        "gpu_memory_utilization": 0.90, "is_channel": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95},
    },
    "STAR1-GPT-OSS-20B": {
        "model_path": "models/final/STAR1-GPT-OSS-20B",
        "size_class": "20b", "batch_size_hf": 2, "max_model_len": 8192,
        "gpu_memory_utilization": 0.88, "is_channel": True,
        "stream_with_cache": True, "stream_chunk_size": 128,
        "sampling": {"temperature": 1.0, "top_p": 1.0},
    },
    "SafeKey-Qwen3-8B": {
        "model_path": "models/final/SafeKey-Qwen3-8B",
        "size_class": "8b", "batch_size_hf": 8, "max_model_len": 24576,
        "max_new_tokens": 16384,  # universal 2026-05-16
        "gpu_memory_utilization": 0.90, "is_channel": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    },
    "SafeKey-Olmo-3-7B-Think": {
        "model_path": "models/final/SafeKey-Olmo-3-7B-Think",
        "size_class": "8b", "batch_size_hf": 8, "max_model_len": 16384,
        "gpu_memory_utilization": 0.90, "is_channel": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95},
    },
    "SafeKey-Phi-4-reasoning": {
        "model_path": "models/final/SafeKey-Phi-4-reasoning",
        "size_class": "14b", "batch_size_hf": 4, "max_model_len": 16384,
        "gpu_memory_utilization": 0.90, "is_channel": False,
        "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
    },
    "SafeKey-GPT-OSS-20B": {
        "model_path": "models/final/SafeKey-GPT-OSS-20B",
        "size_class": "20b", "batch_size_hf": 2, "max_model_len": 8192,
        "gpu_memory_utilization": 0.88, "is_channel": True,
        "stream_with_cache": True, "stream_chunk_size": 128,
        "sampling": {"temperature": 1.0, "top_p": 1.0},
    },
    "R1ACT-Qwen3-8B": {
        "model_path": "models/final/R1ACT-Qwen3-8B",
        "size_class": "8b", "batch_size_hf": 8, "max_model_len": 16384,
        "gpu_memory_utilization": 0.90, "is_channel": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    },
    "R1ACT-Olmo-3-7B-Think": {
        "model_path": "models/final/R1ACT-Olmo-3-7B-Think",
        "size_class": "8b", "batch_size_hf": 8, "max_model_len": 16384,
        "gpu_memory_utilization": 0.90, "is_channel": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95},
    },
    "R1ACT-Phi-4-reasoning": {
        "model_path": "models/final/R1ACT-Phi-4-reasoning",
        "size_class": "14b", "batch_size_hf": 4, "max_model_len": 16384,
        "gpu_memory_utilization": 0.90, "is_channel": False,
        "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
    },
    "R1ACT-GPT-OSS-20B": {
        "model_path": "models/final/R1ACT-GPT-OSS-20B",
        "size_class": "20b", "batch_size_hf": 2, "max_model_len": 8192,
        "gpu_memory_utilization": 0.88, "is_channel": True,
        "stream_with_cache": True, "stream_chunk_size": 128,
        "sampling": {"temperature": 1.0, "top_p": 1.0},
    },
    "ThinkSafe-Qwen3-8B": {
        "model_path": "models/final/ThinkSafe-Qwen3-8B",
        "size_class": "8b", "batch_size_hf": 8, "max_model_len": 24576,
        "max_new_tokens": 16384,  # universal 2026-05-16
        "gpu_memory_utilization": 0.90, "is_channel": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    },
    "ThinkSafe-Olmo-3-7B-Think": {
        "model_path": "models/final/ThinkSafe-Olmo-3-7B-Think",
        "size_class": "8b", "batch_size_hf": 8, "max_model_len": 16384,
        "gpu_memory_utilization": 0.90, "is_channel": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95},
    },
    "RAPO-Qwen3-8B": {
        "model_path": "models/final/RAPO-Qwen3-8B",
        "size_class": "8b", "batch_size_hf": 8, "max_model_len": 16384,
        "gpu_memory_utilization": 0.90, "is_channel": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    },
    "RAPO-Olmo-3-7B-Think": {
        "model_path": "models/final/RAPO-Olmo-3-7B-Think",
        "size_class": "8b", "batch_size_hf": 8, "max_model_len": 16384,
        "gpu_memory_utilization": 0.90, "is_channel": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95},
    },
    "RAPO-Phi-4-reasoning": {
        "model_path": "models/final/RAPO-Phi-4-reasoning",
        "size_class": "14b", "batch_size_hf": 4, "max_model_len": 24576,
        "max_new_tokens": 16384,  # universal 2026-05-16; force-close handles RAPO runaway-loop
        "gpu_memory_utilization": 0.85, "is_channel": False,
        "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
    },
    "RAPO-GPT-OSS-20B": {
        "model_path": "models/final/RAPO-GPT-OSS-20B",
        "size_class": "20b", "batch_size_hf": 2, "max_model_len": 8192,
        "gpu_memory_utilization": 0.88, "is_channel": True,
        "stream_with_cache": True, "stream_chunk_size": 128,
        "sampling": {"temperature": 1.0, "top_p": 1.0},
    },
    "STAR1-Phi-4-reasoning": {
        "model_path": "models/final/STAR1-Phi-4-reasoning",
        "size_class": "14b", "batch_size_hf": 4, "max_model_len": 16384,
        "gpu_memory_utilization": 0.90, "is_channel": False,
        "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
    },
    "STAIR-Qwen3-8B": {
        "model_path": "models/final/STAIR-Qwen3-8B",
        "size_class": "8b", "batch_size_hf": 8, "max_model_len": 16384,
        "gpu_memory_utilization": 0.90, "is_channel": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    },
    "STAIR-Olmo-3-7B-Think": {
        "model_path": "models/final/STAIR-Olmo-3-7B-Think",
        "size_class": "8b", "batch_size_hf": 8, "max_model_len": 16384,
        "gpu_memory_utilization": 0.90, "is_channel": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95},
    },
    "STAIR-Phi-4-reasoning": {
        "model_path": "models/final/STAIR-Phi-4-reasoning",
        "size_class": "14b", "batch_size_hf": 4, "max_model_len": 24576,
        "max_new_tokens": 16384,  # universal 2026-05-16
        "gpu_memory_utilization": 0.85, "is_channel": False,
        "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
    },
    "STAIR-GPT-OSS-20B": {
        "model_path": "models/final/STAIR-GPT-OSS-20B",
        "size_class": "20b", "batch_size_hf": 2, "max_model_len": 8192,
        "gpu_memory_utilization": 0.88, "is_channel": True,
        "stream_with_cache": True, "stream_chunk_size": 128,
        "sampling": {"temperature": 1.0, "top_p": 1.0},
    },
    "ThinkSafe-Phi-4-reasoning": {
        "model_path": "models/final/ThinkSafe-Phi-4-reasoning",
        "size_class": "14b", "batch_size_hf": 4, "max_model_len": 24576,
        "max_new_tokens": 16384,  # universal 2026-05-16
        "gpu_memory_utilization": 0.85, "is_channel": False,
        "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
    },
    "ThinkSafe-GPT-OSS-20B": {
        "model_path": "models/final/ThinkSafe-GPT-OSS-20B",
        "size_class": "20b", "batch_size_hf": 2, "max_model_len": 8192,
        "gpu_memory_utilization": 0.88, "is_channel": True,
        "stream_with_cache": True, "stream_chunk_size": 128,
        "sampling": {"temperature": 1.0, "top_p": 1.0},
    },
}

DEFENSE_MODELS = [
    "STAR1-Qwen3-8B", "STAR1-Olmo-3-7B-Think", "STAR1-Phi-4-reasoning", "STAR1-GPT-OSS-20B",
    "SafeKey-Qwen3-8B", "SafeKey-Olmo-3-7B-Think", "SafeKey-Phi-4-reasoning", "SafeKey-GPT-OSS-20B",
    "R1ACT-Qwen3-8B", "R1ACT-Olmo-3-7B-Think", "R1ACT-Phi-4-reasoning", "R1ACT-GPT-OSS-20B",
    "ThinkSafe-Qwen3-8B", "ThinkSafe-Olmo-3-7B-Think", "ThinkSafe-Phi-4-reasoning", "ThinkSafe-GPT-OSS-20B",
    "RAPO-Qwen3-8B", "RAPO-Olmo-3-7B-Think", "RAPO-Phi-4-reasoning", "RAPO-GPT-OSS-20B",
    "STAIR-Qwen3-8B", "STAIR-Olmo-3-7B-Think", "STAIR-Phi-4-reasoning", "STAIR-GPT-OSS-20B",
]

SAFETY_MODELS = [
    "Qwen3-8B", "Qwen3-32B", "Olmo-3-7B-Think", "Phi-4-reasoning",
    "GPT-OSS-20B", "DeepSeek-R1-0528-Qwen3-8B", "DeepSeek-R1-Distill-Llama-8B",
    "Phi-4-reasoning-plus", "Qwen3.5-9B",
]
PRETRAINED_MODELS = [
    "Qwen3-8B-Base", "OLMo-3-1025-7B", "phi-4",
    "pythia-6.9b", "Mistral-7B-v0.1", "Llama-3.1-8B",
    "safelm-1.7b",
]
NEGCTRL_MODELS = SAFETY_MODELS  # math/gpqa negative control uses thinking models

SUITES["safety"]["models"] = SAFETY_MODELS
SUITES["pretrained"]["models"] = PRETRAINED_MODELS
SUITES["negctrl"]["models"] = NEGCTRL_MODELS
SUITES["supplementary"]["models"] = SAFETY_MODELS  # thinking models on PHTest + ORFuzzSet


def suite_out_root(suite: str) -> Path:
    return SUITES[suite]["out_root"]


def suite_models(suite: str) -> list[str]:
    return SUITES[suite]["models"]


def suite_intents(suite: str) -> list[tuple[str, str]]:
    """Return list of (intent_name, split_name). For safety/pretrained both
    intents map to distinct splits (asr/orr). For negctrl, both intents share
    the math_gpqa split but are separated post-hoc by benchmark."""
    return SUITES[suite]["intents"]

ASR_BENCHMARKS = {"wj2k", "fortress"}
ORR_BENCHMARKS = {"or_bench", "false_reject_test", "coconot_benign"}


def model_out_dir(model_name: str) -> Path:
    return OUT_ROOT / model_name


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def majority_tie_negative(votes: list[bool]) -> bool:
    if not votes:
        return False
    return sum(1 for v in votes if v) > len(votes) / 2.0
